import os
import openai
from openai import OpenAI
import sys
import time
import logging
from tqdm import tqdm
from typing import Union, Tuple
import re
import numpy as np
from transformers import AutoTokenizer

from rag.language_model.base_lm import BaseLM
import tiktoken


class OpenaiModel(BaseLM):
    def __init__(self,args):
        super().__init__(args)
        self.generation_stop = args.generation_stop
        if self.generation_stop == '':
            self.generation_stop = None
        self.llm_name = args.llm_name
        self.api_key_path = args.api_key_path
        self.api_base = args.api_base
        self.api_logprobs = args.api_logprobs
        self.api_top_logprobs = args.api_top_logprobs

    def load_model(self):
        # load api key
        key_path = self.api_key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        self.client = OpenAI(api_key=api_key.strip(),base_url=self.api_base)
        print(f'current api base: {self.api_base}')
        print(f"Current API: {api_key.strip()}")
        if 'gpt' in self.llm_name:
            self.chat=True
            self.tokenizer = tiktoken.encoding_for_model(self.llm_name)
            self.tokenizer_type='tiktoken'
        else:
            self.chat=False
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.tokenizer_type = 'hf'

    def generate(self, inputs: Union[str,list[str]], **kwargs)-> list[BaseLM.Outputs]:
        '''
        Current version of OpenaiModel batch inference was not implemented
        '''
        if isinstance(inputs,str):
            inputs = [inputs]
        apioutputs_list = []
        for input_text in inputs:
            # original_text = input_text
            # input_text, is_modified = self.remove_sensitive_words(input_text)
            
            # if is_modified:
            #     logging.info(f"Remove sensitive text! origanl text: {original_text[:50]}... revised text: {input_text[:50]}...")

            if self.api_logprobs is False:
                response = self.call_llm(input_text, model_name=self.llm_name, max_len=self.generate_maxlength, temp=self.temperature, top_p=self.top_p, stop = self.generation_stop, **kwargs)
                # collate Apioutputs
                if response is None:
                    continue
                apioutput = self.Outputs()
                apioutput.text = response["choices"][0]["message"]["content"]
                apioutput.tokens_ids = self.tokenizer.encode(apioutput.text, add_special_tokens=False)
                apioutput.prompt_tokens_num = response["usage"]["prompt_tokens"]
                apioutput.tokens_num = len(apioutput.tokens_ids)
                apioutputs_list.append(apioutput)
            else:
                for i in range(1,50): # max time of recall is 10 times
                    print(f'The {i}-th API call')
                    response = self.call_llm(input_text, model_name=self.llm_name, max_len=self.generate_maxlength, temp=self.temperature, top_p=self.top_p, stop = self.generation_stop, logprobs=self.api_logprobs, top_logprobs=self.api_top_logprobs, **kwargs)
                    # collate Apioutputs
                    if response is None:
                        continue
                    if 'logprobs' in response["choices"][0]:
                        if response["choices"][0]['logprobs'] is not None:
                            apioutput = self.Outputs()
                            apioutput.text = response["choices"][0]["message"]["content"]
                            if self.tokenizer_type == 'tiktoken':
                                filtered=[]
                                apioutput.tokens_ids = []
                                for c in response["choices"][0]['logprobs']['content']:
                                    try:
                                        apioutput.tokens_ids.append(self.tokenizer.encode_single_token(c['token']))
                                        filtered.append(c)
                                    except KeyError:
                                        print(f"Token '{c['token']}' not found in vocabulary")
                                        continue
                                response["choices"][0]['logprobs']['content']=filtered

                            elif self.tokenizer_type == 'hf':
                                apioutput.tokens_ids = [self.tokenizer.convert_tokens_to_ids(content['token']) for
                                                        content in response["choices"][0]['logprobs']['content']]
                            else:
                                raise NotImplementedError
                            apioutput.tokens_num = len(apioutput.tokens_ids)
                            apioutput.prompt_tokens_num = response["usage"]["prompt_tokens"]
                            apioutput.tokens = [content['token'] for content in response["choices"][0]['logprobs']['content']]
                            apioutput.tokens_logprob = [content['logprob'] for content in response["choices"][0]['logprobs']['content']]
                            apioutput.tokens_prob = np.exp(apioutput.tokens_logprob).tolist()
                            # seems a bug, we modify it
                            # apioutput.cumulative_logprob = float(np.prod(apioutput.tokens_prob) / max(len(apioutput.tokens_prob), 1))
                            apioutput.cumulative_logprob = float(np.sum(apioutput.tokens_logprob))
                            apioutput.logprobs = []
                            apioutput.text_logprobs = []
                            for content in response["choices"][0]['logprobs']['content']: # content:dict[token/logprobs/top_logprobs] 每个content都包含一个 token 的信息
                                top_logprobs = content['top_logprobs']
                                one_token_vocab = {}
                                text_token_vocab = {}
                                for log_prob in top_logprobs: # top_logprobs:list[dict[token/logprobs/bytes]]
                                    token_str = log_prob['token']
                                    try:
                                        if self.tokenizer_type == 'tiktoken':
                                            token_id = self.tokenizer.encode_single_token(token_str)
                                        elif self.tokenizer_type == 'hf':
                                            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
                                        else:
                                            raise NotImplementedError
                                    except KeyError:
                                        print(f"Token '{token_str}' not found in vocabulary")
                                        continue
                                    token_logprob = log_prob['logprob']
                                    one_token_vocab[token_id] = float(token_logprob)
                                    text_token_vocab[token_str] = float(token_logprob)
                                apioutput.logprobs.append(one_token_vocab)
                                apioutput.text_logprobs.append(text_token_vocab)
                            assert len(apioutput.tokens_ids)==len(apioutput.tokens)==apioutput.tokens_num, [len(apioutput.tokens_ids),len(apioutput.tokens),apioutput.tokens_num,apioutput.text,apioutput.tokens]
                            # end of for loop
                            apioutputs_list.append(apioutput)
                            print(f'API call success')
                            break
                        else:
                            pass # logprob is None so recall chatgpt in next turn
                    else:
                        pass
                # --> end of recall loop
            # --> end of else
        # --> end of main loop
        return apioutputs_list

    def call_llm(self, input_text, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, top_p = 1.0, stop = None, logprobs = False, top_logprobs = None, verbose=False, **kwargs):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                if self.chat:
                    message = [{"role": "user", "content": input_text}]
                    response = self.client.chat.completions.create(model=model_name,
                                                            messages=message,
                                                            max_tokens=max_len,
                                                            temperature=temp,
                                                            top_p = top_p,
                                                            stop = stop,
                                                            logprobs = logprobs,
                                                            top_logprobs = top_logprobs,
                                                            seed = 2024,
                                                            extra_body=kwargs)
                    response = response.model_dump()
                else:
                    if not logprobs:
                        top_logprobs = 0
                    response = self.client.completions.create(model=model_name,
                                                            prompt=input_text,
                                                            max_tokens=max_len,
                                                            temperature=temp,
                                                            top_p = top_p,
                                                            stop = stop,
                                                            logprobs = top_logprobs,
                                                            seed = 2024,
                                                            extra_body=kwargs)
                    response = response.model_dump()
                    response = self.convert_legacy_to_chatcompletion(response)

                received = True
            except:
                num_rate_errors += 1
                error = sys.exc_info()[0]
                if error == openai.BadRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    return None
                else:
                    print(error)
                logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
                time.sleep(np.power(2, num_rate_errors))
        return response

    def convert_legacy_to_chatcompletion(self, legacy_response: dict) -> dict:
        chat_response = {
            "id": legacy_response["id"].replace("cmpl", "chatcmpl"),
            "object": "chat.completion",
            "created": legacy_response["created"],
            "model": legacy_response["model"],  # 模型保持不变
            "choices": [],
            "usage": legacy_response.get("usage", {})  # 保留 usage 信息
        }

        # 处理choices字段，将每个choice转换为ChatCompletion格式
        for choice in legacy_response["choices"]:
            chat_choice = {
                "index": choice["index"],
                "message": {
                    "role": "assistant",
                    "content": choice["text"].strip()
                },
                "finish_reason": choice.get("finish_reason", "stop")
            }


            if choice.get("logprobs") is not None:
                chat_choice["logprobs"] = self.convert_logprobs(choice["logprobs"])

            chat_response["choices"].append(chat_choice)

        return chat_response

    def convert_logprobs(self, legacy_logprobs: dict) -> dict:
        chat_logprobs = {"content": []}

        for token, logprob, top_logprobs, offset in zip(
                legacy_logprobs["tokens"],
                legacy_logprobs["token_logprobs"],
                legacy_logprobs["top_logprobs"],
                legacy_logprobs["text_offset"]
        ):

            token_bytes = list(token.encode("utf-8"))


            token_logprob_entry = {
                "token": token,
                "logprob": logprob,
                "bytes": token_bytes,
                "top_logprobs": []
            }


            for top_token, top_token_logprob in top_logprobs.items():
                token_logprob_entry["top_logprobs"].append({
                    "token": top_token,
                    "logprob": top_token_logprob,
                    "bytes": list(top_token.encode("utf-8"))
                })

            chat_logprobs["content"].append(token_logprob_entry)

        return chat_logprobs