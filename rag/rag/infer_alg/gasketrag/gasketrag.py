import re
import time
from collections import defaultdict
from typing import Optional, Any

import spacy
from sympy.core.evalf import fastlog
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from openai import OpenAI

from rag.rag.infer_alg.naive_rag.naiverag import NaiveRag


class GasketRAG(NaiveRag):
    def __init__(self, args):
        super().__init__(args)
        self.gasket_model_type = args.gasket_model_type
        self.gasket_model_name = args.gasket_model_name
        self.gasket_iter_num = args.gasket_iter_num
        self.api_key_path = args.api_key_path
        self.gasket_base = args.gasket_base if hasattr(args, 'gasket_base') else 'https://api.openai.com/v1/'
        if self.gasket_model_type == 'Openai_api':
            with open(args.api_key_path, 'r') as f:
                api_key = f.readline()
            self.client = OpenAI(api_key=api_key.strip(), base_url=self.gasket_base)
        elif self.gasket_model_type == 'vllm':
            self.samplingparams = SamplingParams(top_k=1, max_tokens=200)
            if hasattr(args, 'gasket_adapter_path'):
                self.lora_request = LoRARequest("gasket", 1, lora_path=args.gasket_adapter_path)
            else:
                self.lora_request = None
            self.vllm = LLM(model=self.gasket_model_name, enable_lora=True, gpu_memory_utilization=0.6)
        else:
            raise NotImplementedError
        self.nlp = spacy.load("en_core_web_sm")

    def call_gasket(self, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        if self.gasket_model_type == 'Openai_api':
            max_retries = 10
            wait_time = 10
            attempt = 0
            while attempt < max_retries:
                try:
                    responses = self.client.chat.completions.create(model=self.gasket_model_name,
                                                                    messages=messages,
                                                                    temperature=0,
                                                                    max_tokens=200, )
                    responses = [responses.choices[0].message.content.strip()]
                    return responses
                except Exception as e:
                    attempt += 1
                    print(e)
                    if attempt < max_retries:
                        time.sleep(wait_time)
                    else:
                        raise e

        else:
            if self.lora_request is None:
                responses = self.vllm.chat(messages, sampling_params=self.samplingparams)
            else:
                responses = self.vllm.chat(messages, sampling_params=self.samplingparams,
                                           lora_request=self.lora_request)
            return [responses[0].outputs[0].text.strip()]

    def post_process_choices(self, choices):
        sids_list = []
        sids_strings = []
        sids_set = set()
        pattern = r's_\d+'
        for c in choices:
            c = c.lower().strip()
            sids = re.findall(pattern, c)
            sids = list(dict.fromkeys(sids))
            sids_string = ','.join(sids)
            if sids_string not in sids_set:
                sids_set.add(sids_string)
                sids_list.append(sids)
                sids_strings.append(sids_string)
        return sids_list, sids_strings

    def split_sents(self, passages):
        p_sents = {}
        offset = 0
        for pid, p in enumerate(passages):
            doc = self.nlp(p)
            sents = {f's_{sid + offset}': s.text for sid, s in enumerate(doc.sents)}
            p_sents[f'p_{pid}'] = sents
            offset += len(sents)
        return p_sents

    def selection_phase_prompt_gen(self, query, p_sents):
        template = (
            'Review the passages provided and select the snippet IDs that give a comprehensive background for the posed question. Each snippet ID is marked at the start of the sentence within square brackets. Please list the IDs in a comma-separated format. For example, the output should look like this: s_17,s_22,s_46.\n'
            '### Passages\n'
            '[[P]]\n'
            '### Question\n'
            '[[Q]]\n'
            '### Snippet IDs\n')
        processed_passages = ''
        for pid in p_sents:
            p = p_sents[pid]
            for sid in p:
                s = p[sid]
                joined = f' [{sid}] {s} '
                processed_passages = processed_passages + joined
            processed_passages = processed_passages + '\n'
        return template.replace('[[P]]', processed_passages).replace('[[Q]]', query)

    def retrieve_for_gasket(self, query):
        passages = self.retrieval.search(query)
        precessed_passages = []
        for i, p in passages.items():
            text = p['text']
            title = p['title']
            p = f'Title: {title}. {text}'
            precessed_passages.append(p)
        return precessed_passages

    def gasket_inter(self, query, aug_query):
        passages = self.retrieve_for_gasket(aug_query)
        p_sents = self.split_sents(passages)
        selection_prompt = self.selection_phase_prompt_gen(query, p_sents)
        choices = self.call_gasket(selection_prompt)
        sids_list, _ = self.post_process_choices(choices)
        sids = sids_list[0]
        flatten_p_sents = defaultdict(str)
        for pid in p_sents:
            flatten_p_sents.update(p_sents[pid])
        return flatten_p_sents, sids, passages

    def vanilla_gasketrag(self, query):
        start_time = time.time()
        generation_track = {}
        aug_query = query
        for _ in range(self.gasket_iter_num):
            flatten_p_sents, sids, passages = self.gasket_inter(query, aug_query)
            aug_query = ' '.join([flatten_p_sents[sid] for sid in sids]) + ' ' + query
            selected_sents = '\n'.join([flatten_p_sents[sid] for sid in sids])

        # rag
        # background='\n'.join(passages)+'\n'+'Relevant Sentences:\n'+selected_sents
        target_instruction = self.find_algorithm_instruction('Naive_rag', self.task)
        prompt = target_instruction.format_map({'passages': selected_sents, 'query': query})
        answer = self.llm.generate(prompt)[0].text
        generation_track['final answer'] = answer
        generation_track['time_use'] = time.time() - start_time
        return answer, generation_track

    def iter_gasketrag(self, query):
        # use selected sents and answer to augment query
        start_time = time.time()
        generation_track = {}
        aug_query = query
        for _ in range(self.gasket_iter_num):
            flatten_p_sents, sids, passages = self.gasket_inter(query, aug_query)
            selected_sents = '\n'.join([flatten_p_sents[sid] for sid in sids])
            target_instruction = self.find_algorithm_instruction('Naive_rag', self.task)
            prompt = target_instruction.format_map({'passages': selected_sents, 'query': query})
            answer = self.llm.generate(prompt)[0].text
            aug_query = selected_sents+'\n'+query+'\n'+answer

        # rag
        generation_track['final answer'] = answer
        generation_track['time_use'] = time.time() - start_time
        return answer, generation_track

    def infer(self, query: str) -> tuple[str, dict[str, Any]]:
        answer, generation_track = self.iter_gasketrag(query)
        return answer, generation_track
