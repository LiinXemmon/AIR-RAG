import re
from collections import defaultdict

from datasets import Dataset
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import spacy
import time
import requests
from pprint import pprint


class Labeller:
    def __init__(self, api_key, data_path, out_path, max_samples, gasket_model, gasket_base_url, gasket_use_chat,
                 generator_model, generator_base_url, generator_use_chat, quiet):
        self.api_key = api_key
        self.generator = OpenAI(api_key=api_key, base_url=generator_base_url)
        self.gasket = OpenAI(api_key=api_key, base_url=gasket_base_url)
        self.nlp = spacy.load("en_core_web_sm")
        self.data_path = data_path
        self.out_path = out_path
        self.max_samples = max_samples
        self.gasket_model = gasket_model
        self.gasket_use_chat = gasket_use_chat
        self.generator_model = generator_model
        self.generator_use_chat = generator_use_chat
        self.quiet = quiet

    def labelling(self, query, true_answers, passages):
        p_sents = self.split_sents(passages)
        # select sentences
        selection_prompt = self.selection_phase_prompt_gen(query, p_sents)
        choices = self.call_llm(selection_prompt, 5, 1, 'gasket')
        if choices[0] == '':
            print('LLM returned no sentence selection.')
            return []
        sids_list, sids_strings = self.post_process_choices(choices)
        # generate answers
        answers = []
        for sids in sids_list:
            answer_prompt = self.answer_phase_prompt_gen(query, sids, p_sents)
            answer = self.call_llm(answer_prompt, 1, 0, 'generator')[0]
            answers.append(answer)
        # generate preferences
        preferences = self.compare_answers(query, answers, true_answers)
        hard_level = preferences.count(False)
        labelled_data = []
        if hard_level > 0:
            # re-search using choices leading to right answers
            filtered_idx_and_weight, trues_rank, falses_rank = self.assign_weight(query, sids_list, preferences,
                                                                                  p_sents, true_answers)
            for idx, weight in filtered_idx_and_weight:
                labelled_data.append(
                    {'query': query, 'completion': sids_strings[idx], 'label': preferences[idx],
                     'trues_rank': trues_rank,
                     'falses_rank': falses_rank,
                     'weight': weight,
                     'hard_level': f'{hard_level}/{len(preferences)}',
                     'llm_answer': answers[idx],
                     'true_answers': true_answers,
                     'prompt': selection_prompt})
        if not self.quiet:
            handle_log = {'query': query, 'preferences': preferences, 'hard_level': f'{hard_level}/{len(preferences)}'}
            if hard_level > 0:
                handle_log['trues_rank'] = trues_rank
                handle_log['falses_rank'] = falses_rank
            print('=========================Handle Log========================')
            pprint(handle_log)
            print('===========================================================')

        return labelled_data

    def assign_weight(self, query, sids_list, preferences, p_sents, true_answers):
        if preferences.count(False) == 0:
            return []

        flatten_p_sents = defaultdict(str)
        for pid in p_sents:
            flatten_p_sents.update(p_sents[pid])
        aug_queries = []
        for sids in sids_list:
            aug_query = ' '.join([flatten_p_sents[sid] for sid in sids]) + ' ' + query
            aug_queries.append(aug_query)
        new_passages_list = self.retrieve(aug_queries)

        new_p_sents_list = []
        new_sids_list = []
        for passages in new_passages_list:
            new_p_sents = self.split_sents(passages)
            new_p_sents_list.append(new_p_sents)
            selection_prompt = self.selection_phase_prompt_gen(query, new_p_sents)
            choices = self.call_llm(selection_prompt, 1, 0, 'gasket')
            new_sids, _ = self.post_process_choices(choices)
            new_sids = new_sids[0]
            new_sids_list.append(new_sids)
        assert len(sids_list) == len(new_sids_list) == len(new_passages_list), (
            len(sids_list), len(new_sids_list), len(new_passages_list))

        # for falses: retrieve with augmented query and determine the weight by LLM answer.
        falses = [i for i, x in enumerate(preferences) if x == False]
        new_answers = []
        for idx in falses:
            answer_prompt = self.answer_phase_prompt_gen(query, new_sids_list[idx], new_p_sents_list[idx])
            answer = self.call_llm(answer_prompt, 1, 0, 'generator')[0]
            new_answers.append(answer)
        falses_new_preferences = self.compare_answers(query, new_answers, true_answers)

        # for trues: rank by s_id sum
        trues = [i for i, x in enumerate(preferences) if x == True]
        trues_rank = []
        for idx in trues:
            sids = new_sids_list[idx]
            sum = 0
            for sid in sids:
                n = sid.split('_')[1]
                sum += int(n)
            trues_rank.append((idx, sum / len(sids) if sum != 0 else 0))
        trues_rank.sort(key=lambda x: x[1])

        falses_rank = list(zip(falses, falses_new_preferences))
        falses_rank.sort(key=lambda x: x[1])

        filtered_idx_and_weight = []
        if len(trues_rank) > 0:
            filtered_idx_and_weight.append((trues_rank[0][0], 1.0))
        if len(trues_rank) > 1:
            filtered_idx_and_weight.append((trues_rank[-1][0], 0.5))

        if len(trues_rank) > 0 or falses_rank[-1][1]:
            if len(falses_rank) > 0:
                filtered_idx_and_weight.append((falses_rank[0][0], 0.5 if falses_rank[0][1] else 1.0))
            if len(falses_rank) > 1:
                filtered_idx_and_weight.append((falses_rank[-1][0], 0.5 if falses_rank[-1][1] else 1.0))
        return filtered_idx_and_weight, trues_rank, falses_rank

    def compare_answers(self, query, answers, true_answers):
        preferences = []
        for a in answers:
            if a == '':
                preferences.append(False)
                continue
            if a.strip().lower() in true_answers:
                preferences.append(True)
                continue
            prompt = f'Given a question, determine whether the two answers are consistent, and output True or False.\nQuestion: {query}\nAnswer 1: {a}\nAnswer 2: {true_answers[0]}\nResult:'
            result = self.call_llm(prompt, 1, 0, 'judge_answer')[0]
            result = re.sub(r'[^\w\s]', '', result)
            if 'true' in result.lower().strip():
                preferences.append(True)
            else:
                preferences.append(False)
        return preferences

    def post_process_choices(self, choices):
        sids_list = []
        sids_strings = []
        sids_set = set()
        pattern = r's_\d+'
        for c in choices:
            c = c.lower().strip()
            sids = re.findall(pattern, c)
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

    def answer_phase_prompt_gen(self, query, selected_sents, p_sents):
        # template = ('Provide a brief, direct answer to the question without reasoning or explanation.\n'
        #             '[[E]]\n'
        #             '### Background\n'
        #             '[[B]]\n'
        #             '### Question\n'
        #             '[[Q]]\n'
        #             '### Answer\n')
        # examples = (
        #     '### Background\n'
        #     'Hops are added during boiling as a source of bitterness, flavour and aroma.\n'
        #     'Hops may be added at more than one point during the boil.\n'
        #     'The finishing hops are added last, toward the end of or after the boil.\n'
        #     'In general, hops provide the most flavouring when boiled for approximately 15 minutes, and the most aroma when not boiled at all.\n'
        #     '### Question\n'
        #     'When are hops added to the brewing process?\n'
        #     '### Answer\n'
        #     'The boiling process\n'
        #     '### Background\n'
        #     'The 2017 Cleveland mayoral election took place on November 7, 2017, to elect the Mayor of Cleveland, Ohio.\n'
        #     'The election was officially nonpartisan, with the top two candidates from the September 12 primary election advancing to the general election, regardless of party.\n'
        #     'Incumbent Democratic Mayor Frank G. Jackson won reelection to a fourth term.\n'
        #     '### Question\n'
        #     'Who won the election for mayor of cleveland?\n'
        #     '### Answer\n'
        #     'Incumbent Democratic Mayor Frank G. Jackson\n'
        # )
        template = "### Instruction:\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n [[B]] \n\n## Input:\n\n[[Q]]\n\n### Response:\n"

        flatten_p_sents = defaultdict(str)
        for pid in p_sents:
            flatten_p_sents.update(p_sents[pid])
        processed_sents = '\n'.join([flatten_p_sents[sid] for sid in selected_sents])
        # prompt=template.replace('[[E]]', examples).replace('[[B]]', processed_sents).replace('[[Q]]', query)
        prompt = template.replace('[[B]]', processed_sents).replace('[[Q]]', query)
        return prompt

    def call_llm(self, prompt, n, temperature, calltype):
        max_retries = 10
        wait_time = 10
        attempt = 0
        if calltype == 'gasket':
            client = self.gasket
            model = self.gasket_model
            use_chat = self.gasket_use_chat
        elif calltype == 'generator':
            client = self.generator
            model = self.generator_model
            use_chat = self.generator_use_chat
        elif calltype == 'judge_answer':
            client = OpenAI(api_key=self.api_key)
            model = 'gpt-4o-mini'
            use_chat = True
        while attempt < max_retries:
            try:
                if use_chat:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{'role': 'user', 'content': prompt}],
                        max_tokens=100,
                        n=n,
                        temperature=temperature,
                    )
                    response = [c.message.content.strip() for c in response.choices]
                else:
                    response = client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=100,
                        n=n,
                        temperature=temperature,
                    )
                    response = [c.text.strip() for c in response.choices]
                return response
            except Exception as e:
                attempt += 1
                print(e)
                if attempt < max_retries:
                    time.sleep(wait_time)
        return ['']

    def retrieve(self, query_list):
        passages_list = []
        for q in query_list:
            k = 10
            url = f"http://localhost:8893/api/search?query={q}&k={k}"
            response = requests.get(url)
            response = response.json()
            precessed_response = []
            for i in range(1, k + 1):
                text = response[f'{i}']['text']
                title = response[f'{i}']['title']
                p = f'Title: {title}. {text}'
                precessed_response.append(p)
            passages_list.append(precessed_response)
        return passages_list

    def get_dataloader(self, datapath):
        raise NotImplementedError

    def run_labelling(self, start_from):
        dataloader = self.get_dataloader(self.data_path)
        samples = 0
        with open(self.out_path, "a") as f:
            for i, (query_chunk, answers_chunk) in enumerate(dataloader):
                if i < start_from:
                    continue
                print(f'Start from chunk {i}.')
                passages_chunk = self.retrieve(query_chunk)
                with ThreadPoolExecutor(max_workers=40) as executor:
                    labelled_data = list(
                        tqdm(executor.map(self.labelling, query_chunk, answers_chunk, passages_chunk),
                             total=len(query_chunk), desc="Labeling"))
                filtered_labelled_data = []
                for data in labelled_data:
                    if len(data) != 0:
                        for d in data:
                            filtered_labelled_data.append(d)
                if not self.quiet and len(filtered_labelled_data) != 0:
                    print('====================Labelled Data Sample======================')
                    pprint(filtered_labelled_data[0])
                    print('==============================================================')

                samples += len(filtered_labelled_data)
                for d in filtered_labelled_data:
                    json.dump(d, f, ensure_ascii=False)
                    f.write("\n")
                print(f'{samples} samples has been collected.')
                if samples >= self.max_samples:
                    print(f'Stopped at chunk {i}/{len(dataloader)}. Chunk size {len(query_chunk)}.')
                    break
                else:
                    print(f'Chunk {i}/{len(dataloader)} finished. Chunk size {len(query_chunk)}.')


class TriviaQALabeller(Labeller):

    def get_dataloader(self, datapath):
        with open(datapath, "r") as f:
            triviaqa_data = json.load(f)['Data']
        data = []
        for i, d in enumerate(triviaqa_data):
            query = d['Question']
            answers = [d['Answer']['Value'], d['Answer']['NormalizedValue']] + d['Answer']['Aliases'] + d['Answer'][
                'NormalizedAliases']
            answers = [a.lower() for a in answers]
            data.append({'query': query, 'answers': answers})
        dataset = Dataset.from_list(data).shuffle(seed=42)
        print('Data Example: {}'.format(dataset[0]))

        def collate_fn(batch):
            query_list = []
            answers_list = []
            for d in batch:
                query_list.append(d['query'])
                answers_list.append(d['answers'])
            return query_list, answers_list

        return DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)


class HotPotQALabeller(Labeller):
    def get_dataloader(self, datapath):
        data = []
        with open(datapath, "r") as f:
            hotpot_data = json.load(f)
        for d in hotpot_data:
            query = d['question']
            answers = [d['answer'].lower().strip()]
            data.append({'query': query, 'answers': answers})
        dataset = Dataset.from_list(data).shuffle(seed=42)
        print('Data Example: {}'.format(dataset[0]))

        def collate_fn(batch):
            query_list = []
            answers_list = []
            for d in batch:
                query_list.append(d['query'])
                answers_list.append(d['answers'])
            return query_list, answers_list

        return DataLoader(dataset, batch_size=100, shuffle=False, collate_fn=collate_fn)


if __name__ == '__main__':
    api_key = 'YOUR_KEY' # openai api key
    quiet = True
    gasket_model = 'gpt-3.5-turbo'
    gasket_base_url = 'https://api.openai.com/v1'
    gasket_use_chat = True # use chat template
    generator_model = 'Llama3-8B-baseline'
    generator_base_url = 'http://localhost:8000/v1'
    generator_use_chat = False

    # Labelling TriviaQA
    datapath = 'data/triviaqa-unfiltered/unfiltered-web-train.json'
    out_filepath = 'data/labelled_training_data/triviaqa-labelled.jsonl'
    max_samples = 5000
    labeller = TriviaQALabeller(api_key, datapath, out_filepath, max_samples, gasket_model, gasket_base_url,
                                gasket_use_chat,
                                generator_model, generator_base_url, generator_use_chat, quiet)
    start_from = 0
    labeller.run_labelling(start_from)

    # Labelling HotpotQA
    datapath = 'data/hotpot/hotpot_train_v1.1.json'
    out_filepath = 'data/labelled_training_data/hotpot-labelled.jsonl'
    max_samples = 12000
    labeller = HotPotQALabeller(api_key, datapath, out_filepath, max_samples, gasket_model, gasket_base_url,
                                gasket_use_chat,
                                generator_model, generator_base_url, generator_use_chat, quiet)
    start_from = 0
    labeller.run_labelling(start_from)
