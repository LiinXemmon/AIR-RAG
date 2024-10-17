import os
from datetime import datetime
import jsonlines
import json
from ruamel.yaml import YAML
import pdb

TASK_LIST = ['PopQA','PubHealth', 'TriviaQA',
             'HotPotQA', 'StrategyQA', '2WikiMultiHopQA']
def load_jsonlines(file:str)-> list[dict]:
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst 

def get_dataset(args) -> object:
    # base class
    from rag.dataset.base_dataset.MultiChoiceQA import MultiChoiceQA
    from rag.dataset.base_dataset.QA import QA
    # advanced dataset class
    from rag.dataset.PopQA import PopQA
    from rag.dataset.PubHealth import PubHealth
    from rag.dataset.TriviaQA import TriviaQA
    from rag.dataset.HotPotQA import HotPotQA
    from rag.dataset.StrategyQA import StrategyQA
    from rag.dataset.WikiMultiHopQA import WikiMultiHopQA

    if 'PopQA' == args.task:
        EvalData = PopQA(args)
    elif 'PubHealth' == args.task:
        EvalData = PubHealth(args)
    elif 'TriviaQA' == args.task:
        EvalData = TriviaQA(args)
    elif 'HotPotQA' == args.task:
        EvalData = HotPotQA(args)
    elif 'StrategyQA' == args.task:
        EvalData = StrategyQA(args)
    elif '2WikiMultiHopQA' == args.task:
        EvalData = WikiMultiHopQA(args)
    else:
        raise TaskNotFoundError("Task not recognized. Please provide a valid args.task.")
    return EvalData

def get_args_form_config(yml)-> dict:
    if yml == '':
        return
    yaml = YAML(typ='rt') # rt is (round-trip) mode
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read())
    return dic

class TaskNotFoundError(Exception):
    pass