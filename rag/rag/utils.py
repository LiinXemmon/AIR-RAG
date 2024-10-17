
from rag.rag.infer_alg.gasketrag.gasketrag import GasketRAG
from rag.rag.infer_alg.naive_rag import NaiveRag

def get_algorithm(args):
    if args.algorithm_name == 'naive_rag':
        Rag = NaiveRag(args)
    elif args.algorithm_name == 'gasketrag':
        Rag = GasketRAG(args)
    else:
        raise AlgorithmNotFoundError("Algorithm not recognized. Please provide a valid algorithm name.")
    return Rag


class AlgorithmNotFoundError(Exception):
    pass
