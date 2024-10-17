ALGORITHM_INSTRUCTIONS = [
    {
        "algorithm_name": "-------------------Template-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "Rules of naming:'-' seperate for naming. For example: Algorithm_name-mode-specific_stage",
        "dataset_name": "dataset name",
        "instruction": "Fill in your instruction here"
    },
    {
        "algorithm_name": "-------------------Naive Rag-------------------------",
        "dataset_name": ""
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "",
        "instruction": "### Instruction:\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PopQA-posterior_instruction",
        "instruction": "### Instruction:\n Now, based on the passages and your internal knowledge, please answer the question more succinctly and professionally. ### Retrieved Knowledge:\n {passages}\n \n## Input:\n\n{query}\n\n ### Response:\n"
    }, 
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "PopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "TriviaQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "StrategyQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n You are only allowed to answer True or False, and generating other types of responses is prohibited. ### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "HotPotQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n Now, based on the following passages and your knowledge, please answer the question more succinctly and professionally. ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "2WikiMultiHopQA",
        "instruction": "### Instruction:\n## Input:\n\n{query}\n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n Determine the statement based on the following passages and your knowledge ### Background Knowledge:\n {passages} \n\n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag",
        "dataset_name": "PubHealth-posterior_instruction",
        "instruction": "### Instruction\nDetermine the statement based on the passages and your internal knowledge. ### Retrieved Knowledge:\n {passages}\n  Is the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n  n### Response:\n"
    },
    {
        "algorithm_name": "Naive_rag-without_retrieval",
        "dataset_name": "PubHealth",
        "instruction": "### Instruction:\nIs the following statement correct or not? Say true if it's correct; otherwise say false\n\n## Input:\n\n{query}\n\n### Response:\n"
    }
]