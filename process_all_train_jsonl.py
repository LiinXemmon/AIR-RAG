import json

all_data = []
with open('data/labelled_training_data/train_all.jsonl', 'r') as f:
    lines = f.readlines()
    for l in lines:
        j = json.loads(l)
        all_data.append(
            {'prompt': j['prompt'], 'completion': j['completion'], 'label': j['label'], 'weight': j['weight']})
with open('data/labelled_training_data/train_all.jsonl', 'w', encoding='utf-8') as f:
    for d in all_data:
        json.dump(d, f)
        f.write('\n')