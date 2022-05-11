from datasets import load_dataset
import pandas as pd
from os.path import join

def parse_split(boolq, split, with_passage):
    data = []
    for row in boolq[split]:
        passage = row['passage']
        passage = passage.replace('{', '(')
        passage = passage.replace('}', ')')
        
        if with_passage:
            prompt = "passage: {}\n".format(passage) + "question: {}"
        else:
            prompt = "question: {}"

        subject = row['question']
        label = "False" if row['label'] == 0 else "True"
        
        
        data.append({"prompt": prompt,
            "subject":subject,
            "target_new":{"str": label}})
    return pd.DataFrame(data)
    
def main():

    paths = {True:"data/boolq", False:"data/boolq-no-context"}
    boolq = load_dataset('super_glue', 'boolq')

    for split in ("train", "validation"):
        for with_passage in (True, False):
            df = parse_split(boolq, split, with_passage)
            output_dir = paths[with_passage]
            df.to_json(join(output_dir, f'{split}.jsonl'), lines=True, orient='records')
    
    print('Done!')


if __name__ == '__main__':
    main()