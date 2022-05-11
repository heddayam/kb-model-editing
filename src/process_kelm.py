import re
import jsonlines
import pandas as pd

from string import punctuation 
from numpy.random import default_rng

def format_sentences(dict_line):
    '''Converting a given json object to ROME format '''
    
    gen_sent = dict_line['gen_sentence']
    gen_sent = gen_sent.replace('+', '') # remove '+' symbol before numbers
    gen_sent = gen_sent.replace('(', '') # remove '+' symbol before numbers
    gen_sent = gen_sent.replace(')', '') # remove '+' symbol before numbers

    # Get triple object 
    triple_elements = dict_line['triples'][0]
    triple_object = triple_elements[-1].strip(punctuation)
    triple_object = triple_object.replace('(', '')
    triple_object = triple_object.replace(')', '')
    
    triple_subject = triple_elements[0]
    triple_subject = triple_subject.translate(str.maketrans('', '', punctuation))
    pattern = None

    # Match object component in sentence and extract substring till EOS in the orig sentence
    for object_part in triple_object.split():
        if object_part.lower() in gen_sent.strip(punctuation).lower().split():
            pattern = object_part + '.*'
            break

    if pattern == None:
        return None

    p = re.compile('(' + pattern + ')')
    object_matched_str = p.search(gen_sent, re.IGNORECASE)


    if object_matched_str is None:
        return None

    obj_completion = object_matched_str.group(0)

    # Match subject component 
    pattern = triple_subject.lower()
    p = re.compile('(' + pattern + ')')
    subject_matched_str = p.search(gen_sent, re.IGNORECASE)
    
    if subject_matched_str is None:
        return None
    
    sub_completion = subject_matched_str.group(0)
    masked_sent = gen_sent.replace(obj_completion, "")
    
    # Remove instances where subject appears multiple times
    if masked_sent.count(sub_completion) > 1 :
        return None
    
    masked_sent = masked_sent.replace(sub_completion, "{}")
    
    if len(masked_sent.split()) < len(obj_completion.split()):
        return None
        
    # Construct tuple to be insert into df
    out_tuple = {
        'prompt': masked_sent, 
        'subject': triple_subject, 
        'target_new': {
            "str": obj_completion
        }, 
        'triple': triple_elements
    }

    return out_tuple
            
def process_random_triples(rand_frac):
    '''Processing random subset of the original dataset'''

    rand_count = rand_frac * line_count

    with jsonlines.open("../data/kelm_triples_only_corpus.jsonl", 'r') as ifile, jsonlines.open("../data/formatted/kelm_random_triples_processed.jsonl", 'w') as ofile:
        for dict_line in ifile:
            if rng.integers(line_count) < rand_count:
                processed_line = format_sentences(dict_line)
                if processed_line is not None:
                    ofile.write(processed_line)

def train_test_split(dev_frac, test_frac, infile = "../data/formatted/kelm_random_triples_processed.jsonl"):
    '''Creating Train/Test/Dev splits on the input file'''
    rand_count = rand_frac * line_count

    with jsonlines.open(infile, 'r') as ifile, jsonlines.open("../data/formatted/kelm_train_set.jsonl", 'w') as train_file, jsonlines.open("../data/formatted/kelm_dev_set.jsonl", 'w') as dev_file, jsonlines.open("../data/formatted/kelm_test_set.jsonl", 'w') as test_file:
        for dict_line in ifile:
            gen_rand = rng.uniform()
            if gen_rand <= 1.0 - (dev_frac + test_frac):
                train_file.write(dict_line)
            elif 1.0 - (dev_frac + test_frac) < gen_rand <= 1.0 - test_frac:
                dev_file.write(dict_line)
            else:
                test_file.write(dict_line)

if __name__ == "__main__":

    line_count = 0
    rng = default_rng(42)
    rand_frac = 0.3

    # Extract only triples
    with jsonlines.open("../data/raw/kelm_generated_corpus.jsonl", 'r') as ifile, jsonlines.open("../data/kelm_triples_only_corpus.jsonl", 'w') as ofile:
        for dict_line in ifile:
            if 'triples'  in dict_line:
                if len(dict_line['triples']) != 1:
                    continue
                
                ofile.write(dict_line)
                line_count += 1
                
    process_random_triples(rand_frac)  
    train_test_split(0.1, 0.2)