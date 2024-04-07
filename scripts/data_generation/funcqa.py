import re
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict
from fuse.base.tokenizer import AugmentedTokenizer


def get_tokenizer():
    augmentation_config = OmegaConf.load('./configs/runs/augmentation/funcqa_specific.yaml')
    tokenizer = AugmentedTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            augmentation_config=augmentation_config,
            )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _ends_with_equal(tokenizer, tokenized_text):
    if tokenizer.decode(tokenized_text).strip().endswith('='):
        return True
    return False


def _remove_precedding_equal(tokenized_text, start_idxs, tokenizer):
    offset = 0
    for s in start_idxs:
        
        left_side = tokenized_text[:s + offset]
        right_side = tokenized_text[s + offset:]
        
        if _ends_with_equal(tokenizer, left_side):
            
            # Option 1: Remove equal sign
            # tokenized_text = left_side[:-1] + right_side
            # offset -= 1
            
            # Option 2: Replace equal sign with space
            text = tokenizer.decode(left_side[1:-1]).strip("=") + " " + tokenizer.decode(right_side)
            tokenized_text = tokenizer.encode(text)
            
            # left_side_text = tokenizer.decode(left_side[1:])
            # right_side_text = tokenizer.decode(right_side)
            # left_side_text = left_side_text.strip().strip('=')
            # left_side = tokenizer.encode(left_side_text)
            # right_side = tokenizer.encode(right_side_text)[1:]
            # offset -= 1
            # s -= 1
            # e -= 1
    
    return tokenized_text


def _helper_process_call(eq):
    op = re.search(r"(<.*?>)", eq[:-2]).group(1)
    eq = re.sub(op, "<<" + op + ">>", eq)
    eq = re.sub('<eoe>', "<<<EOR>>>", eq)
    eq = re.sub('=', "<<<EOC>>>", eq)
    return eq


def process_calls(eqs, tokenizer):
    eoc_id = tokenizer.eoc_id
    out_record = defaultdict(list)
    for eq in eqs:    
        eq = _helper_process_call(eq)
        tar_eq_tokens = tokenizer.encode(eq)
        call_name_id = tar_eq_tokens[1]
        eoc_offset = tar_eq_tokens.index(eoc_id)
        
        out_record['calls'].append(eq)
        out_record['func_name_ids'].append(call_name_id)
        out_record['eoc_offsets'].append(eoc_offset)
        
        # print(tokenizer.decode(tar_eq_tokens[1:eoc_offset + 1]))
        # print(tokenizer.convert_ids_to_tokens(tar_eq_tokens[1:eoc_offset + 1]))
    
    return out_record


def get_call_answers(tokenized_text, start_tok_idxs, end_token_idxs, tokenizer):
    call_results = [
        tokenizer.decode(tokenized_text[s:e]) 
        for s, e in zip(start_tok_idxs, end_token_idxs)
    ]
    return call_results


def process_text(tokenized_text, start_tok_idxs, tokenizer):
    
    # tokenized_text = _remove_precedding_equal(tokenized_text, start_tok_idxs, tokenizer)
    
    out_text = tokenizer.decode(tokenized_text[1:])
    
    return out_text


def add_call_to_text(text, call_answers, calls):
    out_text = text
    for call_ans, call in zip(call_answers, calls):
        out_text = re.sub(call_ans, call, out_text)
    return out_text


def process_record(tokenizer, record):
    text = record['text']
    tokenized_text = tokenizer.encode(text)
    
    call_answers = get_call_answers(tokenized_text, 
                                    start_tok_idxs=record['start_token_idx'], 
                                    end_token_idxs=record['end_token_idx'], 
                                    tokenizer=tokenizer)
    
    text = process_text(tokenized_text=tokenized_text, 
                        start_tok_idxs=record['start_token_idx'],
                        tokenizer=tokenizer)
    
    out_record = process_calls(record['tar_eq'], tokenizer)
    
    out_text = add_call_to_text(text, call_answers, out_record['calls'])
    
    out_record['text'] = out_text
    out_record['token_len'] = len(tokenizer.encode(out_text)) - 1
        
    return out_record


def print_calls_from_text(tokenizer, text, func_name_ids, eoc_offsets, **kwargs):
    print("Test extraction of calls from text:")
    text_tokens = tokenizer.encode(text)
    for call_id, call_len in zip(func_name_ids, eoc_offsets):
        idx = text_tokens.index(call_id)
        call_tokens = text_tokens[idx:idx + call_len]
        print("\t", tokenizer.decode(call_tokens))
       
        
def print_examples(tokenizer, data, indices=[3, 19, 90]):
    from pprint import pprint
    
    print("="*20 + " Data Examples " + "="*20)
    for idx in indices:
        print(f"\n--------> {idx=} <--------")
        out_record = data[idx]
        pprint(out_record)
        print_calls_from_text(tokenizer=tokenizer, **out_record)
        

def main(tokenizer, input_path, output_path):

    with open(input_path, 'r') as fp:
        data = json.load(fp)
        
    out_data = []
    for record in tqdm(data):
        out_record = process_record(tokenizer, record)
        out_data.append(out_record)
        
    print_examples(tokenizer, out_data)
    
    
    
    with open(output_path, 'w') as fp:
        json.dump(out_data, fp)


if __name__ == "__main__":
    
    tokenizer = get_tokenizer()
    
    main(tokenizer=tokenizer,
         input_path='../data/funcqa/train.json', 
         output_path='../augmented_data/funcqa/train.json')