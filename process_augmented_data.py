import json
from pprint import pprint
from src.fuse.base.tokenizer import AugmentedTokenizer
import re
import random
from collections import defaultdict
from omegaconf import OmegaConf

def ends_with_equal(tokenizer, tokenized_text):
    if tokenizer.decode(tokenized_text).strip()[-1] == '=':
        return True
    return False


def main(args):
    augmentation_config = OmegaConf.load('./configs/runs/augmentation/funcqa_specific.yaml')
    tokenizer = AugmentedTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            augmentation_config=augmentation_config,
            )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open('../data/funcqa/train.json', 'r') as fp:
        data = json.load(fp)
        
    # random.shuffle(data)
    out_data = []
    for record in data:
        text = record['text']
        # print("\n\nInput record: ", text)
        cum_offset = 0
        tokenized_text = tokenizer.encode(text)
        # print(tokenized_text)
        out_record = {
            'start_token_idx': [],
            'end_token_idx': []
        }
        for s, e, eq in zip(record['start_token_idx'], record['end_token_idx'], record['tar_eq']):
            # print(s, e)
            s, e = s + cum_offset, e + cum_offset
            left_side = tokenized_text[:s]
            right_side = tokenized_text[e:]
            
            # get start of question idx
            idx = left_side[3:].index(29901)
            if left_side[3:][idx-1] == 319 or left_side[3:][idx-1] == 29909:
                out_record['answer_start_idx'] = idx + 1 + 3
                print('\n\n')
                print(tokenizer.convert_ids_to_tokens(left_side[out_record['answer_start_idx']-2:out_record['answer_start_idx']+2]))
            else:
                raise ValueError(f"Not found in {text}")
            
            
            if ends_with_equal(tokenizer, left_side):
                print("Moving equal sign")
                left_side_text = tokenizer.decode(left_side[1:])
                right_side_text = tokenizer.decode(right_side)
                left_side_text = left_side_text.strip().strip('=')
                # right_side_text = '=' + right_side_text
                left_side = tokenizer.encode(left_side_text)
                right_side = tokenizer.encode(right_side_text)[1:]
                cum_offset -= 1
                s -= 1
                e -= 1
                
            op = re.search(r"(<.*?>)", eq[:-2]).group(1)
            eq = re.sub(op, "<<" + op + ">>", eq)
            eq = re.sub('<eoe>', "<<<EOR>>>", eq)
            eq = re.sub('=', "<<<EOC>>>", eq)
            tar_eq_tokens = tokenizer.encode(eq)
            tar_eq_tokens = tar_eq_tokens[2:] if tar_eq_tokens[1] == 660 else tar_eq_tokens[1:]
            
            idx = tar_eq_tokens.index(tokenizer.eoc_id)
            # print(s, s+idx)
            out_record['start_token_idx'].append(s)
            out_record['end_token_idx'].append(s + idx)
            # out_record['end_token_idx'].append(s + len(tar_eq_tokens))
            tokenized_text = left_side + tar_eq_tokens + right_side
            print("Tokenized text: ", tokenizer.decode(tokenized_text))
            print("Call target: ", tokenizer.decode(tokenized_text[out_record['start_token_idx'][0]:out_record['end_token_idx'][0]]))
            cum_offset += len(tar_eq_tokens) - (e - s)
        out_record['text'] = tokenizer.decode(tokenized_text[1:])
        out_data.append(out_record)
        # print("\nTransformed record: ", out_record['text'])

    # with open('../data/funcqa/augmented_train_call_only_equal_after_call.json', 'w') as fp:
    with open('../data/funcqa/augmented_train_call_only_equal_after_call_w_ans_start_idx.json', 'w') as fp:
        json.dump(out_data, fp)

if __name__ == "__main__":
    main(None)