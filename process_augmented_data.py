import json
from pprint import pprint
from src.fuse.base.tokenizer import AugmentedTokenizer
import re
import random
from collections import defaultdict
from omegaconf import OmegaConf

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
            s, e = s + cum_offset, e + cum_offset
            op = re.search(r"(<.*?>)", eq[:-2]).group(1)
            eq = re.sub(op, "<<" + op + ">>", eq)
            eq = re.sub('<eoe>', "<<<EOR>>>", eq)
            eq = re.sub('=', "<<<EOC>>>", eq)
            tar_eq_tokens = tokenizer.encode(eq)
            tar_eq_tokens = tar_eq_tokens[2:] if tar_eq_tokens[1] == 660 else tar_eq_tokens[1:]
            idx = tar_eq_tokens.index(tokenizer.eoc_id)
            out_record['start_token_idx'].append(s)
            out_record['end_token_idx'].append(s + idx)
            # out_record['end_token_idx'].append(s + len(tar_eq_tokens))
            tokenized_text = tokenized_text[:s] + tar_eq_tokens + tokenized_text[e:]
            # print("Tokenized text: ", tokenizer.decode(tokenized_text))
            # print("Call target: ", tokenizer.decode(tokenized_text[out_record['start_token_idx'][0]:out_record['end_token_idx'][0]]))
            cum_offset += len(tar_eq_tokens) - (e - s)
        out_record['text'] = tokenizer.decode(tokenized_text[1:])
        out_data.append(out_record)
        # print("\nTransformed record: ", out_record['text'])

    with open('../data/funcqa/augmented_train_call_only.json', 'w') as fp:
        json.dump(out_data, fp)

if __name__ == "__main__":
    main(None)