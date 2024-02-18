import re
import json
import torch
import os.path as osp
from pathlib2 import Path
from llama_augmented.tokenizer import AugmentedTokenizer


def process_call_annotation(annotation):
    annotation = re.sub(r"=", "<EOC><BOR>", annotation)
    annotation = re.sub(r"\<eoe\>", "<EOR>", annotation)
    annotation = "<BOC>" + annotation
    return annotation

def main(data):
    
    tokenizer = AugmentedTokenizer('../ToolkenGPT/augmented_tokenizer')
    
    out_data = []
    for record in data:
        text = record['text']
        for tar_number, tar_eq in zip(record['tar_number'], record['tar_eq']):
            tar_eq = process_call_annotation(tar_eq)
            text = re.sub(tar_number, tar_eq + " " + tar_number, text)
        tokens = torch.Tensor(tokenizer.encode(text, bos=True, eos=True)).long()
        
        boc_ids = torch.nonzero(tokens==tokenizer.boc_id, as_tuple=True)[0].tolist()
        # eoc_ids = torch.nonzero(tokens==tokenizer.eoc_id, as_tuple=True)[0].tolist()
        bor_ids = torch.nonzero(tokens==tokenizer.bor_id, as_tuple=True)[0].tolist()
        eor_ids = torch.nonzero(tokens==tokenizer.eor_id, as_tuple=True)[0].tolist()
        
        tokens = tokens.tolist()
        
        out_data.append({
            'text': text, 
            'tar_eq': record['tar_eq'], 
            'tar_number': record['tar_number'],
            'api_ids': [tokens[idx + 1] for idx in boc_ids],
            # 'eoc_ids': eoc_ids,
            'bor_idxs': bor_ids,
            'eor_idxs': eor_ids
            })
        # break
    return out_data
    

if __name__ == '__main__':
    in_path = '../data/funcqa/train.json'
    out_dir = '../augmented_data/funcqa/'
    out_fn = 'train.json'
    
    with open(in_path, 'r') as fp:
        data = json.load(fp)

    out_data = main(data)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(osp.join(out_dir, out_fn), 'w') as fp:
        json.dump(out_data, fp)
