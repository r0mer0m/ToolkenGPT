import re
import json
import torch
import argparse
import os.path as osp
from pathlib2 import Path
from fuse.base.tokenizer import AugmentedTokenizer

SYMBOL_SYNTHAX = '<{api_name}>'

def get_tokenizer():
    tokenizer = AugmentedTokenizer.from_pretrained('../ToolkenGPT/augmented_tokenizer', 
                                                   augmentation_config_path='./augmented_tokenizer/augmentation_config.json')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_call(annotation):
    annotation = re.sub(r"=", SYMBOL_SYNTHAX.format(api_name='EOC') + SYMBOL_SYNTHAX.format(api_name='BOR'), annotation)
    annotation = re.sub(r"\<eoe\>", SYMBOL_SYNTHAX.format(api_name='EOR'), annotation)
    annotation = SYMBOL_SYNTHAX.format(api_name='BOC') + annotation
    return annotation


def process_record(tokenizer, record):
    text = record.get('text')
    for tar_number, tar_eq in zip(record['tar_number'], record['tar_eq']):
        tar_eq = get_call(tar_eq)
        text = re.sub(tar_number, tar_eq + " " + tar_number, text)
    tokens = torch.Tensor(tokenizer.encode(text, bos=True, eos=True)).long()
    
    boc_ids = torch.nonzero(tokens==tokenizer.boc_id, as_tuple=True)[0].tolist()
    # eoc_ids = torch.nonzero(tokens==tokenizer.eoc_id, as_tuple=True)[0].tolist()
    bor_ids = torch.nonzero(tokens==tokenizer.bor_id, as_tuple=True)[0].tolist()
    eor_ids = torch.nonzero(tokens==tokenizer.eor_id, as_tuple=True)[0].tolist()
    
    tokens = tokens.tolist()
    
    out = {
        'text': text, 
        'tar_eq': record['tar_eq'], 
        'tar_number': record['tar_number'],
        'api_ids': [tokens[idx + 1] for idx in boc_ids],
        # 'eoc_ids': eoc_ids,
        'bor_idxs': bor_ids,
        'eor_idxs': eor_ids
        }
    
    return out


def main(tokenizer, data):
    out_data = []
    for record in data:
        out_record = process_record(tokenizer, record)
        out_data.append(out_record)
    return out_data
    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in-path', type=str, default='../data/funcqa/train.json')
    argparser.add_argument('--out-dir', type=str, default='../augmented_data/funcqa/')
    args = argparser.parse_args()
    tokenizer = get_tokenizer()
    
    # Load data
    with open(args.in_path, 'r') as fp:
        data = json.load(fp)

    # Process data
    out_data = main(tokenizer, data)

    # Save data
    filename = osp.basename(args.in_path)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(osp.join(args.out_dir, filename), 'w') as fp:
        json.dump(out_data, fp)
