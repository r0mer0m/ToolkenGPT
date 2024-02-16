import json
import argparse
from pathlib2 import Path
from os import path as osp
from sentencepiece import SentencePieceProcessor
from sentencepiece import sentencepiece_model_pb2 as model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_tok_path', type=str, default='/home/karypisg/romer333/projects/LLM-tools/models/llama_checkpoints/tokenizer.model')
    parser.add_argument('--new_tokens', type=str, nargs='+', default=['<BOC>', '<EOC>', '<BOR>', '<EOR>'])
    parser.add_argument('--out_tok_dir', type=str, default='/home/karypisg/romer333/projects/LLM-tools/ToolkenGPT/augmented_tokenizer/')
    parser.add_argument('--insertion_index', type=int, default=3)
    return parser.parse_args()

def augment_tokenizer(in_tok_path, 
                      new_tokens, 
                      out_tok_dir, 
                      insertion_index=3):

    out_model_path = osp.join(out_tok_dir, 'tokenizer.model')
    out_config_path = osp.join(out_tok_dir, 'tokenizer.config')
    
    mp = model.ModelProto()
    mp.ParseFromString(open(in_tok_path, 'rb').read())

    # mp.ParseFromString(open(model_file, 'rb').read())

    print("Inserting additional tokens....")
    print(f'Original model pieces: {len(mp.pieces)}')

    for i, sym in enumerate(new_tokens, insertion_index):
        new_sym = mp.SentencePiece()
        new_sym.piece = sym 
        new_sym.score = 0.0 # default score for USER_DEFINED
        new_sym.type = 4 # type value for USER_DEFINED
        mp.pieces.insert(i, new_sym) # position after default control symbols ("<unk>", "<s>", "</s>")
        print(f'\tadded {new_sym.piece} ...')

    print(f'New model pieces: {len(mp.pieces)}')
    
    print(f"Writting augmented tokenizer to {out_tok_dir}....")
    
    # write tokenizer model 
    with open(out_model_path, 'wb') as fp:
        fp.write(mp.SerializeToString())
    
    # write augmented config
    config = {
        'insertion_index': insertion_index,
        'n_aug_words': len(new_tokens),
        'n_base_words': len(mp.pieces) - len(new_tokens)
    }
    with open(out_config_path, 'w') as fp:
        json.dump(config, fp)
        
        
def test_augmented_tokenizer(out_tok_dir):
    
    out_model_path = osp.join(out_tok_dir, 'tokenizer.model')
    out_config_path = osp.join(out_tok_dir, 'tokenizer.config')
    
    with open(out_config_path, 'r') as fp:
        config = json.load(fp)
    sp_model = SentencePieceProcessor(model_file=out_model_path)
    
    inserted_tokens = [sp_model.IdToPiece(id) for id in range(config['insertion_index'], config['insertion_index'] + config['n_aug_words'])]
    print('[TEST] Inserted tokens:', inserted_tokens)
    out = sp_model.encode_as_pieces(f"This is a test calling function {inserted_tokens[2]}(arg1, arg2) see what happens.")
    assert inserted_tokens[2] in out, f"Token {inserted_tokens[2]} not found in {out}"
        
    
def main(in_tok_path='/home/karypisg/romer333/projects/LLM-tools/models/llama_checkpoints/tokenizer.model',
         new_tokens=['<BOC>', '<EOC>', '<BOR>', '<EOR>'], 
         out_tok_dir='/home/karypisg/romer333/projects/LLM-tools/ToolkenGPT/augmented_tokenizer/',
         insertion_index=3
         ):

    Path(out_tok_dir).mkdir(parents=True, exist_ok=True)
        
    # augment tokenizer
    augment_tokenizer(in_tok_path, new_tokens, out_tok_dir, insertion_index)
    
    test_augmented_tokenizer(out_tok_dir)

if __name__ == '__main__':
    args = get_args()
    main(
        args.in_tok_path,
        args.new_tokens,
        args.out_tok_dir,
        args.insertion_index
    )