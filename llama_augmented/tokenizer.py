# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os
import os.path as osp
import json
import numpy as np

logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


class AugmentedTokenizer:
    def __init__(self, tokenizer_dir: str):
        # reload tokenizer
        assert os.path.isdir(tokenizer_dir), tokenizer_dir
        (self.sp_model, 
         self.augmentation_config) = self._load(tokenizer_dir)
        logger.info(f"Reloaded Augmented SentencePiece model from {tokenizer_dir}")

        # BOS / EOS token IDs
        self.n_base_words: int = self.augmentation_config['n_base_words']
        self.n_aug_words: int = self.augmentation_config['n_aug_words']
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#base words: {self.n_base_words} - #augmented words: {self.n_aug_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def _load(self, tokenizer_dir: str):
        out_model_path = osp.join(tokenizer_dir, 'tokenizer.model')
        out_config_path = osp.join(tokenizer_dir, 'tokenizer.config')
        
        with open(out_config_path, 'r') as fp:
            config = json.load(fp)
        sp_model = SentencePieceProcessor(model_file=out_model_path)
        
        return sp_model, config
    
    def _encode_remap(self, t: List[int]) -> List[int]:
        numpy_t = np.array(t)
        new_tokens_start_idx = self.augmentation_config['insertion_index']
        base_tokens_start_idx = self.augmentation_config['insertion_index'] + self.augmentation_config['n_aug_words']
        
        # identify token types (base non-control and new)
        mask_base_non_control = (
            numpy_t >= base_tokens_start_idx
        )
        mask_new_tokens = (
            (numpy_t >= new_tokens_start_idx) & 
            (numpy_t < base_tokens_start_idx)
        )
        
        # re-map
        numpy_t[mask_base_non_control] -= self.augmentation_config['n_aug_words']
        numpy_t[mask_new_tokens] += (self.augmentation_config['n_base_words'] - (self.augmentation_config['insertion_index'] - 1))
        
        return numpy_t.tolist()
    
    def _decode_remap(self, t: List[int]) -> List[int]:
        numpy_t = np.array(t)
        # identify token types (base non-control and new)
        new_tokens_start_idx = self.augmentation_config['n_base_words'] + 1
        base_tokens_start_idx = self.augmentation_config['insertion_index']
        
        mask_base_non_control = (
            (numpy_t >= base_tokens_start_idx) &
            (numpy_t < new_tokens_start_idx)
        )
        mask_new_tokens = (
            numpy_t >= new_tokens_start_idx
        )
        
        # re-map
        numpy_t[mask_base_non_control] += self.augmentation_config['n_aug_words']
        numpy_t[mask_new_tokens] -= (self.augmentation_config['n_base_words'] - (self.augmentation_config['insertion_index'] - 1))
        
        return numpy_t.tolist()
    
    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        t = self._encode_remap(t)
        return t

    def decode(self, t: List[int]) -> str:
        t = self._decode_remap(t)
        return self.sp_model.decode(t)