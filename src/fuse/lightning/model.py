import re
import torch
from torch import nn
import os.path as osp
from pathlib2 import Path
from typing import Dict, Union
from collections import defaultdict
from pytorch_lightning import LightningModule
from transformers import LlamaForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = []):
      super().__init__()
      self.stops = set(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      idx = input_ids[0, -1]
      if idx in self.stops:
          return True
      return False
      
class PLModel(LightningModule):
    
    def __init__(self, model: nn.Module, tokenizer, rank:int, config: Dict[str, Union[str, int]]):
        super().__init__()
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank
        self.config = config
        self.ignore_index = -100
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        # self.stopping_criteria = StoppingCriteriaList([
        #     StoppingCriteriaSub(stops = [self.tokenizer.eoc_id])
        # ])
        
    # def forward(self, *args, **kwargs):
    #     return self.model(*args, **kwargs)
    
    def on_train_epoch_start(self) -> None:
        self.results = defaultdict(list)
        return super().on_train_epoch_start()
    
    def _forward(self, batch):
        
        input_ids = batch['input_ids'].to("cuda")
        target_ids = batch['target_ids'].to("cuda")
        
        logits = self.model(input_ids).logits
        
        loss = self.loss_fun(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        # print("Loss: ", loss.item())
        # print("logits: ", logits.view(-1, logits.shape[-1])[:, -15:])
        # print("target_ids: ", target_ids.view(-1))
        return loss, logits
    
    def training_step(self, batch, batch_idx):
        if not isinstance(batch, dict):
            assert len(batch) == 1
            batch = batch[0]
        
        loss, logits = self._forward(batch)
        self.log("loss", loss.item())
        return {'loss': loss, 'logits': logits}
        
    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs = []
        self.results = defaultdict(list)
    
    def validation_step(self, batch, batch_idx):
        if not isinstance(batch, dict):
            assert len(batch) == 1
            batch = batch[0]
            
        loss, logits = self._forward(batch) 
        
        self.log("val_loss", loss.item(), sync_dist=True)
        if self.rank == 0: 
            self.validation_step_outputs.append({
                'logits': logits.to("cpu"), 
                'target_ids': batch['target_ids'].to("cpu")}
            )
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        generate = True
        func_calls = []
        question = self.tokenizer.decode(batch['input_ids'])
        try:
            while generate:
                output = self.model.generate(
                    input_ids,
                    max_length = self.args.max_gen_len,
                    temperature = self.args.temperature,
                    top_p = self.args.top_p,
                    stopping_criteria=self.stopping_criteria,
                )
                if output.endswith("<EOC>"):
                    text_call = re.findall(r'(?:<SOC>)(.*)(?:<EOC>)', output)[-1]
                    text_op = text_call.split('(')[0]
                    call = re.replace(text_op, 
                                      self.tokenizer.symbol_to_api[text_op], 
                                      text_call)
                    func_calls.append(call)
                    result = eval(call)
                    result_text = "<SOR>" + result + "<EOR>"
                    result_ids = self._tokenizer.encode(result_text)
                    input_ids = input_ids + result_ids
                
                else:
                    generate = False
                    cur_generation = self.tokenizer.decode(output[batch['template_len']:])
            
            log = {
                "case_idx": batch_idx,
                "question": question,
                "func_calls": func_calls,
                "generation": cur_generation, #.replace("\n", "\\n").strip(),
                "status": "success"
            }
            
        except Exception as e:
            log = {
                "case_idx": batch_idx,
                "question": question,
                "func_calls": func_calls,
                "generation": cur_generation, #.replace("\n", "\\n").strip(),
                "status": str(e)
            }
        return log
    
    def on_train_epoch_start(self,):
        self.model.register_backward_hooks()
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        '''
        Metrics recording and computation
        '''
        self._record_trigger_metric_objects(outputs['logits'], batch['target_ids'])
        if (batch_idx + 1) % 20 == 0:
            self._log_trigger_metrics(stage="train", log_each=False)
            self.results = defaultdict(list)
            
    def on_validation_epoch_end(self) -> None:
        '''
        Metrics recording and computation
        '''
        self.results = defaultdict(list)
        for validation_step_output in self.validation_step_outputs:
            self._record_trigger_metric_objects(
                validation_step_output['logits'], 
                validation_step_output['target_ids']
                )
        self._log_trigger_metrics(stage="test", log_each=False)      
        self.results = defaultdict(list)   
        
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.config.lr)
    
    def _record_trigger_metric_objects(self, logtis, target_ids):
        pred = torch.argmax(logtis, dim=-1)
        pred = pred.view(-1)
        labels = target_ids.view(-1)

        # label_funcs = [labels == idx for idx in ([self.tokenizer.boc_id] + self.tokenizer.api_ids)]
        # pred_funcs = [pred == idx for idx in ([self.tokenizer.boc_id] + self.tokenizer.api_ids)]
        # label_funcs = [labels == idx for idx in [self.tokenizer.boc_id]]
        # pred_funcs = [pred == idx for idx in [self.tokenizer.boc_id]]
        label_funcs = [labels == idx for idx in self.tokenizer.api_ids]
        pred_funcs = [pred == idx for idx in self.tokenizer.api_ids]
        label_funcs = torch.stack(label_funcs, dim=0)
        pred_funcs = torch.stack(pred_funcs, dim=0)
        
        tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        metrics = {
            "tp": tp,
            "pred": pred_funcs,
            "true": true
        }
        
        for i, r in metrics.items(): self.results[i].append(r)  
    
    def _log_trigger_metrics(self, stage="train", log_each=False):
        if log_each:
            for i, api_name in enumerate(self.tokenizer.api_names):
                tp = sum([r[i] for r in self.results["tp"]])
                pred = sum([r[i] for r in self.results["pred"]])
                true = sum([r[i] for r in self.results["true"]])
            
                self.log(f"{stage}/precision-{api_name}", tp / (pred + 1e-8), sync_dist=True)
                self.log(f"{stage}/recall-{api_name}", tp / (true + 1e-8), sync_dist=True)
                self.log(f"{stage}/f1-{api_name}", 2 * tp / (pred + true + 1e-8), sync_dist=True)
        
        tp = sum([r.sum() for r in self.results["tp"]])
        pred = sum([r.sum() for r in self.results["pred"]])
        true = sum([r.sum() for r in self.results["true"]])

        if stage == "train":
            stage = ''
        else:
            stage = stage + '-'
            
        self.log(f"{stage}precision", tp / (pred + 1e-8), sync_dist=True)
        self.log(f"{stage}recall", tp / (true + 1e-8), sync_dist=True)
        self.log(f"{stage}f1", 2 * tp / (pred + true + 1e-8), sync_dist=True)
            
