import torch.nn as nn
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

class LMClassificationModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.transformers = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids: Tensor, att_mask: Tensor, loss_att_mask: Tensor, level='loss'):
        """
        inputs -> dict{input_ids: tensor of shape(batch, seq_len), att_mask: tensor of shpae(batch, seq_len)
        verbalizers -> list of verbalizers, len -> num_of_class, each item is [token_indices]
        labels -> list of labels len -> batch
        First we assume that verbalizers is single token
        This can be changed to multi tokens
        
        outputs
        level 0 -> logits (batch, world_size) bef softmax 
        level 1 -> logits (batch, len(verbalizer)) bef softmax
        level 2 -> prediction (batch)
        level 3 -> acc float
        """
        batch_size, class_num, seq_len = input_ids.shape

        input_ids = input_ids.view(batch_size * class_num, seq_len).cuda() # batch_size * label, seq_len
        att_mask = att_mask.view(batch_size * class_num, seq_len).cuda() # batch_size * label, seq_len
        loss_att_mask = loss_att_mask.view(batch_size * class_num, seq_len).cuda() # batch_size * label, seq_len
        
        tok_label_ids = input_ids[:,1:].reshape(-1) # batch_size * label * (seq_len - 1)
        model_outputs = self.transformers(input_ids=input_ids, attention_mask=att_mask)
        logits = model_outputs.logits[:,:-1,:]
        logits = logits.reshape(batch_size * class_num * (seq_len - 1), -1) # batch_size * label * (seq_len - 1), vocab

        cross_ent_loss = self.cross_entropy(logits, tok_label_ids)
        cross_ent_loss = cross_ent_loss.reshape(batch_size * class_num, seq_len - 1)

        loss = cross_ent_loss * loss_att_mask[:,1:] # batch_size * label, seq_len - 1
        denominator = loss_att_mask.sum(dim=1)
        loss = loss.sum(dim=1) / denominator
        loss = loss.view(batch_size, class_num)

        if level == 'loss':
            return loss
        elif level == 'predict':
            return torch.argmin(loss, dim=1)