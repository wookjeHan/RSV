import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM

class ClassificationModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.transformers = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input_ids: torch.Tensor, att_mask: torch.Tensor, loss_att_mask: torch.Tensor, level='loss'):
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

        input_ids = input_ids.view(batch_size * class_num, seq_len).cuda() # batch_size * class_num, seq_len
        att_mask = att_mask.view(batch_size * class_num, seq_len).cuda() # batch_size * class_num, seq_len
        loss_att_mask = loss_att_mask.view(batch_size * class_num, seq_len).cuda() # batch_size * class_num, seq_len
        
        tok_label_ids = input_ids[:,1:].reshape(-1) # batch_size * class_num * (seq_len - 1)
        model_outputs = self.transformers(input_ids=input_ids, attention_mask=att_mask)
        logits = model_outputs.logits[:,:-1,:]
        logits = logits.reshape(batch_size * class_num * (seq_len - 1), -1) # batch_size * class_num * (seq_len - 1), vocab

        cross_ent_loss = self.cross_entropy(logits, tok_label_ids)
        cross_ent_loss = cross_ent_loss.reshape(batch_size * class_num, seq_len - 1)

        loss = cross_ent_loss * loss_att_mask[:,1:] # batch_size * class_num, seq_len - 1
        denominator = loss_att_mask.sum(dim=1)
        loss = loss.sum(dim=1) / denominator
        loss = loss.view(batch_size, class_num) # batch_size, class_num

        prob = (-loss).exp()
        prob = nn.functional.normalize(prob, p=1, dim=1)
    
        if level == 'loss':
            return loss
        if level == 'prob':
            return prob
        elif level == 'predict':
            return torch.argmin(loss, dim=1)