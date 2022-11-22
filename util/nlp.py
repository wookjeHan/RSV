import torch
import torch.nn.functional as F

def get_input_parameters(tokenizer, inputs, batch_size, class_num, verbalizers):
    tok_result = tokenizer(inputs, padding=True) # batch_size, seq_len
    att_mask = torch.tensor(tok_result['attention_mask']).unsqueeze(1) # batch_size, 1, seq_len
    att_mask = att_mask.expand(-1, class_num, -1) # batch_size, class_num, seq_len
    att_mask = att_mask.reshape(batch_size * class_num, -1)

    verb_inputs = [input + verbalizer for input in inputs for verbalizer in verbalizers]
    verb_tok_result = tokenizer(verb_inputs, padding=True) # batch_size * class_num, seq_len
    verb_input_ids = torch.tensor(verb_tok_result['input_ids']) # batch_size * class_num, seq_len
    verb_input_ids = verb_input_ids.view(batch_size, class_num, -1) # batch_size, class_num, seq_len

    verb_att_mask = torch.tensor(verb_tok_result['attention_mask']) # batch_size * class_num, seq_len

    pad_size = verb_att_mask.shape[-1] - att_mask.shape[-1]
    att_mask = F.pad(att_mask, (0, pad_size, 0, 0), 'constant', 0)
    loss_att_mask = verb_att_mask - att_mask

    verb_att_mask = verb_att_mask.view(batch_size, class_num, -1)
    loss_att_mask = loss_att_mask.view(batch_size, class_num, -1)

    return verb_input_ids, verb_att_mask, loss_att_mask
