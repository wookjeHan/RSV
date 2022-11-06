import argparse
import os
import warnings
import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from transformers import AutoTokenizer
from model import LMClassificationModel
from datasets import load_dataset
from shot_selector import *
from util.dataloader import DataModule
import templates
from tqdm import tqdm

warnings.filterwarnings(
    'ignore', '.*Trying to infer the `batch_size` from an ambiguous collection.*'
)
torch.set_printoptions(threshold=10_000)

def print_title(s):
    print()
    print('=' * 80)
    print(s)
    print('=' * 80)

def composite(batch, shots_list):
    '''
    Composite the batch and shots, which is a list of strings into forwardable sentence.
    The order is prefix-shot1-concatenator-shot2-...-shotn-suffix-resolved_input
    '''
    results = []

    for resolved_input, shots in zip(batch['resolved_input'], shots_list):
        result = batch['prefix']
        result += batch['concatenator'].join(shots)
        result += batch['suffix']
        result += resolved_input
        results.append(result)

    return results

def get_input_paremeters(tokenizer, inputs, batch_size, class_num, verbalizers):
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

def eval(model: LMClassificationModel, eval_dataloader, shot_selector, args):
    predictions = []
    labels = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # for batch in tqdm(eval_dataloader):
    for batch in eval_dataloader:
        # The batch is consisted of prefix, concatenator, suffix, resolved_input, label, and verbalizers
        batch_size = len(batch['resolved_input'])
        verbalizers = batch['verbalizers']
        class_num = len(verbalizers)

        # Shots are selected based on the batch and the verbalizer
        # The datastructure of shots is list of string (batch_size, shot_num)
        shots = shot_selector(batch, templates.superglue_cb.manual)

        # Construct the list of strings based on the components of batch and shots
        inputs = composite(batch, shots)
        
        # get tokenized ids and attention masks
        verb_input_ids, verb_att_mask, loss_att_mask = get_input_paremeters(
            tokenizer, inputs, batch_size, class_num, verbalizers
        )

        with torch.no_grad():
            prediction = model(verb_input_ids, verb_att_mask, loss_att_mask, level='predict')

        labels += batch['label']
        predictions += prediction.tolist()

    print('predictions:', predictions)
    print('labels:     ', labels)

    return torch.sum(torch.tensor(predictions)==torch.tensor(labels)) / len(predictions) 
        

def main(args):
    # Random
    print_title('Seed')
    pl.seed_everything(args.seed)

    # Data
    print_title('Dataset')
    dataset = load_dataset('super_glue', args.dataset_name)    
    trainset = list(dataset['train']) # Train dataset -> list of data(Dictionary)
    evalset = list(dataset['validation']) # Eval dataset -> list of data(Dictionary)
    eval_dataloader = DataModule(evalset, resolver=templates.superglue_cb.manual, batch_size=args.batch_size).get_eval_dataloader()

    # Model
    print_title('Model')
    model = LMClassificationModel(args.model_name)
    shot_selector = globals()[args.shot_selector](trainset, args.shot_num, 3, policy=None)

    # TODO: Build Eval_dataloader from Eval_DS
    acc = eval(model, eval_dataloader, shot_selector, args)
    print('Method : {}, Accuracy : '.format(args.shot_selector), acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, choices= ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg'], default='cb')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_source_len', type=int, default=25)
    parser.add_argument('--max_target_len', type=int, default=10)
    parser.add_argument('--shot_selector', type=str, choices=['random_shot_selector','closest_shot_selector','ours_shot_selector'], default='ours_shot_selector')    
    parser.add_argument('--shot_num', type=int, default=4)    
    args = parser.parse_args()

    main(args)