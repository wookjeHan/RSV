import argparse
import torch
import pytorch_lightning as pl

from transformers import AutoTokenizer
from language_model import ClassificationModel
from datasets import load_dataset

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

from util.dataloader import DataModule
from util.nlp import composite, get_input_parameters

import resolvers
import shot_selectors
from config import GlobalConfig


@torch.no_grad()
def test(language_model, tokenizer, test_dataloader, shot_selector, args):
    predictions = []
    labels = []

    for resolved_batch in test_dataloader:
        # The batch is consisted of prefix, concatenator, suffix, resolved_input, label, and verbalizers
        batch_size = len(resolved_batch['resolved_input'])
        verbalizers = resolved_batch['verbalizers']
        class_num = len(verbalizers)

        # Shots are selected based on the batch
        # The datastructure of shots is a list of string (batch_size, shot_num)
        shots = shot_selector(resolved_batch)

        # Construct the list of strings based on the components of batch and shots
        inputs = composite(resolved_batch, shots)
        
        # get tokenized ids and attention masks
        verb_input_ids, verb_att_mask, loss_att_mask = get_input_parameters(
            tokenizer, inputs, batch_size, class_num, verbalizers
        )

        prediction = language_model(
            verb_input_ids,
            verb_att_mask,
            loss_att_mask,
            level='predict',
        )

        labels += resolved_batch['label']
        predictions += prediction.tolist()

    acc = torch.sum(torch.tensor(predictions) == torch.tensor(labels)) / len(predictions) 
    return acc.item()

def main(args):
    # Random seed
    pl.seed_everything(args.seed)

    # Datasets
    # TODO: split superglue_cb into super_glue and cb
    dataset = load_dataset('super_glue', 'cb')  
    total_trainset = list(dataset['train'])
    total_valset = list(dataset['validation'])
    trainset_size = len(total_trainset)
    split_idx = int(trainset_size * args.tv_split_ratio)

    trainset = total_trainset[:split_idx] # Train dataset -> list of data(Dictionary)
    valset = total_trainset[split_idx:] # Validation dataset -> list of data(Dictionary)
    testset = total_valset # Eval dataset -> list of data(Dictionary)

    # Resolver & Verbalizer
    resolver = getattr(getattr(resolvers, args.dataset), args.prompt)
    sample_shot = resolver(trainset[0:1])
    verbalizers = sample_shot['verbalizers']
    
    # Dataloader
    test_dataloader = DataModule(testset, resolver=resolver, batch_size=args.batch_size).get_dataloader()

    print("Trainset:", len(trainset))
    print("Valset:", len(valset))
    print("Testset:", len(testset))

    # Model
    language_model = ClassificationModel(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    policy = MlpPolicy(len(trainset)).cuda()
    env = ClassificationEnv(
        trainset,
        resolver,
        args.shot_num,
        language_model,
        tokenizer,
        verbalizers,
        2.0,
        1.0,
    )    

    # TODO: Build Eval_dataloader from Eval_DS
    for shot_selector_func in [shot_selectors.random, shot_selectors.closest, shot_selectors.ours]:
        for repeat in range(4):
            shot_selector = shot_selector_func(trainset, args.shot_num, resolver=resolver, policy=policy, env=env)
            acc = test(language_model, tokenizer, test_dataloader, shot_selector, args)
            print(f"[{repeat:1d}/4] Accuracy {acc:.3f}")
            del shot_selector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--language_model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='superglue_cb')
    parser.add_argument('--prompt', type=str, default='manual')
    
    parser.add_argument('--batch_size', type=int, default=GlobalConfig.batch_size)
    parser.add_argument('--shot_num', type=int, default=GlobalConfig.shot_num)
    parser.add_argument('--tv_split_ratio', type=float, default=GlobalConfig.tv_split_ratio)
    args = parser.parse_args()

    main(args)