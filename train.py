import argparse
import pytorch_lightning as pl

from transformers import AutoTokenizer
from language_model import ClassificationModel
from datasets import load_dataset

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv
from rl.trainers.sql_trainer import SQLTrainer

from util.etc import set_device, get_device

import resolvers
from config import GlobalConfig

def main(args):
    # Random Seed
    pl.seed_everything(args.seed)

    # Device
    set_device('cuda')
    device = get_device()

    resolver = getattr(getattr(resolvers, args.dataset), args.prompt)

    # Datasets
    # TODO: split superglue_cb into super_glue and cb
    dataset = load_dataset('super_glue', 'cb')    
    trainset = list(dataset['train'])[0:200] # Train dataset -> list of data(Dictionary)
    valset = list(dataset['train'])[200:] # Validation dataset -> list of data(Dictionary)
    testset = list(dataset['validation']) # Test dataset -> list of data(Dictionary)

    print("Trainset:", len(trainset))
    print("Valset:", len(valset))
    print("Testset:", len(testset))

    # Verbalizer
    resolved_shot = resolver(trainset[0:1])
    verbalizers = resolved_shot['verbalizers']

    # Lauguage related models
    language_model = ClassificationModel(args.language_model)    
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Policies
    policy = MlpPolicy(len(trainset)).to(device=device)
    target_policy = MlpPolicy(len(trainset)).to(device=device)

    # Env
    env = ClassificationEnv(trainset, resolver, args.shot_num, language_model, tokenizer, verbalizers, 2.0, 1.0)

    # Variants
    train_variants = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'soft_update_ratio': 0.3,
        'update_period': 10,
        'num_epochs': 1000,
    }
    test_variants = {
        'batch_size': args.batch_size,
    }
    optimizer_variants = {
        'lr': 3e-4,
    }
    save_variants = {
        'save_results': True,
        'save_dir': None,
        'save_steps': None,
    }

    # Trainer
    trainer = SQLTrainer(
        policy,
        target_policy,
        env,
        tokenizer,
        language_model,
        resolver,
        args.shot_num,
        trainset,
        valset,
        train_variants,
        testset,
        test_variants,
        optimizer_variants,
        save_variants
    )

    trainer.train()

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