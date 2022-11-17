import os
import argparse

from transformers import AutoTokenizer
from language_model import ClassificationModel
from datasets import load_dataset

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv
from rl.trainers.sql_trainer import SQLTrainer
from rl.logger import Logger

from util.etc import set_device, get_device, fix_seed, get_exp_name

import resolvers
from config import GlobalConfig as gc

# torch.set_printoptions(threshold=10_000)

def main(args):
    # Random Seed
    fix_seed(args.seed)

    # Device
    set_device('cuda')
    device = get_device()

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

    print("Trainset:", len(trainset))
    print("Valset:", len(valset))
    print("Testset:", len(testset))

    # Resolver & Verbalizer
    resolver = getattr(getattr(resolvers, args.dataset), args.prompt)
    sample_shot = resolver(trainset[0:1])
    verbalizers = sample_shot['verbalizers']

    # Lauguage related models
    language_model = ClassificationModel(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Policies
    M = len(trainset)
    policy = MlpPolicy(1024, [2 * M, 2 * M], M).to(device=device)
    target_policy = MlpPolicy(1024, [2 * M, 2 * M], M).to(device=device)

    # Env
    env = ClassificationEnv(
        trainset,
        resolver,
        args.shot_num,
        language_model,
        tokenizer,
        verbalizers,
        200.0,
        180.0,
    )

    # Logger
    if args.save_result:
        exp_name = get_exp_name(args)
        result_dir = os.path.join(exp_name, str(args.seed))
    else:
        result_dir = 'DEBUG'
    logger = Logger('result', result_dir, result_dir, result_dir)

    # Variants
    train_variants = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'soft_update_ratio': 1.0, # 1.0 for hard update, 0.0 for no update
        'update_period': 10,
        'num_epochs': 200,
        'temperature': args.temperature,
        'topk': args.topk,
    }
    test_variants = {
        'batch_size': args.batch_size,
    }
    optimizer_variants = {
        'lr': 3e-4,
        'weight_decay': args.weight_decay,
    }
    save_variants = {
        'logger': logger,
        'save_mode': args.save_mode,
        'save_freq': args.save_freq,
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

    parser.add_argument('--batch_size', type=int, default=gc.batch_size)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--shot_num', type=int, default=gc.shot_num)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--tv_split_ratio', type=float, default=gc.tv_split_ratio)

    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--save_mode', type=str, default='freq', choices=['last', 'freq'])
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='OURS')
    args = parser.parse_args()

    main(args)