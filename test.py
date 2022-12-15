import argparse
import torch
import random

from transformers import AutoTokenizer
from language_model import ClassificationModel

from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

from util.dataset import DataModule, get_splited_dataset
from util.nlp import get_input_parameters
from util.truncator import Truncator
from util.etc import fix_seed, get_exp_name, set_device, get_device

import resolvers
import shot_selectors

@torch.no_grad()
def test(language_model, tokenizer, test_dataloader, shot_selector, truncator):
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
        inputs = truncator.truncate(shots, resolved_batch)

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
    fix_seed(args.seed)

    # USe GPU
    set_device('cuda')

    # Datasets
    trainset, valset, testset = get_splited_dataset(args)
    trainset = trainset[:100]
    set_size = len(testset) // 2

    valset = testset[:set_size]
    testset = testset[set_size:]

    # Resolver & Verbalizer
    resolver_name = "_".join(args.dataset.split(","))
    resolver = getattr(getattr(resolvers, resolver_name), args.prompt)

    # Dataloader
    random.shuffle(testset)
    val_dataloader = DataModule(valset, resolver=resolver, batch_size=args.batch_size).get_dataloader()
    test_dataloader = DataModule(testset, resolver=resolver, batch_size=args.batch_size).get_dataloader()

    # Model
    language_model = ClassificationModel(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Truncator
    truncator = Truncator(tokenizer, 1024, 256)

    M = len(trainset)
    policy = MlpPolicy(1024, [M, M], M, True).cuda()

    print(f">>>shot num {args.shot_num}>>>")
    env = ClassificationEnv(
        trainset,
        True,
        False,
        tokenizer,
        truncator,
        language_model,
        resolver,
        args.shot_num,
        200.0,
        180.0,
    )

    for shot_selector_func in [shot_selectors.ours]:
        print(shot_selector_func)
        if shot_selector_func == shot_selectors.ours:
            for seed in range(4):
                snapshot_path = f"result/OURS/{args.dataset}_sr0.0/{args.prompt}/sn{args.shot_num}_k5_lr0.001_bs4_inner/{seed}/itr_90.pt"
                state_dict = torch.load(snapshot_path, map_location=get_device())
                policy.load_state_dict(state_dict)

                shot_selector = shot_selector_func(trainset, args.shot_num, resolver=resolver, policy=policy, env=env)
                val_acc = test(language_model, tokenizer, val_dataloader, shot_selector, truncator)
                test_acc = test(language_model, tokenizer, test_dataloader, shot_selector, truncator)
                print(f"[{seed}/4] Val Acc {val_acc * 100:.1f} Test Accuracy {test_acc * 100:.1f}")
                del shot_selector

        elif shot_selector_func == shot_selectors.random:
            for repeat in range(20):
                shot_selector = shot_selector_func(trainset, args.shot_num, resolver=resolver)
                val_acc = test(language_model, tokenizer, val_dataloader, shot_selector, truncator)
                test_acc = test(language_model, tokenizer, test_dataloader, shot_selector, truncator)
                print(f"[{repeat}/20] Val Acc {val_acc * 100:.1f} Test Accuracy {test_acc * 100:.1f}")
                del shot_selector

        elif shot_selector_func == shot_selectors.closest:
            shot_selector = shot_selector_func(trainset, args.shot_num, resolver=resolver)
            val_acc = test(language_model, tokenizer, val_dataloader, shot_selector, truncator)
            test_acc = test(language_model, tokenizer, test_dataloader, shot_selector, truncator)
            print(f"[0/1] Val Acc {val_acc * 100:.1f} Test Accuracy {test_acc * 100:.1f}")
            del shot_selector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--language_model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='super_glue,cb')
    parser.add_argument('--prompt', type=str, default='manual')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shot_num', type=int, default=2)
    args = parser.parse_args()

    main(args)