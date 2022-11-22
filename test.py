import argparse
import torch

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
from config import GlobalConfig, TestSnapshotIndex as tsi


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
    shot_selector_trainset, _, _ = get_splited_dataset(tsi)

    # Resolver & Verbalizer
    resolver = getattr(getattr(resolvers, args.dataset), args.prompt)
    sample_shot = resolver(trainset[0:1])
    verbalizers = sample_shot['verbalizers']

    # Dataloader
    test_dataloader = DataModule(testset, resolver=resolver, batch_size=args.batch_size).get_dataloader()

    # Model
    language_model = ClassificationModel(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Truncator
    truncator = Truncator(tokenizer, 1024, 256)

    M = len(shot_selector_trainset)
    policy = MlpPolicy(1024, [2 * M, 2 * M], M, tsi.replace).cuda()
    exp_name = get_exp_name(tsi)

    for shot_num in [12, 16]:
        print(f">>>shot num {shot_num}>>>")
        env = ClassificationEnv(
            shot_selector_trainset,
            tsi.tv_split_ratio == 0.0,
            tokenizer,
            truncator,
            language_model,
            resolver,
            shot_num,
            200.0,
            180.0,
        )

        for shot_selector_func in [shot_selectors.ours, shot_selectors.random, shot_selectors.closest]:
            print(shot_selector_func)
            for seed in range(4):
                if shot_selector_func == shot_selectors.ours:
                    tsi.seed = seed
                    snapshot_path = f"result/{exp_name}/{tsi.seed}/itr_{tsi.epoch}.pt"
                    state_dict = torch.load(snapshot_path, map_location=get_device())
                    policy.load_state_dict(state_dict)
                    shot_selector = shot_selector_func(shot_selector_trainset, shot_num, resolver=resolver, policy=policy, env=env)
                else:
                    shot_selector = shot_selector_func(trainset, shot_num, resolver=resolver, policy=policy, env=env)
                acc = test(language_model, tokenizer, test_dataloader, shot_selector, truncator)
                print(f"[{seed:1d}/4] Accuracy {acc:.3f}")
                del shot_selector

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--language_model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='superglue_cb')
    parser.add_argument('--prompt', type=str, default='manual')

    parser.add_argument('--batch_size', type=int, default=GlobalConfig.batch_size)
    args = parser.parse_args()

    main(args)