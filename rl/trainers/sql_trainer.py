import torch
import torch.optim as optim
from torch import autograd

from rl.sampler import sample
from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv
from rl.loss import loss1, loss2
from rl.logger import Logger

from util.dataloader import DataModule
from util.nlp import get_input_parameters, composite
from shot_selectors import ours

class SQLTrainer:
    def __init__(
        self,
        policy: MlpPolicy,
        target_policy: MlpPolicy,
        env: ClassificationEnv,
        tokenizer,
        language_model,
        resolver,
        shot_num,

        # Train params
        trainset,
        valset,
        train_variants,
        testset,
        test_variants,
        optimizer_variants,
        save_variants,
    ):
        self.policy = policy
        self.target_policy = target_policy
        self.env = env
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.resolver = resolver
        self.shot_num = shot_num

        self.trainset = trainset
        self.valset = valset
        self.train_batch_size = train_variants['batch_size']
        self.train_shuffle = train_variants['shuffle']
        self.soft_update_ratio = train_variants['soft_update_ratio']
        self.update_period = train_variants['update_period']
        self.num_train_epochs = train_variants['num_epochs']
        self.temperature = train_variants['temperature']
        self.topk = train_variants['topk']

        self.testset = testset
        self.test_batch_size = test_variants['batch_size']

        self.logger: Logger = save_variants['logger']

        self.optimizer = optim.Adam(policy.parameters(), lr=optimizer_variants['lr'])

        self.traindataloader_generator = DataModule(valset, resolver, self.train_batch_size, self.train_shuffle)
        self.testdataloader_generator = DataModule(testset, resolver, self.test_batch_size, False)

    def compute_loss(self, batch):
        logits, target_logits, actions, rewards = sample(
            self.policy, self.target_policy, self.env, batch, self.shot_num
        )

        l1 = loss1(logits, target_logits, actions, rewards, self.topk, self.temperature)
        l2 = loss2(logits, target_logits, actions, rewards, self.topk, self.temperature)

        l1r = loss1(target_logits, logits, actions, rewards, self.topk, self.temperature)
        l2r = loss2(target_logits, logits, actions, rewards, self.topk, self.temperature)

        self.logger.add_array_stat('loss1', l1)
        self.logger.add_array_stat('loss2', l2)
        self.logger.add_array_stat('loss1 reverse', l1r)
        self.logger.add_array_stat('loss2 reverse', l2r)
        self.logger.add_array_stat('train reward', rewards)

        return 0.25 * (l1 + l2 + l1r + l2r).mean(), rewards > 0

    def update_target_policy(self):
        target_parameters = self.target_policy.parameters()
        source_parameters = self.policy.parameters()
        tau = self.soft_update_ratio

        for target_param, source_param in zip(target_parameters, source_parameters):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def train(self):
        train_dataloader = self.traindataloader_generator.get_dataloader()

        self.total_steps = 0
        self.total_update_num = 0

        # Determine whether to train by epoch or steps
        for self.epoch in range(self.num_train_epochs):
            self.policy.train()
            epoch_loss = 0
            corrects = 0
            dataset_size = 0
            for step, batch in enumerate(train_dataloader):
                loss, correct = self.compute_loss(batch)

                epoch_loss += loss.item() * len(correct)
                corrects += correct.long().sum()
                dataset_size += len(correct)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.total_steps += 1

                if self.total_steps % self.update_period == 0:
                    self.update_target_policy()
                    self.total_update_num += 1

            self.logger.add_scalar('Train Loss', epoch_loss / dataset_size)
            self.logger.add_scalar('Train Acc', corrects / dataset_size * 100)
            self.test()
            self.logger.add_scalar('Epoch', self.epoch)
            self.logger.flush()

    @torch.no_grad()
    def test(self):
        shot_selector = ours(self.trainset, self.shot_num, self.resolver, self.policy, self.env)
        predictions = []
        labels = []
        dataloader = self.testdataloader_generator.get_dataloader()

        for resolved_batch in dataloader:
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
                self.tokenizer, inputs, batch_size, class_num, verbalizers
            )

            prediction = self.language_model(
                verb_input_ids,
                verb_att_mask,
                loss_att_mask,
                level='predict',
            )

            labels += resolved_batch['label']
            predictions += prediction.tolist()

        acc = torch.sum(torch.tensor(predictions) == torch.tensor(labels)) / len(predictions) * 100
        self.logger.add_scalar('Test Acc', acc)
