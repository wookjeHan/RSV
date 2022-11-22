import torch.optim as optim

from rl.sampler import sample
from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv
from rl.loss import loss1, loss2
from rl.logger import Logger

from util.dataset import DataModule
from shot_selectors import ours
from test import test

class SQLTrainer:
    def __init__(
        self,
        policy: MlpPolicy,
        target_policy: MlpPolicy,
        train_env: ClassificationEnv,
        test_env: ClassificationEnv,

        tokenizer,
        truncator,
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
        self.train_env = train_env
        self.test_env = test_env

        self.tokenizer = tokenizer
        self.truncator = truncator
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
        self.save_mode = save_variants['save_mode']
        self.save_freq = save_variants['save_freq']

        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=optimizer_variants['lr'],
            weight_decay=optimizer_variants['weight_decay']
        )

        self.traindataloader_generator = DataModule(valset, resolver, self.train_batch_size, self.train_shuffle)
        self.testdataloader_generator = DataModule(testset, resolver, self.test_batch_size, False)

    def _compute_loss(self, batch):
        logits, target_logits, actions, rewards = sample(
            self.policy, self.target_policy, self.train_env, batch, self.shot_num
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

    def _update_target_policy(self):
        target_parameters = self.target_policy.parameters()
        source_parameters = self.policy.parameters()
        tau = self.soft_update_ratio

        for target_param, source_param in zip(target_parameters, source_parameters):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    def _save_snapshot(self):
        if self.save_mode == 'freq' and self.epoch % self.save_freq == 0 or \
            self.save_mode == 'all' and self.epoch == self.num_train_epochs - 1:
            self.logger.save_snapshot(self.policy.state_dict(), self.epoch)

    def train(self):
        self.total_steps = 0
        for self.epoch in range(self.num_train_epochs):
            self._run_epoch()

    def _train(self):
        # Create a dataloader
        dataloader = self.traindataloader_generator.get_dataloader()

        # Epoch statistics
        epoch_loss = 0
        corrects = 0
        dataset_size = 0

        for batch in dataloader:
            loss, correct = self._compute_loss(batch)

            epoch_loss += loss.item() * len(correct)
            corrects += correct.long().sum()
            dataset_size += len(correct)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_steps += 1

            if self.total_steps % self.update_period == 0:
                self._update_target_policy()

        self.logger.add_scalar('Total Weight', self.policy.get_weight_sum())
        self.logger.add_scalar('Train Loss', epoch_loss / dataset_size)
        self.logger.add_scalar('Train Acc', corrects / dataset_size * 100)

    def _test(self):
        # Create a shot selector based on the current policy
        shot_selector = ours(self.trainset, self.shot_num, self.resolver, self.policy, self.test_env)

        # Create a dataloader
        dataloader = self.testdataloader_generator.get_dataloader()

        acc = test(self.language_model, self.tokenizer, dataloader, shot_selector, self.truncator)
        self.logger.add_scalar('Test Acc', acc * 100)

    def _run_epoch(self):
        # Set self.policy as train mode
        self.policy.train()

        self._train()
        self._test()

        # End epoch
        self.logger.add_scalar('Total Steps', self.total_steps)
        self.logger.add_scalar('Epoch', self.epoch)
        self.logger.flush()
        self._save_snapshot()
