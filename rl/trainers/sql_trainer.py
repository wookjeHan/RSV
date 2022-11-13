import torch
import torch.optim as optim
import torch.nn.functional as F

from rl.samplers.sample import sample
from rl.policies.mlp_policy import MlpPolicy
from rl.envs.classification_env import ClassificationEnv

from util.dataloader import DataModule

class SQLTrainer:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

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

        self.testset = testset
        self.test_batch_size = test_variants['batch_size']

        self.save_results = save_variants['save_results']
        self.save_dir = save_variants['save_dir']
        self.save_steps = save_variants['save_steps']

        self.optimizer = optim.Adam(policy.parameters(), lr=optimizer_variants['lr'])

        self.traindataloader_generator = DataModule(valset, resolver, self.train_batch_size, self.train_shuffle)
        self.testdataloader_generator = DataModule(testset, resolver, self.test_batch_size, False)

    def compute_loss(self, batch):
        # TODO: implement more refined loss
        logits, target_logits, rewards = sample(
            self.policy, self.target_policy, self.env, batch, self.shot_num
        )

        value = logits.logsumexp(dim=-1)
        target_value = target_logits.logsumexp(dim=-1)
        return F.mse_loss(value, target_value)

    def update_target_policy(self):
        target_parameters = self.target_policy.parameters()
        source_parameters = self.policy.parameters()
        tau = self.soft_update_ratio

        for target_param, param in zip(target_parameters, source_parameters):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def train(self):
        train_dataloader = self.traindataloader_generator.get_dataloader()

        self.total_steps = 0

        # Determine whether to train by epoch or steps
        for epoch in range(self.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                loss = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.total_steps += 1

                if self.total_steps % self.update_period == 0:
                    self.update_target_policy()

            self.test()
            

    def test(self):
        test_dataloader = self.testdataloader_generator.get_dataloader()

        self.policy.eval()
        correct = []
        total_rewards = []

        for resolved_inputs in test_dataloader:
            states = self.env.reset(resolved_inputs, 'train')

            for step in range(self.shot_num):
                actions = self.policy.get_actions(states, max_action=True)
                states, rewards = self.env.step(actions)

            correct += (rewards > 0).detach().tolist()
            total_rewards += rewards.detach().tolist()

        print(f"Test Set Acc {sum(correct) / len(correct) * 100}% | Mean Reward {sum(total_rewards) / len(total_rewards)}")
