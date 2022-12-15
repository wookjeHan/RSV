import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from util.nlp import get_input_parameters
from util.etc import get_device

class ClassificationEnv:
    '''
    Classification Env which can manage multiple rollout
    '''
    def __init__(self, trainset, inner_sample, label_balance, tokenizer, truncator, language_model, resolver, shot_num, correct_coeff, incorrect_coeff):
        self.trainset = trainset
        self.trainset_size = len(trainset)
        self.inner_sample = inner_sample
        self.label_balance = label_balance

        self.tokenizer = tokenizer
        self.truncator = truncator
        self.language_model = language_model
        self.resolver = resolver
        self.shot_num = shot_num

        sample_shot = resolver(trainset[0:1])
        self.verbalizers = sample_shot['verbalizers']
        self.class_num = len(self.verbalizers)
        self.correct_coeff = correct_coeff
        self.incorrect_coeff = incorrect_coeff

        # TODO: fix it to recieve an embedder as a parameter
        self.embedder = SentenceTransformer('stsb-roberta-large')
        self.init_label_mask()

    def init_label_mask(self):
        label_mask = torch.zeros(self.trainset_size, self.class_num)
        trainset_label = torch.zeros(self.trainset_size, dtype=torch.long)
        for idx, data in enumerate(self.trainset):
            label = data['label']
            label_mask[idx, label] = 1.0
            trainset_label[idx] = label

        self.base_label_mask = label_mask.unsqueeze(0)
        self.trainset_label = trainset_label

    def set_label_mask(self, batch_size):
        self.batch_label_mask = self.base_label_mask.repeat(batch_size, 1, 1)

    def reset(self, resolved_batch, mode):
        '''
        mode: 'train' | 'test'
        train: env is called during training, reward is calculated
        test: env is called during test, reward is set to zero

        inner_sample: boolean
        inner_sample = True: batch is sampled from trainset (=shotset), action mask is enabled
        inner_sample = False: batch is exclusive to trainset (=shotset)
        '''
        self.stepn = 0
        self.mode = mode
        self.input_batch = resolved_batch

        resolved_inputs = resolved_batch['resolved_input']
        indices = resolved_batch['idx']

        self.batch_size = len(resolved_inputs)
        action_mask = torch.ones(self.batch_size, len(self.trainset), device=get_device())
        selected_indices = np.zeros((self.batch_size, self.shot_num))

        if self.inner_sample and mode == 'train':
            action_mask[range(self.batch_size), indices] = 0

        self.set_label_mask(self.batch_size)
        selected_class_mask = torch.zeros(self.batch_size, self.class_num)

        self.state = (0, resolved_inputs, action_mask, selected_indices, selected_class_mask)

        embeddings = self.embedder.encode(resolved_inputs, convert_to_tensor=True, show_progress_bar=False)

        if mode == 'train':
            self.label = resolved_batch['label']
        else:
            self.label = None

        return embeddings, action_mask

    def step(self, action):
        stepn, cur_inputs, action_mask, selected_indices, selected_class_mask = self.state
        indices, replace = action
        selected_indices[:, self.shot_num - 1 - stepn] = indices.detach().cpu().numpy()

        if not replace:
            action_mask[range(self.batch_size), indices] = 0

        newshots = [self.trainset[idx] for idx in indices]
        resolved_result = self.resolver(newshots, include_label=True)

        resolved_newshots = resolved_result['resolved_input']
        prefix = resolved_result['prefix']
        suffix = resolved_result['suffix']
        concatenator = resolved_result['concatenator']

        if stepn == 0:
            cur_inputs = [newshot + suffix + cur_input for newshot, cur_input in zip(resolved_newshots, cur_inputs)]
        else:
            cur_inputs = [newshot + concatenator + cur_input for newshot, cur_input in zip(resolved_newshots, cur_inputs)]

        stepn += 1

        selected_classes = self.trainset_label[indices]
        selected_class_mask[range(self.batch_size), selected_classes] = 1.0

        if self.label_balance:
            not_free = selected_class_mask.sum(dim=1) + (self.shot_num - stepn) <= self.class_num
            not_free = not_free.long().unsqueeze(-1) # batch_size, 1

            # self.batch_label_mask : batch_size, trainset_size, class_num
            # selected_class_mask   : batch_size, class_num, 1

            to_deactivate = torch.matmul(self.batch_label_mask, selected_class_mask.unsqueeze(-1)) # batch_size, trainset_size, 1
            to_deactivate = to_deactivate.squeeze(-1) # batch_size, trainset_size
            activate_mask = 1.0 - to_deactivate * not_free
            # activate_mask = (1 - to_deactivate) * not_free  + (1.0 - not_free) # batch_size, trainset_size
            #                 [1]  [bs, ts]         [bs, 1]      [1]   [bs, 1] 
            action_mask = action_mask * activate_mask.cuda()

        self.state = (stepn, cur_inputs, action_mask, selected_indices, selected_class_mask)
        done = stepn == self.shot_num

        if done and self.mode == 'train':
            resolved_shots_batch = []
            for i in range(self.batch_size):
                shots = [self.trainset[int(idx)] for idx in selected_indices[i]]
                resolved_shots_batch.append(self.resolver(shots))

            truncated_batch = self.truncator.truncate(resolved_shots_batch, self.input_batch)
            rewards = self._get_reward(truncated_batch)
        else:
            rewards = torch.zeros(self.batch_size)

        embeddings = self.embedder.encode(cur_inputs, convert_to_tensor=True, show_progress_bar=False)

        return (embeddings, action_mask), rewards

    @torch.no_grad()
    def _get_reward(self, inputs):
        assert self.mode == 'train'
        verb_input_ids, verb_att_mask, loss_att_mask = get_input_parameters(
            self.tokenizer, inputs, batch_size=self.batch_size, class_num=self.class_num, verbalizers=self.verbalizers
        )

        prob = self.language_model(verb_input_ids, verb_att_mask, loss_att_mask, level='prob')
        prob_ans = prob[range(self.batch_size), self.label]

        prob_copy = prob.clone()
        prob_copy[range(self.batch_size), self.label] = 0
        prob_adv, _ = prob_copy.max(dim=1)

        prob_gap = prob_ans - prob_adv
        correct = (prob_gap > 0).long()

        return prob_gap * (self.correct_coeff * correct + self.incorrect_coeff * (1 - correct))
