import torch
from sentence_transformers import SentenceTransformer

from util.nlp import get_input_parameters
from util.etc import get_device

class ClassificationEnv():
    '''
    Classification Env which can manage multiple rollout
    '''
    def __init__(self, trainset, resolver, shot_num, language_model, tokenizer, verbalizers, correct_coeff, incorrect_coeff):
        self.trainset = trainset
        self.resolver = resolver
        self.shot_num = shot_num
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.verbalizers = verbalizers
        self.class_num = len(verbalizers)
        self.correct_coeff = correct_coeff
        self.incorrect_coeff = incorrect_coeff

        # TODO: fix it to recieve an embedder as a parameter
        self.embedder = SentenceTransformer('stsb-roberta-large')

    def reset(self, resolved_batch, endo_train, mode):
        '''
        mode: 'train' | 'test'
        train: env is called during training, reward is calculated
        test: env is called during test, reward is set to zero

        endo_train: boolean
        endo_train = True: batch is sampled from trainset (=shotset), action mask is enabled
        endo_train = False: batch is exclusive to trainset (=shotset)

        '''
        self.stepn = 0
        self.mode = mode

        resolved_inputs = resolved_batch['resolved_input']
        indices = resolved_batch['idx']

        batch_size = len(resolved_inputs)
        action_mask = torch.ones(batch_size, len(self.trainset), device=get_device())

        if endo_train:
            action_mask[range(batch_size), indices] = 0

        self.state = (0, resolved_inputs, action_mask)

        self.threadn = batch_size
        embeddings = self.embedder.encode(resolved_inputs, convert_to_tensor=True,show_progress_bar=False)

        if mode == 'train':
            self.label = resolved_batch['label']
        else:
            self.label = None


        return embeddings, action_mask

    def step(self, action):
        stepn, cur_inputs, action_mask = self.state
        indices, replace = action

        if not replace:
            action_mask[range(self.threadn), indices] = 0

        newshots = [self.trainset[index] for index in indices]
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
        self.state = (stepn, cur_inputs, action_mask)
        done = stepn == self.shot_num

        if done and self.mode == 'train':
            cur_inputs = [prefix + cur_input for cur_input in cur_inputs]
            rewards = self._get_reward(cur_inputs)
        else:
            rewards = torch.zeros(self.threadn)

        embeddings = self.embedder.encode(cur_inputs, convert_to_tensor=True,show_progress_bar=False)

        return (embeddings, action_mask), rewards

    @torch.no_grad()
    def _get_reward(self, inputs):
        assert self.mode == 'train'
        verb_input_ids, verb_att_mask, loss_att_mask = get_input_parameters(
            self.tokenizer, inputs, batch_size=self.threadn, class_num=self.class_num, verbalizers=self.verbalizers
        )

        prob = self.language_model(verb_input_ids, verb_att_mask, loss_att_mask, level='prob')
        prob_ans = prob[range(self.threadn), self.label]

        prob_copy = prob.clone()
        prob_copy[range(self.threadn), self.label] = 0
        prob_adv, _ = prob_copy.max(dim=1)

        prob_gap = prob_ans - prob_adv
        correct = (prob_gap > 0).long()

        return prob_gap * (self.correct_coeff * correct + self.incorrect_coeff * (1 - correct))
