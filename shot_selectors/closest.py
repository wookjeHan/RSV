from sentence_transformers import SentenceTransformer, util
import torch

class ClosestShotSelector():
    def __init__(self, trainset, shot_num, resolver, **kwargs):
        self.trainset = trainset
        self.shot_num = shot_num
        self.resolver = resolver

        self.embedder = SentenceTransformer('stsb-roberta-large')
        self.embeddings = []
        self._calculate_embedding()

    def _calculate_embedding(self):
        inputs = self.resolver(self.trainset, include_label=False)
        self.embeddings = self.embedder.encode(inputs['resolved_input'], convert_to_tensor=True, show_progress_bar=False)

    def __call__(self, batch):
        emb_batch = self.embedder.encode(batch['resolved_input'], convert_to_tensor=True, show_progress_bar=False)
        cosine_score = util.pytorch_cos_sim(emb_batch, self.embeddings)
        topk_values, topk_indices = torch.topk(cosine_score, k=self.shot_num)

        shots = []
        for topk_idx in topk_indices:
            selected_data = [self.trainset[idx] for idx in topk_idx]
            shots.append(self.resolver(selected_data, include_label=True)['resolved_input'])

        return shots