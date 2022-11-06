from torch.utils.data import DataLoader

class DataModule():
    def __init__(self, dataset, resolver, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.resolver = resolver

    def get_eval_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.resolver)