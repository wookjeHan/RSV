from torch.utils.data import DataLoader

class DataModule():
    def __init__(self, dataset, resolver, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.resolver = resolver
        self.shuffle = shuffle

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.resolver)