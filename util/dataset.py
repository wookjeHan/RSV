from torch.utils.data import DataLoader
from datasets import load_dataset

class DataModule():
    def __init__(self, dataset, resolver, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.resolver = resolver
        self.shuffle = shuffle

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.resolver)

def _fix_index(dataset):
    for idx, data in enumerate(dataset):
        data['idx'] = idx

def get_splited_dataset(args):
    # TODO: get super_glue and cb from args
    dataset_args = args.dataset.split(",")
    dataset = load_dataset(*dataset_args)

    total_trainset = list(dataset['train'])
    total_valset = list(dataset['validation'])

    # split total_trainset into trainset and valset
    if hasattr(args, 'tv_split_ratio') and args.tv_split_ratio > 0.0:
        trainset_size = len(total_trainset)
        split_idx = int(trainset_size * args.tv_split_ratio)

        trainset = total_trainset[:split_idx] # Train dataset -> list of data(Dictionary)
        valset = total_trainset[split_idx:] # Validation dataset -> list of data(Dictionary)
        testset = total_valset # Test dataset -> list of data(Dictionary)
        _fix_index(trainset)
        _fix_index(valset)
    # use total_trainset as both trainset and valset
    else:
        trainset = total_trainset
        valset = total_trainset
        testset = total_valset

    print("Trainset:", len(trainset))
    print("Valset:", len(valset))
    print("Testset:", len(testset))

    return trainset, valset, testset
