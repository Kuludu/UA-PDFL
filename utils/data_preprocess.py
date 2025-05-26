import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from medmnist import PathMNIST
from torchvision import datasets
from torchvision.transforms import transforms


class DatasetHelper(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]

        return image, label


def iid_partition(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    user_dataset_indexes, all_idxs = list(), [i for i in range(len(dataset))]
    for i in range(num_users):
        selected_idxs = np.random.choice(all_idxs, num_items, replace=False)
        user_dataset_indexes.append(selected_idxs)
        all_idxs = list(set(all_idxs) - set(selected_idxs))

    return user_dataset_indexes


def dirichlet_partition(dataset, num_users, beta):
    y_labels = np.array(dataset.targets)
    n_classes = len(set(y_labels))

    min_samples = 0
    min_required_samples = 15
    user_idxs = []
    while min_samples < min_required_samples:
        user_idxs = [[] for _ in range(num_users)]
        for k in range(n_classes):
            data_idx_k = np.where(y_labels == k)[0]
            np.random.shuffle(data_idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, num_users))
            proportions = np.array([p * (len(idx) < len(dataset) / num_users)
                                    for p, idx in zip(proportions, user_idxs)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(data_idx_k)).astype(int)[:-1]
            user_idxs = [uidx + idx.tolist() for uidx, idx in zip(user_idxs, np.split(data_idx_k, proportions))]
            min_samples = min([len(idx) for idx in user_idxs])
    for j in range(num_users):
        np.random.shuffle(user_idxs[j])

    return user_idxs


def fix_class_noniid(dataset, num_users, num_classes=2):
    # Modified for PathMNIST
    num_samples_per_client = len(dataset) // num_users
    class_idx = list(range(9))
    targets = np.array(dataset.targets)

    idx = targets.argsort()
    idxs = {}
    start = 0
    class_size = [9366, 9509, 10360, 10401, 8006, 12182, 7886, 9401, 1288]
    for i in range(9):
        idxs[i] = idx[start:start + class_size[i]]
        np.random.shuffle(idxs[i])
        start += class_size[i]

    user_dataset_indexes = [None for _ in range(num_users)]

    for i in range(num_users):
        selected_class = np.random.choice(class_idx, num_classes, replace=False)
        user_dataset_indexes[i] = np.concatenate(
            [np.random.choice(idxs[selected_class[j]], num_samples_per_client // num_classes)
             for j in range(num_classes)])

    return user_dataset_indexes


def split_train_test(user_data_indexes, test_ratio=0.2):
    train_index, test_index = list(), list()
    for user_data_index in user_data_indexes:
        random.shuffle(user_data_index)
        len_user_data_index = len(user_data_index)
        test_set_size = int(len_user_data_index * test_ratio)
        train_index.append(user_data_index[test_set_size:])
        test_index.append(user_data_index[:test_set_size])

    return train_index, test_index


def split_few_shot_train_test(user_data_indexes, num_way, num_shot, test_shot_ratio=2):
    train_index, test_index = list(), list()
    for user_data_index in user_data_indexes:
        train_index.append(list())
        test_index.append(list())
        for i in range(num_way):
            train_index[-1].extend(user_data_index[i * num_shot + i * num_shot * (1 + test_shot_ratio):
                                                   (i + 1) * num_shot + i * num_shot * (1 + test_shot_ratio)])
            test_index[-1].extend(user_data_index[(i + 1) * num_shot + i * num_shot * (1 + test_shot_ratio):
                                                  (i + 1) * num_shot + (i + 1) * num_shot * (1 + test_shot_ratio)])

    return train_index, test_index


def get_dataset(path, dataset):
    train_data, test_data, num_classes = None, None, int()

    if dataset == "cifar10":
        cifar10_transforms = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010))])
        train_data = datasets.CIFAR10(path + "data/cifar10", train=True, download=True, transform=cifar10_transforms)
        num_classes = 10
    elif dataset == "cifar100":
        cifar100_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                       (0.2675, 0.2565, 0.2761))])
        train_data = datasets.CIFAR100(path + "data/cifar100", train=True, download=True, transform=cifar100_transforms)
        num_classes = 100
    elif dataset == "svhn":
        svhn_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4376, 0.4437, 0.4728),
                                                                   (0.1980, 0.2010, 0.1970))])
        train_data = datasets.SVHN(path + "data/svhn", split="train", download=True, transform=svhn_transforms)
        # Alias
        train_data.targets = train_data.labels
        num_classes = 10
    elif dataset == "pathmnist":
        pathmnist_transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                     (0.5, 0.5, 0.5))])
        train_data = PathMNIST(split="train", root=path + "data/pathmnist", download=True, transform=pathmnist_transforms)
        # Alias
        train_data.targets = train_data.labels.reshape(-1)
        num_classes = 10

    assert train_data is not None, "Dataset not defined."

    print("Dataset %s selected, train_data: %d." % (dataset, len(train_data)))

    return train_data, num_classes


def get_loaders(train_data, num_clients, distribution, distribution_parameter=tuple(), batch_size=128):
    user_data_indexes = None
    if distribution == "dirichlet":
        user_data_indexes = dirichlet_partition(train_data, num_clients, *distribution_parameter)
    elif distribution == "iid":
        user_data_indexes = iid_partition(train_data, num_clients)
    elif distribution == "fix_class":
        user_data_indexes = fix_class_noniid(train_data, num_clients)

    assert user_data_indexes is not None, "Distribution not defined."
    print("Distribution %s selected, train_batch: %d" % (distribution, batch_size))

    train_loaders, test_loaders = list(), list()

    user_train_dataset, user_test_dataset = split_train_test(user_data_indexes)
    train_loaders = [
        DataLoader(DatasetHelper(train_data, user_train_dataset[i]),
                    batch_size=batch_size, shuffle=True)
        for i in range(num_clients)
    ]
    test_loaders = [
        DataLoader(DatasetHelper(train_data, user_test_dataset[i]),
                    batch_size=batch_size, shuffle=False)
        for i in range(num_clients)
    ]

    return train_loaders, test_loaders
