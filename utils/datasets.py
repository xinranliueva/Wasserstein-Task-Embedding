import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import os


class loaders_NIST():
    def __init__(self, dataset_name, toy_dataset=False):
        self.dataset = dataset_name
        assert self.dataset in ['MNIST', 'FashionMNIST',
                                'EMNIST', 'KMNIST', 'QMNIST', 'USPS']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(28)
        ])
        self.subset = toy_dataset

    def get_dataset(self):
        if not self.subset:
            if self.dataset == 'MNIST':
                train = datasets.MNIST(
                    root='./dataset', train=True, transform=self.transform, download=True)
                test = datasets.MNIST(
                    root='./dataset', train=False, transform=self.transform, download=True)
            elif self.dataset == 'FashionMNIST':
                train = datasets.FashionMNIST(
                    root='./dataset', train=True, transform=self.transform, download=True)
                test = datasets.FashionMNIST(
                    root='./dataset', train=False, transform=self.transform, download=True)
            elif self.dataset == 'EMNIST':
                train = datasets.EMNIST(root='./dataset', split='mnist',
                                        train=True, transform=self.transform, download=True)
                test = datasets.EMNIST(root='./dataset', split='mnist',
                                       train=False, transform=self.transform, download=True)
            elif self.dataset == 'KMNIST':
                train = datasets.KMNIST(
                    root='./dataset', train=True, transform=self.transform, download=True)
                test = datasets.KMNIST(
                    root='./dataset', train=False, transform=self.transform, download=True)
            elif self.dataset == 'QMNIST':
                train = datasets.QMNIST(
                    root='./dataset', compat=False, train=True, transform=self.transform, download=True)
                test = datasets.QMNIST(
                    root='./dataset', compat=False, train=False, transform=self.transform, download=True)
            elif self.dataset == 'USPS':
                train = datasets.USPS(
                    root='./dataset', train=True, transform=self.transform, download=True)
                test = datasets.USPS(root='./dataset', train=False,
                                     transform=self.transform, download=True)
        else:
            train_, test_ = loaders_NIST(self.dataset).get_dataset()
            train, test = self.get_pair_digits(train_, test_)

        return train, test

    def get_pair_digits(self, train, test):
        datasets_train = []
        datasets_test = []

        for i in range(10):
            for j in range(i+1, 10):
                for k in range(3):
                    datasets_i = [
                        (data, i) for data in train.data[torch.tensor(train.targets) == i]]
                    datasets_j = [
                        (data, j) for data in train.data[torch.tensor(train.targets) == j]]
                    datasets_train.append(datasets_i+datasets_j)

                datasets_i = [(data, i)
                              for data in test.data[torch.tensor(test.targets) == i]]
                datasets_j = [(data, j)
                              for data in test.data[torch.tensor(test.targets) == j]]
                datasets_test.append(datasets_i+datasets_j)

        return datasets_train, datasets_test


class loaders_CIFAR():
    def __init__(self, seed=19):
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                 [0.2675, 0.2565, 0.2761])
        ])

    def get_loaders(self):
        torch.manual_seed(self.seed)
        coarse_cifar100 = torch.randperm(100).reshape(10, 10)
        train_loaders = []
        test_loaders = []

        for i in range(10):
            train_loaders.append(self.create_cifar100_task(
                coarse_cifar100, [i], train=True)[1])
            test_loaders.append(self.create_cifar100_task(
                coarse_cifar100, [i], train=False)[1])
        return train_loaders, test_loaders

    def create_cifar100_task(self, split, tasks, train=True, shuffle=False, bs=256):
        tmap, cmap = {}, {}
        for tid, task in enumerate(split):
            for lid, lab in enumerate(task):
                tmap[lab.item()], cmap[lab.item()] = tid, lid
        dataset = datasets.CIFAR100(
            root='dataset', train=train, download=False, transform=self.transform)

        task_labels = []
        for task_id in tasks:
            task_labels += split[task_id]
        idx = np.where(np.isin(dataset.targets, task_labels))[0]

        dataset.targets = torch.tensor([cmap[dataset.targets[j]] for j in idx])
        dataset.data = dataset.data[idx]

        dataloader = DataLoader(
            dataset, batch_size=bs, shuffle=shuffle,
            num_workers=0, pin_memory=True)

        return dataset, dataloader


class loaders_TIN():
    def __init__(self, seed=19):
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.train = TinyImageNetDataset(
            root_dir='./dataset/tin', mode='train', transform=self.transform)
        self.val = TinyImageNetDataset(
            root_dir='./dataset/tin', mode='val', transform=self.transform)

    def get_loaders(self):
        torch.manual_seed(self.seed)
        coarse_tin = torch.randperm(200).reshape(10, 20)

        train_loaders = []
        test_loaders = []

        train_loaders = []
        val_loaders = []

        for i in range(10):
            train_loaders.append(self.create_tin_task(
                coarse_tin, [i], mode='train')[1])
            val_loaders.append(self.create_tin_task(
                coarse_tin, [i], mode='val')[1])
        return train_loaders, test_loaders

    def create_tin_task(self, split, tasks, mode='train', bs=64):
        tmap, cmap = {}, {}
        for tid, task in enumerate(split):
            for lid, lab in enumerate(task):
                tmap[lab.item()], cmap[lab.item()] = tid, lid
        dataset_full = self.train if mode == 'train' else self.val
        shuffle = True if mode == 'train' else False
        # Select a subset of the data-points, depending on the task
        task_labels = []
        for task_id in tasks:
            task_labels += split[task_id]
        idx = np.where(np.isin(dataset_full.label_data, task_labels))[0]
        dataset = Subset(dataset_full, idx)
        # dataset.label_data = [(tmap[dataset.label_data[j]], cmap[dataset.label_data[j]]) for j in idx]
        # dataset.img_data = dataset.img_data[idx]

        dataloader = DataLoader(
            dataset, batch_size=bs, shuffle=shuffle,
            num_workers=0, pin_memory=True)

        return dataset, dataloader


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
                 transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (3, 224, 224)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(
                self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros(
                (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32)
            self.label_data = np.zeros((self.samples_num, 1), dtype=int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = Image.open(s[0]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = Image.open(s[0]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            lbl = None if self.mode == 'test' else s[self.label_idx]
        sample = img, lbl
        return sample
