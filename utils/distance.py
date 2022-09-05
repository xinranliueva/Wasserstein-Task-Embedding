# from .poolings import SWE
# from .torchswd import SlicedWD
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch.nn as nn
import numpy as np
import ot
from tqdm.autonotebook import tqdm
from .bures_wasserstein import *
from sklearn.manifold import MDS


class label_mds:
    def __init__(self, emb_dim, device, class_num=None, lbl_transform=None, gaussian_assumption=True, maxsamples=None):
        self.dim = emb_dim
        self.device = device
        self.assumption = gaussian_assumption
        self.maxsamples = maxsamples
        self.lbl_transform = lbl_transform
        self.class_num = class_num

    def preprocess_dataset(self, X):
        if isinstance(X, DataLoader):
            loader = X
        elif isinstance(X, Dataset):
            if self.maxsamples and len(X) > self.maxsamples:
                idxs = np.sort(np.random.choice(
                    len(X), self.maxsamples, replace=False))
                sampler = SubsetRandomSampler(idxs)
                loader = DataLoader(X, sampler=sampler, batch_size=64)
            else:
                # No subsampling
                loader = DataLoader(X, batch_size=64)

        X = []
        Y = []

        for batch in tqdm(loader, leave=False):
            x = batch[0]
            y = batch[1]
            if self.lbl_transform:
                y = torch.stack([torch.tensor(self.lbl_transform[l.item()])
                                for l in y], dim=0).squeeze(0)

            X.append(x.squeeze().view(x.shape[0], -1))
            Y.append(y.squeeze())

        X = torch.cat(X).to(self.device)
        Y = torch.cat(Y).to(self.device)

        return X, Y

    def precompute_dissimilarity(self, X1, X2=None, symmetric=True):
        X1, Y1 = self.preprocess_dataset(X1)
        if X2 is not None:
            X2, Y2 = self.preprocess_dataset(X2)
        else:
            Y2 = None
            X2 = None
            M2 = None
            S2 = None
        if self.assumption:
            M1, S1 = self.get_gaussian_stats(X1, Y1)
            if X2 is not None:
                M2, S2 = self.get_gaussian_stats(X2, Y2)

            D = efficient_pwdist_gauss(M1, S1, M2, S2, sqrtS1=None, sqrtS2=None,
                                       symmetric=symmetric, diagonal_cov=False, commute=False,
                                       sqrt_method='ns', sqrt_niters=20, sqrt_pref=0,
                                       device=self.device, nworkers=1,
                                       return_dmeans=False, return_sqrts=False)
        else:
            def distance(Xa, Xb):
                C = ot.dist(Xa, Xb, metric='euclidean').cpu().numpy()
                return torch.tensor(ot.emd2(ot.unif(Xa.shape[0]), ot.unif(Xb.shape[0]), C))
            D = torch.zeros((self.class_num, self.class_num),
                            device=self.device)
            if symmetric:
                for i in range(self.class_num):
                    for j in range(i+1, self.class_num):
                        D[i, j] = distance(X1[Y1 == i], X1[Y1 == j]).item()
                        D[j, i] = D[i, j]
            else:
                for i in range(self.class_num):
                    for j in range(self.class_num):
                        D[i, j] = distance(X1[Y1 == i], X2[Y2 == j]).item()
        return D

    def get_gaussian_stats(self, X, Y):
        labels, _ = torch.sort(torch.unique(Y))
        means = torch.stack([torch.mean(X[Y == y].float(), dim=0)
                            for y in labels], dim=0)
        cov = torch.stack([torch.cov(X[Y == y].T) for y in labels], dim=0)
        return means, cov

    def dissimilarity_for_all(self, datasets):
        l = len(datasets)
        if not self.class_num:
            self.class_num = torch.unique(datasets[0].targets).shape[0]
        distance_array = np.full((l*self.class_num, l*self.class_num), 0)
        for i in range(l-1):
            for j in range(i + 1, l):
                distance_array[(self.class_num * i):(self.class_num * (i + 1)), (self.class_num * j):(self.class_num * (j + 1))] = \
                    np.asarray(self.precompute_dissimilarity(
                        datasets[i], datasets[j], symmetric=False).cpu())

        distance_array = distance_array + distance_array.T - \
            np.diag(np.diag(distance_array))
        for i in range(l):
            distance_array[(self.class_num * i):(self.class_num * (i + 1)), (self.class_num * i):(self.class_num * (i + 1))] = \
                np.asarray(self.precompute_dissimilarity(datasets[i]).cpu())
        self.dm = distance_array
        return distance_array

    def embedding(self, datasets):
        distance_array = self.dissimilarity_for_all(datasets)
        embedding = MDS(n_components=self.dim, n_init=10,
                        max_iter=10000, dissimilarity='precomputed')
        mds = embedding.fit_transform(distance_array)
        data_label_pairs = []
        # reference = []
        for idx, dataset in enumerate(datasets):
            X, Y = self.preprocess_dataset(dataset)
            label_emb = mds[self.class_num*idx:self.class_num*(idx+1)]
            labels = torch.stack([torch.from_numpy(label_emb[target])
                                 for target in Y], dim=0).squeeze(1).to(self.device)
            Z = torch.cat((X.to(self.device), labels), dim=1)
            data_label_pairs.append(Z)
        # Y_unique = torch.unique(Y)
        # for i in Y_unique:
        #   idx = torch.randperm(len(Z[Y==i]))[:4]
        #   reference.extend(Z[Y==i][idx])

        return mds, data_label_pairs  # , reference


def WTE(datasets, label_dim, device, class_num=None, ref=None, maxsamples=None):
    print('Embedding labels...')
    emb = label_mds(emb_dim=label_dim, class_num=class_num,
                    device=device, maxsamples=maxsamples)
    mds, data_label_pairs = emb.embedding(datasets)
    assert ref is not None, f'Reference not found.'
    ref_size = ref.shape[0]
    task_embs = []
    print('Wasserstein embedding...')
    for X in data_label_pairs:
        X = X.float()
        C = ot.dist(X.cpu(), ref).cpu().numpy()
        # Calculating the transport plan
        gamma = torch.from_numpy(ot.emd(ot.unif(X.shape[0]), ot.unif(
            ref_size), C, numItermax=1000000)).float()
        # Calculating the transport map via barycenter projection
        f = (torch.matmul((ref_size*gamma).T, X.cpu())-ref)/np.sqrt(ref_size)
        task_embs.append(f)
    return torch.stack(task_embs, dim=0).squeeze(0)
