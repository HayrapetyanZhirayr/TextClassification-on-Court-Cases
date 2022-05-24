"""
This Module is designed for auxilliary functions such as data_loaders and
    train wrappers for convenient usage of torch models
"""
from typing import Union
import os  # to utilize bash commands
import collections
import matplotlib.pyplot as plt

import numpy as np
import sklearn
import models
from models import AverageMeter
from IPython import display
import torch
from torch import nn
from torch.utils.data import Dataset
import scipy
from scipy.sparse import (random,
                          coo_matrix,
                          csr_matrix,
                          vstack)


def dataloader(X, Y, batch_size):
    """
    Yields data in batches

    # Arguments
      X: torch.FloatTensor of shape (n_points, emb_dim)
      Y: torch.LongTensor of shape (n_points, )
      batch_size: int, size of the batch slisi

     # Yields
       Slices of X and Y

    """
    max_idx = len(X)
    i = 0
    while i < max_idx:
        yield X[i:i+batch_size, :], Y[i:i+batch_size]
        i += batch_size


def csr_to_torch(S):
    """
    scipy.sparse.csr_matrix -> torch.sparse.FloatTensor (type coo)

    # Arguments
      S: scipy.sparse.csr_matrix, array to cast to torch.Tensor

     # Returns
       torch.sparse.FloatTensor (type coo)
    """
    S_coo = scipy.sparse.coo_matrix(S)  # scipy.csr_matrix to scipy.coo_matrix
    values = S_coo.data
    indices = np.vstack((S_coo.row, S_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = S_coo.shape
    S_t = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return S_t


def dataloader_sparse(X, Y, batch_size):
    """
    Yields data in batches for sparse csr matrices

    # Arguments
      X: scipy.sparce.csr_matrix of shape (n_points, emb_dim)
      Y: torch.LongTensor of shape (n_points, )
    """
    max_idx = X.shape[0]
    i = 0
    while i < max_idx:
        yield csr_to_torch(X[i:i+batch_size, :]), Y[i:i+batch_size]
        i += batch_size


def calculate_accuracy(prediction, target):
    """
    Caclulates accuracy given prediction Tensor and Ground Trooth targets
    """
    # Note that prediction.shape == target.shape == [B, ]

    matching = (prediction == target).float()
    return matching.mean()


def train(
    NUM_EPOCH,
    BATCH_SIZE,
    DEVICE,
    optimizer,
    HISTORY,
    X_train, Y_train,
    X_test, Y_test,
    model,
    verbose):
    """
    Training wrapper
    """
    criterion = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCH):

        train_loss_meter = AverageMeter()
        train_accuracy_meter = AverageMeter()
        test_loss_meter = AverageMeter()
        test_accuracy_meter = AverageMeter()

        for train_batch in dataloader(X_train, Y_train, BATCH_SIZE):
            x, labels = train_batch
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model.forward(x)

            prediction = torch.argmax(logits, 1)

            loss = criterion(logits, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculate_accuracy(prediction, labels)
            train_loss_meter.update(loss.item())
            train_accuracy_meter.update(acc.item())


        HISTORY['train_loss'].append(train_loss_meter.avg)
        HISTORY['train_accuracy'].append(train_accuracy_meter.avg)

        for test_batch in dataloader(X_test, Y_test, BATCH_SIZE):
            x, labels = test_batch
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.no_grad():
                logits = model.forward(x)
                prediction = torch.argmax(logits, 1)
                loss = criterion(logits, labels)

            acc = calculate_accuracy(prediction, labels)
            test_loss_meter.update(loss.item())
            test_accuracy_meter.update(acc.item())

        HISTORY['test_loss'].append(test_loss_meter.avg)
        HISTORY['test_accuracy'].append(test_accuracy_meter.avg)
        if verbose:
            # visualize all together
            display.clear_output()
            fig, axes = plt.subplots(1, 2, figsize=(20, 7))
            axes[0].set_title('Loss (Cross Entropy)')
            axes[0].plot(HISTORY['train_loss'], label='Train Loss')
            axes[0].plot(HISTORY['test_loss'], label='Test Loss')
            axes[0].grid()
            axes[0].legend(fontsize=20)

            axes[1].set_title('Accuracy')
            axes[1].plot(HISTORY['train_accuracy'], label='Train Accuracy')
            axes[1].plot(HISTORY['test_accuracy'], label='Test Accuracy')
            axes[1].grid()
            axes[1].legend(fontsize=20)

            plt.show()

    return model, HISTORY, test_accuracy_meter


def custom_train_test_split(X, Y_i):
    """
    Train test wrapper for SMALL_NAVEC model
    """
    (X_train, X_test,
        Y_train_i, Y_test_i) = sklearn.model_selection.train_test_split(
            X, Y_i, test_size=.2)

    X_train_t = torch.Tensor(X_train)
    X_test_t = torch.Tensor(X_test)

    Y_train_it = torch.Tensor(Y_train_i).type(torch.long)
    Y_test_it = torch.Tensor(Y_test_i).type(torch.long)

    return X_train_t, X_test_t, Y_train_it, Y_test_it


def custom_train_test_split2(X, Y_id):
    """
    Train test wrapper for SMALL_FIXED_TFIDF + SMALL_MIRKIN model
    """

    indices = np.arange(len(Y_id))
    train_indices, test_indices = sklearn.model_selection.train_test_split(
        indices, test_size=.2)
    skip_indices = set(np.where(Y_id == -1)[0])
    keep_mask = (Y_id != -1)

    id2i = dict(zip(set(Y_id[keep_mask]), range(len(set(Y_id[keep_mask])))))
    i2id = {v:k for k, v in id2i.items()}

    train_indices=np.array(list(set(train_indices) - skip_indices))
    test_indices = np.array(list(set(test_indices) - skip_indices))

    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train_id = Y_id[train_indices]
    Y_test_id = Y_id[test_indices]

    Y_train_i = np.array(list(map(
        lambda x: id2i[x],
        Y_train_id
    )))
    Y_test_i = np.array(list(map(
        lambda x: id2i[x],
        Y_test_id
    )))

    X_train_t = torch.Tensor(X_train)
    X_test_t = torch.Tensor(X_test)

    Y_train_it = torch.Tensor(Y_train_i).type(torch.long)
    Y_test_it = torch.Tensor(Y_test_i).type(torch.long)

    return (train_indices, test_indices, X_train, X_test, Y_train_id, Y_test_id,
            Y_train_i, Y_test_i, X_train_t, X_test_t, Y_train_it, Y_test_it)


def network_configs(X_train_t, Y_i, verbose):
    """
    Wrapper for Network Connfiguration for SMALL_NAVEC model

    """
    INPUT_DIM = X_train_t.shape[-1]
    if verbose: print(f'INPUT_DIM : {INPUT_DIM}')
    OUTPUT_DIM = len(np.unique(Y_i))  # _classes
    if verbose: print(f'OUTPUT_DIM : {OUTPUT_DIM}')

    BATCH_SIZE = 2**12
    if verbose: print(f'BATCH_SIZE : {BATCH_SIZE}')

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose: print(DEVICE)

    # linear_model = models.LinearModelCustom2(INPUT_DIM, OUTPUT_DIM)
    # linear_model = models.LinearModel(INPUT_DIM, OUTPUT_DIM)
    model = models.MLP2(INPUT_DIM, OUTPUT_DIM)
    model.to(DEVICE)

    HISTORY = collections.defaultdict(list)
    return INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, DEVICE, model, HISTORY


def network_configs2(X_train_t, Y_train_i, Y_test_i, verbose):
    """
    Wrapper for Network Connfiguration for SMALL_FIXED_TFIDF + SMALL_MIRKIN model

    """
    INPUT_DIM = X_train_t.shape[-1]
    if verbose: print(f'INPUT_DIM : {INPUT_DIM}')
    OUTPUT_DIM = len(np.unique(np.concatenate((Y_train_i, Y_test_i))))  # _classes
    if verbose: print(f'OUTPUT_DIM : {OUTPUT_DIM}')

    BATCH_SIZE = 2**12
    if verbose: print(f'BATCH_SIZE : {BATCH_SIZE}')

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose: print(DEVICE)

    # linear_model = models.LinearModelCustom2(INPUT_DIM, OUTPUT_DIM)
    # linear_model = models.LinearModel(INPUT_DIM, OUTPUT_DIM)
    model = models.MLP2(INPUT_DIM, OUTPUT_DIM)
    model.to(DEVICE)

    HISTORY = collections.defaultdict(list)

    return INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, DEVICE, model, HISTORY


class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix],
                 targets:Union[np.ndarray, coo_matrix, csr_matrix],
                 transform:bool = None):

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

        self.transform = transform # Can be removed

    def __getitem__(self, index:int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)


def sparse_batch_collate(batch:list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # return batch
    data_batch, targets_batch = zip(*batch)
    # return data_batch
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        # data_batch = torch.FloatTensor(data_batch)
        with open('check_else.txt', 'w') as f:
            f.write('im here')
        data_batch = torch.vstack(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch


class SplitLoader:
    """
    This class is designed to speed up the process of loading data by saving
      and reading splits of data from disk rather then converting scipy sparse
      matrix to torch sparse tensor and slicing on it. Yes, it is faster.
    """

    def __init__(self, X, Y, n_splits, save_dir):
        """
        X is scipy.sparse.csr coo_matrix
        Y is torch.LongTensor
        """
        self.X = X
        self.Y = Y
        self.n_splits = n_splits
        self.save_dir = save_dir

        self.split_size = None

    def __iter__(self):
        y_start = 0
        for split_id in range(self.n_splits):
            x_split = torch.load(f"{self.save_dir}/split_{split_id}.pt")
            y_split = self.Y[y_start:y_start+len(x_split)]
            y_start = y_start + len(x_split)
            yield x_split, y_split


    def split_and_save(self):

        bashCommand = f"rm -r {self.save_dir}/*"
        os.system(bashCommand)
        n_points = self.X.shape[0]

        split_size = n_points // self.n_splits

        self.split_size = split_size

        first_split_size = split_size + n_points%self.n_splits
        split_is_first = True

        last_index = first_split_size
        for split_id in range(self.n_splits):
            if split_is_first:
                split_tensor = torch.FloatTensor(
                    self.X[:first_split_size, :]
                        .toarray().astype(np.float32)
                )
                split_is_first = False
            else:
                split_tensor = torch.FloatTensor(
                        self.X[last_index:last_index + split_size, :]
                            .toarray().astype(np.float32)
                )
                last_index = last_index + split_size

            torch.save(split_tensor, f"{self.save_dir}/split_{split_id}.pt")
