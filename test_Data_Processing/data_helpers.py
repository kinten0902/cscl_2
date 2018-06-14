import numpy as np


class DataSet(object):
    @property
    def vectors(self):
        return self._vectors

    @property
    def vectors_pre(self):
        return self._vectors_pre

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __init__(self, vectors, vectors_pre, labels):
        self._num_examples = vectors.shape[0]
        self._vectors = vectors
        self._vectors_pre = vectors_pre
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._vectors = self._vectors[perm]
            self._vectors_pre = self._vectors_pre[perm]
            self._labels = self._labels[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._vectors[start:end], self._vectors_pre[
            start:end], self._labels[start:end]


def read_data_sets(vectors, vectors_pre, labels):
    class DataSets(object):
        pass

    data_sets = DataSets()
    train_flag = np.arange(len(vectors)) % 10 != 0
    train_vectors = vectors[train_flag]
    train_vectors_pre = vectors_pre[train_flag]
    train_labels = labels[train_flag]
    test_vectors = vectors[~train_flag]
    test_vectors_pre = vectors_pre[~train_flag]
    test_labels = labels[~train_flag]
    data_sets.train = DataSet(train_vectors, train_vectors_pre, train_labels)
    data_sets.test = DataSet(test_vectors, test_vectors_pre, test_labels)
    return data_sets


def labels_to_one_hot(labels, num_classes):
    return np.eye(num_classes, dtype=np.int32)[labels]


def sen_to_fv(sen, max_len, model, ravel):
    sen_fv = []
    for s in sen:
        s_pad = np.pad(
            list(map(int, s)), (0, max_len - len(s)),
            'constant',
            constant_values=(0, 0))
        if ravel:
            s_vec = np.array([model.wv[str(w)] for w in s_pad]).ravel()
        else:
            s_vec = np.array([model.wv[str(w)] for w in s_pad])
        sen_fv.append(s_vec)
    return np.array(sen_fv)