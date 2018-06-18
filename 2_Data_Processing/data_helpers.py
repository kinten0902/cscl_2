import numpy as np
from gensim.models import word2vec


class DataSet(object):
    @property
    def vectors_1(self):
        return self._vectors_1

    @property
    def vectors_2(self):
        return self._vectors_2

    @property
    def labels_1(self):
        return self._labels_1

    @property
    def labels_2(self):
        return self._labels_2

    @property
    def labels_3(self):
        return self._labels_3

    @property
    def labels_4(self):
        return self._labels_4

    @property
    def labels_5(self):
        return self._labels_5

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __init__(self, vectors_1, vectors_2):
        self._num_examples = vectors_1.shape[0]
        self._vectors_1 = vectors_1
        self._vectors_2 = vectors_2
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def set_labels_1(self, labels_1):
        self._labels_1 = labels_1

    def set_labels_2(self, labels_2):
        self._labels_2 = labels_2

    def set_labels_3(self, labels_3):
        self._labels_3 = labels_3

    def set_labels_4(self, labels_4):
        self._labels_4 = labels_4

    def set_labels_5(self, labels_5):
        self._labels_5 = labels_5

    def next_batch(self, batch_size, num_label):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._vectors_1 = self._vectors_1[perm]
            self._vectors_2 = self._vectors_2[perm]
            if num_label >= 1:
                self._labels_1 = self._labels_1[perm]
            if num_label >= 2:
                self._labels_2 = self._labels_2[perm]
            if num_label >= 3:
                self._labels_3 = self._labels_3[perm]
            if num_label >= 4:
                self._labels_4 = self._labels_4[perm]
            if num_label >= 5:
                self._labels_5 = self._labels_5[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        if num_label == 1:
            return self._vectors_1[start:end], self._vectors_2[
                start:end], self._labels_1[start:end]

        if num_label == 2:
            return self._vectors_1[start:end], self._vectors_2[
                start:end], self._labels_1[start:end], self._labels_2[start:
                                                                      end]

        if num_label == 3:
            return self._vectors_1[start:end], self._vectors_2[
                start:end], self._labels_1[start:end], self._labels_2[
                    start:end], self._labels_3[start:end]

        if num_label == 4:
            return self._vectors_1[start:end], self._vectors_2[
                start:end], self._labels_1[start:end], self._labels_2[
                    start:end], self._labels_3[start:end], self._labels_4[
                        start:end]

        if num_label == 5:
            return self._vectors_1[start:end], self._vectors_2[
                start:end], self._labels_1[start:end], self._labels_2[
                    start:end], self._labels_3[start:end], self._labels_4[
                        start:end], self._labels_5[start:end]


def set_data_sets(vectors_1, vectors_2, *labels):
    class DataSets(object):
        pass

    num_labels = len(labels)

    data_sets = DataSets()
    train_flag = np.arange(len(vectors_1)) % 10 != 0

    train_vectors_1 = vectors_1[train_flag]
    test_vectors_1 = vectors_1[~train_flag]

    train_vectors_2 = vectors_2[train_flag]
    test_vectors_2 = vectors_2[~train_flag]

    data_sets.train = DataSet(train_vectors_1, train_vectors_2)
    data_sets.test = DataSet(test_vectors_1, test_vectors_2)
    if num_labels >= 1:
        train_labels_1 = labels[0][train_flag]
        test_labels_1 = labels[0][~train_flag]
        data_sets.train.set_labels_1(train_labels_1)
        data_sets.test.set_labels_1(test_labels_1)
    if num_labels >= 2:
        train_labels_2 = labels[1][train_flag]
        test_labels_2 = labels[1][~train_flag]
        data_sets.train.set_labels_2(train_labels_2)
        data_sets.test.set_labels_2(test_labels_2)
    if num_labels >= 3:
        train_labels_3 = labels[2][train_flag]
        test_labels_3 = labels[2][~train_flag]
        data_sets.train.set_labels_3(train_labels_3)
        data_sets.test.set_labels_3(test_labels_3)
    if num_labels >= 4:
        train_labels_4 = labels[3][train_flag]
        test_labels_4 = labels[3][~train_flag]
        data_sets.train.set_labels_4(train_labels_4)
        data_sets.test.set_labels_4(test_labels_4)
    if num_labels >= 5:
        train_labels_5 = labels[4][train_flag]
        test_labels_5 = labels[4][~train_flag]
        data_sets.train.set_labels_5(train_labels_5)
        data_sets.test.set_labels_5(test_labels_5)
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

def sen_to_same_length(sen, max_len):
    sen_sl = []
    for s in sen:
        s_pad = np.pad(
            list(map(int, s)), (0, max_len - len(s)),
            'constant',
            constant_values=(0, 0))
        sen_sl.append(s_pad)
    return np.array(sen_sl)


def get_w2v_model():
    return word2vec.Word2Vec.load("../data/w2v_model")


def train_w2v_model(sents, wv_size):
    model = word2vec.Word2Vec(sents, size=wv_size, min_count=1, window=5)
    model.save("../data/w2v_model")