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
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


    def __init__(self, vectors_1, labels_1, labels_2):
        self._num_examples = vectors_1.shape[0]
        self._vectors_1 = vectors_1
        self._labels_1 = labels_1
        self._labels_2 = labels_2
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    def __init__(self, vectors_1, labels_1, labels_2, labels_3):
        self._num_examples = vectors_1.shape[0]
        self._vectors_1 = vectors_1
        self._labels_1 = labels_1
        self._labels_2 = labels_2
        self._labels_3 = labels_3
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    def __init__(self, vectors_1, vectors_2, labels_1, labels_2, labels_3):
        self._num_examples = vectors_1.shape[0]
        self._vectors_1 = vectors_1
        self._vectors_2 = vectors_2
        self._labels_1 = labels_1
        self._labels_2 = labels_2
        self._labels_3 = labels_3
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
            self._vectors_1 = self._vectors_1[perm]
            self._labels_1 = self._labels_1[perm]
            self._labels_2 = self._labels_2[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._vectors_1[start:end], self._labels_1[start:end], self._labels_2[start:end]
    
    def next_batch_2(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            # shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._vectors_1 = self._vectors_1[perm]
            self._labels_1 = self._labels_1[perm]
            self._labels_2 = self._labels_2[perm]
            self._labels_3 = self._labels_3[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._vectors_1[start:end], self._labels_1[start:end], self._labels_2[start:end], self._labels_3[start:end]

       
    def next_batch_3(self, batch_size):
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
            self._labels_1 = self._labels_1[perm]
            self._labels_2 = self._labels_2[perm]
            self._labels_3 = self._labels_3[perm]
            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._vectors_1[start:end], self._vectors_2[start:end], self._labels_1[start:end], self._labels_2[start:end], self._labels_3[start:end]
    
    
def set_data_sets(vectors_1, labels_1, labels_2):
    class DataSets(object):
        pass

    data_sets = DataSets()
    train_flag = np.arange(len(vectors_1)) % 10 != 0
    
    train_vectors_1 = vectors_1[train_flag]
    train_labels_1 = labels_1[train_flag]
    train_labels_2 = labels_2[train_flag]
    
    test_vectors_1 = vectors_1[~train_flag]
    test_labels_1 = labels_1[~train_flag]
    test_labels_2 = labels_2[~train_flag]
    
    data_sets.train = DataSet(train_vectors_1, train_labels_1, train_labels_2)
    data_sets.test = DataSet(test_vectors_1, test_labels_1, test_labels_2)
    return data_sets

def set_data_sets_2(vectors_1, labels_1, labels_2, labels_3):
    class DataSets(object):
        pass

    data_sets = DataSets()
    train_flag = np.arange(len(vectors_1)) % 10 != 0
    
    train_vectors_1 = vectors_1[train_flag]
    train_labels_1 = labels_1[train_flag]
    train_labels_2 = labels_2[train_flag]
    train_labels_3 = labels_3[train_flag]
    
    test_vectors_1 = vectors_1[~train_flag]
    test_labels_1 = labels_1[~train_flag]
    test_labels_2 = labels_2[~train_flag]
    test_labels_3 = labels_3[~train_flag]
    
    data_sets.train = DataSet(train_vectors_1, train_labels_1, train_labels_2, train_labels_3)
    data_sets.test = DataSet(test_vectors_1, test_labels_1, test_labels_2, test_labels_3)
    return data_sets

def set_data_sets_3(vectors_1, vectors_2, labels_1, labels_2, labels_3):
    class DataSets(object):
        pass

    data_sets = DataSets()
    train_flag = np.arange(len(vectors_1)) % 10 != 0
    
    train_vectors_1 = vectors_1[train_flag]
    train_vectors_2 = vectors_2[train_flag]
    train_labels_1 = labels_1[train_flag]
    train_labels_2 = labels_2[train_flag]
    train_labels_3 = labels_3[train_flag]
    
    test_vectors_1 = vectors_1[~train_flag]
    test_vectors_2 = vectors_2[~train_flag]
    test_labels_1 = labels_1[~train_flag]
    test_labels_2 = labels_2[~train_flag]
    test_labels_3 = labels_3[~train_flag]
    
    data_sets.train = DataSet(train_vectors_1, train_vectors_2, train_labels_1, train_labels_2, train_labels_3)
    data_sets.test = DataSet(test_vectors_1, test_vectors_2, test_labels_1, test_labels_2, test_labels_3)
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


def get_w2v_model():
    return word2vec.Word2Vec.load("../data/w2v_model")

def train_w2v_model(sents, wv_size):
    model = word2vec.Word2Vec(sents, size=wv_size, min_count=1, window=5)
    model.save("../data/w2v_model")
    
    