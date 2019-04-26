import collections
import io
import random

import numpy

from glob import glob
from progressbar import ProgressBar

import chainer
from chainer.backends import cuda


def split_text(text, char_based=False):
    if char_based:
        return list(text)
    else:
        return text.split()


def normalize_text(text):
    return text.strip().lower()


def read_kernel(fi_name):
    rl = []
    with io.open(fi_name, encoding='utf-8') as fi:
        for line in fi:
            if line.startswith('#'):
                continue
            l_lst = line.strip().split(',')
            rl.append([float(l_lst[0]), float(l_lst[1])])
    return rl


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return numpy.array(ids, numpy.int32)


def convert_seq3(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if with_label:
        return {'xs1': to_device_batch([x1 for x1, _, _, _ in batch]),
                'xs2': to_device_batch([x2 for _, x2, _, _ in batch]),
                'xs3': to_device_batch([x3 for _, _, x3, _ in batch]),
                'ys': to_device_batch([y for _, _, _, y in batch])}
    else:
        return {'xs1': to_device_batch([x1 for x1, _ in batch]),
                'xs2': to_device_batch([x2 for _, x2 in batch])}


def load_data_using_dataset_api(fi_name, vocab):
    EOS, UNK = 0, 1

    def _transform_line(content):
        words = content.strip().split()
        return numpy.array(
            [vocab.get(w, UNK) for w in words], numpy.int32)

    def _transform(line):
        l_lst = line.strip().split('\t')
        return(
            _transform_line(l_lst[0]),
            _transform_line(l_lst[1]),
            _transform_line(l_lst[2]),
            numpy.array([float(l_lst[3])], numpy.float32)
        )

    def _load_single_data_using_dataset_api(fi):
        return chainer.datasets.TransformDataset(
            chainer.datasets.TextDataset(fi, encoding='utf-8'), _transform)

    train_path = glob(fi_name)
    p = ProgressBar(0, len(train_path))

    datasets = []
    for n, fi_path in enumerate(train_path):
        p.update(n+1)
        datasets.append(_load_single_data_using_dataset_api(fi_path))

    return chainer.datasets.ConcatenatedDataset(*datasets)


def make_vocab(dataset, vocabsize, min_freq=2):
    counts = collections.defaultdict(int)
    for t1, t2, _ in dataset:
        tokens = t1 + t2
        for token in tokens:
            counts[token] += 1

    vocab = {'<eos>': 0, '<unk>': 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= vocabsize or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def transform_to_array3(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(t1, vocab, False), make_array(t2, vocab, False),
                 make_array(t3, vocab, False), numpy.array([cls], numpy.float32))
                for t1, t2, t3, cls in dataset]
    else:
        return [(make_array(t1, vocab), make_array(t2, vocab)) for t1, t2 in dataset]


def load_word2vec_model(fi_name, units):
    print('load {} word2vec model'.format(fi_name))
    with open(fi_name, encoding='utf-8') as fi:
        vocab = {'<eos>': 0, '<unk>': 1}
        vector = []
        for n, line in enumerate(fi):
            l_lst = line.strip().split()
            if n == 0:
                # vocabsize = int(l_lst[0])
                v_size = int(l_lst[1])
                assert(units == v_size)

                vector.append([random.uniform(-0.5, 0.5) for _ in range(v_size)])
                vector.append([random.uniform(-0.5, 0.5) for _ in range(v_size)])
            else:
                v = l_lst[0]
                vec = [float(i) for i in l_lst[1:]]
                vocab[v] = n + 1
                vector.append(vec)

    return vocab, numpy.array(vector, numpy.float32)


def load_input_file(fi_name):
    rl = []
    with open(fi_name, encoding='utf-8') as fi:
        for line in fi:
            l_lst = line.strip().split('\t')
            if len(l_lst) < 4:
                continue
            i1, i2, i3 = split_text(l_lst[0]), split_text(l_lst[1]), split_text(l_lst[2])
            label = l_lst[3]

            rl.append((i1, i2, i3, label,))

    return rl


def get_input_dataset(fi_name, vocab):
    dataset = load_input_file(fi_name)
    train = transform_to_array3(dataset, vocab)

    return train
