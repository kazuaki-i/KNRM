import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from ranking_utils import convert_seq3


def sequence_embed(embed, xs, dropout=0.):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.reshape(e, (e.shape[0], 1, e.shape[1], e.shape[2]))
    return e


class EvaluationPairwise(chainer.training.Extension):
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, vocab, key, batch=1000, device=-1):
        self.model = model
        self.test_data = test_data
        self.vocab = vocab
        self.key = key
        self.batch = batch
        self.device = device

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            for i in range(0, len(self.test_data), self.batch):
                d = convert_seq3(self.test_data[i:i + self.batch], device=self.device)

                loss, accuracy = self.model.forward(
                    d['xs1'], d['xs2'], d['xs3'], d['ys'], train=False)

                chainer.report({'{}/loss'.format(self.key): loss})
                chainer.report({'{}/accuracy'.format(self.key): accuracy})


class PairwiseRanker(chainer.Chain):
    def __init__(self, encoder, dropout=0.1, debug=False):
        super(PairwiseRanker, self).__init__()
        with self.init_scope():
            self.encoder = encoder
        self.dropout = dropout
        self.debug = bool(debug)

    def forward(self, xs1, xs2, xs3, ys, train=True):
        with chainer.using_config('debug', self.debug):
            # initialization
            batch_size = len(ys)
            direction = F.reshape(F.concat(ys, axis=0), (batch_size, 1))
            zeros = self.xp.zeros((batch_size, 1), self.xp.float32)
            label = F.concat(self.xp.array(self.xp.array(ys, self.xp.int32) < 0., self.xp.int32),  axis=0)

            # calculate ranking score each pair
            f1 = self.encoder(xs1, xs2)
            f2 = self.encoder(xs1, xs3)

            # reflect direction of higher or lower ranking
            ps = (f1 - f2) * direction
            x = F.concat([f1, f2, ps], axis=1)

            # calculate loss
            loss = F.sum(F.max(F.concat((zeros, 1 - ps), axis=1), axis=1))
            # calculate pair-wise accuracy
            accuracy = F.accuracy(x, label)

            # print(loss, accuracy)

            if train:
                reporter.report({'loss': loss}, self)
                reporter.report({'accuracy': accuracy}, self)
                return loss
            else:
                return loss, accuracy


class Utils:
    def __init__(self, out_units, batch_size, column, row, xp, minute_num):
        self.out_units = out_units
        self.batch_size = batch_size
        self.column = column
        self.row = row
        self.xp = xp
        self.minute_num = minute_num

    def normalize(self, x):
        n = self.xp.linalg.norm(x.data, axis=-1)
        n = n[:, :,  None]
        n = F.tile(n, (1, 1, x.shape[-1]))
        return x / (n + self.minute_num)

    def masking(self, x):
        m = self.xp.sum(self.xp.absolute(x.data), axis=-1) > 0.
        m = m[:, :, None]
        m = self.xp.tile(m, (1, 1, x.shape[-1]))
        return m

    def kernel_shaping(self, x):
        return F.concat([self.xp.full((self.batch_size, self.column, self.row),
                                      _x, self.xp.float32) for _x in x], axis=0)

    def cross_match(self, x1, x2, m1, m2):
        x1 = self.normalize(x1)
        x2 = self.normalize(x2)

        x1 = F.repeat(x1, self.column, axis=1)
        x2 = F.tile(x2, (1, self.row, 1))

        m1 = self.xp.repeat(m1, self.column, axis=1)
        m2 = self.xp.tile(m2, (1, self.row, 1))

        x = F.sum(x1 * x2, axis=-1)
        x = F.reshape(x, (self.batch_size, self.column, self.row))

        mask = self.xp.all(self.xp.logical_and(m1, m2), axis=-1)
        mask = self.xp.reshape(mask, (self.batch_size, self.column, self.row))

        return x, mask

    def kernel_pooling(self, x, m, mean, variance):
        # calculate RBF kernel
        x = F.tile(x, (self.out_units, 1, 1))
        mask = self.xp.tile(m, (self.out_units, 1, 1))

        x = F.exp(-1 * (x - mean)**2 / (2 * variance**2))
        x = F.where(mask, x, self.xp.full(x.shape, 0., self.xp.float32))

        # pooling (b x k x n x m) -> (b x k x n)
        s = (self.batch_size, self.out_units, self.column, self.row,)
        x = F.sum(F.reshape(x, s), axis=-1)
        mask = self.xp.all(self.xp.reshape(mask, s), axis=-1)

        # pooling (b x k x n) -> (b x k)
        x = F.log(x + self.minute_num)
        x = F.where(mask, x, self.xp.full(x.shape, 0., self.xp.float32))
        x = F.sum(x, axis=-1)

        return x


class KernelEncoder(chainer.Chain):
    def __init__(self, kernel, n_vocab, n_units, hidden_units=0, embed_init=None, dropout=0.1,  minute_num=0.00001):
        super(KernelEncoder, self).__init__()
        with self.init_scope():
            if embed_init is not None:
                embed_init = chainer.initializers.Uniform(.5)
            self.embed = L.EmbedID(n_vocab, n_units, initialW=embed_init, ignore_label=-1)
            self.liner = L.Linear(len(kernel), 1)

        self.n_units = n_units
        self.dropout = dropout
        self.kernels = kernel
        self.out_units = len(kernel)
        self.means = [m for m, _ in self.kernels]
        self.variances = [v for _, v in self.kernels]
        self.minute_num = minute_num

    def forward(self, xs1, xs2):
        # padding inputs
        x1 = F.pad_sequence(xs1, padding=-1)
        x2 = F.pad_sequence(xs2, padding=-1)

        # word idx -> word vector
        ex1 = F.dropout(self.embed(x1), self.dropout)
        ex2 = F.dropout(self.embed(x2), self.dropout)

        # this mini batch parameters definition
        batch_size = len(xs1)
        row, column = ex1.shape[1], ex2.shape[1]
        utils = Utils(self.out_units, batch_size, column, row, self.xp, self.minute_num)

        m1, m2 = utils.masking(ex1), utils.masking(ex2)
        mean = utils.kernel_shaping(self.means)
        variance = utils.kernel_shaping(self.variances)

        # cross match and kernel pooling
        h, mask = utils.cross_match(ex1, ex2, m1, m2)
        h = utils.kernel_pooling(h, mask, mean, variance)

        # calculate ranking score
        h = F.tanh(self.liner(h))

        return h


class KernelEncoderCNN(chainer.Chain):
    def __init__(self, kernel, n_vocab, n_units, hidden_units=0, embed_init=None, dropout=0.1, minute_num=0.0001):
        super(KernelEncoderCNN, self).__init__()
        with self.init_scope():
            if embed_init is not None:
                embed_init = chainer.initializers.Uniform(.25)
            self.embed = L.EmbedID(n_vocab, n_units, initialW=embed_init, ignore_label=-1)
            self.c1 = L.Convolution2D(
                1, hidden_units, ksize=(1, n_units), stride=1)
            self.c2 = L.Convolution2D(
                1, hidden_units, ksize=(2, n_units), stride=1)
            self.cnn = [self.c1, self.c2]
            self.liner = L.Linear(len(kernel)*len(self.cnn)*len(self.cnn), 1)

        self.dropout = dropout
        self.n_units = n_units
        self.kernels = kernel
        self.out_units = len(kernel)
        self.means = [m for m, _ in self.kernels]
        self.variances = [v for _, v in self.kernels]
        self.minute_num = minute_num

    def forward(self, xs1, xs2):
        # padding inputs
        x1 = chainer.dataset.convert.concat_examples(xs1, padding=-1)
        x2 = chainer.dataset.convert.concat_examples(xs2, padding=-1)

        batch_size = len(xs1)
        row, column = x1.shape[-1], x2.shape[-1]

        utils = Utils(self.out_units, batch_size, column, row, self.xp, self.minute_num)

        mean = utils.kernel_shaping(self.means)
        variance = utils.kernel_shaping(self.variances)

        # word idx -> word vector
        x1 = block_embed(self.embed, x1, self.dropout)
        x2 = block_embed(self.embed, x2, self.dropout)
        pad = self.xp.full((batch_size, 1, len(self.cnn)-1, self.n_units), 0., self.xp.float32)

        # CNN applying
        h1_lst, h2_lst = [], []
        for n, c in enumerate(self.cnn):
            h1 = c(F.concat([x1, pad[:, :, :n, :]], axis=2))
            h2 = c(F.concat([x2, pad[:, :, :n, :]], axis=2))

            h1 = F.transpose(F.reshape(h1, h1.shape[:-1]), (0, 2, 1))
            h2 = F.transpose(F.reshape(h2, h2.shape[:-1]), (0, 2, 1))

            m1, m2 = utils.masking(h1), utils.masking(h2)
            h1, h2 = F.relu(h1), F.relu(h2)

            h1_lst.append([h1, m1])
            h2_lst.append([h2, m2])

        # cross match
        cross_lst = []
        for x1, m1 in h1_lst:
            for x2, m2 in h2_lst:
                x, mask = utils.cross_match(x1, x2, m1, m2)
                cross_lst.append([x, mask])

        # kernel pooling
        stf_lst = []
        for h, m in cross_lst:
            x = utils.kernel_pooling(h, m, mean, variance)
            stf_lst.append(x)

        # calculate ranking score
        h = F.tanh(self.liner(F.concat(stf_lst, axis=-1)))

        return h
