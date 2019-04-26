import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from ranking_utils import convert_seq3


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
    def __init__(self, encoder, dropout=0.1):
        super(PairwiseRanker, self).__init__()
        with self.init_scope():
            self.encoder = encoder
        self.dropout = dropout

    def forward(self, xs1, xs2, xs3, ys, train=True):
        # initialization
        batch_size = len(ys)
        direction = F.reshape(F.concat(ys, axis=0), (batch_size, 1))
        zeros = F.reshape(self.xp.zeros(len(ys), self.xp.float32), (batch_size, 1))

        # calculate ranking score each pair
        f1 = self.encoder(xs1, xs2)
        f2 = self.encoder(xs1, xs3)

        # reflect direction of higher or lower ranking
        ps = (f1 - f2) * direction

        # calculate loss
        loss = F.sum(F.max(F.concat((zeros, 1 - ps), axis=1), axis=1))
        # calculate pair-wise accuracy
        accuracy = F.sum((ps.data > 0).astype(self.xp.float32)) / batch_size

        if train:
            reporter.report({'loss': loss}, self)
            reporter.report({'accuracy': accuracy}, self)
            return loss
        else:
            return loss, accuracy


class KernelEncoder(chainer.Chain):
    def __init__(self, kernel, n_vocab, n_units, embed_init=None, dropout=0.1, avoid_inf=0.00001, avoid_nan=0.00001):
        super(KernelEncoder, self).__init__()
        with self.init_scope():
            if embed_init is not None:
                embed_init = chainer.initializers.Uniform(.5)
            self.embed = L.EmbedID(n_vocab, n_units, initialW=embed_init, ignore_label=-1)
            self.liner = L.Linear(len(kernel), 1)

        self.dropout = dropout
        self.kernels = kernel
        self.out_units = len(kernel)
        self.means = [m for m, _ in self.kernels]
        self.variances = [v for _, v in self.kernels]
        self.avoid_inf = avoid_inf
        self.avoid_nan = avoid_nan

    def forward(self, xs1, xs2):
        # padding inputs
        x1 = F.pad_sequence(xs1, padding=-1)
        x2 = F.pad_sequence(xs2, padding=-1)

        # word idx -> word vector
        ex1 = F.dropout(self.embed(x1), ratio=self.dropout)
        ex2 = F.dropout(self.embed(x2), ratio=self.dropout)

        # Take this batch's parameters
        batch_size = len(xs1)
        row, column = ex1.shape[1], ex2.shape[1]
        m1, m2 = self._mask(ex1), self._mask(ex2)

        # calculate norm for cosine similarity
        ex1 = F.repeat(ex1, column, axis=1)
        ex2 = F.tile(ex2, (1, row, 1))
        norm = F.sqrt(F.sum(ex1 * ex1, axis=-1)) + F.sqrt(F.sum(ex2 * ex2, axis=-1)) + self.avoid_nan

        # transfer matrix creation
        h = F.reshape(F.sum(ex1 * ex2, axis=-1) / norm, (batch_size, column, row))

        # reshape the mask to transfer matrix
        m1 = F.repeat(m1, column, axis=1)
        m2 = F.tile(m2, (1, row, 1))
        mask = F.reshape(F.max(m1 * m2, axis=-1), (batch_size, column, row))

        # kernel matrix shaping and tiling for kernel pooling
        m = self._kernel_matrix_shaping(self.means, h.shape)
        v = self._kernel_matrix_shaping(self.variances, h.shape)
        h = F.tile(h, (self.out_units, 1, 1))
        mask = F.tile(mask, (self.out_units, 1, 1))

        # calculate RBF kernel
        h = F.exp(-1 * (h - m)**2 / (2 * v**2))
        h = h * mask    # masking padding

        # pooling (b x k x n x m) -> (b x k x n)
        h = F.sum(F.reshape(h, (batch_size, self.out_units, column, row)), axis=-1)
        mask = F.max(F.reshape(mask, (batch_size, self.out_units, column, row)), axis=-1)

        # pooling (b x k x n) -> (b x k)
        h = F.log(h + self.avoid_inf)
        h = F.sum(h * mask, axis=-1)

        # calculate ranking score
        h = F.tanh(self.liner(h))

        return h

    def _mask(self, x):
        m = F.sum(F.absolute(x), axis=-1)
        m = (m.data > 0.).astype(self.xp.float32)
        m = m[:, :, None]
        m = F.tile(m, (1, 1, x.shape[-1]))
        return m

    def _kernel_matrix_shaping(self, x, target_shape):
        return F.concat([self.xp.full(target_shape, _x, self.xp.float32) for _x in x], axis=0)
