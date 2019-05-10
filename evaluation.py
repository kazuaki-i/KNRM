import sys
import argparse
import json

import chainer
import numpy
import math

from collections import defaultdict
from operator import itemgetter
import ranking_nets as nets

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file')
parser.add_argument('-o', '--output', dest='output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                    help='output file')
parser.add_argument('-p', '--parameter', default='args.json',
                    help='parameter files args.json')
parser.add_argument('-v', '--vocab', default='vocab.json',
                    help='parameter files args.json')
parser.add_argument('-m', '--model', dest='model', default='',
                    help='model file')
parser.add_argument('-t', '--type', dest='type', default='pair', choices=['pair', 'list'],
                    help='evaluation type')
parser.add_argument('-b', '--batch', default=512, type=int,
                    help='batch size for evaluation')
parser.add_argument('-', '--', dest='', default='',
                    help='')
args = parser.parse_args()


class Scoring:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.rvocab = {i:w for w, i in vocab.items()}

    def __call__(self, i1_lst, i2_lst):
        xs1 = self._input_to_array(i1_lst)
        xs2 = self._input_to_array(i2_lst)

        y = self.model.encoder(xs1, xs2)
        return y.data

    def _input_to_array(self, lst):
        xs = [numpy.array([self.vocab.get(w, 0) for w in i.split()]) for i in lst]
        return xs


def pairwise_evaluation(score_model, fi):
    def batch_scoring(d):
        p1 = score_model(d['i1'], d['i2'])
        p2 = score_model(d['i1'], d['i3'])

        tf = (p1 - p2) * label

        batch_size = len(tf)
        p_count = numpy.sum(tf > 0)

        return p_count / batch_size

    accuracy = 0.
    count = 0
    d = defaultdict(list)
    for n, line in enumerate(fi):
        l_lst = line.strip().split('\t')
        if len(l_lst) < 4:
            continue
        i1, i2, i3, label = l_lst
        label = int(label)

        if n % args.batch == args.batch - 1:
            accuracy += batch_scoring(d)
            count += 1
            d = defaultdict(list)

        d['i1'].append(i1)
        d['i2'].append(i2)
        d['i3'].append(i3)
        d['label'].append(label)

    accuracy += batch_scoring(d)
    count += 1

    print('pairwise accuracy: {}'.format(accuracy / count))


def ndcg(A, k=3):
    R = numpy.arange(1, min(k, len(A))+1)

    def dcg(target):
        return numpy.sum((2 ** A[target][:, 0] - 1) / numpy.log2(R + 1))

    F = numpy.argsort(-A, axis=0)

    t = dcg(F[:, 0][:k])
    p = dcg(F[:, 1][:k])

    return p / t


def mrr(A):
    F = numpy.argsort(-A, axis=0)
    t_idx = F[:, 1][0]
    p_idx = F[t_idx, :]
    return 1 / float(p_idx[0] + 1)


def listwise_evaluation(score_model, fi):
    def batch_scoring(l, d):
        p_score = score_model(l['i1'], l['i2'])
        kv_score = numpy.reshape(numpy.array(l['kv_score']), (len(l['kv_score']), 1))
        score_array = numpy.concatenate([p_score, kv_score], axis=1)
        d['mrr'] += mrr(score_array)
        d['ndcg1'] += ndcg(score_array, k=1)
        d['ndcg3'] += ndcg(score_array, k=3)
        d['ndcg10'] += ndcg(score_array, k=10)

    query = ''
    q_count = 0
    d = defaultdict(int)
    l = defaultdict(list)
    for n, line in enumerate(fi):
        l_lst = line.strip().split('\t')
        if len(l_lst) < 3:
            continue

        i1, i2, kv_score = l_lst
        if l and query != i1:
            batch_scoring(l, d)
            q_count += 1
            l = defaultdict(list)

        l['i1'].append(i1)
        l['i2'].append(i2)
        l['kv_score'].append(float(kv_score))

        query = i1

    batch_scoring(l, d)
    q_count += 1

    for k, v in d.items():
        print('{}: {}'.format(k, v/q_count))


def main():
    vocab = json.load(open(args.vocab, encoding='utf-8'))
    p = json.load(open(args.parameter, encoding='utf-8'))
    vocab_size = len(vocab)

    encoder = nets.KernelEncoder(kernel=p['kernel'], n_vocab=len(vocab), n_units=p['unit'],
                                 dropout=p['dropout'], hidden_units=128)
    model = nets.PairwiseRanker(encoder, debug=True)
    chainer.serializers.load_npz(args.model, model)

    if args.type == 'pair':
        pairwise_evaluation(Scoring(model, vocab), args.input)

    elif args.type == 'list':
        listwise_evaluation(Scoring(model, vocab), args.input)

    return None


if __name__ == '__main__':
    main()

