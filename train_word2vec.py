import sys
import argparse
import time

import logging
from gensim.models import word2vec

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file')
parser.add_argument('-o', '--output', dest='output', default='word2vec.model',
                    help='output file')
parser.add_argument('-v', '--vocab', dest='vocab', default=100000, type=int,
                    help='vocabulary_size')
parser.add_argument('-e', '--embed', dest='embed', default=200, type=int,
                    help='embedding_size')
parser.add_argument('-w', '--window', dest='window', default=5, type=int,
                    help='window size')
parser.add_argument('-I', '--iter', dest='iter', default=5, type=int,
                    help='number of iteration')
parser.add_argument('-m', '--min', dest='min', default=3, type=int,
                    help='min count of train token freq')
parser.add_argument('-', '--', dest='', default='',
                    help='')
args = parser.parse_args()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
start = time.time()


def load_train_data(fi):
    rl = []
    for line in fi:
        l_lst = line.strip().split('\t')
        if len(l_lst) < 2:
            continue
        rl.append(l_lst[0].split())
        rl.append(l_lst[1].split())

    return rl


def main():
    print('load train data...')
    train_data = load_train_data(args.input)
    print('fin.({})'.format(time.time()-start))

    model = word2vec.Word2Vec(train_data, sg=1, size=args.embed, window=args.window,
                              iter=args.iter, min_count=args.min)

    model.wv.save_word2vec_format(args.output)

    return None


if __name__ == '__main__':
    main()

