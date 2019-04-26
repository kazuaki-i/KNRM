import sys
import argparse
import random

from collections import defaultdict
from progressbar import ProgressBar

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file')
parser.add_argument('-o', '--output', dest='output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                    help='output file')
parser.add_argument('-r', '--table-row', dest='table_row', default=3, type=int,
                    help='table row length')
parser.add_argument('-t', '--test', dest='test', default='pointwise.test',
                    help='test data file name')
parser.add_argument('-d', '--dev', dest='dev', default='pointwise.dev',
                    help='development data file name')
parser.add_argument('-s', '--size', dest='size', default=1000, type=int,
                    help='query size of test and dev data')
parser.add_argument('-', '--', dest='', default='',
                    help='')
args = parser.parse_args()


def save_data(fo, D, size):
    K = list(D.keys())
    random.shuffle(K)

    n = 0
    p = ProgressBar(0, size)
    for key in K:
        if n > size:
            break
        values = list(D.get(key, {}).items())
        if len(values) > 1:
            l = ['{}\t{}\t{}'.format(key, v, s) for v, s in values]
            print('\n'.join(l), file=fo)
            p.update(n)
            n += 1
        D.pop(key, None)


def load_data(fi):
    rd = defaultdict(dict)
    for line in fi:
        l_lst = line.strip().split('\t')
        if len(l_lst) != args.table_row:
            continue
        i1, i2, score = l_lst

        if i1 and i2:
            try:
                rd[i1].setdefault(i2, float(score))
            except ValueError:
                print(line, file=sys.stderr)
    return rd


def main():
    D = load_data(args.input)

    if args.test:
        save_data(open(args.test, 'w', encoding='utf-8'), D, size=args.size)
    if args.dev:
        save_data(open(args.dev, 'w', encoding='utf-8'), D, size=args.size)
    save_data(sys.stdout, D, size=len(D))

    return None


if __name__ == '__main__':
    main()

