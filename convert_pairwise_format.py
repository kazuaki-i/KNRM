import sys
import argparse
import random

from collections import defaultdict
from itertools import combinations

random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                    help='input file')
parser.add_argument('-o', '--output', dest='output', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                    help='output file')
parser.add_argument('-', '--', dest='', default='',
                    help='')
args = parser.parse_args()


def convert_pairwise(key, values):
    rl = []
    for v1, v2 in combinations(values, 2):
        label = 1 if v1[1] > v2[1] else -1
        rl.append('{}\t{}\t{}\t{}'.format(key, v1[0], v2[0], label))
    return rl


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
    for key, d in D.items():
        l = convert_pairwise(key, list(d.items()))
        if l:
            print('\n'.join(l))

    return None


if __name__ == '__main__':
    main()

