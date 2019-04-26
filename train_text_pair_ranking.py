import argparse
import datetime
import json
import os
import time

import chainer
from chainer import training
from chainer.training import extensions, triggers

import ranking_nets as nets
from ranking_utils import *


def main():
    start = time.time()
    current_datetime = '{}'.format(datetime.datetime.today())
    parser = argparse.ArgumentParser(description='Chainer Text Ranking')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=200,
                        help='Number of units')
    parser.add_argument('--vocabsize', type=int, default=50000,
                        help='Number of max vocabulary')
    parser.add_argument('--dropout', '-d', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--train', '-T', required=True, help='train dataset')
    parser.add_argument('--dev', '-D', required=True, help='dev dataset')
    parser.add_argument('--vocab', '-v', help='word2vec format vocab file (not binary)')
    parser.add_argument('--vocab-source', '-vs', help='source file of creating vocabulary')
    parser.add_argument('--kernel', '-k', help='kernel parameter file (csv)')
    parser.add_argument('--use-dataset-api', default=False, action='store_true',
                        help='use TextDataset API to reduce CPU memory usage')
    parser.add_argument('--validation-interval', type=int, default=10000,
                        help='number of iteration to evaluate the model with validation dataset')
    parser.add_argument('--resume', '-r', type=str,
                        help='resume the training from snapshot')
    parser.add_argument('--save-fin', '-sf', type=str,
                        help='save a snapshot at the training finished time')
    # parser.add_argument('--model', '-model', default='transfer',
    #                     choices=['cnn', 'transfer', 'gru'],
    #                     help='Name of encoder model type.') Coming soon!
    parser.add_argument('--early-stop', action='store_true', help='use early stopping method')
    parser.add_argument('--save-init', action='store_true', help='save init model')
    parser.add_argument('--save-snapshot', action='store_true', help='save snapshot per validation')
    parser.add_argument('--progressbar', action='store_true', help='show training progressbar')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # load vocabulary
    vocab, embed_init = None, None
    if args.vocab:
        vocab, init_vector = load_word2vec_model(args.vocab, units=args.unit)
    elif args.vocab_source:
        vocab = make_vocab(args.vocab_source, args.vocabsize)

    assert(vocab is not None)

    # load data sets
    print('load data sets')
    if args.use_dataset_api:
        # if you get out of  CPU memory, this potion can reduce the memory usage
        train = load_data_using_dataset_api(fi_name=args.train, vocab=vocab)
    else:
        train = get_input_dataset(fi_name=args.train, vocab=vocab)

    dev = get_input_dataset(fi_name=args.dev, vocab=vocab)

    if args.kernel:
        kernel = read_kernel(args.kernel)
    else:
        kernel = [(1.0, 0.001)] + [(m10 / 10., 0.1) for m10 in range(-9, 11, 2)]

    print('# train  data: {}'.format(len(train)))
    print('# dev    data: {}'.format(len(dev)))
    print('# vocab  size: {}'.format(len(vocab)))
    print('# kernel size: {}'.format(len(kernel)))

    # Setup your defined model
    Encoder = nets.KernelEncoder
    encoder = Encoder(kernel=kernel, n_vocab=len(vocab), n_units=args.unit,
                      dropout=args.dropout, embed_init=embed_init)
    model = nets.PairwiseRanker(encoder)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=convert_seq3, device=args.gpu)

    # early Stopping
    if args.early_stop:
        stop_trigger = triggers.EarlyStoppingTrigger(monitor='validation/main/loss', max_trigger=(args.epoch, 'epoch'))
    else:
        stop_trigger = (args.epoch, 'epoch')

    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch and validation_interval
    trainer.extend(nets.EvaluationPairwise(model, dev, vocab, key='validation/main', device=args.gpu),
                   trigger=(args.validation_interval, 'iteration'))
    trainer.extend(nets.EvaluationPairwise(model, dev, vocab, key='validation/main', device=args.gpu),
                   trigger=(1, 'epoch'))

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger('validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)
    if args.save_snapshot:
        trainer.extend(extensions.snapshot(filename='snapshot_latest'), trigger=(args.validation_interval, 'iteration'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.validation_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy',
         'validation/main/accuracy', 'elapsed_time']), trigger=(args.validation_interval, 'iteration'))

    if args.progressbar:
        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    vocab_path = os.path.join(args.out, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(args.out, 'best_model.npz')
    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

    if args.save_init:
        chainer.serializers.save_npz(os.path.join(args.out, 'init_model.npz'), model)

    if args.resume is not None:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    print('Start trainer.run: {}'.format(current_datetime))
    trainer.run()
    print('Elapsed_time: {}'.format(datetime.timedelta(seconds=time.time()-start)))

    if args.save_fin:
        # save latest epoch model
        chainer.serializers.save_npz(os.path.join(args.out, 'epoch{}_model.npz'.format(args.epoch)), model)


if __name__ == '__main__':
    main()
