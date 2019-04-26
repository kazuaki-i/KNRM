
# K-NRM: Kernel-based Neural Ranking Model

This is an ranking for query-document pairs using kernel based neural ranking model (K-NRM) coding by [Chainer](https://chainer.org/).
To leaning K-NRM, see [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://arxiv.org/abs/1706.06613).

# Setup dataset

To train this program, you must prepare dataset:

```
input1(words separated space) \t input2(words separated space) \t score
```

Split dataset by query:
```
cat dataset | python split_dataset.py > pointwise.train
```


At default, development dataset is saved as pointwise.dev and test datasets as pointwise.test


Convert dataset for pairwise format:
```
cat pointwise.train | python convert_pairwise_format.py > pairwise.train
cat pointwise.dev | python convert_pairwise_format.py > pairwise.dev
cat pointwise.test | python convert_pairwise_format.py > pairwise.test  
```

pairwise.train and pairwise.dev are used at training time
pairwise.test is used at evaluation time

## Optional
If you want to save diskspace, you should compress pointwise.train and pairwise.train:

```
cat dataset | python split_dataset.py | gzip -c > pointwise.train.gz
zcat pointwise.train.gz | python convert_pairwise_format.py > pairwise.train.gz
```

If your dataset will cause out-of-memory, you should split pairwise.train:
```
mkdir train; cd train
split -l 100000 ../pairwise.train -d
```

If you want to train efficiently, you pre-training word vector by word2vec.
Option -v of `train_text_pair_ranking.py` can load word2vec vector file. 


# How to RUN

To train model:

```
python train_text_pair_ranking.py -T "train/x*" -D pairwise.dev -vs pointwise.train --use-dataset-api -k kernels.csv -g 0
```

train_text_pair_ranking.py is REQUIRED -T, -D and (-vs or -v) options.

Important options are below:
- vocab-source(-vs): vocab source file (e.g. pointwise.train)
- v(-v): pre-training word2vec data (not binary)
- kernel(-k): RBF-kernel parameters you defined by csv format (See kernels.csv)

The output directory result contains:

- best_model.npz: a model snapshot, which won the best accuracy for validation data during training
- vocab.json: model's vocabulary dictionary as a json file
- args.json: model's setup as a json file, which also contains paths of the model and vocabulary
- snapshot_latest: latest snapshot data (you can resume training using -r option) 


To evaluate:
```
Coming soon!
```

---

# K-NRM: Kernel-based Neural Ranking Model について

このプログラムはKernel-based Neural Ranking Model(K-NRM)を利用したランキング学習のモデルです。

元論文はこちら： [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://arxiv.org/abs/1706.06613)


# Setup dataset
まず以下のフォーマットのデータが必要です。


```
input1(words separated space) \t input2(words separated space) \t score
```

まず、データセットをクエリ（input1）単位で学習・開発・テストセットに分割します。
```
cat dataset | python split_dataset.py > pointwise.train
```
初期設定では、開発セットが「pointwise.dev」、テストセットが「pointwise.test」として保存されます。


次に、pairwise学習用のフォーマットに書き換えます

```
cat pointwise.train | python convert_pairwise_format.py > pairwise.train
cat pointwise.dev | python convert_pairwise_format.py > pairwise.dev
cat pointwise.test | python convert_pairwise_format.py > pairwise.test  
```

pairwise.train と pairwise.dev が学習で、pairwise.test が評価に使われます。


## Optional
ディスク容量を節約したい場合は、圧縮しつつ処理してください
```
(z)cat dataset | python split_dataset.py | gzip -c > pointwise.train.gz
zcat pointwise.train.gz | python convert_pairwise_format.py > pairwise.train.gz
```

学習データがメモリに乗りきらない場合は、ファイルを分割しておくとロード時のログが出るようになります。
```
mkdir train; cd train
split -l 100000 ../pairwise.train -d
```

学習を効率的に進めたい場合は、word2vecで単語ベクトルをあらかじめ学習してください。
-v オプションでword2vecの単語ベクトルのファイルを呼び出せます。

# How to RUN

学習：

```
python train_text_pair_ranking.py -T "train/x*" -D pairwise.dev -vs pointwise.train --use-dataset-api -k kernels.csv -g 0
```
-T と -D は必須、-vs と -v はどちらか必須のオプションです。


重要なオプションは以下の通りです:
- vocab-source(-vs): vocabularyの元となるファイルです。単語ベクトルの初期値はランダムになります
- v(-v): word2vecで学習した単語ベクトルのデータです (not binary)
- kernel(-k): RBF-kernel のパラメータです (See kernels.csv)

出力は以下の通りです：
- best_model.npz: a model snapshot, which won the best accuracy for validation data during training
- vocab.json: model's vocabulary dictionary as a json file
- args.json: model's setup as a json file, which also contains paths of the model and vocabulary
- snapshot_latest: latest snapshot data (you can resume training using -r option) 


評価:
```
準備中!
```
