# ChinaBirds

## Setup Environment
```bash
pip install -r requirements.txt
```

## Data Preparation
We utilized [XenoPy](https://github.com/realzza/xenopy) to query data from Xeno-Canto 2.0. 

## Extract Features
Extract acoustic features with `extract_feature.py`. Params are the folowing,
```
usage: extract_feature.py [-h] [--data DATA] [--output OUTPUT] [--sr SR] [--nproc NPROC]

extract features to h5 struct

optional arguments:
  -h, --help       show this help message and exit
  --data DATA      path to the input/output dir
  --output OUTPUT  path to the output dir
  --sr SR
  --nproc NPROC    number of process to extract features
```

## Generate Index
It is necessary to generate index files for model training. Generated index files are in directories `index-try` and `index-new`. We can refer to `indexGen_multi.py` and `indexGen_multi_bad.py` to generate index files.
`_bad` suffix mean that bird activity detection is integrated in the index generation process. Params are the following,
```
usage: indexGen_multi_bad.py [-h] [--dataDir DATADIR] [--outDir OUTDIR] [--process PROCESS] [--test-size TEST_SIZE]
                             [--threshold THRESHOLD] [--class2count CLASS2COUNT] [--isTrain] --model-bad MODEL_BAD
                             [--device DEVICE] [--id ID]

align smaplerate of dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataDir DATADIR     path to the input dir
  --outDir OUTDIR       path to the output dir
  --process PROCESS     number of process running
  --test-size TEST_SIZE
                        ratio of test set from all
  --threshold THRESHOLD
                        length of each species
  --class2count CLASS2COUNT
                        class2count json file
  --isTrain             parse the train dataset
  --model-bad MODEL_BAD
  --device DEVICE
  --id ID               id for your index files
```

## Train
After index files are generated, we can train deep learning models with `train.py` script. A sample command is 
```bash
python train.py --config conf/logfbank_train.json --name some_name
```

## Inference
After the training script runs successfully, we can find the trained models in `exp/some_name/chkpt/`, two models are saved, `chkpt_best.pth` and `chkpt_last.pth`. Two groups of inference script can be run, with and without BAD. 

#### Without BAD
```text
usage: infer-vote.py [-h] --utt2wav UTT2WAV --utt2lab UTT2LAB --label2int LABEL2INT [--output OUTPUT] --model-clf
                     MODEL_CLF [--device DEVICE]

infer labels

optional arguments:
  -h, --help            show this help message and exit
  --utt2wav UTT2WAV
  --utt2lab UTT2LAB
  --label2int LABEL2INT
  --output OUTPUT       path to the output file
  --model-clf MODEL_CLF
  --device DEVICE
```

#### With BAD
Inference with BAD at `infer-vote-bad.py`.
```text
usage: infer-vote-bad.py [-h] --utt2wav UTT2WAV --utt2lab UTT2LAB --label2int LABEL2INT [--output OUTPUT] --model-clf
                         MODEL_CLF --model-bad MODEL_BAD [--device DEVICE]

infer labels

optional arguments:
  -h, --help            show this help message and exit
  --utt2wav UTT2WAV
  --utt2lab UTT2LAB
  --label2int LABEL2INT
  --output OUTPUT       path to the output file
  --model-clf MODEL_CLF
  --model-bad MODEL_BAD
  --device DEVICE
```

## Visualizations
Visualizations in my SW can be found at directory `visualization`.


## Todo
- [x] prepare a balanced dataloader
- [x] prepare a utt2wav and utt2label. [utt] [wav] [start]
- [x] write in multithread style
- [ ] run the VAD model
- [x] prepare a customized dataset
- [ ] make sure label2int's order is the same as initializing the sampler.
- [ ] the object that the sampler is working on is merely a feature, using a window to sample. There must be an encoding&decoding mechanism to sample from the same file.
- [ ] remove unnecessary part in `train.py` and config files.