import os
import sys
import h5py
import librosa
import numpy as np
import random
from scipy.io import wavfile
import soundfile as sf
import python_speech_features
from tqdm import tqdm
from shutil import copyfile
import argparse

import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, AveragePooling2D
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.regularizers import l2

# parse argument into our program
def parse_args():
    desc="tease noise"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, default=None, help="path to the input dir")
    parser.add_argument('--noise', type=str, default=None, help="path to the output dir")
    parser.add_argument('--model', type=str, 
                        default='/Netdata/2020/ziang/code/ukybirddet/trained_model/baseline/model97.h5', 
                        help='path/to/model')
    parser.add_argument('--gpuid', type=str, default='8', help="support single gpu id")
    parser.add_argument('--threshold', type=float, default=0.5, help="threshold to judged as noise, activate only when strict_floor is set")
    parser.add_argument('--cutoff-freq', type=int, default=10)
    parser.add_argument('--remove', type=bool, default=False, help="whether to remove noise after detected")
    return parser.parse_args()

args = parse_args()
# select gpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
thre = args.threshold
isremove = args.remove

# build model
model = Sequential()
# augmentation generator
# code from baseline : "augment:Rotation|augment:Shift(low=-1,high=1,axis=3)"
# keras augmentation:
#preprocessing_function

# convolution layers
model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(700, 80, 1), ))  # low: try different kernel_initializer
model.add(BatchNormalization())  # explore order of Batchnorm and activation
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 3)))  # experiment with using smaller pooling along frequency axis
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 1)))
model.add(Conv2D(16, (3, 3), padding='valid', kernel_regularizer=l2(0.01)))  # drfault 0.01. Try 0.001 and 0.001
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 1)))

# dense layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))  # leaky relu value is very small experiment with bigger ones
model.add(Dropout(0.5))  # experiment with removing this dropout
model.add(Dense(1, activation='sigmoid'))

# normalize each clip
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def wav2spec(audioDir, sampleRate):
    y, sr = librosa.load(audioDir, sr=sampleRate)
    logFeat = python_speech_features.base.logfbank(y, samplerate=44100, winlen=0.046, winstep=0.014, nfilt=80, nfft=2048, lowfreq=50, highfreq=12000, preemph=0.97)
    logFeat = np.resize(logFeat, (700,80))
    logFeat = NormalizeData(logFeat)
    return logFeat

# load trained model
modelDir = args.model
modelTest = load_model(modelDir)
noiseDir = args.noise
if not os.path.isdir(noiseDir):
    os.mkdir(noiseDir)
datasetDir = args.data
cutoff = args.cutoff_freq
allBirds = os.listdir(datasetDir)
allSegs = []
for bird in allBirds:
    if len(os.listdir(datasetDir+bird)) < cutoff:
        continue
    allSegs += [datasetDir+bird+'/'+x for x in os.listdir(datasetDir+bird)]
# allSegs = [datasetDir + x for x in os.listdir(datasetDir)]

allNoise = []
for segment in tqdm(allSegs, desc='data noise detecting'):
    segName = segment.split('/')[-1]
    try:
        tmpFeat = wav2spec(segment, sampleRate=44100).reshape(1,700,80,1)
        prob = modelTest.predict(tmpFeat)[0][0]
        if prob < thre:
            allNoise.append(segment)
            if isremove:
                copyfile(segment, noiseDir+segName)
                os.remove(segment)
    except:
        continue
print('... %d songs are determined as noise ...'%len(allNoise))
