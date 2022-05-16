import os 
import numpy as np 
import json
import math 
import random 
import soundfile
import h5py
import torch 
from copy import deepcopy
import multiprocessing
from multiprocessing import Process
from tqdm import tqdm
from .augmentation import *
from torch.utils.data import Dataset 
from python_speech_features import logfbank 


class LogfbankDataset(Dataset):
    def __init__(self, wav_scp, utt2label=None, spk2int=None, samplerate=None, padding="wrap", cmn=True, tdur=5, pad0=False, p_aug=0, nproc=40):
        """
        Params:
            wav_scp         - <utt> <wavpath>
            utt2label       - <utt> <lab>
            sample_rate     - config of logfbank features
            padding         - "wrap" or "constant"(zeros), for feature truncation, 
                              effective only if waveform length is less than truncated length.
            cmn             - whether perform mean normalization for feats.
            tdur            - time duration of each utterance
        """    
        self.utt2wavpath = {x.split()[0]:(x.split()[1], float(x.split()[2])) for x in open(wav_scp)} # utt: (wav, t_start)
        self.utt2label = self.init_label(utt2label, p_aug, spk2int)
        self.utts = sorted(list(self.utt2wavpath.keys()))
        self.sample_rate = samplerate
        self.padding = 'wrap' if padding == "wrap" else 'constant'
        self.cmn = cmn
        self.len = len(self.utts)
        self.tdur = tdur * 100 # 16k sample rate
        self.ispad = pad0
        self.augs = ['_fm', '_tm', '_cb']
#         self.feats_all = []
#         self.labels_all = []
#         self.utts_all = []
        # multiprocess data loading
        with open('kill.sh','w') as f:
            f.write('')
        
        worker_pool = []
        manager = multiprocessing.Manager()
        return_feats = manager.list()
        return_labels = manager.list()
        return_utts = manager.list()
        for i in range(nproc):
            p = Process(target=self.load_stuff, args=(i, self.utts, return_feats, return_labels, return_utts, nproc))
            worker_pool.append(p)
            p.start()
        for p in worker_pool:
            p.join()
            
        self.labels_all = return_labels
        self.feats_all = return_feats
        self.utts_all = return_utts
        print("... Stuff preloading finished, %d stuff in total ..."%(len(self.labels_all)))
        assert len(self.labels_all) == len(self.feats_all) == len(self.utts_all), "unequal stuff"

    def load_stuff(self, pid, org_utts, return_feats, return_labels, return_utts, nproc):
        isFirst = True
        portion = len(org_utts) // nproc
        if pid == 0:
            curr_utts = org_utts[:portion]
        elif pid == nproc - 1:
            curr_utts = org_utts[portion*pid:]
        else:
            curr_utts = org_utts[portion*pid:portion*(pid+1)]
        
        tmp_feats = []
        tmp_labels = []
        tmp_utts = []
        
        
        for utt_each in tqdm(curr_utts, desc="process %d"%os.getpid()):
            if isFirst:
                with open('kill.sh','a') as f:
                    f.write('kill -9 %d\n'%(os.getpid()))
                    isFirst = False
            tmp_feats.append(self.extract_feature(utt_each))
            tmp_labels.append(self.utt2label[utt_each])
            tmp_utts.append(utt_each)
        
#         if "feats_all" not in return_dict:
#             return_dict["feats_all"] = []
#         if "labels_all" not in return_dict:
#             return_dict["labels_all"] = []
#         if "utts_all" not in return_dict:
#             return_dict["utts_all"] = []
        
#         print(len(tmp_feats), len(tmp_labels), len(tmp_utts))
        return_labels += tmp_labels
        return_feats += tmp_feats
        
        return_utts += tmp_utts
        print(len(return_utts))
#         print(len(self.feats_all), len(self.labels_all), len(self.utts_all))
            
            
    def init_label(self, utt2label, p_aug, spk2int=None):
        utt2lab = {x.split()[0]:x.split()[1] for x in open(utt2label)}
        utt2lab_aug = deepcopy(utt2lab)
        # expand utt2lab
        if p_aug:
            for utt in tqdm(utt2lab, desc="preparing augmentation"):
                # freq mask
                if random.random() <= p_aug:
                    utt2lab_aug[utt+"_fm"] = utt2lab_aug[utt]
                    self.utt2wavpath[utt+"_fm"] = self.utt2wavpath[utt]
                if random.random() <= p_aug:
                    utt2lab_aug[utt+"_tm"] = utt2lab_aug[utt]
                    self.utt2wavpath[utt+'_tm'] = self.utt2wavpath[utt]
                if random.random() <= p_aug:
                    utt2lab_aug[utt+'_cb'] = utt2lab_aug[utt]
                    self.utt2wavpath[utt+'_cb'] = self.utt2wavpath[utt]
        if spk2int is None:
            spks = sorted(set(utt2lab.values()))
            spk2int = {spk:i for i, spk in enumerate(spks)}
        else:
            with open(spk2int,'r') as f:
                spk2int = json.load(f)
        utt2label = {utt:spk2int[spk] for utt, spk in utt2lab_aug.items()}
        return utt2label
        
        
    def trun_wav(self, y, tlen, padding, ispad0):
        """
        Truncation, zero padding or wrap padding for waveform.
        """
        # no need for truncation or padding
        if tlen is None:
            return y
        n = len(y)
        # needs truncation
        if n > tlen:
            offset = random.randint(0, n - tlen)
            y = y[offset:offset+tlen]
            return y
        # needs padding (zero/repeat padding)
        if ispad0:
            zeros = np.zeros((tlen, 256))
            zeros[:y.shape[0],:y.shape[1]] = y
            return zeros
        else:
            y = np.resize(y, (tlen, 256))
            return y
    
    def tailor_feature(self, y, t_dur, t_start):
        y = y[int(t_start*100):int(t_start*100)+t_dur]
        return np.resize(y, (500,80))


    def extract_feature(self, h5_utt):
        h5Dir = self.utt2wavpath[h5_utt]
        hf = h5py.File(h5Dir[0], 'r')
        logfbankFeat = np.array(hf.get('logfbank'))
        hf.close()
        if any(h5_utt.endswith(x) for x in self.augs):
            aug_method = h5_utt.split('_')[-1]
            if aug_method == "fm":
#                 print('... applying fm ...')
                logfbankFeat = freq_mask(logfbankFeat, num_masks=2)
            elif aug_method == "tm":
#                 print('... applying tm ...')
                logfbankFeat = time_mask(logfbankFeat, num_masks=2)
            elif aug_method == "cb":
#                 print('... applying cb ...')
                logfbankFeat = time_mask(freq_mask(logfbankFeat, num_masks=2), num_masks=2)

        logfbankFeat = self.tailor_feature(logfbankFeat, self.tdur, h5Dir[1])
        return logfbankFeat.astype('float32')


    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
            
        utt = self.utts_all[index]
        feat = self.feats_all[index]
        label = self.labels_all[index]
#         feat = self.extract_feature(utt)
#         label = self.utt2label[utt]
        return utt, feat, label
#         return utt, feat


    def __len__(self):
        return self.len

