import os
import sys
import sox
import json
import h5py
import time
import torch
import shutil
import GPUtil
import argparse
import numpy as np
import pandas as pd
import audiofile as af

from tqdm import tqdm
from scipy.special import softmax
from python_speech_features import logfbank

from module.model import Gvector


# config:
mdl_clf_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 821
}

mdl_bad_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 2
}

logfbank_kwargs = {
    "winlen": 0.025, 
    "winstep": 0.01, 
    "nfilt": 80, 
    "nfft": 2048, 
    "lowfreq": 50, 
    "highfreq": None, 
    "preemph": 0.97    
}


# parse args
def parse_args():
    desc="infer labels"
    parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument('--feat-dir', type=str, required=True, help="path to the feature of dataset dir for inference")
#     parser.add_argument('--data-dir', type=str, required=True, help="path to the dataset dir")
    parser.add_argument('--utt2wav', type=str, required=True)
    parser.add_argument('--utt2lab', type=str, required=True)
    parser.add_argument('--label2int', type=str, required=True)
    parser.add_argument('--output', type=str, default=None, help="path to the output file")
    parser.add_argument('--model-clf', type=str, required=True)
    parser.add_argument('--model-bad', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    return parser.parse_args()

def extract_feat(wav_path, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 80,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 8000,
        "preemph": 0.97
    }
    y, sr = af.read(wav_path)
    logfbankFeat = logfbank(y, sr, **kwargs)
    if cmn:
        logfbankFeat -= logfbankFeat.mean(axis=0, keepdims=True)
    return logfbankFeat.astype('float32')
    

class SVExtractor():
    def __init__(self, mdl_kwargs, model_path, device):
        self.model = self.load_model(mdl_kwargs, model_path, device)
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)

    def load_model(self, mdl_kwargs, model_path, device):
        model = Gvector(**mdl_kwargs)
        state_dict = torch.load(model_path, map_location=device)
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.load_state_dict(state_dict)
        return model

    def __call__(self, frame_feats):
        feat = torch.from_numpy(frame_feats).unsqueeze(0)
        feat = feat.float().to(self.device)
        with torch.no_grad():
            embd = self.model(feat)
        embd = embd.squeeze(0).cpu().numpy()
        return embd
    
def most_common(lst):
    return max(set(lst), key=lst.count)
    
def infer_wav(wav, detector, featExtractor, int2label_dict):
#     wav_len = sox.file_info.duration(wav)
    wav_feats = extract_feat(wav)
    feat_len = len(wav_feats)
    start_ts = 0 # starting time stamp
    window_len = 500
    window_shift = 100 # customize here
    
    # record results
    results = []
    while start_ts <= (feat_len - window_len) and len(results)<=50:
        frame_feats = wav_feats[start_ts:start_ts+window_len]
        # apply detector
        hasBird = np.argmax(softmax(detector(frame_feats)))
        if hasBird:
            softmax_embd = softmax(featExtractor(frame_feats))
            results.append(np.argmax(softmax_embd))
        start_ts += window_shift
    vote_species = most_common(results)
    return int2label_dict[vote_species]
    
if __name__ == "__main__":
    
    with open('kill.sh','w') as f:
        f.write('')
        
    args = parse_args()
    model_clf_path = args.model_clf
    model_bad_path = args.model_bad
    utt2wav_path = args.utt2wav
    with open(utt2wav_path,'r') as f:
        all_utts_h5 = [line.split()[1] for line in f.read().split('\n') if line]
    all_utts_h5 = list(set(all_utts_h5))
    all_utts_wavs = [line.replace('3h-feats','3h-aligned').replace('.h5','.wav') for line in all_utts_h5]
    with open(args.label2int,'r') as f:
        label2int = json.load(f)
    int2label = {v:k for k,v in label2int.items()}
    
    # outdir
    outDir = '/'.join(args.output.split('/')[:-1])
    os.makedirs(outDir, exist_ok=True)
    
    with open(args.utt2lab, 'r') as f:
        all_utt2label = f.read().split('\n')
        all_utt2label = [line.split('_')[0]+' '+line.split()[1] for line in all_utt2label if line]
        all_utt2label = list(set(all_utt2label))
        utt2label_dict = {line.split()[0]:line.split()[1] for line in all_utt2label if line}
    
        
    print('... loading activity detector ...')
    bad_extractor = SVExtractor(mdl_bad_kwargs, model_bad_path, device=args.device)
    print('... loaded ...')
    
    print('... loading classifer ...')
    clf_extractor = SVExtractor(mdl_clf_kwargs, model_clf_path, device=args.device)
    print('... loaded ...')
    
    pred_dict = {}
    
    for wav in tqdm(all_utts_wavs):
        try:
            species_pred = infer_wav(wav, bad_extractor, clf_extractor, int2label)
        except:
            print("all noise wav %s"%wav)
            continue
        species_real = utt2label_dict[wav.split('/')[-1][:-4]]
#         print(species_pred, species_real, species_pred==species_real)
        if not species_real in pred_dict:
            pred_dict[species_real] = [0,0,0]
        pred_dict[species_real][1] += 1
        if species_pred == species_real:
            pred_dict[species_real][0] += 1
        pred_dict[species_real][2] = pred_dict[species_real][0]/pred_dict[species_real][1]
    
    pred_dict = {k: v for k, v in sorted(pred_dict.items(), key=lambda item: item[1][2], reverse=True)}
    
    with open(args.output,'w') as f:
        json.dump(pred_dict, f)