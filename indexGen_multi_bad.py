import os
import sox
import json
import math
import h5py
import torch
import random
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.special import softmax
from sox.file_info import duration
from multiprocessing import Process, set_start_method
from python_speech_features import logfbank

from module.model import Gvector

def parse_args():
    desc="align smaplerate of dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataDir', type=str, default=None, help="path to the input dir")
    parser.add_argument('--outDir', type=str, default=None, help="path to the output dir")
    parser.add_argument('--process', type=int, default=20, help="number of process running")
    parser.add_argument('--test-size', type=float, default=0.15, help="ratio of test set from all")
    parser.add_argument('--threshold', type=int, default=1000, help="length of each species")
    parser.add_argument('--class2count', type=str, default="class2count.json", help="class2count json file")
    parser.add_argument('--isTrain', action='store_true', help="parse the train dataset")
    parser.add_argument('--model-bad', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--id', type=str, default="", help="id for your index files")
    return parser.parse_args()

mdl_bad_kwargs = {
    "channels": 16, 
    "block": "BasicBlock", 
    "num_blocks": [2,2,2,2], 
    "embd_dim": 1024, 
    "drop": 0.3, 
    "n_class": 2
}

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

def extract_feat(y, sr, cmn=True):
    kwargs = {
        "winlen": 0.025,
        "winstep": 0.01,
        "nfilt": 80,
        "nfft": 2048,
        "lowfreq": 50,
        "highfreq": 8000,
        "preemph": 0.97
    }
    logfbankFeat = logfbank(y, sr, **kwargs)
    if cmn:
        logfbankFeat -= logfbankFeat.mean(axis=0, keepdims=True)
    return logfbankFeat.astype('float32')

def get_floor(x, n):
    return math.floor(x * 10**n)/10**n

def gen(pid, args, all_records, class2nsample, detector):
    nproc = args.process
    outDir = args.outDir
    isTrain = args.isTrain
    
    isFirst = True
    
    portion = len(all_records) // nproc
    
    if pid == 0:
        all_records = all_records[:portion]
    elif pid == nproc - 1:
        all_records = all_records[portion*pid:]
    else:
        all_records = all_records[portion*pid:portion*(pid+1)]
    
    tagged_utt2wav = []
    tagged_utt2label = []
    
    for rec in tqdm(all_records, desc="process %d"%os.getpid()):
        if isFirst:
            with open('kill.sh','a') as f:
                f.write('kill -9 %d\n'%os.getpid())
            isFirst = False
        
        rec_len = sox.file_info.duration(rec)
        if rec_len <= 5:
            continue
        feat_path = rec.replace('aligned','feats').replace('.wav','.h5')
        rec_h5 = h5py.File(feat_path, 'r')
        feat_rec = np.array(rec_h5.get('logfbank'))
        rec_h5.close()
        # y, sr = sf.read(rec)
        feat_len = len(feat_rec)
        if feat_len <=500:
            continue
        bird_species = rec.split('/')[-2]
        bird_samples = class2nsample[bird_species]
        max_attempt = 50 * bird_samples
        curr_attempt = 0
        nstarts = []
        while len(nstarts)<bird_samples:
            # start_sec = get_floor(random.uniform(0, rec_len-5),2)
            start_sec = random.randint(0, (feat_len-500))
            rand_start = int(start_sec*100)
            # y_tmp = y[rand_start:(rand_start+sr*5)]
#             feats_tmp = extract_feat(y_tmp, sr)
            feats_tmp = feat_rec[start_sec:(start_sec+500)]
            hasBird = np.argmax(softmax(detector(feats_tmp)))
            if hasBird:
                curr_attempt = 0
                nstarts.append(start_sec)
            curr_attempt += 1
            if curr_attempt >= max_attempt:
                print("wav %s stopped bouncing (%d/%d)"%(rec.split('/')[-1], len(nstarts), bird_samples))
                break
#         nstarts = [get_floor(random.uniform(0, rec_len-5),2) for i in range(bird_samples)]
        tagged_utt2wav += [rec.split('/')[-1].split('.')[0] + '_%d '%i + rec.replace("aligned","feats").replace('.wav','.h5') + ' %.2f'%(fn/100) for i,fn in enumerate(nstarts)]
        tagged_utt2label += [rec.split('/')[-1].split('.')[0] + '_%d '%i + bird_species for i in range(len(nstarts))]
    
    if isTrain:
        sec = 'train'
    else:
        sec='devel'
    
    with open('%s%s_utt2wav_%s'%(outDir, sec, args.id),'a') as f:
        f.write('\n'.join(tagged_utt2wav)+'\n')
    with open('%s%s_utt2label_%s'%(outDir, sec, args.id),'a') as f:
        f.write('\n'.join(tagged_utt2label)+'\n')
    
    
if __name__ == '__main__':
    with open('kill.sh','w') as f:
        f.write('')
    
    # load args and clips
    args = parse_args()
    dataDir = args.dataDir
    allClips = []
    with open(args.class2count,'r') as f:
        all_birds_class2count = json.load(f)
    all_birds = list(all_birds_class2count.keys())
    all_birds_class2sample = {k:int(args.threshold//all_birds_class2count[k])+1 for k in all_birds_class2count}
        
    # init output directories
    outDir = args.outDir.rstrip('/') + '/'
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        
    # load bad model
    print('... loading activity detector ...')
    bad_extractor = SVExtractor(mdl_bad_kwargs, args.model_bad, device=args.device)
    print('... loaded ...')
    
    # train/test set
    if args.isTrain:
        with open('%strain_utt2wav_%s'%(outDir, args.id),'w') as f:
            f.write('')
        with open('%strain_utt2label_%s'%(outDir, args.id),'w') as f:
            f.write('')
        for bird in all_birds:
            tmpClips = os.listdir(dataDir+bird)
            if len(tmpClips) > args.threshold:
                random.seed(42)
                tmpClips = random.sample(os.listdir(dataDir+bird), args.threshold)
            tmpClips = tmpClips[int(args.test_size*len(tmpClips)):]
            allClips += [dataDir + bird + '/' + x for x in tmpClips if x.endswith('mp3') or x.endswith('wav')]
    else:
        with open('%sdevel_utt2wav_%s'%(outDir, args.id),'w') as f:
            f.write('')
        with open('%sdevel_utt2label_%s'%(outDir, args.id),'w') as f:
            f.write('')
        for bird in all_birds:
            tmpClips = os.listdir(dataDir+bird)
            if len(tmpClips) > args.threshold:
                random.seed(42)
                tmpClips = random.sample(os.listdir(dataDir+bird), args.threshold)
            tmpClips = tmpClips[:int(args.test_size*len(tmpClips))]
            allClips += [dataDir + bird + '/' + x for x in tmpClips if x.endswith('mp3') or x.endswith('wav')]
        
    
    # multiprocessing transformation
    worker_count = args.process
    worker_pool = []
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    for i in range(worker_count):
        p = Process(target=gen, args=(i, args, allClips, all_birds_class2sample, bad_extractor))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")
