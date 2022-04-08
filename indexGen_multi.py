import os
import sox
import json
import math
import random
import argparse
from tqdm import tqdm
from sox.file_info import duration
from multiprocessing import Process

def parse_args():
    desc="align smaplerate of dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataDir', type=str, default=None, help="path to the input dir")
    parser.add_argument('--outDir', type=str, default=None, help="path to the output dir")
    parser.add_argument('--process', type=int, default=20, help="number of process running")
    parser.add_argument('--test-size', type=float, default=0.15, help="ratio of test set from all")
    parser.add_argument('--threshold', type=int, default=1000, help="number of samples from each wav file")
    parser.add_argument('--class2count', type=str, default="class2count.json", help="class2count json file")
    parser.add_argument('--isTrain', action='store_true', help="parse the train dataset")
    parser.add_argument('--id', type=str, default="", help="id for your index files")
    return parser.parse_args()

def get_floor(x, n):
    return math.floor(x * 10**n)/10**n

def gen(pid, args, all_records, class2nsample):
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
        bird_species = rec.split('/')[-2]
        bird_samples = class2nsample[bird_species]
#         if rec_len > 5:
        nstarts = [get_floor(random.uniform(0, rec_len-5),2) for i in range(bird_samples)]
        tagged_utt2wav += [rec.split('/')[-1].split('.')[0] + '_%d '%i + rec.replace("aligned","feats").replace('.wav','.h5') + ' %.2f'%fn for i,fn in enumerate(nstarts)]
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
    for i in range(worker_count):
        p = Process(target=gen, args=(i, args, allClips, all_birds_class2sample))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")
