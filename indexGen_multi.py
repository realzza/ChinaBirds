import os
import sox
import json
import argparse
from tqdm import tqdm
from multiprocessing import Process

def parse_args():
    desc="align smaplerate of dataset"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataDir', type=str, default=None, help="path to the input dir")
    parser.add_argument('--outDir', type=str, default=None, help="path to the output dir")
    parser.add_argument('--process', type=int, default=20, help="number of process running")
    return parser.parse_args()


def gen(pid, nproc, all_records, outDir):
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
        if rec_len > 5:
            max_start = int(rec_len) - 4
            tagged_utt2wav += [rec.split('/')[-1].split('.')[0] + '_%d '%i + rec + ' %d'%i for i in range(max_start)]
            tagged_utt2label += [rec.split('/')[-1].split('.')[0] + '_%d '%i + rec.split('/')[-2] for i in range(max_start)]
    with open('%sutt2wav_id_%d'%(outDir, pid),'w') as f:
        f.write('\n'.join(tagged_utt2wav))
    with open('%sutt2label_id_%d'%(outDir, pid),'w') as f:
        f.write('\n'.join(tagged_utt2label))
    
    
if __name__ == '__main__':
    with open('kill.sh','w') as f:
        f.write('')
    
    # load args and clips
    args = parse_args()
    dataDir = args.dataDir
    allClips = []
    with open('class2weight.json','r') as f:
        all_birds = list(json.load(f).keys())
    for bird in all_birds:
        tmpClips = os.listdir(dataDir+bird)
        allClips += [dataDir + bird + '/' + x for x in tmpClips if x.endswith('mp3') or x.endswith('wav')]
        
    # init output directories
    outDir = args.outDir.strip('/') + '/'
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        
    # multiprocessing transformation
    worker_count = args.process
    worker_pool = []
    for i in range(worker_count):
        p = Process(target=gen, args=(i, worker_count, allClips, outDir))
        p.start()
        worker_pool.append(p)
    for p in worker_pool:
        p.join()  # Wait for all of the workers to finish.

    # Allow time to view results before program terminates.
    a = input("Finished")