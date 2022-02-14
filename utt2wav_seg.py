import os
import sox
import json
import argparse
from tqdm import tqdm

with open('class2weight.json','r') as f:
    all_birds = list(json.load(f).keys())
all_records = []
database = '/DATA1/ziang/data/1480birds/wavs/'
# get all recordings
for bird in tqdm(all_birds):
    all_records += [database+bird+'/'+rec for rec in os.listdir(database+bird) if rec.endswith('.wav')]

# tag starting point
tagged_recordings = []
for rec in tqdm(all_records):
    rec_len = sox.file_info.duration(rec)
    if rec_len > 5:
        max_start = int(rec_len) - 4
        tagged_recordings += [rec.split('/')[-1].split('.')[0] + ' ' + rec + ' %d'%i for i in range(max_start)]

with open('index/utt2wav_id','w') as f:
    f.write('\n'.join(tagged_recordings))