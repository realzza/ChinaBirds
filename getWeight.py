import os
import json
import argparse

def parse_args():
    desc="calculate weight used in balanced dataloader"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--bird2dur', type=str, required=True, help="[1] file of bird2duration")
    parser.add_argument('--save','-o', type=str, default='./', help="[2] output dir")
    parser.add_argument('--min-len', type=float, default=10, help="[3] low threshold for total length")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()

    # load json file
    with open(args.bird2dur,'r') as f:
        bird2duration = json.load(f)
    
    # calculating weights
    bird2count = {k:int((int(v)-4)*0.9) for k,v in bird2duration.items() if v > args.min_len}
    countSum = sum(list(bird2count.values()))
    bird2weight = {k:(countSum/v) for k,v in bird2count.items()}
    
    # save weights:
    with open(args.save+"class2weight.json",'w') as f:
        json.dump(bird2weight, f)