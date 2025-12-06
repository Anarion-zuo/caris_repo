import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--a', type=float, required=True)
    parser.add_argument('--max_id', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--seq_num', type=int, required=True)
    args = parser.parse_args()
    
    seqs = np.remainder(np.random.zipf(a=args.a, size=(args.seq_num, args.seq_len,)), args.max_id)
    with open(args.output, 'w') as writer:
        for seq_i, seq in tqdm(enumerate(seqs)):
            writer.write(str(seq_i) + ' ')
            writer.write(' '.join([
                str(x) 
                if x < args.max_id 
                else str(args.max_id - 1)
                for x in seq
            ]))
