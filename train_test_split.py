import argparse
import numpy as np


def dump_seqs(seqs, dump_path: str):
    with open(dump_path, 'w') as f:
        for seq_i, seq in enumerate(seqs):
            f.write(' '.join([str(seq_i)] + [str(x) for x in seq]))
            f.write('\n')

def load_seqs(load_path: str):
    seqs = []
    with open(load_path, 'r') as f:
        for line in f:
            seqs.append([int(x) for x in line.strip().split(' ')[1:]])
    return seqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_output', type=str, required=True)
    parser.add_argument('--test_output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_tests', type=int, required=True)
    args = parser.parse_args()

    seqs = load_seqs(args.input)
    np.random.shuffle(seqs)
    train_seqs = seqs[:-args.num_tests]
    test_seqs = seqs[-args.num_tests:]

    dump_seqs(train_seqs, args.train_output)
    dump_seqs(test_seqs, args.test_output)
