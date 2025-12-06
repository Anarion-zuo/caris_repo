import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--min_len', type=int, required=True)
    args = parser.parse_args()

    writer = open(args.output, 'w')
    cur_seq = list()
    num_output_seq = 0
    num_reqs = 0
    item_set = set()

    with open(args.input, 'r') as reader:
        for line in reader:
            line = line.strip()
            if line == '':
                continue
            seq = [int(x) for x in line.split(' ')][1:]
            item_set.update(seq)
            num_reqs += len(seq)
            if len(cur_seq) >= args.min_len:
                writer.write(' '.join([str(num_output_seq)] + [str(x) for x in cur_seq]) + '\n')
                cur_seq = list()
                num_output_seq += 1
            cur_seq += seq

    writer.write(' '.join([str(num_output_seq)] + [str(x) for x in cur_seq]) + '\n')
    num_output_seq += 1
    writer.close()

    print("num_reqs:", num_reqs, "num_output_seq:", num_output_seq, "num_items:", len(item_set))
