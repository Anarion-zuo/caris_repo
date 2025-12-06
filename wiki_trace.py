import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import urllib.parse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--max_item_id', type=int, required=True)
    args = parser.parse_args()
    
    item_id_map = dict()
    trace = list()
    
    with open(args.input, 'rb') as reader:
        reader.seek(0)
        raw_num_lines = sum(1 for _ in reader)
        
    with open(args.input, 'r', encoding='iso-8859-1') as reader:
        reader.seek(0)
        for line in tqdm(reader, total=raw_num_lines, desc='loading'):
            line = line.strip(' \r\n')
            splitted = line.split(' ')
            url = ''.join(splitted[2:-1])
            req_path = urllib.parse.urlparse(url).path
            req_path_splitted = req_path.split('/')
            if len(req_path_splitted[0]) != 0:
                print("skipping line [{}], format not expected, req_path {}".format(line, req_path_splitted))
            if len(req_path_splitted) != 3:
                continue
            if req_path_splitted[1] != 'wiki':
                continue
            item_id_raw = (hash(req_path_splitted[2]) % args.max_item_id)
            if item_id_raw not in item_id_map:
                item_id = len(item_id_map)
                item_id_map[item_id_raw] = item_id
            else:
                item_id = item_id_map[item_id_raw]
            
            timestamp = float(splitted[1])
            trace.append((timestamp, item_id,))
    sorted_trace = [item_id for _, item_id in sorted(trace, key=lambda x: x[0])]
    print("#items {}, #reqs".format(len(item_id_map)), len(trace))
    with open(args.output, 'w') as writer:
        seq_list = np.array_split(sorted_trace, len(sorted_trace) // args.seq_len)
        for seq_i, seq in tqdm(enumerate(seq_list), desc='write seqs'):
            writer.write(' '.join([str(x) for x in [seq_i] + seq]) + '\n')
    