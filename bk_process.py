import argparse, dateutil
import dateutil.parser
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    loc_id_map = dict()
    
    user_seq_map = dict()
    with open(args.input, "rbU") as reader:
        reader.seek(0)
        raw_num_lines = sum(1 for _ in reader)
    with open(args.input, 'r') as reader:
        reader.seek(0)
        for line in tqdm(reader, desc='read user seqs from file', total=raw_num_lines):
            line = line.strip(' \r\n')
            user_id, date, _, _, loc_raw_id = line.split('\t')
            if loc_raw_id not in loc_id_map:
                loc_id = len(loc_id_map)
                loc_id_map[loc_raw_id] = loc_id
            else:
                loc_id = loc_id_map[loc_raw_id]
            try:
                timestamp = round(datetime.timestamp(dateutil.parser.parse(date)))
            except dateutil.parser.ParserError:
                print("Skipping line due to parser error, date: [{}], line: [{}]".format(date, line))
                continue
            if user_id in user_seq_map:
                user_seq_map[user_id].append((timestamp, loc_id, ))
            else:
                user_seq_map[user_id] = [(timestamp, loc_id,)]
    user_sorted_seq_map = {
        user_id: [
            t[1]
            for t in sorted(seq, key=lambda x: x[0])
        ]
        for user_id, seq in tqdm(user_seq_map.items(), desc="sort user seqs")
    }
    seq_len_list = list()
    item_counter = Counter()
    with open(args.output, 'w') as writer:
        for user_id, seq in tqdm(user_sorted_seq_map.items(), desc='dump user seqs'):
            writer.write(' '.join([str(x) for x in [user_id] + seq]) + '\n')
            seq_len_list.append(len(seq))
            item_counter.update(seq)
    item_counter_values = list(item_counter.values())
            
    print("#users: {}, #locs: {}".format(len(user_sorted_seq_map), len(loc_id_map)))
    print("seq len: avg {}, p50 {}, p99 {}, min {}, max {}".format(
        np.mean(seq_len_list), np.quantile(seq_len_list, 0.50), np.quantile(seq_len_list, 0.99), 
        np.min(seq_len_list), np.max(seq_len_list),
    ))
    print("#visits: avg {}, p50 {}, p99 {}, min {}, max {}".format(
        np.mean(item_counter_values), np.quantile(item_counter_values, 0.50), 
        np.quantile(item_counter_values, 0.99), 
        np.min(item_counter_values), np.max(item_counter_values),
    ))
    