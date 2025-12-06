import argparse, dateutil
import dateutil.parser
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    table = pd.read_csv(args.input)[['starttime', 'start station id', 'bikeid']]
    
    station_id_map, bike_id_map = dict(), dict()
    
    seq_map = dict()
    for row in tqdm(table.itertuples(), desc='read user seqs from file', total=table.shape[0]):
        # print(row)
        _, start_time, raw_station_id, raw_bike_id = row
        if raw_station_id not in station_id_map:
            station_id = len(station_id_map)
            station_id_map[raw_station_id] = station_id
        else:
            station_id = station_id_map[raw_station_id]
        if raw_bike_id not in bike_id_map:
            bike_id = len(bike_id_map)
            bike_id_map[raw_bike_id] = bike_id
        else:
            bike_id = bike_id_map[raw_bike_id]
        try:
            timestamp = round(datetime.timestamp(dateutil.parser.parse(start_time)))
        except dateutil.parser.ParserError:
            print("Skipping line due to parser error, date str: [{}]".format(start_time))
            continue
        # print(timestamp)
        if bike_id in seq_map:
            seq_map[bike_id].append((timestamp, station_id, ))
        else:
            seq_map[bike_id] = [(timestamp, station_id,)]
    sorted_seq_map = {
        bike_id: [
            t[1]
            for t in sorted(seq, key=lambda x: x[0])
        ]
        for bike_id, seq in tqdm(seq_map.items(), desc="sort seqs")
    }
    seq_len_list = list()
    item_counter = Counter()
    with open(args.output, 'w') as writer:
        for user_id, seq in tqdm(sorted_seq_map.items(), desc='dump user seqs'):
            writer.write(' '.join([str(x) for x in [user_id] + seq]) + '\n')
            seq_len_list.append(len(seq))
            item_counter.update(seq)
    item_counter_values = list(item_counter.values())
            
    print("#bikes: {}, #stations: {}".format(len(sorted_seq_map), len(station_id_map)))
    print("seq len: avg {}, p50 {}, p99 {}, min {}, max {}".format(
        np.mean(seq_len_list), np.quantile(seq_len_list, 0.50), np.quantile(seq_len_list, 0.99), 
        np.min(seq_len_list), np.max(seq_len_list),
    ))
    print("#visits: avg {}, p50 {}, p99 {}, min {}, max {}".format(
        np.mean(item_counter_values), np.quantile(item_counter_values, 0.50), 
        np.quantile(item_counter_values, 0.99), 
        np.min(item_counter_values), np.max(item_counter_values),
    ))
    