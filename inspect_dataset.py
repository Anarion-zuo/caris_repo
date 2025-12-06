import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from collections import Counter
from env import compute_reuse_distances


def report_list(name: str, data):
    print("{}: mean [{}], min [{}], max [{}], p50 [{}], p90 [{}]".format(
        name, np.mean(data), np.min(data), np.max(data),
        np.quantile(data, 0.50), np.quantile(data, 0.90)
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--reuse_hist', type=str, required=False, default='')
    args = parser.parse_args()

    seq_num_items, seq_len = [], []
    item_set = set()
    item_counter = Counter()
    reuse_distances = list()
    total_num_once = 0
    max_reuse_distances = dict()

    with open(args.input, 'r') as reader:
        for line in reader:
            line = line.strip(' \r\n')
            splitted = line.split()
            seq_len.append(len(splitted)-1)
            seq_num_items.append(len(set(splitted[1:])))
            item_set.update(splitted[1:])
            item_counter.update(splitted[1:])
            
            cur_reuse_distances, cur_num_once = compute_reuse_distances(splitted[1:])
            reuse_distances += [x for x in cur_reuse_distances if x != 0]
            total_num_once += cur_num_once

            if len(args.reuse_hist) > 0:
                reuse_counter = Counter(splitted)
                for item, num in reuse_counter.items():
                    if item not in max_reuse_distances:
                        max_reuse_distances[item] = num
                    else:
                        max_reuse_distances[item] = max(max_reuse_distances[item], num)

    item_num_requested = np.array([float(x) for x in item_counter.values()])
    
    report_list("seq_num_items", seq_num_items)
    report_list("seq_len", seq_len)
    print("#items", len(item_set))
    print("#num_once", total_num_once, total_num_once / item_num_requested.sum())
    try:
        report_list("reuse_distances", reuse_distances)
    except:
        print("reuse_distance: ", reuse_distances)
    
    print("#requested: total [{}], mean [{}], p50 [{}], p75 [{}], p90 [{}], p99 [{}], min [{}], max[{}]".format(
        item_num_requested.sum(), item_num_requested.mean(), 
        np.quantile(item_num_requested, 0.5), np.quantile(item_num_requested, 0.75),
        np.quantile(item_num_requested, 0.90), np.quantile(item_num_requested, 0.99),
        item_num_requested.min(), item_num_requested.max(),
    ))

    if len(args.reuse_hist) > 0:
        plt.clf()
        max_reuse_vals = np.array(list(max_reuse_distances.values()))
        # max_reuse_vals = max_reuse_vals / max_reuse_vals.sum()
        counts, bins = np.histogram(max_reuse_vals, bins=100)
        plt.stairs(counts, bins)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(len(item_set)))

        plt.savefig("max_reuse_distances.png")
