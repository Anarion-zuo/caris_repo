import argparse, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
from collections import Counter
from tqdm import tqdm

matplotlib.rc("font", **{"size": 16})
sns.set_theme(style="whitegrid", palette="bright", font_scale=1.4)


def load_dataset(path: str):
    max_reuse_distances = dict()
    with open(path, 'r') as reader:
        for line in reader:
            line = line.strip(' \r\n')
            splitted = line.split()
            reuse_counter = Counter(splitted)
            for item, num in reuse_counter.items():
                if item not in max_reuse_distances:
                    max_reuse_distances[item] = num
                else:
                    max_reuse_distances[item] = max(max_reuse_distances[item], num)
    return max_reuse_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--input_names', type=str, required=True)
    parser.add_argument('--output', type=str, required=False, default='')
    args = parser.parse_args()

    input_files = args.input.split(',')
    names = args.input_names.split(',')
    assert len(names) == len(input_files)
    max_reuse_distances_list = [load_dataset(file) for file in tqdm(input_files, desc='loading')]
    plt.clf()
    for name, data in zip(names, max_reuse_distances_list):
        sns_plot = sns.histplot(data={name: data}, stat='percent', binwidth=1)
        sns_plot.set_xlabel(name)
        sns_plot.get_figure().savefig(os.path.join(args.output, name + '.png'))
        plt.clf()
