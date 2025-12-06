import argparse, shutil
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    with open(args.input, 'r') as reader:
        for line_i, line in enumerate(reader):
            line = line.strip(' \r\n')
            splitted = line.split(' ')[1:]
            # time id size
            time_arr = range(len(splitted))
            size_arr = [1] * len(splitted)
            with open(output_dir.joinpath("{}.txt".format(line_i)), 'w') as writer:
                for time, id_str, size in zip(time_arr, splitted, size_arr):
                    writer.write("{} {} {}\n".format(time, id_str, size))
    