import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--max_item_id', type=int, required=True)
    args = parser.parse_args()

    item_id_map = dict()
    trace_dict = dict()
    
    with open(args.input, 'rb') as reader:
        reader.seek(0)
        raw_num_lines = sum(1 for _ in reader)
        
    with open(args.input, 'r') as reader:
        reader.seek(0)
        for line in tqdm(reader, total=raw_num_lines, desc='loading'):
            line = line.strip(' \r\n')
            splitted = line.split(',')
            timestamp, item_id, client_id = splitted
            item_id_raw = (hash(item_id) % args.max_item_id)
            if item_id_raw not in item_id_map:
                item_id = len(item_id_map)
                item_id_map[item_id_raw] = item_id
            else:
                item_id = item_id_map[item_id_raw]
            
            timestamp = int(timestamp)
            client_id = int(client_id)
            if client_id not in trace_dict:
                trace_dict[client_id] = [item_id]
            else:
                trace_dict[client_id].append(item_id)
    # sorted_trace = [item_id for _, item_id in sorted(trace, key=lambda x: x[0])]
    print("#items {}, #reqs {}".format(len(item_id_map), raw_num_lines))
    with open(args.output, 'w') as writer:
        for seq_i, seq in tqdm(enumerate(trace_dict.values()), desc='writing seqs'):
            writer.write(' '.join([str(x) for x in [seq_i] + seq]) + '\n')
