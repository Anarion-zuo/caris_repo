from typing import List
import argparse, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from model.rl import CacheWorld
from env import CacheEnv
from utils import parse_args


def checkout_this_seq(model: CacheWorld, args, seq: List[int], device) -> bool:
    input_ids = np.array(seq)
    
    # model
    env = CacheEnv(args=args, input_ids=input_ids)
    sampled_actions = model.sample_admit_actions(input_ids=torch.tensor(input_ids, device=device))
    ret_tup = env.multistep(admit_actor=CacheEnv.DeterministictActor(sampled_actions.cpu().numpy()), prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=getattr(env, args.replace_policy_name), num_last_steps=0, num_total_multisteps=len(seq))
    mask, hit, reward, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, held_ids, cache_held_size = ret_tup
    model_num_hits = (hit * mask).sum().item()
    
    # no model
    env = CacheEnv(args=args, input_ids=input_ids)
    ret_tup = env.multistep(admit_actor=CacheEnv.always_admit_actor, prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=getattr(env, args.replace_policy_name), num_last_steps=0, num_total_multisteps=len(seq))
    mask, hit, reward, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, held_ids, cache_held_size = ret_tup
    no_model_num_hits = (hit * mask).sum().item()
    
    # lru
    # env = CacheEnv(args=args, input_ids=input_ids)
    # ret_tup = env.multistep(admit_actor=CacheEnv.always_admit_actor, prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=env.lru_actor, num_last_steps=0, num_total_multisteps=len(seq))
    # mask, hit, reward, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, held_ids, cache_held_size = ret_tup
    # lru_num_hits = (hit * mask).sum().item()
    
    # lfu
    # env = CacheEnv(args=args, input_ids=input_ids)
    # ret_tup = env.multistep(admit_actor=CacheEnv.always_admit_actor, prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=env.lfu_actor, num_last_steps=0, num_total_multisteps=len(seq))
    # mask, hit, reward, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, held_ids, cache_held_size = ret_tup
    # lfu_num_hits = (hit * mask).sum().item()
    
    # random
    # env = CacheEnv(args=args, input_ids=input_ids)
    # ret_tup = env.multistep(admit_actor=CacheEnv.always_admit_actor, prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=env.random_actor, num_last_steps=0, num_total_multisteps=len(seq))
    # mask, hit, reward, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, held_ids, cache_held_size = ret_tup
    # random_num_hits = (hit * mask).sum().item()
    
    # decide
    num_not_masked = (1 - mask).sum().item()
    is_better = model_num_hits > no_model_num_hits
    if np.random.rand() <= args.use_better_prob:
        use_for_test = True
    else:
        use_for_test = False
    return use_for_test, is_better, model_num_hits, no_model_num_hits, num_not_masked


if __name__ == "__main__":
    args = parse_args()
    
    test_writer = open(os.path.join("./data_selected/", args.test_output + '.txt'), 'w')
    train_writer = open(os.path.join("./data_selected/", args.train_output + '.txt'), 'w')
    
    train_num_not_masked, test_num_not_masked = 0, 0
    train_model_num_hits, test_model_num_hits = 0, 0
    train_no_model_num_hits, test_no_model_num_hits = 0, 0
    num_is_better = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CacheWorld(args)
    model.load_state_dict(torch.load(args.preload_rl_pt, map_location=device))
    
    data_path = os.path.join("./data/", args.data_name + '.txt')
    with open(data_path, "rb") as f:
        f.seek(0)
        data_file_num_lines = sum(1 for _ in f)
    with open(data_path, 'r') as reader:
        reader.seek(0)
        for line in tqdm(reader, total=data_file_num_lines):
            line = line.strip(' \r\n')
            splitted = line.split(' ')
            user_id = splitted[0]
            seq = splitted[1:]
            seq = seq[-args.max_seq_length :]
            seq = [int(x) for x in seq]
            if len(seq) < args.max_seq_length:
                seq = [0] * (args.max_seq_length - len(seq)) + seq
            
            should_use, is_better, model_num_hits, no_model_num_hits, num_not_masked = checkout_this_seq(model=model, args=args, seq=seq, device=device)
            if should_use:
                test_writer.write(line + '\n')
                test_model_num_hits += model_num_hits
                test_no_model_num_hits += no_model_num_hits
                test_num_not_masked += num_not_masked
            else:
                train_writer.write(line + '\n')
                train_model_num_hits += model_num_hits
                train_no_model_num_hits += no_model_num_hits
                train_num_not_masked += num_not_masked
            if is_better:
                num_is_better += 1
    
    test_writer.close()
    train_writer.close()
    
    print("test: model HR {}, no model HR {}, model #hits {}, no model #hits {}, not masked num {}".format(
        test_model_num_hits / test_num_not_masked, 
        test_no_model_num_hits / test_num_not_masked,
        test_model_num_hits, test_no_model_num_hits,
        num_not_masked,
    ))
    print("train: model HR {}, no model HR {}, model #hits {}, no model #hits {}, not masked num {}".format(
        train_model_num_hits / train_num_not_masked, 
        train_no_model_num_hits / train_num_not_masked,
        train_model_num_hits, train_no_model_num_hits,
        train_num_not_masked,
    ))
    print("#is_better", is_better)
    