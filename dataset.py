import tqdm, copy
import numpy as np
import torch
import torch.nn as nn
import os
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset
import random
from env import CacheEnv
# from memory_profiler import profile


class RecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.contrastive_learning = args.model_type.lower() in ['fearec', 'duorec']
        self.data_type = data_type

        if self.data_type=='train':
            for user, seq in enumerate(user_seq):
                input_ids = seq[-(self.max_len + 2):-2]
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
                    self.user_ids.append(user)
        elif self.data_type=='valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            # test
            self.user_seq = user_seq
        
        print("RecDataset data_type", data_type, "#seqs", len(self.user_seq))

        self.test_neg_items = test_neg_items

        if self.contrastive_learning and self.data_type=='train':
            if os.path.exists(args.same_target_path):
                self.same_target_index = np.load(args.same_target_path, allow_pickle=True)
            else:
                print("Start making same_target_index for contrastive learning")
                self.same_target_index = self.get_same_target_index()
                self.same_target_index = np.array(self.same_target_index)
                np.save(args.same_target_path, self.same_target_index)

    def get_same_target_index(self):
        num_items = max([max(v) for v in self.user_seq]) + 2
        same_target_index = [[] for _ in range(num_items)]
        
        user_seq = self.user_seq[:]
        tmp_user_seq = []
        for i in tqdm.tqdm(range(1, num_items)):
            for j in range(len(user_seq)):
                if user_seq[j][-1] == i:
                    same_target_index[i].append(user_seq[j])
                else:
                    tmp_user_seq.append(user_seq[j])
            user_seq = tmp_user_seq
            tmp_user_seq = []

        return same_target_index

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.args.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        if self.data_type in ['valid', 'test']:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
                torch.zeros(0, dtype=torch.long), # not used
            )

        elif self.contrastive_learning:
            sem_augs = self.same_target_index[answer]
            sem_aug = random.choice(sem_augs)
            keep_random = False
            for i in range(len(sem_augs)):
                if sem_augs[0] != sem_augs[i]:
                    keep_random = True

            while keep_random and sem_aug == items:
                sem_aug = random.choice(sem_augs)

            sem_aug = sem_aug[:-1]
            pad_len = self.max_len - len(sem_aug)
            sem_aug = [0] * pad_len + sem_aug
            sem_aug = sem_aug[-self.max_len:]
            assert len(sem_aug) == self.max_len

            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(sem_aug, dtype=torch.long)
            )

        else:
            cur_tensors = (
                torch.tensor(self.user_ids[index], dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.zeros(0, dtype=torch.long), # not used
            )

        return cur_tensors


class CachePredDataset(Dataset):
    def __init__(self, args, user_seq, num_future_steps, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.num_future_steps = num_future_steps
        self.max_len = args.max_seq_length
        self.user_ids = []
        self.data_type = data_type
        self.is_rl = args.is_rl

        if self.data_type=='train':
            for user, seq in enumerate(user_seq):
                input_ids = seq[-(self.max_len + 2):-2]
                for i in range(len(input_ids)):
                    cur_ids = input_ids[:i + 1]
                    if len(cur_ids) > self.num_future_steps:
                        # print("input len", len(cur_ids), self.num_future_steps, self.args.sim_cache_size)
                        if not (self.args.is_rl and 
                                self.args.sim_cache_size > len(cur_ids) - self.num_future_steps):
                            self.user_seq.append(cur_ids)
                            self.user_ids.append(user)
                    # else:
                    #     print("skipping len", len(cur_ids), "num_future_steps", self.num_future_steps)
        elif self.data_type=='valid':
            for sequence in user_seq:
                if len(sequence) > self.num_future_steps and not (
                    self.args.is_rl and 
                    self.args.sim_cache_size > len(sequence) - self.num_future_steps
                ):
                    self.user_seq.append(sequence)
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items

    def __len__(self):
        return len(self.user_seq)
    
    def set_explore_model(self, explore_model: nn.Module):
        self.explore_model = explore_model

    def __getitem__(self, index):
        items = self.user_seq[index]
        # input_ids = items[:-1]
        # answer = items[-1]
        assert len(items) > 0
        num_future_steps = self.num_future_steps
        assert 1 <= num_future_steps <= len(items), f"num_future_steps not in range, expected [{len(items)}], got [{num_future_steps}]"
        if self.data_type in ['valid', 'test']:
            user_id_t = torch.tensor(index, dtype=torch.long)
        else:
            user_id_t = torch.tensor(self.user_ids[index], dtype=torch.long)
        # if num_future_steps == len(items):
        #     num_future_steps -= 1
        # elif num_future_steps == 0:
        #     num_future_steps = 1
        input_ids = items[:-num_future_steps]
        answers = items[-num_future_steps:]

        # seq_set = set(items)
        # neg_answer = neg_sample(seq_set, self.args.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        cur_tensors = (
            user_id_t,  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(answers, dtype=torch.long),
        )
        # print(items)
        if self.is_rl:
            input_ids = np.array(input_ids)
            # input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            env = CacheEnv(args=self.args, input_ids=input_ids)
            
            # admit actions
            """
            sampled_actions = self.explore_model.sample_admit_actions(input_ids).detach().clone()
            random_explore_mask = torch.multinomial(
                torch.as_tensor([[
                    1 - self.args.explore_epsilon,
                    self.args.explore_epsilon,
                ]]*sampled_actions.shape[0]), 
                num_samples=1,
            ).squeeze(1).to(self.explore_model.device)
            random_explore_actions = torch.multinomial(
                torch.as_tensor([[0.5, 0.5]] * sampled_actions.shape[0]),
                num_samples=1,
            ).squeeze(1).to(self.explore_model.device)
            real_actions = random_explore_mask * random_explore_actions + (1 - random_explore_mask) * sampled_actions
            """
            
            sampled_actions = self.explore_model.sample_admit_actions(input_ids=torch.as_tensor(input_ids))
            if self.args.explore_epsilon > 0:
                random_admit_actions = torch.randint_like(input=sampled_actions, low=0, high=2, device=sampled_actions.device)
                random_selector = torch.as_tensor(
                    [self.args.explore_epsilon, 1 - self.args.explore_epsilon], 
                    device=sampled_actions.device
                ).multinomial(
                    num_samples=sampled_actions.shape[0], 
                    replacement=True,
                ).to(sampled_actions.device)
                sampled_actions = sampled_actions * random_selector + random_admit_actions * (1 - random_selector)
            
            # sampled_replace_actions = self.explore_model.sample_replace_imitator(input_ids=torch.as_tensor(input_ids), held_ids=None)
            # cpu_explore_model = copy.deepcopy(self.explore_model.state_dict()).cpu()
            with torch.no_grad():
                # ret_tup = env.multistep(admit_actor=CacheEnv.always_admit_actor, prefetch_actor=CacheEnv.DeterminisictActor(self.explore_model.sample_prefetch_actions(input_ids)), replace_actor=getattr(env, self.args.replace_policy_name), num_last_steps=self.args.num_future_steps, num_total_multisteps=input_ids.shape[0])
                # ret_tup = env.multistep(admit_actor=CacheEnv.always_admit_actor, prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=CacheEnv.DeterminisictActor(sampled_replace_actions), num_last_steps=self.args.num_future_steps, num_total_multisteps=input_ids.shape[0])
                ret_tup = env.multistep(admit_actor=CacheEnv.DeterministictActor(sampled_actions.cpu().numpy()), prefetch_actor=CacheEnv.no_prefetch_actor, replace_actor=getattr(env, self.args.replace_policy_name), num_last_steps=self.args.num_future_steps, num_total_multisteps=input_ids.shape[0])
            
            env = CacheEnv(args=self.args, input_ids=input_ids)
            lru_tup = env.multistep(
                admit_actor=CacheEnv.always_admit_actor,
                prefetch_actor=CacheEnv.no_prefetch_actor,
                replace_actor=env.lru_actor, num_last_steps=0,
                num_total_multisteps=input_ids.shape[0],
            )
            env = CacheEnv(args=self.args, input_ids=input_ids)
            lfu_tup = env.multistep(
                admit_actor=CacheEnv.always_admit_actor,
                prefetch_actor=CacheEnv.no_prefetch_actor,
                replace_actor=env.lfu_actor, num_last_steps=0,
                num_total_multisteps=input_ids.shape[0],
            )
            env = CacheEnv(args=self.args, input_ids=input_ids)
            random_tup = env.multistep(
                admit_actor=CacheEnv.always_admit_actor,
                prefetch_actor=CacheEnv.no_prefetch_actor,
                replace_actor=env.random_actor, num_last_steps=0,
                num_total_multisteps=input_ids.shape[0],
            )
            env = CacheEnv(args=self.args, input_ids=input_ids)
            lru_k_tup = env.multistep(
                admit_actor=CacheEnv.always_admit_actor,
                prefetch_actor=CacheEnv.no_prefetch_actor,
                replace_actor=env.lru_k_actor, num_last_steps=0,
                num_total_multisteps=input_ids.shape[0],
            )
            env = CacheEnv(args=self.args, input_ids=input_ids)
            lfuda_tup = env.multistep(
                admit_actor=CacheEnv.always_admit_actor,
                prefetch_actor=CacheEnv.no_prefetch_actor,
                replace_actor=env.lfuda_actor, num_last_steps=0,
                num_total_multisteps=input_ids.shape[0],
            )
            env = CacheEnv(args=self.args, input_ids=input_ids)
            fifo_tup = env.multistep(
                admit_actor=CacheEnv.always_admit_actor,
                prefetch_actor=CacheEnv.no_prefetch_actor,
                replace_actor=env.fifo_actor, num_last_steps=0,
                num_total_multisteps=input_ids.shape[0],
            )
            cur_tensors += ret_tup + fifo_tup + lru_tup + lfu_tup + random_tup + lru_k_tup + lfuda_tup
        # print(cur_tensors)
        return cur_tensors


class ReplayBuffer:
    def __init__(self, args, dataset: Dataset):
        self.args = args
        self.pool_size = args.replay_size
        self.fifo_pool = list()
        self.dataset = dataset
        # self.inner_iter = inner_iter
        # self.ds_i = 0
        
        self.add_new_experience_count = 0
        
    def sample(self):
        return random.sample(self.fifo_pool, 1)[0]
    
    def add_new_experience(self, expr):
        self.fifo_pool.append(expr)
        if len(self.fifo_pool) > self.pool_size:
            if (self.add_new_experience_count % 30) == 0 and self.add_new_experience_count != 0:
                self.fifo_pool = [x for x in self.fifo_pool[-self.pool_size:]]
            else:
                self.fifo_pool = self.fifo_pool[-self.pool_size:]
            self.add_new_experience_count += 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        ds_i = np.random.randint(low=0, high=len(self.dataset))
        # if self.ds_i >= len(self.dataset):
        #     raise StopIteration()
        self.add_new_experience(self.dataset[ds_i])
        return self.sample()


class ReplayCacheDataset(IterableDataset):
    def __init__(self, args, dataset: CachePredDataset):
        super(ReplayCacheDataset).__init__()
        self.args = args
        
        self.dataset = dataset
        self.replay_buffer = ReplayBuffer(args=args, dataset=dataset)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.replay_buffer)
    
    def __len__(self):
        return len(self.dataset)
    
    def set_explore_model(self, explore_model):
        self.dataset.set_explore_model(explore_model)


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item

def get_user_seqs(data_file, min_len=0):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    num_users = 0
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        if min_len > 0 and len(items) < min_len:
            print(f"skipping line [{line}]")
            continue
        user_seq.append(items)
        item_set = item_set | set(items)
        num_users += 1
    max_item = max(item_set)
    # num_users = len(lines)
    print("get_user_seqs num_users", num_users, "max_item", max_item, "#seqs", len(user_seq))

    return user_seq, max_item, num_users

def get_seq_dic(args, min_len=0):

    args.data_file = args.data_dir + args.data_name + '_train.txt'
    args.test_data_file = args.data_dir + args.data_name + '_test.txt'
    user_seq, max_item, num_users = get_user_seqs(args.data_file, min_len=min_len)
    test_user_seq, test_max_item, test_num_users = get_user_seqs(args.test_data_file, min_len=min_len)
    seq_dic = {'user_seq':user_seq, 'test_user_seq': test_user_seq, 'num_users':num_users }

    return seq_dic, max(max_item, test_max_item), num_users

def dump_seqs(seqs, dump_path: str):
    with open(dump_path, 'w') as f:
        for seq_i, seq in enumerate(seqs):
            f.write(' '.join([str(seq_i)] + [str(x) for x in seq]))
            f.write('\n')

def get_dataloader(args, seq_dic, is_cache):
    # num_users = len(seq_dic)
    train_seqs = seq_dic['user_seq']
    test_seqs = seq_dic['test_user_seq']
    # np.random.shuffle(user_seq)
    # test_seqs = user_seq[: args.num_test_seqs]
    # train_seqs = user_seq[args.num_test_seqs :]
    # if len(args.train_dump_path) > 0:
    #     dump_seqs(train_seqs, args.train_dump_path)
    # if len(args.test_dump_path) > 0:
    #     dump_seqs(test_seqs, args.test_dump_path)
    if not is_cache:
        train_dataset = RecDataset(args, seq_dic['user_seq'], data_type='train')
        # eval_dataset = RecDataset(args, seq_dic['user_seq'], data_type='valid')
        # test_dataset = RecDataset(args, seq_dic['user_seq'], data_type='test')
    else:
        print('user_seq', len(seq_dic['user_seq']))
        train_dataset_raw = CachePredDataset(args, train_seqs, num_future_steps=args.num_future_steps, data_type='train')
        train_dataset = ReplayCacheDataset(args=args, dataset=train_dataset_raw)
        test_dataset = CachePredDataset(args, test_seqs, num_future_steps=args.num_future_steps, data_type='test')
        # eval_dataset = CachePredDataset(args, seq_dic['user_seq'], num_future_steps=args.num_future_steps, data_type='valid')
        # test_dataset = CachePredDataset(args, seq_dic['user_seq'], num_future_steps=args.num_future_steps, data_type='test')

    # def collate_fn(batch):
    #     return torch.utils.data.dataloader.default_collate(list(filter(lambda x: x is not None, batch)))

    train_dataloader = DataLoader(train_dataset, sampler=None, batch_size=args.batch_size, num_workers=args.num_workers)

    # eval_sampler = SequentialSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.num_workers)

    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size, num_workers=args.num_workers)

    return train_dataloader, (train_dataset, test_dataset)


if __name__ == "__main__":
    from utils import parse_args
    from model.rl import CacheWorld
    args = parse_args()
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    seq_dic, max_item, num_users = get_seq_dic(args, min_len=args.num_future_steps)
    args.item_size = max_item + 1
    
    (train_dataloader, eval_dataloader, test_dataloader), (train_dataset, eval_dataset, test_dataset) = get_dataloader(args, seq_dic, True)
    train_dataset.set_explore_model(CacheWorld(args))
    for batch in tqdm.tqdm(train_dataloader):
        pass
