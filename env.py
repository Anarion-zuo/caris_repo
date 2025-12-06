from typing import List, Dict, Any, Callable, Set
from collections import namedtuple
import numpy as np
import torch, random, copy
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class CacheSimulator:
    class Entry:
        def __init__(self, item_id: int, loaded_time: int, last_used_time: int, lfuda_age: int):
            self.item_id = item_id
            self.loaded_time = loaded_time
            self.last_used_time = last_used_time
            self.freq = 0
            self.lfuda_prio = lfuda_age
    
    def __init__(self, cache_size: int, held_items: Set = None) -> None:
        # print("CacheSimulator constructor")
        self.cache_size = cache_size
        if held_items is None:
            self.cache_table: Dict[int, CacheSimulator.Entry] = dict()
        else:
            self.cache_table = {
                item_id: CacheSimulator.Entry(item_id=item_id, loaded_time=0, last_used_time=0)
                for item_id in held_items
            }

        # last is newest
        self.item_recency_order = list()
        
        # LFU-DA
        self.lfuda_age = 0
        
        self.item_freq_map = dict()
        self.load_order = list()  # for FILO

    def size(self) -> int:
        return len(self.cache_table)

    def is_full(self) -> bool:
        return len(self.cache_table) >= self.cache_size
    
    def add_item_when_not_full(self, item_id: int, time: int) -> None:
        assert not self.is_full()
        assert item_id not in self.cache_table
        # print("add item to cache", item_id)
        self.load_order.append(item_id)
        self.cache_table[item_id] = CacheSimulator.Entry(
            item_id=item_id, loaded_time=time, last_used_time=time, lfuda_age=self.lfuda_age,
        )
    
    def requested_item_mark(self, item_id: int, time: int) -> None:
        try:
            self.item_recency_order.remove(item_id)
        except ValueError:
            pass
        self.item_recency_order.append(item_id)
        if item_id in self.item_freq_map:
            self.item_freq_map[item_id] += 1
        else:
            self.item_freq_map[item_id] = 1
        # if item_id in self.cache_table:
        entry = self.cache_table[item_id]
        entry.last_used_time = time
        entry.freq += 1
        entry.lfuda_prio = entry.freq + self.lfuda_age

    def has_item(self, item_id: int) -> bool:
        return item_id in self.cache_table
    
    def remove_item(self, item_id: int):
        assert item_id in self.cache_table
        # print("remove item", item_id)
        assert item_id in self.cache_table
        # self.item_recency_order = list(filter(lambda x: x != item_id, self.item_recency_order))
        if item_id in self.item_recency_order:
            self.item_recency_order.remove(item_id)
        if item_id in self.load_order:
            self.load_order.remove(item_id)
        # item_id show up only once, so no need for a full scan

        self.item_freq_map[item_id] = 0
        self.item_freq_map.pop(item_id)
        
        entry = self.cache_table.pop(item_id)
        self.lfuda_age = entry.lfuda_prio
        return entry

    def bitmap(self, max_item_id: int) -> torch.Tensor:
        bm = torch.zeros(size=(max_item_id,), dtype=torch.long)
        for item_id, _ in self.cache_table.items():
            bm[item_id] = 1
        return bm
    
    def get_least_recent(self) -> int:
        if len(self.item_recency_order) == 0:
            return random.sample(self.cache_table.keys(), k=1)[0]
        return self.item_recency_order[0]
    
    def get_lru_k_evict(self, k: int) -> int:
        # find first accessed in correct time
        for item_id in self.item_recency_order:
            if self.cache_table[item_id].freq < k:
                return item_id
        return self.get_least_recent()
    
    def get_least_frequent(self) -> int:
        if len(self.item_freq_map) == 0:
            return random.sample(self.cache_table.keys(), k=1)[0]
        return min(self.item_freq_map.items(), key=lambda x: x[1])[0]
    
    def get_lfuda_evict(self) -> int:
        result_id, min_pri = 0, np.inf
        for item_id, entry in self.cache_table.items():
            if min_pri > entry.freq + self.lfuda_age:
                result_id = item_id
                min_pri = entry.freq + self.lfuda_age
        if result_id == 0:
            return self.get_least_frequent()
        return result_id
    
    def get_random_held(self) -> int:
        # return np.random.choice(self.cache_table.keys())
        # return random.choice(self.cache_table.keys())
        return random.sample(self.cache_table.keys(), k=1)[0]
    
    def get_first_loaded(self) -> int:
        return self.load_order[0]
    
    def get_last_used_time_reward_penalty(self, time: int, window_weight: float) -> float:
        per_item = [time - entry.last_used_time for _, entry in self.cache_table.items()]
        return sum(per_item) / self.cache_size / window_weight, per_item
    
    def get_replace_last_used_time_penalty(self, time: int, entry, window_weight: float) -> float:
        # print("get_replace_last_used_time_penalty", time, entry.last_used_time)
        used_time_diff = time - entry.last_used_time
        return used_time_diff / window_weight, used_time_diff


def compute_reuse_distances(seq: np.array):
    use_indices = dict()
    for i, x in enumerate(seq):
        if x in use_indices:
            use_indices[x].append(i)
        else:
            use_indices[x] = [i]
    all_right_distances = [0] * len(seq)
    num_once = 0
    for x, l in use_indices.items():
        sorted_l = np.array(sorted(l))
        if len(sorted_l) == 1:
            num_once += 1
        else:
            diff_list = np.diff(sorted_l)
            for right_distance, cur_index in zip(diff_list, l[:-1]):
                all_right_distances[cur_index] = right_distance
    return all_right_distances, num_once


class CacheEnv:
    def __init__(self, args, input_ids: np.array, preload_held_items: Set = None) -> None:
        self.max_item_id = args.item_size+1
        self.cache_size = args.sim_cache_size
        self.input_ids = input_ids
        self.first_not_zero_index = np.equal(input_ids, 0).astype(int).sum()
        self.belady_results = None
        reuse_distances, num_used_once = compute_reuse_distances(input_ids[self.first_not_zero_index :])
        self.reuse_distances, self.num_used_once = np.concatenate([np.zeros(self.first_not_zero_index), reuse_distances]), num_used_once
        
        # self.belady_labels = self.actions_to_held_ids_index(self.belady_results, self.belady_cache_held_tensor_trace)
        # self.use_distances = torch.as_tensor(use_distances)
        self.preload_held_items = preload_held_items
        # running state
        self.cur_pos = 0
        self.args = args
        self.reset()
        
    def lazy_compute_belady(self) -> None:
        if self.belady_results is None:
            belady_results, use_distances, belady_cache_held_trace = compute_best_replacement(input_ids=self.input_ids, cache_size=self.cache_size)
            self.belady_results = belady_results
            # self.belady_cache_held_trace = belady_cache_held_trace
            # self.belady_cache_held_tensor_trace = torch.stack([
            #     CacheEnv.cache_held_ids_set_to_tensor(held_set, self.cache_size)
            #     for held_set in belady_cache_held_trace
            # ], dim=0)

    def set_explore_model(self, explore_model) -> None:
        self.explore_model = explore_model

    def observe(self) -> int:
        obs = self.input_ids[self.cur_pos]
        # print("pos", self.cur_pos, "obs", obs)
        return obs
    
    def observe_history(self) -> torch.Tensor:
        return self.input_ids[:self.cur_pos]
    
    @staticmethod
    def cache_held_ids_set_to_tensor(cache_held_set: List[int], cache_size: int) -> List[int]:
        held_id_list = cache_held_set
        pad = [0] * (cache_size - len(held_id_list))
        # prepend 0 as placeholder
        return held_id_list + pad
    
    @staticmethod
    def cache_held_ids_tensor_to_bitmap(held_ids: torch.Tensor, max_item_id: int) -> torch.Tensor:
        # held_ids [batch, time, cache_size]
        # bitmap [batch, time, #items]
        bitmap = torch.zeros(size=(held_ids.shape[0], held_ids.shape[1], max_item_id,), dtype=held_ids.dtype, device=held_ids.device)
        return bitmap.scatter_(dim=2, index=held_ids, value=1)
    
    def cache_held_ids(self) -> np.array:
        return CacheEnv.cache_held_ids_set_to_tensor(list(self.cache_simulator.cache_table.keys()), self.cache_size)
    
    def reset(self) -> None:
        self.cache_simulator = CacheSimulator(self.cache_size, held_items=self.preload_held_items)
        self.bitmap = np.zeros((self.max_item_id,), dtype=int)
        if self.preload_held_items is not None:
            for x in self.preload_held_items:
                self.bitmap[x] = 1
        self.cur_pos = 0

    def fifo_actor(self, cur_pos, obs) -> int:
        return self.cache_simulator.get_first_loaded()

    def lru_actor(self, cur_pos, obs) -> int:
        return self.cache_simulator.get_least_recent()
    
    def lfu_actor(self, cur_pos, obs) -> int:
        return self.cache_simulator.get_least_frequent()

    def random_actor(self, cur_pos, obs) -> int:
        return self.cache_simulator.get_random_held()
        # return random.sample(bitmap.argwhere().squeeze(1).tolist(), k=1)[0]
        
    def lfuda_actor(self, cur_pos, obs) -> int:
        return self.cache_simulator.get_lfuda_evict()
    
    def lru_k_actor(self, cur_pos, obs) -> int:
        return self.cache_simulator.get_lru_k_evict(k=self.args.lru_k)
        
    def belady_actor(self, cur_pos, obs) -> int:
        self.lazy_compute_belady()
        if cur_pos == self.input_ids.shape[0] - 1:
            return 0
        return self.belady_results[cur_pos]
    
    @staticmethod
    def always_admit_actor(cur_pos, obs) -> int:
        return 1
    
    @staticmethod
    def no_prefetch_actor(cur_pos, obs) -> int:
        return 0
    
    def random_prefetch_actor(self, cur_pos, obs) -> int:
        while True:
            res = np.random.randint(0, self.max_item_id)
            if res not in self.cache_simulator.cache_table:
                break
    
    class DeterministictActor:
        def __init__(self, seq: np.array):
            self.action_seq = seq
            
        def __call__(self, cur_pos, obs) -> int:
            return self.action_seq[cur_pos]
        
    def actions_to_held_ids_index(self, actions: torch.Tensor, held_ids: torch.Tensor):
        # actions [time]
        # held_ids [time, cache_size]
        action_indices = list()
        for action, cur_held_ids in zip(actions, held_ids):
            if action == 0:
                action_indices.append(0)
            else:
                try:
                    action_index = cur_held_ids.tolist().index(action)
                except ValueError:
                    action_index = 0
                action_indices.append(action_index)
        return torch.as_tensor(action_indices)

    def step(self, admit_actor: Callable[[int, int], int], prefetch_actor: Callable[[int, int], int], replace_actor: Callable[[int, int], int]) -> tuple:
        # actor returns the item to be replaced
        obs = self.observe()
        if obs == 0:
            self.cur_pos += 1
            # print("cache step obs 0")
            return None
        terminated = self.cur_pos + 1 >= len(self.input_ids)

        replace_action_hit = 0
        replace_action = 0
        admit_action = 1
        is_load_action = 0
        is_full = 1 if self.cache_simulator.is_full() else 0
        is_prefetch = 0
        prefetch_action = 0
        hit = 0
        replaced_entry = None
        # held_ids = self.cache_held_ids()
        if self.cache_simulator.has_item(obs):
            # print("cache hit", obs)
            assert obs in self.cache_simulator.cache_table
            self.cache_simulator.requested_item_mark(obs, self.cur_pos)
            hit = 1
            load_item = prefetch_actor(self.cur_pos, obs)
            prefetch_action = load_item
            is_prefetch = 1 if prefetch_action != 0 else 0
        else:
            load_item = obs
        if load_item != 0:
            is_load_action = 1
            hit = 0
            admit_action = admit_actor(self.cur_pos, load_item)
            # replace logic
            if not self.cache_simulator.is_full():
                # cache not full, no need to replace
                if admit_action == 1:
                    self.cache_simulator.add_item_when_not_full(load_item, self.cur_pos)
                    self.bitmap[load_item] = 1
            else:
                if admit_action == 1:
                    replace_action = replace_actor(self.cur_pos, obs)
                    if self.cache_simulator.has_item(replace_action):
                        # cache full, and the action is in the cache, can replace according to action
                        replaced_entry = self.cache_simulator.remove_item(replace_action)
                        self.bitmap[replace_action] = 0
                        replace_action_hit = 1
                    else:
                        # cache full, and the action is not in the cache, fall back to random replacement
                        replace_action = self.cache_simulator.get_random_held()
                        replaced_entry = self.cache_simulator.remove_item(replace_action)
                        self.bitmap[replace_action] = 0
                        replace_action_hit = 0
                        # reward -= 1  # action not hit penalty
                    self.cache_simulator.add_item_when_not_full(load_item, self.cur_pos)
                    self.bitmap[load_item] = 1
                    if is_prefetch == 0:
                        self.cache_simulator.requested_item_mark(load_item, self.cur_pos)
        # try to compute bitmap less expensively
        # if last_bitmap is None:
        # bitmap = self.cache_simulator.bitmap(max_item_id=self.max_item_id)
        
        # else:
        #     bitmap = last_bitmap.clone()
        #     bitmap[obs] = 1
        #     bitmap[action] = 0
        self.cur_pos += 1
        
        # the reward rule
        # a cache hit is encouraged +1
        # a cache miss is discouraged -1
        # a prefetch is costly -0.5
        # there is no 0 reward
        reward = 2 * hit - 1 - 0.5 * is_prefetch
        # if replaced_entry is not None:
        #     occu_pen, replace_occu_time = self.cache_simulator.get_replace_last_used_time_penalty(time=self.cur_pos, entry=replaced_entry, window_weight=self.args.occu_pen_window)
        #     reward -= occu_pen
        # else:
        replace_occu_time = 0
        return obs, hit, reward, is_load_action, is_prefetch, prefetch_action, admit_action, replace_action, replace_action_hit, is_full, replace_occu_time, self.cache_held_ids(), self.cache_simulator.size(), terminated
    
    def multistep(self, admit_actor: Callable[[int, int], int], prefetch_actor: Callable[[int, int], int], replace_actor: Callable[[torch.Tensor, int, torch.Tensor], int], num_last_steps: int, num_total_multisteps: int = None) -> tuple:
        if num_total_multisteps is None:
            num_total_multisteps = len(self.input_ids) - self.cur_pos
        mask_list, obs_list, hit_list, reward_list, is_load_action_list, is_prefetch_list, prefetch_action_list, admit_action_list, replace_action_list, replace_action_hit_list, is_full_list, replace_occu_time_list, held_ids_list, cache_held_size_list =\
            [], [], [], [], [], [], [], [], [], [], [], [], [], []
        if admit_actor is None:
            admit_actor = self.always_admit_actor
        for step_i in range(num_total_multisteps):
            ret_tup = self.step(admit_actor, prefetch_actor, replace_actor)
            if ret_tup is None:
                mask, obs, hit, reward, is_load_action, is_prefetch, prefetch_action, admit_action, replace_action, replace_action_hit, is_full, replace_occu_time, cache_held_size, terminated = \
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                held_ids = np.zeros((self.cache_size,), dtype=int)
                # held_ids = torch.zeros(self.cache_size+1)
            else:
                mask = 1
                obs, hit, reward, is_load_action, is_prefetch, prefetch_action, admit_action, replace_action, replace_action_hit, is_full, replace_occu_time, held_ids, cache_held_size, terminated = ret_tup
                # print("per_item_occu_list", len(per_item_occu), self.cache_size, )
            mask_list.append(mask)
            obs_list.append(obs)
            hit_list.append(hit)
            reward_list.append(reward)
            is_load_action_list.append(is_load_action)
            # held_ids_list.append(held_ids)
            is_prefetch_list.append(is_prefetch)
            prefetch_action_list.append(prefetch_action)
            admit_action_list.append(admit_action)
            replace_action_list.append(replace_action)
            replace_action_hit_list.append(replace_action_hit)
            is_full_list.append(is_full)
            replace_occu_time_list.append(replace_occu_time)
            held_ids_list.append(held_ids)  # [time, #items]
            cache_held_size_list.append(cache_held_size)
            if terminated:
                break
        # print("multistep mask", mask_list)
        # print("multistep reward", reward_list)
        # print("multistep prefetch", prefetch_action_list)
        # print("bitmap list", bitmap_list)
        # print("per_item_occu_list", [(len(l), l.dtype, self.cache_size,) for l in per_item_occu_list])
        return torch.as_tensor(mask_list), \
            torch.as_tensor(hit_list), \
            torch.as_tensor(reward_list), \
            torch.as_tensor(is_load_action_list), \
            torch.as_tensor(is_prefetch_list), \
            torch.as_tensor(prefetch_action_list), \
            torch.as_tensor(admit_action_list, dtype=torch.long), \
            torch.as_tensor(replace_action_list), \
            torch.as_tensor(replace_action_hit_list), \
            torch.as_tensor(is_full_list), \
            torch.as_tensor(replace_occu_time_list, dtype=torch.long), \
            torch.as_tensor(np.array(held_ids_list, dtype=int), dtype=torch.long), \
            torch.as_tensor(cache_held_size_list), \
            torch.as_tensor(self.reuse_distances), \
            self.num_used_once
            # self.belady_results, \
            # self.use_distances, \
            # self.belady_labels, \
            # torch.stack(held_ids_list, dim=0).long(),

    def multistep_action_logits(
            self, hat_e_t: torch.Tensor, 
            num_last_steps: int, 
            num_total_multisteps: int = None,
            greedy: bool = False) -> tuple:
        # action_logits [time, #items/sampling k, #items m]
        def sampling_actor(obs_history, obs, bitmap):
            time_step = len(obs_history)
            cur_hat_e_t = hat_e_t[time_step].unsqueeze(0).unsqueeze(1)
            bitmap = bitmap.to(self.explore_model.device)
            held_items = bitmap.argwhere().squeeze(1).unsqueeze(0).unsqueeze(1)
            requested_items = torch.as_tensor([obs], device=self.explore_model.device).reshape(1, 1, 1)
            action_logits = self.explore_model.calculate_policy_use(
                hat_e_t=cur_hat_e_t, held_items=held_items,
                requested_items=requested_items,
            ).squeeze(0).squeeze(0)  # [#held]
            if not greedy:
                action_sampled_index = F.softmax(action_logits, dim=-1).multinomial(num_samples=1, replacement=False)[0]
                # print("action_sampled_index", action_sampled_index, held_items)
                return held_items.squeeze(0).squeeze(0)[action_sampled_index].item()
            else:
                return held_items.squeeze(0).squeeze(0)[action_logits.argmax()].item()
        return self.multistep(
            replace_actor=sampling_actor, num_last_steps=num_last_steps, 
            num_total_multisteps=num_total_multisteps,
        )


MAX_REUSE_DISTANCE = 100000000


def compute_best_replacement(input_ids: np.array, cache_size: int):
    # Belady's policy
    
    # compute use distance at each time step backwardly
    use_distance_list = [MAX_REUSE_DISTANCE] * len(input_ids)  # for the current item
    use_distance_map_list = [dict()] * len(input_ids)
    for i in range(len(input_ids) - 1):
        cur_pos = len(input_ids) - i - 2  # L - 2 ~ 0
        cur_item = input_ids[cur_pos]
        if cur_item == 0:
            break
        next_item = input_ids[cur_pos + 1]
        use_distance_map_list[cur_pos] = {
            k: v + 1
            for k, v in use_distance_map_list[cur_pos + 1].items()
        }
        use_distance_map_list[cur_pos][next_item] = 1
        
        if cur_item in use_distance_map_list[cur_pos]:
            use_distance_list[cur_pos] = use_distance_map_list[cur_pos][cur_item]
        
    held_set = set()
    replace_actions = list()
    # held_set_list = list()
    for cur_pos, item in enumerate(input_ids[:-1]):
        item = item.item()
        # held_set_list.append(copy.deepcopy(held_set))
        if item == 0:
            replace_actions.append(0)
            continue
        if len(held_set) < cache_size:
            held_set.add(item)
            replace_actions.append(0)
        else:
            if item in held_set:
                replace_action = 0
            else:
                never_requested = held_set.difference(use_distance_map_list[cur_pos].keys())
                if len(never_requested) > 0:
                    replace_action = never_requested.pop()
                else:
                    intersected = list(held_set.intersection(use_distance_map_list[cur_pos].keys()))
                    if len(intersected) == 0:
                        replace_action = held_set[0]
                    else:
                        distances = [use_distance_map_list[cur_pos][item_id] for item_id in intersected]
                        replace_action = intersected[np.argmax(distances)]
                held_set.remove(replace_action)
            replace_actions.append(replace_action)
            held_set.add(item)
    # held_set_list.append(copy.deepcopy(held_set))
    return replace_actions, use_distance_list, []  # [time-1]


def compute_best_admit(input_ids: np.array, cur_pos: int, cache_size: int):
    class Info:
        def __init__(self):
            self.admit = 0
    info_list = [Info() for _ in range(input_ids.shape[0])]
    for i in range(len(input_ids) - 1):
        cur_pos = len(input_ids) - i - 1  # L - 1 ~ 0
        item_id = input_ids[cur_pos]
        

def evaluation_to_tb(tb_writer: SummaryWriter, run_step: int, name: str, cache_size: int, batch_tup, num_cache_steps: int, tag_prefix: str) -> None:
    def keep_eval_steps_only(t: torch.Tensor):
        if len(t.shape) < 2:
            return t
        if num_cache_steps > 0:
            return t[:, -num_cache_steps:]
        return t
    # [batch, ...]
    mask, hit, reward, is_load_action, is_prefech, prefetch_action, admit_action, replace_action, replace_action_hit, is_full, replace_occu_time, bitmap, cache_held_size, reuse_distances, num_used_once =\
        (keep_eval_steps_only(tup) for tup in batch_tup)
    num_not_masked = mask.sum()
    if tb_writer is not None:
        tb_writer.add_scalars(
            main_tag="{}/{}CacheNumbers".format(tag_prefix, name),
            tag_scalar_dict={
                "#Hits": hit.sum().float() / num_not_masked,
                "#ReplaceActionHits": replace_action_hit.sum().float() / num_not_masked,
                "#CacheHoldsBegin": cache_held_size[:, 0].float().mean(),
                "#ValidReplaceActions": (replace_action != 0).sum().float() / num_not_masked,
                "CacheHeldSize": cache_held_size[:, -1].float().mean(),
                "#Masked": (mask == 0).long().float().mean(),
                "#IsFull": is_full.sum().float() / num_not_masked,
                # "CacheTotalSize": cache_size,  
                "#Admits": (admit_action * mask * is_load_action).sum().float() / (mask * is_load_action).sum(),
                "#DidPrefetch": is_prefech.sum().float() / num_not_masked,
                "#AskedButNotPrefetching": ((hit == 0).long() * (is_prefech == 0).long()).sum().float() / num_not_masked,
                "ReplaceOccuTime": replace_occu_time.sum().float() / num_not_masked,
                "#LoadActions": is_load_action.sum().float() / num_not_masked,
                "AdmitUsedDistance":
                    (mask * is_load_action * admit_action * reuse_distances).sum().float() 
                    / (mask * is_load_action * admit_action * (reuse_distances != 0).long()).sum(),
                "NotAdmitUsedDistance":
                    (mask * is_load_action * (1 - admit_action) * reuse_distances).sum().float() 
                    / (mask * is_load_action * (1 - admit_action) * (reuse_distances != 0).long()).sum(),
            },
            global_step=run_step,
        )
    num_hits = hit.sum(dim=1)
    num_items = mask.sum(dim=1)
    hr = ((num_hits+1).float() / (num_items+1).float()).mean()
    if tb_writer is not None:
        tb_writer.add_scalars(
            main_tag="{}/{}CacheRates".format(tag_prefix, name),
            tag_scalar_dict={
                "HR": hr,
                "#AdmitsRate": (
                    (
                        admit_action.sum(dim=1).float()+1
                    ) / (
                        mask.sum(dim=1).float()+1
                    )
                ).mean(),
                "AskedToPrefetchRate": (
                    (
                        is_prefech.sum(dim=1)+1
                    ).float() / (
                        num_items.float()+1
                    )
                ).mean(),
                "PrefetchGiveupRate": (
                    (
                        (is_prefech * (prefetch_action == 0).long()).sum(dim=1)+1
                    ).float() / (
                        is_prefech.sum(dim=1)+1
                    ).float()
                ).mean(),
                "NotUsedRate": ((reuse_distances == 0).long() * mask * is_load_action).sum().float() 
                                / (mask * is_load_action).sum(),
                "AdmitNotUsedRate":
                    (mask * is_load_action * admit_action * (reuse_distances == 0).long()).sum().float() 
                    / (mask * is_load_action * admit_action).sum(),
                "AdmitUsedRate":
                    (mask * is_load_action * admit_action * (reuse_distances != 0).long()).sum().float() 
                    / (mask * is_load_action * admit_action).sum(),
                "NotAdmitNotUsedRate":
                    (mask * is_load_action * (1 - admit_action) * (reuse_distances == 0).long()).sum().float() 
                    / (mask * is_load_action * (1 - admit_action)).sum(),
                "NotAdmitUsedRate":
                    (mask * is_load_action * (1 - admit_action) * (reuse_distances != 0).long()).sum().float() 
                    / (mask * is_load_action * (1 - admit_action)).sum(),
                
            },
            global_step=run_step,
        )
    return hr, num_hits, num_items


def evaluate_steps_with_actor(tb_writer: SummaryWriter, run_step: int,
                            args, name: str, admit_actor: str, replace_actor_name: str, future_obs: torch.Tensor, 
                            bitmap: torch.Tensor, num_cache_steps: int, tag_prefix: str):
    # hat_e_t: [batch, time, dim]
    # future_obs: [batch, time]
    # bitmap: [batch, #items]
    assert len(future_obs.shape) == 2
    if bitmap is not None:
        assert len(bitmap.shape) == 2
        assert future_obs.shape[0] == bitmap.shape[0]

    tup_list = list()
    for b in range(future_obs.shape[0]):
        if bitmap is None:
            item_set = set()
        else:
            item_set = bitmap[b, :].argwhere().squeeze(1)
            item_set = set(item_set.cpu().tolist())
        env = CacheEnv(args=args, input_ids=future_obs[b].cpu().numpy(), preload_held_items=item_set)
        tup = env.multistep(
            admit_actor=admit_actor,
            prefetch_actor=CacheEnv.no_prefetch_actor,
            replace_actor=getattr(env, replace_actor_name), num_last_steps=0,
            num_total_multisteps=future_obs.shape[1],
        )
        tup_list.append(tup)
    return evaluation_to_tb(tb_writer, run_step, name, env.cache_size, torch.utils.data.default_collate(tup_list), num_cache_steps, tag_prefix)


def evaluate_steps_with_replace_actor(tb_writer: SummaryWriter, run_step: int,
                            args, name: str, replace_actions: torch.Tensor, future_obs: torch.Tensor, 
                            bitmap: torch.Tensor, num_cache_steps: int, tag_prefix: str):
    # hat_e_t: [batch, time, dim]
    # future_obs: [batch, time]
    # bitmap: [batch, #items]
    assert len(future_obs.shape) == 2
    if bitmap is not None:
        assert len(bitmap.shape) == 2
        assert future_obs.shape[0] == bitmap.shape[0]

    tup_list = list()
    for b in range(future_obs.shape[0]):
        if bitmap is None:
            item_set = set()
        else:
            item_set = bitmap[b, :].argwhere().squeeze(1)
            item_set = set(item_set.cpu().tolist())
        env = CacheEnv(args=args, input_ids=future_obs[b].cpu().numpy(), preload_held_items=item_set)
        tup = env.multistep(
            admit_actor=CacheEnv.always_admit_actor,
            prefetch_actor=CacheEnv.no_prefetch_actor,
            replace_actor=CacheEnv.DeterministictActor(replace_actions[b].cpu().numpy()), num_last_steps=0,
            num_total_multisteps=future_obs.shape[1],
        )
        tup_list.append(tup)
    return evaluation_to_tb(tb_writer, run_step, name, env.cache_size, torch.utils.data.default_collate(tup_list), num_cache_steps, tag_prefix)


def evaluate_steps_with_admit_actor(tb_writer: SummaryWriter, run_step: int,
                            args, name: str, admit_actions: torch.Tensor, replace_actor_name: str, future_obs: torch.Tensor, 
                            bitmap: torch.Tensor, num_cache_steps: int, tag_prefix: str):
    # hat_e_t: [batch, time, dim]
    # future_obs: [batch, time]
    # bitmap: [batch, #items]
    assert len(future_obs.shape) == 2
    if bitmap is not None:
        assert len(bitmap.shape) == 2
        assert future_obs.shape[0] == bitmap.shape[0]

    tup_list = list()
    for b in range(future_obs.shape[0]):
        if bitmap is None:
            item_set = set()
        else:
            item_set = bitmap[b, :].argwhere().squeeze(1)
            item_set = set(item_set.cpu().tolist())
        env = CacheEnv(args=args, input_ids=future_obs[b].cpu().numpy(), preload_held_items=item_set)
        tup = env.multistep(
            admit_actor=CacheEnv.DeterministictActor(admit_actions[b].cpu().numpy()),
            prefetch_actor=CacheEnv.no_prefetch_actor,
            replace_actor=getattr(env, replace_actor_name), num_last_steps=0,
            num_total_multisteps=future_obs.shape[1],
        )
        tup_list.append(tup)
    return evaluation_to_tb(tb_writer, run_step, name, env.cache_size, torch.utils.data.default_collate(tup_list), num_cache_steps, tag_prefix)


def evaluate_steps_with_prefetch_actor(tb_writer: SummaryWriter, run_step: int,
                            args, name: str, prefetch_actions: torch.Tensor, replace_actor_name: str, future_obs: torch.Tensor, 
                            bitmap: torch.Tensor, num_cache_steps: int, tag_prefix: str):
    # hat_e_t: [batch, time, dim]
    # future_obs: [batch, time]
    # bitmap: [batch, #items]
    assert len(future_obs.shape) == 2
    if bitmap is not None:
        assert len(bitmap.shape) == 2
        assert future_obs.shape[0] == bitmap.shape[0]

    tup_list = list()
    for b in range(future_obs.shape[0]):
        if bitmap is None:
            item_set = set()
        else:
            item_set = bitmap[b, :].argwhere().squeeze(1)
            item_set = set(item_set.cpu().tolist())
        env = CacheEnv(args=args, input_ids=future_obs[b].cpu().numpy(), preload_held_items=item_set)
        tup = env.multistep(
            admit_actor=CacheEnv.always_admit_actor,
            prefetch_actor=CacheEnv.DeterministictActor(prefetch_actions[b].cpu().numpy()),
            replace_actor=getattr(env, replace_actor_name), num_last_steps=0,
            num_total_multisteps=future_obs.shape[1],
        )
        tup_list.append(tup)
    return evaluation_to_tb(tb_writer, run_step, name, env.cache_size, torch.utils.data.default_collate(tup_list), num_cache_steps, tag_prefix)


def evaluate_nn_steps(tb_writer: SummaryWriter, run_step: int,
                    args, future_obs: torch.Tensor, 
                    hat_e_t: torch.Tensor, 
                    bitmap: torch.Tensor, model: nn.Module, greedy: bool, tag_prefix: str):
    # hat_e_t: [batch, time, dim]
    # future_obs: [batch, time]
    # bitmap: [batch, #items]
    assert len(future_obs.shape) == 2
    assert len(hat_e_t.shape) == 3
    assert len(bitmap.shape) == 2
    assert future_obs.shape[0] == hat_e_t.shape[0]
    assert future_obs.shape[0] == bitmap.shape[0]
    assert future_obs.shape[1] == hat_e_t.shape[1]

    tup_list = list()
    for b in range(future_obs.shape[0]):
        item_set = bitmap[b, :].argwhere().squeeze(1)
        if item_set.shape[0] == 0:
            item_set = set()
        else:
            item_set = set(item_set.cpu().tolist())
        env = CacheEnv(args=args, input_ids=future_obs[b].cpu().numpy(), preload_held_items=item_set)
        env.set_explore_model(model)
        tup = env.multistep_action_logits(
            hat_e_t=hat_e_t[b], num_last_steps=0, 
            num_total_multisteps=future_obs.shape[1]-1,
            greedy=greedy,
        )
        tup_list.append(tup)
    return evaluation_to_tb(tb_writer, run_step, "NN", env.cache_size, torch.utils.data.default_collate(tup_list), tag_prefix)


if __name__ == "__main__":
    # Args = namedtuple("Args", "item_size sim_cache_size")
    # args = Args(item_size=1000, sim_cache_size=32)
    # input_ids = torch.tensor([int(s) for s in "1 2 3 4 5 6 7 8 9 10 4 11 4 12 13 14 15 16 17 18 19 20 21 22 23 4 24 4 25 26 27 28 29 30 31 32 33 34 35 4 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 4 56 57 58 4 59 60 61 62 22 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 83 87 59 88 89 90 91 92 93 94 83 95 96 97 15 98 99 100 101 83 102 103 28 104 105 106 107 83 108 83 109 110 111 112 113 114 115 116 117 118 119 83 120 121 122 123 124 83 125 126 127 128 83 129 130 131 83 132 133 134 135 136 137 83 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 33 169 170 171 172 173 174 175 176 177 178 179".split(' ')])
    # env = CacheEnv(
    #     args=args,
    #     input_ids=input_ids,
    #     # preload_held_items={6, 7}
    # )
    # mask, reward, action, action_hit, is_full, bitmap, cache_held_size = env.multistep(
    #     replace_actor=env.lru_actor, num_last_steps=0, num_total_multisteps=len(input_ids),
    # )
    # print(action_hit.sum())
    
    print(compute_best_replacement(
        input_ids=torch.as_tensor([
            1, 2, # cache full
            2, # cache hit
            1, # cache hit
            3, # cache miss, should replace 1
            2, # cache hit
            4, # cache miss, should replace 2
            1, # do nothing
        ]),
        cache_size=2,
    ))
