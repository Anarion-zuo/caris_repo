import tqdm
import torch
import numpy as np

from torch.optim import Adam
from dataset import ReplayBuffer
from metrics import recall_at_k, ndcg_k
from env import evaluate_steps_with_actor, evaluate_nn_steps, evaluate_steps_with_admit_actor, evaluate_steps_with_prefetch_actor, evaluate_steps_with_replace_actor, evaluation_to_tb, CacheEnv, MAX_REUSE_DISTANCE
import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score, binary_recall, binary_precision, binary_auroc
from torch import autograd
autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, IterableDataset

from model.rl import scan_nan, assert_tensor_not_nan
from utils import EarlyStopping
# from memory_profiler import profile

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataset, args, logger, is_multistep):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.is_multistep = is_multistep
        self.num_cache_steps = args.num_cache_steps

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        # self.eval_dataloader = eval_dataloader
        self.test_dataset = test_dataset
        self.test_dataset.set_explore_model(model)

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")
        self.tb_writer = SummaryWriter(args.tb_logdir)
        
    def new_test_dataloader_iter(self):
        test_sampler = SequentialSampler(self.test_dataset)
        dl = DataLoader(self.test_dataset, sampler=test_sampler, batch_size=self.args.test_batch_size, num_workers=1)
        def _iter():
            for batch in dl:
                yield batch
        return _iter()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name)
        self.logger.info(new_dict.keys())
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out, max_item_id):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred[:, : max_item_id]

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]),  "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
        # return [recall[0], recall[1], recall[3],], str(post_fix)

    def iteration(self, epoch, dataloader, train=True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                if not self.is_multistep:
                    user_ids, input_ids, answers, neg_answer, same_target = batch
                    loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)
                else:
                    user_ids, input_ids, answers = batch
                    loss = self.model.calculate_multistep_loss(input_ids, answers, self.args.num_future_steps)
                    
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                if self.is_multistep:
                    user_ids, input_ids, answers = batch
                    A = answers.shape[1]
                    answers = answers.reshape(-1)
                    user_ids = user_ids.repeat(1, A).reshape(-1)
                else:
                    user_ids, input_ids, answers, _, _ = batch
                if self.is_multistep:
                    recommend_output = self.model.forward(input_ids, num_multisteps=self.args.num_future_steps)
                    E = recommend_output.shape[-1]
                    recommend_output = recommend_output[:, -A :, :].reshape(-1, E)
                    # last n outputs of each batch
                else:
                    recommend_output = self.model.predict(input_ids, user_ids)
                    recommend_output = recommend_output[:, -1, :]
                    # last output of each batch
                
                rating_pred = self.predict_full(recommend_output, self.args.item_size)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                try:
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                except: # bert4rec
                    rating_pred = rating_pred[:, :-1]
                    # print(self.args.train_matrix.shape, self.args.train_matrix[batch_user_index].shape, rating_pred.shape)
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition time complexity O(n)  argsort O(nlogn)
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                # Take the corresponding values from the corresponding dimension 
                # according to the returned subscript to get the sub-table of each row of topk
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            return self.get_full_sort_score(epoch, answer_list, pred_list)
    
    def train_rl_imitate_belady(self, epoch, run_step: int):
        replay_buffer = ReplayBuffer(self.args, self.train_dataloader)
        rl_data_iter = tqdm.tqdm(enumerate(replay_buffer),
                                  desc="Mode_train_rl:%d" % (epoch,),
                                  total=len(self.train_dataloader),
                                  bar_format="{l_bar}{r_bar}")
        model: model.rl.CacheWorld = self.model
        self.model.train()
        early_stopping = EarlyStopping(self.args.checkpoint_path, logger=self.logger, patience=self.args.patience, verbose=True)

        for i, batch in rl_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)

            # user_ids, input_ids, answers, mask, hit, reward, held_ids, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, bitmap, belady_replace_actions, belady_labels, use_distances, cache_held_size = batch
            user_ids, input_ids, answers, mask, hit, reward, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, bitmap, belady_replace_actions, use_distances, cache_held_size = batch
            if len(user_ids) == 0:
                continue
            loss, imitate_loss, reuse_loss = self.model.calculate_loss_replace_imitate_belady(
                input_ids=input_ids, held_items=None, 
                belady_actions=belady_replace_actions, 
                belady_labels=None,
                use_distances=use_distances,
            )
            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2, error_if_nonfinite=True)
            self.optim.step()
            
            assert_tensor_not_nan(loss)
            assert_tensor_not_nan(prefetch_action)
            assert_tensor_not_nan(reward)
            assert_tensor_not_nan(input_ids)
            scan_nan(self.model)
            
            lru_hr, _, _ = evaluate_steps_with_actor(
                tb_writer=self.tb_writer, run_step=run_step,
                args=self.args, name='LRUHistory', 
                admit_actor=CacheEnv.always_admit_actor,
                replace_actor_name="lru_actor",
                future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
            )
            lfu_hr, _, _ = evaluate_steps_with_actor(
                tb_writer=self.tb_writer, run_step=run_step,
                args=self.args, name='LFUHistory', 
                admit_actor=CacheEnv.always_admit_actor,
                replace_actor_name="lfu_actor",
                future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
            )
            random_hr, _, _ = evaluate_steps_with_actor(
                tb_writer=self.tb_writer, run_step=run_step,
                admit_actor=CacheEnv.always_admit_actor,
                args=self.args, name='RandomHistory', replace_actor_name="random_actor",
                future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
            )
            belady_hr, _, _ = evaluate_steps_with_actor(
                tb_writer=self.tb_writer, run_step=run_step,
                admit_actor=CacheEnv.always_admit_actor,
                args=self.args, name='BeladyHistory', replace_actor_name="belady_actor",
                future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
            )
            evaluate_steps_with_prefetch_actor(
                tb_writer=self.tb_writer, run_step=run_step,
                prefetch_actions=torch.zeros(prefetch_action.shape, dtype=torch.long, device=self.device),
                args=self.args, name='LFUMockHistory', replace_actor_name="lfu_actor",
                future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
            )
            hr_score, _, _ = evaluate_steps_with_replace_actor(
                tb_writer=self.tb_writer, run_step=run_step,
                args=self.args, name='NNReplaceImitateHistory',
                replace_actions=replace_action,
                future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
            )
            if run_step % 100 == 0:
                early_stopping([hr_score.item()], self.model, run_step)
            if run_step % 10 == 0:
                self.tb_writer.flush()
            num_not_masked = mask.sum()
            self.tb_writer.add_scalars(
                main_tag="Train/Model",
                tag_scalar_dict={
                    "loss": loss,
                    "imitate_loss": imitate_loss,
                    # "reuse_loss": reuse_loss,
                },
                global_step=run_step,
            )
            self.tb_writer.add_scalars(
                main_tag="Train/Values",
                tag_scalar_dict={
                    "is_prefetch": is_prefetch.sum().float() / num_not_masked,
                    "mask": mask.float().mean(),
                    "reward": reward.sum().float() / num_not_masked,
                },
                global_step=run_step,
            )
            self.tb_writer.add_scalars(
                main_tag="Train/LargerValues",
                tag_scalar_dict={
                    "#max_reuse_distances": (use_distances == MAX_REUSE_DISTANCE).long().sum().float() / num_not_masked,
                    "use_distances_mean_no_max": (use_distances * mask * (use_distances != MAX_REUSE_DISTANCE).long()).sum().float() / (mask * (use_distances != MAX_REUSE_DISTANCE).long()).sum().float(),
                },
                global_step=run_step,
            )
            self.tb_writer.add_scalars(
                main_tag="Train/ScoreCompare",
                tag_scalar_dict={
                    "random": random_hr / random_hr,
                    "lru": lru_hr / random_hr,
                    "lfu": lfu_hr / random_hr,
                    "belady": belady_hr / random_hr,
                    "nn_imitate_replace": hr_score / random_hr,
                },
                global_step=run_step,
            )

            run_step += 1

        post_fix = {
            "epoch": epoch,
            # "rl_loss": '{:.4f}'.format(rl_loss / len(rl_data_iter)),
        }

        if (epoch + 1) % self.args.log_freq == 0:
            self.logger.info(str(post_fix))

        return run_step
    
    # @profile
    def train_rl(self, epoch, run_step: int):
        # replay_buffer = ReplayBuffer(self.args, self.train_dataloader)
        rl_data_train_iter = tqdm.tqdm(self.train_dataloader,
                                  desc="Mode_train_rl:%d" % (epoch,),
                                  total=len(self.train_dataloader),
                                  bar_format="{l_bar}{r_bar}")
        model: model.rl.CacheWorld = self.model
        self.model.train()
        early_stopping = EarlyStopping(self.args.checkpoint_path, logger=self.logger, patience=self.args.patience, verbose=True)
        
        def unpack_batch(batch):
            user_ids, input_ids, answers = batch[:3]
            cur_i = 3
            tup_list = list()
            batch_num_ts = 15
            for _ in range(7):
                tup_list.append(batch[cur_i : cur_i + batch_num_ts])
                cur_i += batch_num_ts
            return (user_ids, input_ids, answers), tup_list

        def rl_data_iter_f():
            test_dataloader = self.new_test_dataloader_iter()
            for iter_i, train_batch in enumerate(rl_data_train_iter):
                if iter_i % self.args.test_every_n == 0:
                    test_batch = next(test_dataloader, None)
                    if test_batch is None:
                        test_dataloader = self.new_test_dataloader_iter()
                        test_batch = next(test_dataloader)
                    yield (True, test_batch,)
                yield (False, train_batch,)
        rl_data_iter = rl_data_iter_f()
        
        for is_test, batch in rl_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            tag_prefix = "Test" if is_test else "Train"

            (user_ids, input_ids, answers), batch_tup_list = unpack_batch(batch)
            mask, hit, reward, is_load_action, is_prefetch, prefetch_action, admit_action, replace_action, action_hit, is_full, replace_occu_time, held_ids, cache_held_size, reuse_distances, num_used_once = batch_tup_list[0]
            fifo_tup, lru_tup, lfu_tup, random_tup, lru_k_tup, lfuda_tup = batch_tup_list[1:]
            if len(user_ids) == 0:
                continue
            bitmap = CacheEnv.cache_held_ids_tensor_to_bitmap(
                held_ids=held_ids, max_item_id=self.args.item_size+1,
            )
            if self.args.model_policy == "admit_ppo":
                (loss, ppo_loss, q_loss, entropy, bitmap_loss, is_reused_loss, uplift_admit_loss), (action_prob, action_not_admit_prob, inaction_prob, action_likelihood, q, q_update, v, adv, bitmap_pred, is_reused_pred, is_reused_label), (admit_logits_0, admit_logits_1, admit_logits_inaction, is_reused_mask,) = self.model.calculate_loss_admit_ppo_clip(input_ids=input_ids, actions=admit_action, rewards=reward, bitmap=bitmap, mask=mask, is_load_action=is_load_action, reuse_distances=reuse_distances)
            elif self.args.model_policy == "replace_ppo":
                (loss, ppo_loss, q_loss, entropy, bitmap_loss), (action_prob, action_likelihood, q, q_update, v, adv, bitmap_pred) = self.model.calculate_loss_admit_ppo_clip(input_ids=input_ids, actions=admit_action, rewards=reward, bitmap=bitmap, mask=mask,)
            else:
                raise RuntimeError("Unknown model_policy {}".format(self.args.model_policy))
            
            if not is_test:
                self.optim.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2, error_if_nonfinite=True)
                self.optim.step()
            
            assert_tensor_not_nan(loss)
            assert_tensor_not_nan(reward)
            assert_tensor_not_nan(input_ids)
            scan_nan(self.model)
            
            num_not_masked = mask.sum()
            if self.args.model_policy == "admit_ppo":
                is_reused_pred_list, is_reused_label_list = list(), list()
                for cur_is_reused_pred, cur_is_reused_label, cur_mask in zip(is_reused_pred, is_reused_label, mask):
                    masked_len = (cur_mask == 0).long().sum()
                    not_masked_is_reused_pred = cur_is_reused_pred[masked_len:]
                    not_masked_is_reused_label = cur_is_reused_label[masked_len:]
                    is_reused_pred_list.append(not_masked_is_reused_pred)
                    is_reused_label_list.append(not_masked_is_reused_label)
                reuse_f1 = binary_f1_score(
                    input=torch.cat(is_reused_pred_list),
                    target=torch.cat(is_reused_label_list),
                )
                reuse_prec = binary_precision(
                    input=torch.cat(is_reused_pred_list),
                    target=torch.cat(is_reused_label_list),
                )
                reuse_recall = binary_recall(
                    input=torch.cat(is_reused_pred_list),
                    target=torch.cat(is_reused_label_list),
                )
                reuse_auroc = binary_auroc(
                    input=torch.cat(is_reused_pred_list),
                    target=torch.cat(is_reused_label_list),
                )
                probs_info = {
                    "mask": mask.float().mean(),
                    
                    # action prob without conditions
                    "action_0": (action_not_admit_prob * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "action_1": (action_prob * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    
                    # action prob is_load or not
                    "inaction_is_load": (inaction_prob * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "inaction_not_load": (inaction_prob * mask * (1 - is_load_action)).sum() / (mask * (1 - is_load_action)).sum(),
                    "action_0_is_load": (action_not_admit_prob * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "action_1_is_load": (action_prob * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "action_0_is_not_load": (action_not_admit_prob * mask * (1 - is_load_action)).sum() / (mask * (1 - is_load_action)).sum(),
                    "action_1_is_not_load": (action_prob * mask * (1 - is_load_action)).sum() / (mask * (1 - is_load_action)).sum(),
                    
                    # action prob is_load true and reused or not
                    "action_1_is_load_reuse": (action_prob * mask * is_load_action * is_reused_mask).sum() / (mask * is_load_action * is_reused_mask).sum(),
                    "action_1_is_load_not_reuse": (action_prob * mask * is_load_action * (1 - is_reused_mask)).sum() / (mask * is_load_action * (1 - is_reused_mask)).sum(),
                    "action_0_is_load_reuse": (action_not_admit_prob * mask * is_load_action * is_reused_mask).sum() / (mask * is_load_action * is_reused_mask).sum(),
                    "action_0_is_load_not_reuse": (action_not_admit_prob * mask * is_load_action * (1 - is_reused_mask)).sum() / (mask * is_load_action * (1 - is_reused_mask)).sum(),
                    
                    "bitmap_acc": ((bitmap_pred.round() == bitmap).long().sum(dim=2) * mask).sum().float() / (num_not_masked * (self.args.item_size+1)),
                    "reuse_acc": (((is_reused_mask > 0.5).long() == is_reused_label).long() * mask).sum().float() / num_not_masked,
                    "reuse_f1": reuse_f1,
                    "reuse_precision": reuse_prec,
                    "reuse_recall": reuse_recall,
                    "reuse_auroc": reuse_auroc,
                    "reuse_likelihood": (is_reused_mask * mask).sum().float() / num_not_masked,
                }
                values_info = {
                    "adv": adv.sum() / num_not_masked,
                    "v": v.sum() / num_not_masked,
                    "q_update": q_update.sum() / num_not_masked,
                    "q": q.sum() / num_not_masked,
                    "#admits": admit_action.sum().float() / num_not_masked,
                    "reward": reward.sum().float() / num_not_masked,
                }
                large_values_info = {
                    # action prob without conditions
                    "action_0": (admit_logits_0 * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "action_1": (admit_logits_1 * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    
                    # action prob is_load or not
                    "inaction_is_load": (admit_logits_inaction * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "inaction_not_load": (admit_logits_inaction * mask * (1 - is_load_action)).sum() / (mask * (1 - is_load_action)).sum(),
                    "action_0_is_load": (admit_logits_0 * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "action_1_is_load": (admit_logits_1 * mask * is_load_action).sum() / (mask * is_load_action).sum(),
                    "action_0_is_not_load": (admit_logits_0 * mask * (1 - is_load_action)).sum() / (mask * (1 - is_load_action)).sum(),
                    "action_1_is_not_load": (admit_logits_1 * mask * (1 - is_load_action)).sum() / (mask * (1 - is_load_action)).sum(),
                    
                    # action prob is_load true and reused or not
                    "action_1_is_load_reuse": (admit_logits_1 * mask * is_load_action * is_reused_pred).sum() / (mask * is_load_action * is_reused_pred).sum(),
                    "action_1_is_load_not_reuse": (admit_logits_1 * mask * is_load_action * (1 - is_reused_pred)).sum() / (mask * is_load_action * (1 - is_reused_pred)).sum(),
                    "action_0_is_load_reuse": (admit_logits_0 * mask * is_load_action * is_reused_pred).sum() / (mask * is_load_action * is_reused_pred).sum(),
                    "action_0_is_load_not_reuse": (admit_logits_0 * mask * is_load_action * (1 - is_reused_pred)).sum() / (mask * is_load_action * (1 - is_reused_pred)).sum(),
                    
                    # "admit_logits_1_not_reused": (admit_logits_1 * mask * is_load_action * (1 - is_reused_pred)).sum().float() / (mask * is_load_action * (1 - is_reused_pred)).sum(),
                    # "admit_logits_1_reused": (admit_logits_1 * mask * is_load_action * is_reused_pred).sum().float() / (mask * is_load_action * is_reused_pred).sum(),
                    # "admit_logits_0_not_reused": (admit_logits_0 * mask * is_load_action * (1 - is_reused_pred)).sum().float() / (mask * is_load_action * (1 - is_reused_pred)).sum(),
                    # "admit_logits_0_reused": (admit_logits_0 * mask * is_load_action * is_reused_pred).sum().float() / (mask * is_load_action * is_reused_pred).sum(),
                    # "admit_logits_inaction_is_load": (admit_logits_inaction * mask * is_load_action).sum().float() / (mask * is_load_action).sum(),
                    # "admit_logits_inaction_not_load": (admit_logits_inaction * mask * (1 - is_load_action)).sum().float() / (mask * (1 - is_load_action)).sum(),
                }
                loss_info = {
                    "loss": loss,
                    "ppo_loss": ppo_loss,
                    "q_loss": q_loss,
                    "entropy": (entropy * mask).sum() / num_not_masked,
                    "bitmap_loss": bitmap_loss,
                    "is_reused_loss": is_reused_loss,
                    "uplift_admit_loss": uplift_admit_loss,
                    # "admit_predict": admit_preds.mean(),
                    # "random_explore_ratio": (random_explore_mask.sum(dim=1).float() / random_explore_mask.shape[1]).mean(),
                    # "random_explore_ne_ratio": (((random_explore_actions != orig_actions).long() * random_explore_mask).sum(dim=1).float() / random_explore_mask.shape[1]).mean()
                    # "rl_loss": rl_loss,
                    # "seq_loss": seq_loss,
                }
                # grad_info = {
                #     "admit_1_logits_grad": admit_1_logits_grad.grad,
                #     "is_reused_pred_grad": is_reused_pred_grad.grad,
                # }
                hr_score, _, _ = evaluate_steps_with_admit_actor(
                    tb_writer=self.tb_writer, run_step=run_step,
                    args=self.args, name='NNAdmit', replace_actor_name=self.args.replace_policy_name,
                    admit_actions=admit_action,
                    future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
                    tag_prefix=tag_prefix,
                )
                pred_reused_hr, _, _ = evaluate_steps_with_admit_actor(
                    tb_writer=self.tb_writer, run_step=run_step,
                    args=self.args, name='AdmitPredReused', replace_actor_name=self.args.replace_policy_name,
                    admit_actions=(is_reused_pred > 0.5).long(),
                    future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
                    tag_prefix=tag_prefix,
                )
                is_reused_hr, _, _ = evaluate_steps_with_admit_actor(
                    tb_writer=self.tb_writer, run_step=run_step,
                    args=self.args, name='AdmitIsReused', replace_actor_name=self.args.replace_policy_name,
                    admit_actions=is_reused_label.long(),
                    future_obs=input_ids, bitmap=None, num_cache_steps=self.num_cache_steps,
                    tag_prefix=tag_prefix,
                )
            fifo_hr, _, _ = evaluation_to_tb(
                tb_writer=self.tb_writer, run_step=run_step,
                name='FIFO', 
                cache_size=self.args.sim_cache_size,
                batch_tup=fifo_tup, num_cache_steps=self.num_cache_steps,
                tag_prefix=tag_prefix,
            )
            lru_hr, _, _ = evaluation_to_tb(
                tb_writer=self.tb_writer, run_step=run_step,
                name='LRUHistory', 
                cache_size=self.args.sim_cache_size,
                batch_tup=lru_tup, num_cache_steps=self.num_cache_steps,
                tag_prefix=tag_prefix,
            )
            lru_k_hr, _, _ = evaluation_to_tb(
                tb_writer=self.tb_writer, run_step=run_step,
                name='LRU-{}'.format(self.args.lru_k), 
                cache_size=self.args.sim_cache_size,
                batch_tup=lru_k_tup, num_cache_steps=self.num_cache_steps,
                tag_prefix=tag_prefix,
            )
            lfu_hr, _, _ = evaluation_to_tb(
                tb_writer=self.tb_writer, run_step=run_step,
                name='LFUHistory', 
                cache_size=self.args.sim_cache_size,
                batch_tup=lfu_tup, num_cache_steps=self.num_cache_steps,
                tag_prefix=tag_prefix,
            )
            lfuda_hr, _, _ = evaluation_to_tb(
                tb_writer=self.tb_writer, run_step=run_step,
                name='LFU-DA', 
                cache_size=self.args.sim_cache_size,
                batch_tup=lfuda_tup, num_cache_steps=self.num_cache_steps,
                tag_prefix=tag_prefix,
            )
            random_hr, _, _ = evaluation_to_tb(
                tb_writer=self.tb_writer, run_step=run_step,
                name='RandomHistory',
                cache_size=self.args.sim_cache_size,
                batch_tup=random_tup, num_cache_steps=self.num_cache_steps,
                tag_prefix=tag_prefix,
            )

            if run_step % 500 == 0:
                early_stopping([hr_score.item()], self.model, run_step)
            if run_step % 10 == 0:
                self.tb_writer.flush()
            if run_step % self.model.value.switch_steps == 0 and run_step != 0:
                self.model.value.switch()
                self.model.admit_q.switch()
            self.tb_writer.add_scalars(
                main_tag="{}/Loss".format(tag_prefix),
                tag_scalar_dict=loss_info,
                global_step=run_step,
            )
            if self.args.model_policy == "admit_ppo":
                local_scores_info = {
                    "pred_reused": pred_reused_hr,
                    "is_reused": is_reused_hr,
                }
                local_score_compare_info = {
                    "pred_reused": pred_reused_hr / random_hr,
                    "is_reused": is_reused_hr / random_hr,
                }
            scores_info = {
                "random": random_hr,
                "lru": lru_hr,
                "lruk": lru_k_hr,
                "lfu": lfu_hr,
                "lfuda": lfuda_hr,
                "fifo": fifo_hr,
                # "belady": belady_hr,
                "nn": hr_score,
            }
            score_compare_info = {
                "random": random_hr / random_hr,
                "lru": lru_hr / random_hr,
                "lruk": lru_k_hr / random_hr,
                "lfu": lfu_hr / random_hr,
                "lfuda": lfuda_hr / random_hr,
                "fifo": fifo_hr / random_hr,
                # "belady": belady_hr / random_hr,
                "nn": hr_score / random_hr,
            }
            scores_info.update(local_scores_info)
            self.tb_writer.add_scalars(
                main_tag="{}/Scores".format(tag_prefix),
                tag_scalar_dict=scores_info,
                global_step=run_step,
            )
            score_compare_info.update(local_score_compare_info)
            self.tb_writer.add_scalars(
                main_tag="{}/ScoreCompare".format(tag_prefix),
                tag_scalar_dict=score_compare_info,
                global_step=run_step,
            )
            self.tb_writer.add_scalars(
                main_tag="{}/Values".format(tag_prefix),
                tag_scalar_dict=values_info,
                global_step=run_step,
            )
            self.tb_writer.add_scalars(
                main_tag="{}/LargeValues".format(tag_prefix),
                tag_scalar_dict=large_values_info,
                global_step=run_step,
            )
            self.tb_writer.add_scalars(
                main_tag="{}/Probs".format(tag_prefix),
                tag_scalar_dict=probs_info,
                global_step=run_step,
            )
            # self.tb_writer.add_scalars(
            #     main_tag="Train/Grads",
            #     tag_scalar_dict=grad_info,
            #     global_step=run_step,
            # )
            run_step += 1

        post_fix = {
            "epoch": epoch,
            # "rl_loss": '{:.4f}'.format(rl_loss / len(rl_data_iter)),
        }

        if (epoch + 1) % self.args.log_freq == 0:
            self.logger.info(str(post_fix))

        return run_step

    def eval_rl(self):
        model: model.rl.CacheWorld = self.model
        self.model.eval()
        num_items, num_hits = 0, 0
        num_lru_hits, num_lfu_hits, num_random_hits = 0, 0, 0
        # rl_data_iter = tqdm.tqdm(enumerate(self.eval_dataloader),
        #                           desc="Mode_train_rl",
        #                           total=len(self.eval_dataloader),
        #                           bar_format="{l_bar}{r_bar}")
        rl_data_iter = tqdm.tqdm(enumerate(self.train_dataloader),
                                  desc="Mode_train_rl",
                                  total=len(self.train_dataloader),
                                  bar_format="{l_bar}{r_bar}")
        for i, batch in rl_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            
            user_ids, input_ids, answers, mask, reward, admit_action, replace_action, action_hit, is_full, bitmap, cache_held_size, random_explore_mask, random_explore_actions, orig_actions = batch
            if len(user_ids) == 0:
                continue
            
            num_items += mask.sum().item()
            num_hits += (mask * reward).sum().item()
            
            _, cur_hits, _ = evaluate_steps_with_actor(None, run_step=0, args=self.args, name="", admit_actor=CacheEnv.always_admit_actor, replace_actor_name="lru_actor", future_obs=input_ids, bitmap=bitmap)
            num_lru_hits += cur_hits.sum().item()
            
            _, cur_hits, _ = evaluate_steps_with_actor(None, run_step=0, args=self.args, name="", admit_actor=CacheEnv.always_admit_actor, replace_actor_name="lfu_actor", future_obs=input_ids, bitmap=bitmap)
            num_lfu_hits += cur_hits.sum().item()
            
            _, cur_hits, _ = evaluate_steps_with_actor(None, run_step=0, args=self.args, name="", admit_actor=CacheEnv.always_admit_actor, replace_actor_name="random_actor", future_obs=input_ids, bitmap=bitmap)
            num_random_hits += cur_hits.sum().item()
            
        print("NN #items: {}, #hits: {}, HR: {}".format(num_items, num_hits, num_hits / num_items))
        print("lru #items: {}, #hits: {}, HR: {}".format(num_items, num_lru_hits, num_lru_hits / num_items))
        print("lfu #items: {}, #hits: {}, HR: {}".format(num_items, num_lfu_hits, num_lfu_hits / num_items))
        print("random #items: {}, #hits: {}, HR: {}".format(num_items, num_random_hits, num_random_hits / num_items))
