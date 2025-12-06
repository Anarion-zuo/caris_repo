from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from env import evaluate_nn_steps
from model.bsarec import BSARecModel
from model._modules import LayerNorm
from metrics import recall_at_k, ndcg_k
import sklearn.metrics as skmetrics
import torcheval.metrics.functional as eF


def scan_nan(module):
    for name, param in module.named_parameters(recurse=True):
        if param.isnan().any().item():
            raise RuntimeError("Nan param, name: {} #nan: {} shape {}".format(name, param.isnan().long().sum().item(), param.shape))


def assert_tensor_not_nan(module):
    if module.isnan().any().item():
        raise RuntimeError("tensor contains nan, shape: {}, #nan {}".format(module.shape, module.isnan().long().sum()))

def init_linear_layer(module):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.01)

def init_layers(m: nn.Module):
    def _init(module):
        if isinstance(module, nn.Linear):
            init_linear_layer(module)
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    m.apply(_init)


class RLCommonFFN(nn.Module):
    def _make_layer(self, input_dim, output_dim, middle: bool = True):
        if middle:
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout),
            )
        return nn.Linear(input_dim, output_dim)
    
    def _init_layers(self):
        init_layers(self)

    def __init__(self, args) -> None:
        super(RLCommonFFN, self).__init__()
        self.args = args


class ReplacePolicyFutureFFN(RLCommonFFN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.input_dim = args.hidden_size * 2  # concat 2 vectors
        self.num_layers = args.policy_num_layers
        assert self.num_layers >= 2
        self.hidden_dim = args.hidden_size
        self.dropout = args.policy_dropout

        self.layers = nn.ModuleList(
        # [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        # + [self._make_layer(self.hidden_dim, args.item_size+1)]  # hidden_size is the embedding dim
        )
        self.input1 = nn.Linear(args.hidden_size, self.hidden_dim)
        self.input2 = nn.Linear(args.hidden_size, self.hidden_dim)
        self.layer_norm = LayerNorm(self.hidden_dim)

        self._init_layers()

    def forward(self, hat_e_t: torch.Tensor, e_k: torch.Tensor, E: nn.Embedding):
        # print(hat_e_t.shape, e_k.shape)
        if e_k is None:
            x = hat_e_t
        else:
            x = torch.cat([hat_e_t, e_k], dim=1)
        # print(x.shape)
        # assert len(x.shape) == 2
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        x = self.layers[-1](x)
        x = self.layer_norm(x)
        f_t = x
        # f_t: [batch, N_l, emb_dim]
        pi_logits = torch.matmul(x, E.weight[: self.args.item_size+1, :].transpose(0, 1)) # remove reserved ids
        # [batch, N_l, #items]
        pi = F.softmax(pi_logits, dim=-1)
        # pi: [batch, N_l, #items]
        return f_t, pi
    
    def calculate_policy_likelihoods(self, hat_e_t: torch.Tensor, e_k_t: torch.Tensor, E: nn.Embedding, u_ids_t: torch.Tensor):
        # hat_e_t, e_k_t [batch, time, dim]
        # u_ids_t [batch, time]
        f_t, pi_t = self.forward(hat_e_t=hat_e_t, e_k=e_k_t, E=E)
        return pi_t.gather(dim=2, index=u_ids_t.unsqueeze(2)).squeeze(2) # [batch, time]

    def forward_all_e(self, hat_e_t: torch.Tensor, E: nn.Embedding, world_sg: bool, item_sampled: torch.Tensor = None):
        # item_sampled [batch, N_l, #sampling]
        hat_e_t_shape = hat_e_t.shape  # [batch, N_l, hidden_dim]
        hat_e_t = hat_e_t.reshape(-1, hat_e_t.shape[-1])  # remove batch dim
        # [batch x N_l, hidden_dim]
        x1 = self.input1(hat_e_t)
        if item_sampled is None:
            x1 = x1.unsqueeze(1).expand(-1, E.num_embeddings, -1)
            # [batch x N_l, #items, hidden_dim]
        else:
            x1 = x1.unsqueeze(1).expand(-1, item_sampled.shape[2], -1)
            # [batch x N_l, #sampling, hidden_dim]
        if item_sampled is None:
            E_weight = E.weight  # [#items, emb_dim]
        else:
            E_weight = E(item_sampled)  # [batch x N_l x #sampling, emb_dim]
        if world_sg:
            E_weight = E_weight.detach()
        
        x2 = self.input2(E_weight)  # [batch, N_l, #items/sampling, emb_dim]
        if item_sampled is None:
            x2 = x2.unsqueeze(0).expand(hat_e_t_shape[0] * hat_e_t_shape[1], -1, -1)
            # [batch x N_l, #items, emb_dim]
        else:
            x2 = x2.reshape(-1, item_sampled.shape[2], x2.shape[-1])
            # [batch x N_l, #sampling, emb_dim]
        x = x1 + x2
        x = x.reshape(-1, x.shape[-1])  # [batch x N_l x #items/sampling, input_dim]
        # print("forward_all_e", x.shape, self.input_dim, self.hidden_dim)
        for layer in self.layers:
            x = x + layer(x)
        x = self.layer_norm(x)
        # x: [batch x N_l x #items/sampling, output_dim]
        if item_sampled is None:
            pi_logits = torch.matmul(x, E_weight[: self.args.item_size+1, :].transpose(0, 1)) # remove reserved ids
            # pi: [batch x N_l x #items, #items] -> [batch, N_l, #items, #items]
            # the first #items dim has the reserved ids accounted
            pi = F.softmax(pi_logits, dim=-1)
            return pi.reshape(hat_e_t_shape[0], hat_e_t_shape[1], E.num_embeddings, pi.shape[-1])[:, :, :self.args.item_size+1, :self.args.item_size+1]
        else:
            pi_logits = torch.matmul(x.view(hat_e_t_shape[0], hat_e_t_shape[1], item_sampled.shape[2], -1), E.weight[:self.args.item_size+1, :].transpose(0, 1))
            # [batch, N_l, #sampling, dim] x [#items, dim]^T -> [batch, N_l, #sampling, #items]
            return F.softmax(pi_logits, dim=-1)
        
    def forward_use(self, hat_e_t: torch.Tensor, E: nn.Embedding, held_items: torch.Tensor, requested_items: torch.Tensor):
        # hat_e_t [batch, time, dim]
        x1 = self.input1(hat_e_t)
        # held_items [batch, time, #held]
        # requested_items [batch, time, #requested]
        r_embs = E(requested_items)  # [batch, time, #requested, dim]
        x2 = self.input2(r_embs)  # [batch, time, #requested, dim]
        # print("forward_use", x1.shape, x2.shape, hat_e_t.shape, r_embs.shape)
        x = x2 + x1.unsqueeze(2).expand(-1, -1, x2.shape[2], -1) # [batch, time, #requested, dim]
        x = x.reshape(-1, x.shape[-1])  # [batch x time x #requested, dim]
        for layer in self.layers:
            x = x + layer(x)
        x = self.layer_norm(x)  # [batch x time x #requested, dim]
        held_embs = E(held_items)  # [batch, time, #held, dim]
        pi_logits = torch.matmul(
            x.reshape(-1, r_embs.shape[2], r_embs.shape[3]), 
            held_embs.reshape(-1, held_embs.shape[2], held_embs.shape[3]).transpose(1, 2)
        )  # [batch x time, #requested, #held]
        return pi_logits
    
    def sample_u_t(self, hat_e_t: torch.Tensor, e_k_t: torch.Tensor, E: nn.Embedding, bitmap: torch.Tensor):
        # bitmap: [#items]
        # pi: [batch, N_l, #items]
        f_t, pi = self.forward(hat_e_t, e_k_t, E)
        pi_shape = pi.shape[:-1]
        pi = pi * bitmap.unsqueeze(0).unsqueeze(1).expand(pi.shape[0], pi.shape[1], -1)
        u_t = torch.multinomial(pi.view(-1, self.args.item_size+1), 1).squeeze(1).view(pi_shape)
        # sampled result [batch, N_l]
        return u_t


class ReplacePolicyFFN(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.input_dim = args.hidden_size
        self.num_layers = args.policy_num_layers
        assert self.num_layers >= 2
        self.hidden_dim = args.hidden_size
        self.dropout = args.policy_dropout

        self.layers = nn.ModuleList(
        # [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers-1)
        ]
        + [self._make_layer(self.hidden_dim, 1, middle=False)]  # hidden_size is the embedding dim
        )
        self.layer_norm = LayerNorm(self.hidden_dim)

        self._init_layers()
        scan_nan(self)
        
    def forward(self, seq_output: torch.Tensor, held_item_embs: torch.Tensor, replace_action_indices: torch.Tensor):
        x = self.layers[0](seq_output)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        x = self.layers[-1](x)
        x = self.layer_norm(x)  # [batch, time, dim]
        
        # held_item_embs [batch, time, cache_size, dim]
        logits = torch.einsum("btd,btcd->btc", x, held_item_embs)
        probs = F.softmax(logits, dim=2)  # [batch, time, cache_size]
        if replace_action_indices is None:
            return probs, logits, None
        # compute likelihood
        # replace_action_indices [batch, time]
        lh = probs.gather(dim=2, index=replace_action_indices.unsqueeze(2)).squeeze(2)  # [batch, time]
        return probs, logits, lh
    
    def sample_replace_action_indices(self, seq_output: torch.Tensor, held_item_embs: torch.Tensor):
        probs, _, _ = self.forward(seq_output=seq_output, held_item_embs=held_item_embs, replace_action_indices=None)
        return probs.argmax(dim=2)


class ReplaceQFFN(RLCommonFFN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.num_layers = args.value_num_layers
        assert self.num_layers > 1
        self.input_dim = args.hidden_size*3
        self.hidden_dim = args.hidden_size
        self.dropout = args.value_dropout

        self.layers = nn.ModuleList(
        [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers - 1)
        ]
        )
        # self.value_readout_layer = nn.Linear(self.hidden_dim, 1)
        # self.advantage_readout_layer = nn.Linear(self.hidden_dim, self.num_actions)
        self.q_predictor = nn.Linear(self.hidden_dim, 1)
        self.lambd = args.discount_ratio

        self._init_layers()
        
    def forward(self, seq_output: torch.Tensor, last_action_embs: torch.Tensor, cur_action_embs: torch.Tensor):
        replace_states = torch.cat([seq_output, last_action_embs], dim=1)  # [batch, dim]
        # held_item_embs [batch, dim]
        x = torch.cat([replace_states, cur_action_embs], dim=1)  # [batch, dim]
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = x + layer(x)
        q: torch.Tensor = self.q_predictor(x).squeeze(1)  # [batch]
        return q
    
    def forward_batch_seqs(self, seq_output: torch.Tensor, last_action_embs: torch.Tensor, cur_action_embs: torch.Tensor):
        return self.forward(
            seq_output=seq_output.reshape(-1, seq_output.shape[2]),
            last_action_embs=last_action_embs.reshape(-1, last_action_embs.shape[2]),
            cur_action_embs=cur_action_embs.reshape(-1, cur_action_embs.shape[2]),
        ).reshape(seq_output.shape[0], seq_output.shape[1])
    
    def calculate_max_action_given_states(self, seq_output: torch.Tensor, last_action_embs: torch.Tensor, held_item_embs: torch.Tensor):
        # seq_output, last_action_embs [batch, time, dim]
        # held_item_embs [batch, time, cache_size, dim]

        seq_output_exp = seq_output.unsqueeze(2).expand(-1, -1, held_item_embs.shape[2], -1)  # [batch, time, cache_size, dim]
        last_action_embs_exp = last_action_embs.unsqueeze(2).expand(-1, -1, held_item_embs.shape[2], -1)  # [batch, time, cache_size, dim]
        
        full_q = self.forward(
            seq_output=seq_output_exp.reshape(-1, seq_output_exp.shape[3]),
            last_action_embs=last_action_embs_exp.reshape(-1, last_action_embs_exp.shape[3]),
            cur_action_embs=held_item_embs.reshape(-1, held_item_embs.shape[3]),
        )  # [batch x time x cache_size]
        full_q = full_q.reshape(seq_output.shape[0], seq_output.shape[1], held_item_embs.shape[2])
        # [batch, time, cache_size]
        return full_q.argmax(dim=2)  # [batch, time]


class ReplaceQPairFFN(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.m1 = AdmitQFFN(args)
        self.m2 = AdmitQFFN(args)
        self.switch_steps = args.value_switch_steps
        
    def calculate_td_loss_value_adv(self, seq_output: torch.Tensor, last_action_embs: torch.Tensor, cur_action_embs: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor):
        # m1 to update, m2 to eval
        # compute q update
        a_max = self.m1.calculate_max_action_given_states(admit_states=admit_states, action_full_embs=action_full_embs)  # [batch, time]
        a_max_embs = action_full_embs(a_max)  # [batch, time, dim]
        next_states = admit_states[:, 1:, :]  # [batch, time-1, dim]
        next_action_embs = a_max_embs[:, 1:, :]  # [batch, time-1, dim]
        next_q = self.m2.forward_batch_seqs(  # [batch, time-1]
            admit_states=next_states,
            action_embs=next_action_embs,
        )
        q_update = reward[:, :-1] + next_q * self.m1.lambd
        q_update = q_update.detach()  # [batch, time-1]
        
        # compute q
        action_embs = action_full_embs(actions)  # [batch, time, dim]
        q = self.m1.forward_batch_seqs(  # [batch, time]
            admit_states=admit_states,
            action_embs=action_embs,
        )
        q_loss = nn.MSELoss()((q * mask)[:, :-1], q_update * mask[:, :-1])
        
        # compute v
        v = self.m1.forward_batch_seqs(
            admit_states=admit_states,
            action_embs=a_max_embs,
        )
        
        # compute adv
        adv = q - v
        
        return q_loss, q, q_update, v, adv
    
    def switch(self):
        m = self.m1
        self.m1 = self.m2
        self.m2 = m


def approx_ndcg(scores, relevances, alpha=10., mask=None):
  """Computes differentiable estimate of NDCG of scores as following.

  Uses the approximation framework from Qin et. al., 2008

    IDCG = sum_i (exp(rel[i]) - 1) / ln(i + 1)
    DCG = sum_i (exp(rel[i]) - 1) / ln(pos(score, i) + 1)
    pos(score, i) =
      1 + sum_{j != i} exp(-alpha s_{i, j}) / (1 + exp(-alpha s_{i, j}))
      (differentiable approximate position function)
    s_{i, j} = scores[i] - scores[j]
    NDCG loss = -DCG / IDCG

  Args:
    scores (torch.FloatTensor): tensor of shape (batch_size, num_elems).
    relevances (torch.FloatTensor): tensor of same shape as scores (rel).
    alpha (float): value to use in the approximate position function. The
      approximation becomes exact as alpha tends toward inf.
    mask (torch.ByteTensor | None): tensor of same shape as scores. Masks out
      elements at index [i][j] if mask[i][j] = 0. Defaults to no masking.

  Returns:
    ndcg (torch.FloatTensor): tensor of shape (batch_size).
  """
  def approx_positions(scores, alpha=10.):
    # s_{i, j} (batch_size, num_elems)
    diff = (scores.unsqueeze(-1).expand(-1, -1, scores.shape[1]) -
            scores.unsqueeze(1).expand(-1, scores.shape[1], -1))
    # Add 0.5 instead of 1, because s_{1, i} = 0.5 is included
    return 0.5 + torch.sigmoid(alpha * diff).sum(1)

  if mask is None:
    mask = torch.ones_like(scores)

  # +1 because indexing starts at 1 in IDCG
  idcg = torch.expm1(relevances) * mask.float() / torch.log1p(
      torch.arange(scores.shape[-1]).float() + 1)
  pos = approx_positions(scores, alpha)
  dcg = torch.expm1(relevances) * mask.float() / torch.log1p(pos)
  return -dcg.sum(-1) / (idcg.sum(-1) + 1e-8)


class ReplacePolicyImitatorFFN(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.input_dim = args.hidden_size
        self.num_layers = args.policy_num_layers
        assert self.num_layers >= 2
        self.hidden_dim = args.hidden_size
        self.dropout = args.policy_dropout
        self.num_cache_steps = self.args.num_cache_steps

        self.layers = nn.ModuleList(
        # [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        # + [self._make_layer(self.hidden_dim, args.item_size+1)]  # hidden_size is the embedding dim
        )
        self.layer_norm = LayerNorm(self.hidden_dim)
        self.use_distance_pred = nn.Linear(self.hidden_dim, 1)

        self._init_layers()
        scan_nan(self)
        
    def forward(self, hat_e_t: torch.Tensor, E: nn.Embedding, held_items: torch.Tensor):
        x = hat_e_t
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        use_dist_pred = self.use_distance_pred(x).squeeze(2)
        x = self.layers[-1](x)
        
        x = self.layer_norm(x)  # [batch, time, dim]
        # held_embs = E(held_items)  # [batch, time, cache_size, embs]
        # print("ReplacePolicyImitaqtorFFN forward", x.shape, x[:, :, :].shape, held_embs.shape, held_items.shape)
        # logits = torch.einsum("btd,btcd->btc", x, held_embs)
        logits = torch.einsum("btd,id->bti", x, E.weight[: self.args.item_size+1, :])
        # [batch, time, dim], [batch, time, cache_size, dim] -> [batch, time, cache_size]
        prob = F.softmax(logits, dim=2)
        # entropy = \sum_x p_x \log p_x
        # [batch, time]
        entropy = -(prob * torch.log(prob + 1e-5)).sum(dim=2)
        # assert_tensor_not_nan(logits)
        # assert_tensor_not_nan(prob)
        return logits, prob, entropy, use_dist_pred
    
    def sample_replace(self, hat_e_t: torch.Tensor, E: nn.Embedding, held_items: torch.Tensor):
        _, prob, _, _ = self.forward(hat_e_t, E, held_items)  # [batch, time, cache_size]
        sampled_index = torch.argmax(prob, dim=2)  # [batch, time]
        return sampled_index
        return held_items.gather(dim=2, index=sampled_index.unsqueeze(2)).squeeze(2)
    
    def likelihood_from_sample(self, prob: torch.Tensor, action: torch.Tensor):
        # prob [batch, time, #items]
        # action [batch, time]
        return prob.gather(dim=2, index=action.unsqueeze(2)).squeeze(2)
        # [batch, time]
    
    def calculate_imitate_loss(self, hat_e_t: torch.Tensor, E: nn.Embedding, held_items: torch.Tensor, teacher_actions: torch.Tensor, belady_labels: torch.Tensor, logp1_use_distances: torch.Tensor):
        assert hat_e_t.shape[0] == teacher_actions.shape[0]
        assert hat_e_t.shape[1] - 1 == teacher_actions.shape[1]
        logits, _, _, use_dist_pred = self.forward(hat_e_t=hat_e_t, E=E, held_items=held_items)
        # logits [batch, time, #items]
        # teacher_actions [batch, time-1]
        # reuse_loss = nn.MSELoss()(use_dist_pred, logp1_use_distances)
        reuse_loss = 0
        # imitate_loss = nn.CrossEntropyLoss()(logits[:, -1-self.num_cache_steps : -1, :].reshape(-1, logits.shape[2]), teacher_actions[:, -self.num_cache_steps:].reshape(-1))
        imitate_loss = nn.CrossEntropyLoss(ignore_index=0)(logits[:, : -1, :].reshape(-1, logits.shape[2]), teacher_actions.reshape(-1))
        loss = imitate_loss + reuse_loss
        return loss, imitate_loss, reuse_loss
        rank_loss = approx_ndcg(
            scores=logits[:, -1-self.num_cache_steps:, :].reshape(-1, logits.shape[2]),
            relevances=logp1_use_distances[:, -1-self.num_cache_steps:],
        )
        loss = rank_loss + reuse_loss
        return loss, rank_loss, reuse_loss


class PrefetchPolicyFNN(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.input_dim = args.hidden_size
        self.num_layers = args.policy_num_layers
        assert self.num_layers >= 2
        self.hidden_dim = args.hidden_size
        self.dropout = args.policy_dropout

        self.layers = nn.ModuleList(
        # [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        # + [self._make_layer(self.hidden_dim, args.item_size+1)]  # hidden_size is the embedding dim
        )
        self.layer_norm = LayerNorm(self.hidden_dim)

        self._init_layers()
        scan_nan(self)
        
    def forward(self, hat_e_t: torch.Tensor, E: nn.Embedding):
        x = hat_e_t
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        x = self.layers[-1](x)
        # [batch, time, dim]
        x = self.layer_norm(x)
        logits = torch.einsum("btd,id->bti", x, E.weight[:self.args.item_size+1, :])
        # [batch, time, dim], [#items, dim] -> [batch, time, #items]
        prob = F.softmax(logits, dim=2)
        # entropy = \sum_x p_x \log p_x
        # [batch, time]
        entropy = -(prob * torch.log(prob + 1e-5)).sum(dim=2)
        # assert_tensor_not_nan(logits)
        # assert_tensor_not_nan(prob)
        return prob, entropy
    
    def sample_prefetch(self, hat_e_t: torch.Tensor, E: nn.Embedding):
        prob, _ = self.forward(hat_e_t, E)  # [batch, time, #items]
        # sampled = torch.multinomial(prob.reshape(-1, self.args.item_size+1), num_samples=1).squeeze(1)
        sampled = torch.argmax(prob, dim=2)
        # mask a probability to prefetch actions
        # to control prefetch frequency
        use_mask = torch.multinomial(
            input=torch.as_tensor([1-self.args.prefetch_predicate_use_prob, self.args.prefetch_predicate_use_prob]),
            num_samples=sampled.shape[0]*sampled.shape[1],
            replacement=True,
        ).reshape(sampled.shape).to(sampled.device)
        return sampled * use_mask  # [batch, time]
    
    def likelihood_from_sample(self, prob: torch.Tensor, action: torch.Tensor):
        # prob [batch, time, #items]
        # action [batch, time]
        return prob.gather(dim=2, index=action.unsqueeze(2)).squeeze(2)
        # [batch, time]


class PrefetchPolicyPredictMixture(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.input_dim = args.hidden_size
        self.num_layers = args.policy_num_layers
        assert self.num_layers >= 2
        self.hidden_dim = args.hidden_size
        self.dropout = args.policy_dropout

        self.layers = nn.ModuleList(
        # [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ]
        # + [self._make_layer(self.hidden_dim, args.item_size+1)]  # hidden_size is the embedding dim
        )
        self.layer_norm = LayerNorm(self.hidden_dim)

        self._init_layers()
        scan_nan(self)
        
    def forward(self, hat_e_t: torch.Tensor, E: nn.Embedding):
        x = hat_e_t
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        x = self.layers[-1](x)
        # [batch, time, dim]
        x = self.layer_norm(x)
        logits = torch.einsum("btd,id->bti", x, E.weight[:self.args.item_size+1, :])
        # [batch, time, dim], [#items, dim] -> [batch, time, #items]
        prob = F.softmax(logits, dim=2)
        # entropy = \sum_x p_x \log p_x
        # [batch, time]
        entropy = -(prob * torch.log(prob + 1e-5)).sum(dim=2)
        # assert_tensor_not_nan(logits)
        # assert_tensor_not_nan(prob)
        return prob, entropy
    
    def sample_prefetch(self, hat_e_t: torch.Tensor, E: nn.Embedding):
        prob, _ = self.forward(hat_e_t, E)  # [batch, time, #items]
        # sampled = torch.multinomial(prob.reshape(-1, self.args.item_size+1), num_samples=1).squeeze(1)
        sampled = torch.argmax(prob, dim=2)
        # mask a probability to prefetch actions
        # to control prefetch frequency
        use_mask = torch.multinomial(
            input=torch.as_tensor([1-self.args.prefetch_predicate_use_prob, self.args.prefetch_predicate_use_prob]),
            num_samples=sampled.shape[0]*sampled.shape[1],
            replacement=True,
        ).reshape(sampled.shape).to(sampled.device)
        return sampled * use_mask  # [batch, time]
    
    def likelihood_from_sample(self, prob: torch.Tensor, action: torch.Tensor):
        # prob [batch, time, #items]
        # action [batch, time]
        return prob.gather(dim=2, index=action.unsqueeze(2)).squeeze(2)
        # [batch, time]


INVALID_LOGIT = -1e8


class AdmitPolicyFFN(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.input_dim = args.hidden_size
        self.num_layers = args.policy_num_layers
        assert self.num_layers >= 2
        self.hidden_dim = args.hidden_size
        self.dropout = args.policy_dropout

        self.layers = nn.ModuleList(
        # [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers-1)
        ]
        + [self._make_layer(self.hidden_dim, 3, middle=False)]  # hidden_size is the embedding dim
        )
        self.layer_norm = LayerNorm(self.hidden_dim)
        # self.action_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=self.hidden_dim) # 0, 1

        self._init_layers()
        scan_nan(self)
        
    def forward(self, seq_output: torch.Tensor, action: torch.Tensor, is_load_action: torch.Tensor, is_reused_prob: torch.Tensor, reuse_prob_thresh: float):
        # print(x.shape)
        # assert len(x.shape) == 2
        x = seq_output
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        x = self.layer_norm(x)  # [batch, time, dim]
        x: torch.Tensor = self.layers[-1](x)  # [batch, time, 2]
        if self.args.use_pred_reuse:
            is_reused_mask = (is_reused_prob > reuse_prob_thresh).long()  # [batch, time]
        else:
            is_reused_mask = torch.ones(is_reused_prob.shape, device=is_reused_prob.device).long()
        if self.args.use_nop:
            nop_mask = is_load_action
        else:
            nop_mask = torch.zeros(is_load_action.shape, device=is_load_action.device).long()
        admit_logits = x
        admit_logits_inaction = (1 - nop_mask) * admit_logits[:, :, 2] + nop_mask * INVALID_LOGIT
        # admit_logits_1 = is_load_action * admit_logits[:, :, 1] + (1 - is_load_action) * INVALID_LOGIT  # [batch, time]
        # admit_logits_0 = is_load_action * ((1 - is_reused_mask) * admit_logits[:, :, 0] + is_reused_mask * INVALID_LOGIT) + (1 - is_load_action) * INVALID_LOGIT  # [batch, time]
        admit_logits_1 = nop_mask * (is_reused_mask * admit_logits[:, :, 1] + (1 - is_reused_mask) * INVALID_LOGIT) + (1 - nop_mask) * INVALID_LOGIT  # [batch, time]
        admit_logits_0 = nop_mask * admit_logits[:, :, 0] + (1 - nop_mask) * INVALID_LOGIT  # [batch, time]
        admit_prob_full = torch.stack([admit_logits_0, admit_logits_1, admit_logits_inaction], dim=2).softmax(dim=2)
        admit_prob = admit_prob_full[:, :, 1]
        no_admit_prob = admit_prob_full[:, :, 0]
        inaction_prob = admit_prob_full[:, :, 2]
        entropy = -(admit_prob * torch.log(admit_prob + 1e-5) + no_admit_prob * torch.log(no_admit_prob + 1e-5) + inaction_prob * torch.log(inaction_prob + 1e-5))
        if action is not None:
            # action [batch, time] 0 or 1
            # lh = prob.gather(dim=2, index=action.unsqueeze(2)).squeeze(2)
            lh = is_load_action * (admit_prob * action + no_admit_prob * (1 - action)) + (1 - is_load_action) * inaction_prob
        else:
            lh = None
        return admit_prob, no_admit_prob, inaction_prob, entropy, lh, (admit_logits_0, admit_logits_1, admit_logits_inaction, is_reused_mask,)
    
    def sample_actions(self, seq_output: torch.Tensor, is_reused_prob: torch.Tensor):
        prob, no_admit_prob, inaction_prob, _, _, _ = self.forward(seq_output, action=None, is_load_action=torch.ones(size=seq_output.shape[:2], device=seq_output.device), is_reused_prob=is_reused_prob, reuse_prob_thresh=self.args.reuse_prob_thresh)
        # [batch, time]
        raw_actions = torch.stack([no_admit_prob, prob, inaction_prob], dim=2).argmax(dim=2)
        not_inaction_mask = (raw_actions != 2).long()
        # inaction to not admit 0
        return raw_actions * not_inaction_mask


class ValueFFN(RLCommonFFN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.num_layers = args.value_num_layers
        assert self.num_layers > 1
        self.input_dim = args.hidden_size
        self.hidden_dim = args.hidden_size
        self.num_actions = self.args.item_size+1
        self.dropout = args.value_dropout

        self.layers = nn.ModuleList(
        [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers - 1)
        ]
        )
        self.value_readout_layer = nn.Linear(self.hidden_dim, 1)
        self.advantage_readout_layer = nn.Linear(self.hidden_dim, self.num_actions)
        self.lambd = args.discount_ratio

        self._init_layers()
        self._precompute_weights()
        
    def _precompute_weights(self):
        full_row = [self.lambd**t for t in range(self.args.max_seq_length)]
        row_list = list()
        for row_i in range(self.args.max_seq_length-1):
            new_row = [0] * row_i + full_row
            new_row = new_row[: self.args.max_seq_length]
            row_list.append(new_row)
        self.lambda_matrix = torch.as_tensor(
            row_list,
        )

    def forward_dueling(self, hat_e_t: torch.Tensor):
        x = hat_e_t  # [batch, time, dim]
        for layer in self.layers:
            x = x + layer(x)
        v: torch.Tensor = self.value_readout_layer(x)  # [batch, time, 1]
        a: torch.Tensor = self.advantage_readout_layer(x)  # [batch, time, #items]
        a -= a.max(dim=2)[0].unsqueeze(2).expand(-1, -1, self.args.item_size+1)
        q = v.expand(-1, -1, self.num_actions) + a
        # value, advantage, Q
        return v.squeeze(2), a, q
    
    def forward_value_adv(self, hat_e_t: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor):
        x = hat_e_t  # [batch, time, dim]
        for layer in self.layers:
            x = x + layer(x)
        v: torch.Tensor = self.value_readout_layer(x).squeeze(2)  # [batch, time]
        r_t = rewards[:, :-1]  # [batch, time-1]
        
        if self.args.multistep_value_estimation:
            v_T = v[:, -1]
            value_estimate_term_t = torch.cat([r_t, v_T.unsqueeze(1)], dim=1)  # [batch, time]
            # the masked state actions should not be accounted
            value_estimate_term_t = value_estimate_term_t * mask / torch.float_power(
                input=torch.as_tensor(self.lambd, device=mask.device, dtype=torch.float), exponent=(1-mask).sum(dim=1),
            ).unsqueeze(1).expand(-1, mask.shape[1])
            v_update = torch.einsum("it,bt->bi", self.lambda_matrix.to(value_estimate_term_t.device).double(), value_estimate_term_t).float()
            # [batch, time-1]
            
            adv = v_update - v[:, :-1]
            return adv, v, v_update
        
        v_t1 = v[:, 1:]
        v_update = r_t + self.lambd * v_t1
        adv = v_update - v[:, :-1]
        return adv, v, v_update
    
    # def calculate_td_update(self, hat_e_N: torch.Tensor, h_t: torch.Tensor, lambd: float):
    #     lambd_t = torch.tensor([lambd ** i for i in range(h_t.shape[1])])
    #     H_N = self.forward(hat_e_N)
    #     hat_H_1 = torch.einsum("bk,k->b", h_t, lambd_t) + lambd**h_t.shape[1] * H_N
    #     return hat_H_1
    
    def calculate_td_update_dueling(self, hat_e_t: torch.Tensor, prefetch_prob: torch.Tensor, reward: torch.Tensor):
        # hat_e_t [batch, time, dim]
        # prefetch_prob [batch, time, #items]
        # reward [batch, time]
        v, a, q = self.forward_dueling(hat_e_t)
        q = q.detach()
        next_q = q[:, 1:, :]  # [batch, time-1, #items]
        cur_reward = reward[:, :-1]
        td_update = cur_reward + self.lambd * (next_q * prefetch_prob[:, 1:, :]).sum(dim=2)
        return td_update
    
    def calculate_td_update_value_adv(self, hat_e_t: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor):
        a, v, v_update = self.forward_value_adv(hat_e_t, reward, mask)
        return v_update.detach()
        
        
class ValueFFNPair(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.m1 = ValueFFN(args)
        self.m2 = ValueFFN(args)
        self.switch_steps = args.value_switch_steps
        
    def calculate_td_loss_dueling(self, hat_e_t: torch.Tensor, prefetch_prob: torch.Tensor, reward: torch.Tensor, prefetch_action: torch.Tensor):
        # prefetch_action [batch, time]
        td_update = self.m1.calculate_td_update_dueling(hat_e_t, prefetch_prob, reward)  # [batch, time-1]
        td_update = td_update.detach()
        v, a, q = self.m2.forward_dueling(hat_e_t)
        q = q[:, :-1, :].gather(dim=2, index=prefetch_action[:, :-1].unsqueeze(2)).squeeze(2)
        return nn.MSELoss()(q, td_update), a, q, v, td_update
    
    def calculate_td_loss_value_adv(self, hat_e_t: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor):
        v_update = self.m1.calculate_td_update_value_adv(hat_e_t, reward, mask)
        # [batch, time-1]
        a, v, _ = self.m2.forward_value_adv(hat_e_t, reward, mask)
        v = v[:, :-1]  # [batch, time-1]
        return nn.MSELoss()(v, v_update), a, v, v_update
    
    def switch(self):
        m = self.m1
        self.m1 = self.m2
        self.m2 = m


class AdmitQFFN(RLCommonFFN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.num_layers = args.value_num_layers
        assert self.num_layers > 1
        self.input_dim = args.hidden_size
        self.hidden_dim = args.hidden_size
        self.dropout = args.value_dropout

        self.layers = nn.ModuleList(
        [self._make_layer(self.input_dim, self.hidden_dim)] +
        [
            self._make_layer(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers - 1)
        ]
        )
        # self.value_readout_layer = nn.Linear(self.hidden_dim, 1)
        # self.advantage_readout_layer = nn.Linear(self.hidden_dim, self.num_actions)
        self.q_predictor = nn.Linear(self.hidden_dim, 3)
        self.lambd = args.discount_ratio

        self._init_layers()
        
    def forward(self, seq_output: torch.Tensor):
        # seq_output [batch, dim]
        x = seq_output  # [batch, dim]
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = x + layer(x)
        q: torch.Tensor = self.q_predictor(x)  # [batch, #actions]
        return q
    
    def forward_batch_seqs(self, seq_output: torch.Tensor):
        return self.forward(
            seq_output=seq_output.reshape(-1, seq_output.shape[2]),
        ).reshape(seq_output.shape[0], seq_output.shape[1], 3)  # [batch, time, #actions]
    
    def calculate_max_action_given_states(self, seq_output: torch.Tensor):
        # seq_output [batch, time, dim]
        full_q = self.forward_batch_seqs(
            seq_output=seq_output
        )  # [batch, time, #actions]
        return full_q.argmax(dim=2)  # [batch, time]


def symlog(t: torch.Tensor) -> torch.Tensor:
    return t.sign() * t.abs().log1p()

def symexp(t: torch.Tensor) -> torch.Tensor:
    return t.sign() * t.abs().expm1()

        
class AdmitQPairFFN(RLCommonFFN):
    def __init__(self, args):
        super().__init__(args)
        self.m1 = AdmitQFFN(args)
        self.m2 = AdmitQFFN(args)
        self.switch_steps = args.value_switch_steps
        
    def calculate_td_loss_value_adv(self, seq_output: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor, mask: torch.Tensor):
        # m1 to update, m2 to eval
        # compute q update
        a_max = self.m1.calculate_max_action_given_states(seq_output=seq_output)  # [batch, time]
        next_states = seq_output[:, 1:, :]  # [batch, time-1, dim]
        next_q_log_full = self.m2.forward_batch_seqs(
            seq_output=next_states,
        )  # [batch, time-1, #actions]
        next_q_log = next_q_log_full.gather(dim=2, index=a_max[:, 1:].unsqueeze(2)).squeeze(2)  # [batch, time-1]
        q_update_real = reward[:, :-1] + symexp(next_q_log) * self.m1.lambd
        q_update_real = torch.concat([q_update_real, reward.float()[:, -1].unsqueeze(1)], dim=1)
        q_update_real: torch.Tensor = q_update_real.detach()  # [batch, time]
        
        # compute q
        q_log_full = self.m1.forward_batch_seqs(
            seq_output=seq_output,
        )  # [batch, time, #actions]
        q_log = q_log_full.gather(dim=2, index=actions.unsqueeze(2)).squeeze(2)
        q_real = symexp(q_log)
        q_loss = nn.MSELoss()((q_log * mask), symlog(q_update_real) * mask)
        
        # compute v
        v_log = q_log_full.gather(dim=2, index=a_max.unsqueeze(2)).squeeze(2)
        v_real = symexp(v_log)
        
        # compute adv
        adv = q_real - v_real
        
        return q_loss, q_real, q_update_real, v_real, adv
    
    def switch(self):
        m = self.m1
        self.m1 = self.m2
        self.m2 = m


class CacheWorld(nn.Module):
    def __init__(self, args) -> None:
        super(CacheWorld, self).__init__()
        self.args = args
        # self.actor = ReplacePolicyFFN(args)
        # self.replace_actor = ReplacePolicyImitatorFFN(args)
        self.admit_actor = AdmitPolicyFFN(args)
        # self.prefetch_actor = PrefetchPolicyFNN(args)
        self.value = ValueFFNPair(args)
        self.admit_q = AdmitQPairFFN(args)
        self.world = BSARecModel(args)
        
        self.world_output_to_action_friendly = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.bitmap_predictor = nn.Linear(self.args.hidden_size, self.args.item_size+1)
        self.is_used_predictor = nn.Linear(self.args.hidden_size, 1)

        if len(args.world_pt) > 0:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.world.load_state_dict(torch.load(args.world_pt, map_location=self.device))

        self.E = self.world.item_embeddings
        self.lambd = args.discount_ratio
        self.eta = args.entropy_weight
        self.world_loss_weight = args.world_loss_weight
        
        init_linear_layer(module=self.bitmap_predictor)
        init_linear_layer(module=self.world_output_to_action_friendly)
        init_linear_layer(module=self.is_used_predictor)
        
    def forward_admit_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        # to get the cache state
        seq_output = self.world.forward(input_ids)
        return seq_output  # [batch, time, dim]
        
    def predict_bitmap_admit(self, input_ids: torch.Tensor):
        seq_output = self.forward_admit_state(input_ids=input_ids)
        x = self.bitmap_predictor(seq_output)
        bitmap_pred = F.sigmoid(x)
        return bitmap_pred, seq_output
    
    def predict_item_is_reused(self, seq_output: torch.Tensor, label: torch.Tensor):
        # seq_output [batch, time, dim]
        weight = self.is_used_predictor(seq_output).squeeze(2)
        pred = F.sigmoid(weight)
        if label is not None:
            loss = -(label * torch.log(pred + 1e-5) + (1 - label) * torch.log(1 - pred + 1e-5))
            return loss, pred
        return None, pred
    
    def calculate_bitmap_pred_admit_loss(self, input_ids: torch.Tensor, bitmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bitmap_pred, seq_output = self.predict_bitmap_admit(input_ids=input_ids)
        # bitmap [batch, time, #items]
        loss = bitmap_pred * bitmap + (1 - bitmap_pred) * (1 - bitmap)  # [batch, time]
        return (loss.mean(dim=2) * mask).sum(dim=1).mean(), seq_output, bitmap_pred

    def sample_actions(self, obs_history: torch.Tensor, obs: int, bitmap: torch.Tensor):
        bitmap = bitmap.to(self.device)
        world_input = torch.cat([obs_history, torch.as_tensor([obs], dtype=torch.long)], dim=0).to(self.device)
        # print(obs_history.shape, world_input.shape)
        seq_output = self.world.forward(world_input.unsqueeze(0), num_multisteps=1)
        E_k_t = self.E(torch.as_tensor([[obs]], dtype=torch.long, device=self.device))
        return self.actor.sample_u_t(hat_e_t=seq_output, e_k_t=E_k_t, E=self.E, bitmap=bitmap)[0][0].item()
    
    def sample_replace_imitator(self, input_ids: torch.Tensor, held_ids: torch.Tensor):
        seq_output = self.world.forward(input_ids.to(self.E.weight.device).unsqueeze(0))
        return self.replace_actor.sample_replace(seq_output, self.E, None).squeeze(0)
    
    def sample_admit_actions(self, input_ids: torch.Tensor):
        # input_ids [time]
        seq_output = self.world.forward(input_ids=input_ids.to(self.admit_q.m1.q_predictor.weight.device).unsqueeze(0))
        _, is_reused_prob = self.predict_item_is_reused(seq_output=seq_output, label=None)
        return self.admit_actor.sample_actions(seq_output=seq_output, is_reused_prob=is_reused_prob).squeeze(0)
    
    def sample_prefetch_actions(self, input_ids: torch.Tensor):
        # input_ids [time]
        seq_output = self.world.forward(input_ids.unsqueeze(0).to(self.E.weight.device))
        return self.prefetch_actor.sample_prefetch(seq_output, self.E).squeeze(0)
        

    def caculate_basics_old2(self, input_ids: torch.Tensor, answers: torch.Tensor, num_multisteps: int, cache_bitmap: torch.Tensor, world_sg: bool, num_mc_samples: int = 0):
        e_t = input_ids
        k_t = answers
        # forward world model
        seq_output = self.world.forward(e_t, num_multisteps=num_multisteps)
        # print(seq_output)
        hat_e_all_t = seq_output
        hat_e_t = hat_e_all_t[:, -num_multisteps:, :]
        E_weight = self.E.weight
        if world_sg:
            E_weight = E_weight.detach()
        p_all_t = F.softmax(torch.matmul(hat_e_all_t, E_weight.transpose(0, 1)), dim=-1)
        p_t = p_all_t[:, -num_multisteps:, :]
        # print(p_t.shape, p_t.shape[-1], k_t.shape)
        if world_sg:
            p_t = p_t.detach()
        world_loss = nn.CrossEntropyLoss()(p_t.reshape(-1, p_t.shape[-1]), k_t.reshape(-1))
        # [batch, time, #items] reserved ids accounted
        p_t = p_t[:, :, :self.args.item_size+1]  # now reserved ids are not accounted
        if num_mc_samples > 0:
            items_sampled = torch.multinomial(p_t.reshape(-1, p_t.shape[-1]), num_mc_samples).reshape(p_t.shape[0], p_t.shape[1], num_mc_samples)
            # [batch, time, #sampling]
        else:
            items_sampled = None

        # world model metrics
        def world_hr(k):
            return recall_at_k(
                predicted=p_t.reshape(-1, p_t.shape[-1]).detach().cpu().numpy(), 
                actual=k_t.reshape(-1).detach().cpu().numpy(), topk=k,
            )
        def world_ndcg(k):
            return ndcg_k(
                predicted=p_t.reshape(-1, p_t.shape[-1]).detach().cpu().numpy(), 
                actual=k_t.reshape(-1).detach().cpu().numpy(), topk=k,
            )
        hr5 = world_hr(5)
        hr10 = world_hr(10)
        ndcg5 = world_ndcg(5)
        ndcg10 = world_ndcg(10)

        # forward policy
        pi_t = self.actor.forward_all_e(hat_e_t, self.E, world_sg, items_sampled)  # sampling at t
        pi_n_t = self.actor.forward_all_e(hat_e_t[:, :-1, :], self.E, world_sg, items_sampled[:, 1:, :]) # sampling at t+1
        # needs this one to compute \sum_m p_{t+1}^m c_t^m \pi_t^{me}
        # pi [batch, time, #items/sampling k, #items m]
        # reserved ids are not accounted
        # pi_t^{mk}, m is the 2nd #items index, k is the 1st #items index

        # check shapes
        # print(pi_t.shape, p_t.shape)
        assert pi_t.shape[0] == p_t.shape[0]
        assert pi_t.shape[1] == p_t.shape[1]
        if num_mc_samples == 0:
            assert pi_t.shape[3] == p_t.shape[2]
            assert pi_t.shape[2] >= pi_t.shape[3]
        else:
            assert pi_t.shape[2] == num_mc_samples

        B = p_t.shape[0]

        c_list = list()  # [batch, time, #items], list indexed by time_step
        c_n_list = list()
        h_list = list()
        # this is actually c_{T-1}, not c_T
        # cache_bitmap: [batch, #items]
        c_list.append(cache_bitmap.float())
        c_n_list.append(cache_bitmap.float())
        # compute c_{t+1}, h_{t+1} from pi_{t+1}, p_t, c_t step by step
        for time_step in range(0, p_t.shape[1]-2):
            c_t = c_list[time_step]  # [batch, #items m]
            c_n_t = c_n_list[time_step]
            pi_tt = 1 - pi_t[:, time_step, :, :] # \pi_t^{mk} [batch, #items/sampling k, #items m] sampled at t+1
            pi_n_tt = 1 - pi_n_t[:, time_step, :, :] # \pi_{t+1}^{mk} [batch, #items/sampling k, #items m] sampled at t+1
            pi1_tt = 1 - pi_t[:, time_step+1, :, :] # \pi_{t+1}^{mk} [batch, #items/sampling k, #items m] sampled at t
            pi1_n_tt = 1 - pi_n_t[:, time_step+1, :, :] # \pi_{t+1}^{mk} [batch, #items/sampling k, #items m] sampled at t+1
            p_tt = p_t[:, time_step, :] # p_t^m  [batch, #items m] no sampling
            p1_tt = p_t[:, time_step+1, :] # p_{t+1}^m
            cur_items_sampled = items_sampled[:, time_step, :]

            # if num_mc_samples > 0:
            #     # num_mc_samples [batch, time, #sampling]
            #     cur_num_mc_samples = items_sampled[:, time_step, :]
            #     c_t = c_t.gather(dim=1, index=cur_num_mc_samples)
            #     cur_n_num_mc_samples = items_sampled[:, time_step+1, :]
            #     c_n_t = c_n_t.gather(dim=1, index=cur_n_num_mc_samples)
            x1 = p_tt * (1 - c_t)  # [batch, #items]

            if num_mc_samples == 0:
                x2 = torch.einsum("bkm,bk->bm", pi1_tt, p_tt)  # [batch, #items]
            else:
                # m_sampled = torch.multinomial(input=p_tt, num_samples=num_mc_samples, replacement=True).unsqueeze(2).expand(-1, -1, pi1_tt.shape[1])
                # use sampling results from t
                x2 = pi1_tt.mean(dim=1)
                x2_n = pi1_n_tt.mean(dim=1)
                # [batch, #sampling k]
            c_t1 = x2 * c_t + x1  # [batch, #items]
            c_n_t1 = x2_n * c_n_t + x1  # [batch, #items]
            # [batch, #items]
            c_list.append(c_t1)
            c_n_list.append(c_n_t1)
            
            if num_mc_samples == 0:
                x3 = torch.einsum("bm,bm,bem->be", p1_tt, c_t, pi_tt)  # [batch, #items]
            else:
                # m_sampled = torch.multinomial(input=p1_tt, num_samples=num_mc_samples, replacement=True)
                # use sampling results from t+1
                # [batch, #samples]
                x3 = torch.einsum("bm,bem->be", c_n_t, pi_n_tt) / (self.args.item_size + 1)  # [batch, #sampling]
            x4 = 1 - c_t.gather(dim=1, index=cur_items_sampled) + x3  # [batch, #sampling]
            if num_mc_samples == 0:
                h_t = torch.einsum("be,be->b", p_tt, x4)
            else:
                h_t = x4.mean(dim=1)  # [batch]
            h_list.append(h_t)
        # to tensor
        h_t = torch.stack(h_list, dim=0).transpose(0, 1)  # [batch, time]

        return (
            # basics
            hat_e_t, h_t, pi_t, world_loss,
        ), (
            # metrics
            hr5, hr10, ndcg5, ndcg10,
        )
    
    def calculate_policy_use(self, hat_e_t: torch.Tensor, held_items: torch.Tensor, requested_items: torch.Tensor):
        return self.actor.forward_use(hat_e_t=hat_e_t, E=self.E, held_items=held_items, requested_items=requested_items).detach()

    def caculate_basics(self, input_ids: torch.Tensor, answers: torch.Tensor, 
                        num_multisteps: int, cache_bitmap: torch.Tensor, 
                        last_action: torch.Tensor, world_sg: bool, 
                        num_mc_samples: int = 0):
        e_t = input_ids
        k_t = answers
        # forward world model
        seq_output = self.world.forward(torch.cat([e_t, k_t]), num_multisteps=num_multisteps)
        # print(seq_output)
        hat_e_all_t = seq_output  # [batch, time, hidden_dim]
        hat_e_t = hat_e_all_t[:, -num_multisteps:, :]
        input_hat_e_t = hat_e_all_t[:, :-num_multisteps, :]
        # print("caculate_basics", input_ids, input_ids.shape, num_multisteps, hat_e_all_t.shape, hat_e_t.shape, input_hat_e_t.shape)
        E_weight = self.E.weight
        if world_sg:
            E_weight = E_weight.detach()  # [#items, hidden_dim]
        p_all_t = F.softmax(torch.matmul(hat_e_all_t, E_weight.transpose(0, 1)), dim=-1)  # [batch, time, #items]
        p_t = p_all_t[:, -num_multisteps:, :]  # [batch, time, #items]
        # print(p_t.shape, p_t.shape[-1], k_t.shape)
        if world_sg:
            p_t = p_t.detach()
        world_loss = nn.CrossEntropyLoss()(p_t.reshape(-1, p_t.shape[-1]), k_t.reshape(-1))
        # [batch, time, #items] reserved ids accounted
        p_t = p_t[:, :, :self.args.item_size+1]  # now reserved ids are not accounted
        # print(p_t.isnan().long().sum(), (1 - p_t.isfinite().long()).sum(), (p_t < 0).long().sum())
        if num_mc_samples > 0:
            items_sampled = torch.multinomial(p_t.reshape(-1, p_t.shape[-1]), num_mc_samples).reshape(p_t.shape[0], p_t.shape[1], num_mc_samples)
            # [batch, time, #sampling]
        else:
            items_sampled = None

        # world model metrics
        def world_hr(k):
            return recall_at_k(
                predicted=p_t.reshape(-1, p_t.shape[-1]).detach().cpu().numpy(), 
                actual=k_t.reshape(-1).detach().cpu().numpy(), topk=k,
            )
        def world_ndcg(k):
            return ndcg_k(
                predicted=p_t.reshape(-1, p_t.shape[-1]).detach().cpu().numpy(), 
                actual=k_t.reshape(-1).detach().cpu().numpy(), topk=k,
            )
        hr5 = world_hr(5)
        hr10 = world_hr(10)
        ndcg5 = world_ndcg(5)
        ndcg10 = world_ndcg(10)

        # forward policy
        pi_t = self.actor.forward_all_e(hat_e_t, self.E, world_sg, items_sampled)  # sampling at t
        # with torch.eval():
        #     pi_t_full = self.actor.forward_all_e(hat_e_t, self.E, world_sg, None)  # sampling at t
        # needs this one to compute \sum_m p_{t+1}^m c_t^m \pi_t^{me}
        # pi [batch, time, #items/sampling k, #items m]
        # reserved ids are not accounted
        # pi_t^{mk}, m is the 2nd #items index, k is the 1st #items index

        # check shapes
        # print(pi_t.shape, p_t.shape)
        assert pi_t.shape[0] == p_t.shape[0]
        assert pi_t.shape[1] == p_t.shape[1]
        if num_mc_samples == 0:
            assert pi_t.shape[3] == p_t.shape[2]
            assert pi_t.shape[2] >= pi_t.shape[3]
        else:
            # print(pi_t.shape, num_mc_samples)
            assert pi_t.shape[2] == num_mc_samples

        # timestep 1 as initial state to start off the computation
        c_list = list()  # [batch, time, #items], list indexed by time_step
        h_list = list()  # [batch, time] start at 1
        # this is actually c_{T-1}, not c_T
        # cache_bitmap: [batch, #items]
        # c_0 = the cache bitmap
        # last_action [batch] -> [batch, #items]
        c_1 = cache_bitmap * F.one_hot(last_action, num_classes=self.args.item_size+1)
        c_1 += F.one_hot(e_t[:, -(num_multisteps+1)], num_classes=self.args.item_size+1) * (1 - cache_bitmap)
        c_list.append(c_1.float())
        # compute c_{t+1}, h_{t+1} from pi_{t+1}, p_t, c_t step by step
        for time_step in range(0, p_t.shape[1]-2):
            c_t = c_list[time_step]  # [batch, #items m]
            pi_tt = 1 - pi_t[:, time_step, :, :] # \pi_t^{mk} [batch, #items/sampling k, #items m] sampled at t+1
            p_tt = p_t[:, time_step, :] # p_t^m  [batch, #items m] no sampling

            # if num_mc_samples > 0:
            #     # num_mc_samples [batch, time, #sampling]
            #     cur_num_mc_samples = items_sampled[:, time_step, :]
            #     c_t = c_t.gather(dim=1, index=cur_num_mc_samples)
            #     cur_n_num_mc_samples = items_sampled[:, time_step+1, :]
            #     c_n_t = c_n_t.gather(dim=1, index=cur_n_num_mc_samples)
            x1 = p_tt * (1 - c_t)  # [batch, #items]

            if num_mc_samples == 0:
                x2 = torch.einsum("bkm,bk->bm", pi_tt, p_tt)  # [batch, #items]
            else:
                # m_sampled = torch.multinomial(input=p_tt, num_samples=num_mc_samples, replacement=True).unsqueeze(2).expand(-1, -1, pi1_tt.shape[1])
                # use sampling results from t
                x2 = pi_tt.mean(dim=1)
                # [batch, #sampling k]
            c_t1 = x2 * c_t + x1  # [batch, #items]
            # [batch, #items]
            c_list.append(c_t1)
            
            h_t = torch.einsum("bm,bm->b", p_tt, c_t)  # [batch, #items]
            h_list.append(h_t.unsqueeze(1))
        # to tensor
        h_t = torch.cat(h_list, dim=1)  # [batch, time]

        return (
            # basics
            hat_e_t, input_hat_e_t, h_t, pi_t, world_loss,
        ), (
            # metrics
            hr5, hr10, ndcg5, ndcg10,
        )
    
    def calculate_loss_future_steps_gradient(self, input_ids: torch.Tensor, answers: torch.Tensor, last_action: torch.Tensor, num_multisteps: int, cache_bitmap: torch.Tensor):
        (hat_e_t, _, h_t, pi_t, world_loss), metrics = self.caculate_basics(
            input_ids=input_ids, answers=answers,
            num_multisteps=num_multisteps,
            last_action=last_action,
            cache_bitmap=cache_bitmap,
            world_sg=self.args.world_sg,
            num_mc_samples=self.args.num_mc_samples,
        )
        # cache_not_contains = 1 - cache_bitmap
        # should not replace items not in cache
        # not_contains_loss = -torch.log(1 - pi_t * cache_not_contains.detach())

        actor_loss = -h_t.sum(dim=1).mean()
        total_loss = actor_loss + self.world_loss_weight * world_loss
        return (total_loss, actor_loss, world_loss), metrics, (hat_e_t, h_t, pi_t,)

    def calculate_loss_all_steps_gradient(
            self, input_ids: torch.Tensor, answers: torch.Tensor, 
            actions: torch.Tensor, rewards: torch.Tensor, 
            num_multisteps: int, cache_bitmap: torch.Tensor):
        (hat_e_t, input_hat_e_t, h_t, pi_t, world_loss), metrics = self.caculate_basics(
            input_ids=input_ids, answers=answers,
            num_multisteps=num_multisteps,
            last_action=actions[:, -1],
            cache_bitmap=cache_bitmap,
            world_sg=self.args.world_sg,
            num_mc_samples=self.args.num_mc_samples,
        )
        # print("calculate_loss_all_steps_gradient", input_hat_e_t.shape, actions.shape, input_ids.shape)
        input_pi_t = self.actor.calculate_policy_likelihoods(
            hat_e_t=input_hat_e_t, e_k_t=None, E=self.E,
            u_ids_t=actions,
        )  # [batch, time]
        # \log \pi_t^{u_t e_t} * r_t
        actor_loss = -(torch.log(input_pi_t) * (rewards - 0.5)).sum(dim=1).mean()
        return (actor_loss, actor_loss, world_loss,), metrics, (hat_e_t, h_t, pi_t,)
    
    def calculate_loss_replace_imitate_belady(
        self, input_ids: torch.Tensor,
        held_items: torch.Tensor,
        belady_actions: torch.Tensor,
        belady_labels: torch.Tensor,
        use_distances: torch.Tensor,
    ):
        seq_output = self.world.forward(input_ids)
        loss, imitate_loss, reuse_loss = self.replace_actor.calculate_imitate_loss(hat_e_t=seq_output, E=self.E, held_items=held_items, teacher_actions=belady_actions, belady_labels=belady_labels, logp1_use_distances=torch.log1p(use_distances))
        return loss, imitate_loss, reuse_loss
    
    def calculate_loss_admit_gradient(
        self, input_ids: torch.Tensor, 
        actions: torch.Tensor, rewards: torch.Tensor, 
    ):
        seq_output = self.forward_admit_state()
        # [batch, time, dim]
        admit_preds, ll = self.admit_actor.forward(seq_output, actions)
        actor_loss = -(torch.log(ll+1e-5) * (rewards - 0.5)).mean()
        return admit_preds, actor_loss
    
    def calculate_loss_admit_ppo_clip(
        self, input_ids: torch.Tensor, actions: torch.Tensor,
        rewards: torch.Tensor, bitmap: torch.Tensor, mask: torch.Tensor,
        is_load_action: torch.Tensor, reuse_distances: torch.Tensor,
    ):
        # actions [batch, time]
        # pad = torch.zeros((actions.shape[0], 1,), device=actions.device, dtype=torch.long)
        bitmap_loss, seq_output, bitmap_pred = self.calculate_bitmap_pred_admit_loss(input_ids=input_ids, bitmap=bitmap, mask=mask)
        
        is_reused_label = (1 - (reuse_distances == 0).long())
        is_reused_loss, is_reused_pred = self.predict_item_is_reused(seq_output=seq_output, label=is_reused_label)
        is_reused_loss = (is_reused_loss * mask).sum(dim=1).mean()
        is_reused_pred = is_reused_pred.detach()
        
        q_loss, q, q_update, v, adv = self.admit_q.calculate_td_loss_value_adv(
            seq_output=seq_output, actions=actions, 
            reward=rewards, mask=mask,
        )
        
        action_prob, action_not_admit_prob, inaction_admit_prob, entropy, action_likelihood, (admit_logits_0, admit_logits_1, admit_logits_inaction, is_reused_mask,) = self.admit_actor.forward(seq_output=seq_output, action=actions, is_load_action=is_load_action, is_reused_prob=is_reused_pred, reuse_prob_thresh=self.args.reuse_prob_thresh)
        
        r = (action_likelihood / (action_likelihood.detach() + 1e-5))[:, :-1]
        l1 = r * adv.detach()[:, :-1]
        l2 = adv.detach()[:, :-1] * torch.clip(
            input=r, min=1-self.args.ppo_epsilon, max=1+self.args.ppo_epsilon,
        )
        ppo_loss = -torch.min(
            input=torch.cat([l1.unsqueeze(0), l2.unsqueeze(0)], dim=0),
            dim=0,
        ).values * mask[:, :-1]
        ppo_loss = ppo_loss.sum(dim=1).mean()
        
        # uplift_admit_loss = -torch.log(action_prob * mask * is_load_action + 1e-5).sum(1).mean()
        # uplift_admit_loss = -(admit_logits_1 * mask * is_load_action * is_reused_mask).sum(1).mean()
        uplift_admit_loss = -((1 - action_prob)**2 / 2 * mask * is_load_action * is_reused_mask).sum(1).mean()
        
        loss = ppo_loss + q_loss - self.args.entropy_weight * (entropy * mask).mean() + bitmap_loss * self.args.bitmap_loss_weight + self.args.reuse_weight * is_reused_loss + self.args.uplift_admit_weight * uplift_admit_loss
        return (loss, ppo_loss, q_loss, entropy, bitmap_loss, is_reused_loss, uplift_admit_loss), (action_prob, action_not_admit_prob, inaction_admit_prob, action_likelihood, q, q_update, v, adv, bitmap_pred, is_reused_pred, is_reused_label), (admit_logits_0, admit_logits_1, admit_logits_inaction, is_reused_mask,)
    
    def calculate_loss_prefetch_gradient(
        self, input_ids: torch.Tensor,
        is_prefetch: torch.Tensor, prefetch_actions: torch.Tensor,
        rewards: torch.Tensor, mask: torch.Tensor,
    ):
        seq_output = self.world.forward(input_ids)
        prob = self.prefetch_actor.forward(seq_output, self.E)
        lh = self.prefetch_actor.likelihood_from_sample(prob, prefetch_actions)
        actor_loss = -(torch.log(lh+1e-5) * (rewards - 0.5) * is_prefetch * mask).mean()
        return prob, actor_loss
    
    def calculate_loss_prefetch_dueling_ac_gradient(
        self, input_ids: torch.Tensor,
        is_prefetch: torch.Tensor, prefetch_actions: torch.Tensor,
        rewards: torch.Tensor, mask: torch.Tensor,
    ):
        seq_output = self.world.forward(input_ids)
        # actor
        prob = self.prefetch_actor.forward(seq_output, self.E)
        # [batch, time, #items]
        lh = self.prefetch_actor.likelihood_from_sample(prob, prefetch_actions)  # [batch, time]
        no_prefetch_prob = prob[:, :, 0]  # [batch, time]
        no_prefetch_penalty = torch.float_power(F.relu(0.9 - no_prefetch_prob), 2).mean() / 2
        # critic
        rewards = rewards - prefetch_actions  # prefetch has a penalty
        critic_loss, adv, q, v, q_update = self.value.calculate_td_loss_dueling(seq_output, prob, rewards, prefetch_actions)
        # adv [batch, time, #items] -> [batch, time]
        adv = adv.gather(dim=2, index=prefetch_actions.unsqueeze(2)).squeeze(2)
        
        actor_loss = -(torch.log(lh+1e-5) * adv.detach() * is_prefetch * mask).mean()
        loss = actor_loss + critic_loss # + self.args.prefetch_penalty_weight * no_prefetch_penalty
        return prob, adv, q, v, q_update, loss, actor_loss, critic_loss, no_prefetch_prob, no_prefetch_penalty
    
    def calculate_loss_prefetch_ppo_clip_gradient(
        self, input_ids: torch.Tensor, prefetch_actions: torch.Tensor,
        rewards: torch.Tensor, mask: torch.Tensor,
    ):
        seq_output = self.world.forward(input_ids)
        # print("prefetch_actions", prefetch_actions.shape, prefetch_actions)
        # print("rewards", rewards)
        
        # actor
        prob, entropy = self.prefetch_actor.forward(seq_output, self.E)
        # [batch, time, #items]
        lh = self.prefetch_actor.likelihood_from_sample(prob, prefetch_actions)  # [batch, time]
        no_prefetch_prob = prob[:, :, 0]  # [batch, time]
        
        # critic
        critic_loss, adv, v, v_update = self.value.calculate_td_loss_value_adv(seq_output, rewards, mask)
        
        # ppo clip
        r = (lh / (lh.detach() + 1e-5))[:, :-1]  # [batch, time-1]
        l1 = r * adv.detach()
        l2 = adv.detach() * torch.clip(
            input=r, min=1-self.args.ppo_epsilon, max=1+self.args.ppo_epsilon,
        )
        ppo_loss = -torch.min(
            input=torch.concat([l1.unsqueeze(0), l2.unsqueeze(0),], dim=0),
            dim=0,
        ).values  # [batch, time-1]
        ppo_loss *= mask[:, :-1]  # don't account for the masked timesteps
        ppo_loss = ppo_loss.mean()
        
        loss = ppo_loss + critic_loss - self.args.entropy_weight * entropy.mean()  # [batch, time-1]
        return prob, adv, v, v_update, loss, ppo_loss, critic_loss, entropy, no_prefetch_prob

    def train_one_actor_critic(self, input_ids: torch.Tensor, num_multisteps: int, cache_bitmap: torch.Tensor):
        e_t = input_ids[: -num_multisteps]
        k_t = input_ids[-num_multisteps :]
        e_k_t = self.E(k_t)
        # forward world model, no grad what so ever
        seq_output = self.world.forward(e_t, num_multisteps=num_multisteps)
        E = seq_output.shape[2]
        hat_e_t = seq_output[:, -num_multisteps :, :].reshape(-1, E)
        item_emb = self.E.weight
        p_t = F.softmax(torch.matmul(hat_e_t, item_emb.transpose(0, 1)))
        # [batch, time, #items]
        p_t = p_t.detach()
        # forward policy
        pi_t = self.actor.forward_all_e(hat_e_t, self.E)
        # pi [batch, time, #items, #items]
        # the first #items dim has the reserved ids accounted
        # pi_t^{mk}, m is the 2nd #items index, k is the 1st #items index

        # check shapes
        assert pi_t.shape[0] == p_t.shape[0]
        assert pi_t.shape[1] == p_t.shape[1]
        assert pi_t.shape[2] >= pi_t.shape[3]
        assert pi_t.shape[3] == p_t.shape[2]

        B = p_t.shape[0]

        c_list = list()  # [batch, time, #items], list indexed by time_step
        h_list = list()
        # this is actually c_{T-1}, not c_T
        c_list.append(cache_bitmap.repeat(B, 1))
        # compute c_{t+1}, h_{t+1} from pi_{t+1}, p_t, c_t step by step
        for time_step in range(0, p_t.shape[1]-1):
            c_t = c_list[time_step]
            pi_tt = 1 - pi_t[:, time_step, :, :] # \pi_t^{mk}
            pi1_tt = 1 - pi_t[:, time_step+1, :, :] # \pi_{t+1}^{mk}
            p_tt = p_t[:, time_step, :] # p_t^m
            p1_tt = p_t[:, time_step+1, :] # p_{t+1}^m
            x1 = p_tt * (1 - c_t)
            x2 = torch.einsum("bkm,bk->bm", pi1_tt, p_tt)
            c_t1 = x2 * c_t + x1
            # [batch, #items]
            c_list.append(c_t1)
            
            x3 = 1 - c_t + torch.einsum("bm,bm,bme->be", p1_tt, c_t, pi_tt)
            h_t = torch.einsum("be,be->b", p_tt, x3)
            h_list.append(h_t)
        # to tensor
        h_t = torch.cat(h_list, dim=0)
        # compute advantage A
        A = self.critic.calculate_A(
            hat_e_1=hat_e_t[:, 0, :], hat_e_N=hat_e_t[:, -1, :],
            h_t=h_t, lambd=self.lambd,
        )
        A_normed = A / num_multisteps
        # act
        u_ids_t = self.actor.sample_u_t(hat_e_t=hat_e_t, e_k_t=e_k_t, E=self.E)
        # compute policy log term
        pi_likes = self.actor.calculate_policy_likelihoods(hat_e_t=hat_e_t, e_k_t=e_k_t, E=self.E, u_ids_t=u_ids_t)
        pi_likes = torch.log(pi_likes)
        policy_log_like = A_normed.detach() * pi_likes.sum(dim=1)
        # compute entropy
        # [batch, N_l, #items 1, #items 2] -> [batch, N_l, #items 1]
        # compute for each time step's prediction
        # then select with trajectory
        policy_entropy = torch.einsum("btmk->btm", pi_t * torch.log(pi_t))
        policy_entropy = policy_entropy.gather(dim=2, index=u_ids_t.unsqueeze(2))
        # [batch, N_l]

        # actor loss
        actor_loss = (- policy_log_like + self.eta * policy_entropy).sum(dim=1)

        # critic loss
        hat_H = self.critic.calculate_td_update(
            hat_e_N=hat_e_t[:, -1, :],
            h_t=h_t, lambd=self.lambd,
        )
        H_pred = self.critic.forward(hat_e_t=hat_e_t[:, 0, :])
        critic_loss = F.mse_loss(H_pred, hat_H.detach())

        # total loss
        loss = (actor_loss + critic_loss)

        return loss


def get_cache_world_model(args):
    model = CacheWorld(args)
    if len(args.preload_rl_pt) > 0:
        cuda_condition = torch.cuda.is_available() and not args.no_cuda
        print("Loading model from path", args.preload_rl_pt)
        model.load_state_dict(torch.load(args.preload_rl_pt, map_location=torch.device("cuda" if cuda_condition else "cpu")))
    return model
