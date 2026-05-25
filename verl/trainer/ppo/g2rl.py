import numpy as np
import torch


def _group_indices(index):
    groups = {}
    for i, uid in enumerate(index):
        groups.setdefault(uid, []).append(i)
    return groups


def _group_indices_tensor(index, device):
    groups = _group_indices(index)
    return {k: torch.tensor(v, device=device, dtype=torch.long) for k, v in groups.items()}


def minmax_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def project_reason_scores(
    reason_hs_feat: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 1e-8,
):
    direction = direction.to(reason_hs_feat.device, dtype=reason_hs_feat.dtype)
    direction = direction / (direction.norm() + eps)
    return reason_hs_feat @ direction


def build_reason_mem_masks(
    reason_scores: torch.Tensor,
    index: np.ndarray,
):
    """
    0-threshold split without fallback:
      score > 0   -> reason
      score <= 0  -> mem
    """
    device = reason_scores.device
    B = reason_scores.size(0)

    reason_mask = torch.zeros(B, device=device, dtype=torch.bool)
    mem_mask = torch.zeros(B, device=device, dtype=torch.bool)

    groups = _group_indices(index)

    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        g_scores = reason_scores[ids_t]

        g_reason = g_scores > 0
        g_mem = ~g_reason

        reason_mask[ids_t] = g_reason
        mem_mask[ids_t] = g_mem

    return reason_mask, mem_mask


def compute_g2rl_group_scores(
    seq_feat: torch.Tensor,       # [B, H]
    scalar_rewards: torch.Tensor, # [B]
    index: np.ndarray,
    eps: float = 1e-6,
):
    """
    Original G2RL group-wise nu computed over the provided grouping index.
    Used for non-reason-subgroup branch (and for backward compatibility).
    """
    device = seq_feat.device
    B = seq_feat.size(0)
    nu = torch.zeros(B, device=device, dtype=seq_feat.dtype)

    feat = seq_feat / (seq_feat.norm(dim=-1, keepdim=True) + eps)
    groups = _group_indices(index)

    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        g_feat = feat[ids_t]
        g_r = scalar_rewards[ids_t]

        if len(ids) <= 1:
            nu[ids_t] = 0.0
            continue

        sim = g_feat @ g_feat.T
        sim2 = sim.pow(2)

        reward_logits = torch.exp(g_r)
        m = len(ids)
        w = reward_logits.unsqueeze(0).repeat(m, 1)
        eye = torch.eye(m, device=device, dtype=torch.bool)
        w = w.masked_fill(eye, 0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + eps)

        explained = (w * sim2).sum(dim=-1)
        g_nu = torch.sqrt(torch.clamp(1.0 - explained, min=0.0))
        nu[ids_t] = g_nu

    return nu


def shape_g2rl_rewards(
    scalar_rewards: torch.Tensor,
    nu: torch.Tensor,
    index: np.ndarray,
    lambda_pos: float = 0.5,
    lambda_neg: float = 0.5,
    reward_clip: float = 3.0,
    eps: float = 1e-6,
):
    """Original G2RL shaping (no reason subgroup)."""
    shaped = scalar_rewards.clone()
    nu_bar = torch.zeros_like(nu)

    groups = _group_indices(index)
    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=nu.device, dtype=torch.long)
        g_nu = nu[ids_t]

        g_min = g_nu.min()
        g_max = g_nu.max()

        if torch.allclose(g_max, g_min):
            nu_bar[ids_t] = 0.0
        else:
            nu_bar[ids_t] = (g_nu - g_min) / (g_max - g_min + eps)

    pos_mask = scalar_rewards > 0
    lambda_tensor = torch.full_like(scalar_rewards, lambda_neg)
    lambda_tensor[pos_mask] = lambda_pos

    shaped = scalar_rewards * (1.0 + lambda_tensor * scalar_rewards * nu_bar)

    shaped = shaped.clamp(min=-reward_clip, max=reward_clip)
    return shaped, nu_bar


def compute_nu_in_reason_subgroup(
    seq_feat: torch.Tensor,
    scalar_rewards: torch.Tensor,
    index: np.ndarray,
    reason_mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Compute ν for each R sample using ONLY the R subgroup as comparison set.
    Same semantics as reason1_match's reason-subgroup branch.
    Returns nu_reason_raw [B], with values only at R positions (others are 0).
    """
    device = seq_feat.device
    feat = seq_feat / (seq_feat.norm(dim=-1, keepdim=True) + eps)
    nu_reason_raw = torch.zeros_like(scalar_rewards)

    groups = _group_indices(index)
    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        reason_ids_t = ids_t[reason_mask[ids_t]]

        if reason_ids_t.numel() <= 1:
            continue

        g_feat = feat[reason_ids_t]
        g_r = scalar_rewards[reason_ids_t]

        sim = g_feat @ g_feat.T
        sim2 = sim.pow(2)

        reward_logits = torch.exp(g_r)
        m = reason_ids_t.numel()
        w = reward_logits.unsqueeze(0).repeat(m, 1)
        eye = torch.eye(m, device=device, dtype=torch.bool)
        w = w.masked_fill(eye, 0.0)
        w = w / (w.sum(dim=-1, keepdim=True) + eps)

        explained = (w * sim2).sum(dim=-1)
        g_nu_reason = torch.sqrt(torch.clamp(1.0 - explained, min=0.0))

        nu_reason_raw[reason_ids_t] = g_nu_reason

    return nu_reason_raw


def compute_nu_M_against_R(
    seq_feat: torch.Tensor,
    scalar_rewards: torch.Tensor,
    index: np.ndarray,
    reason_mask: torch.Tensor,
    mem_mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    reason4_match key change vs reason1_match:
    For each M sample, compute ν using ONLY the R subgroup as comparison set.
    Reward weighting is over R samples; no self-mask needed (M_i is not in R).

    Semantically: ν_M[i] = "M sample i's distance from the reward-weighted
    reasoning subgroup" — pure cross-subgroup signal, no M-M noise.

    Returns nu_mem_raw [B], with values only at M positions (others are 0).
    Stays 0 if R or M subgroup is empty in a given prompt's group.
    """
    device = seq_feat.device
    feat = seq_feat / (seq_feat.norm(dim=-1, keepdim=True) + eps)
    nu_mem_raw = torch.zeros_like(scalar_rewards)

    groups = _group_indices(index)
    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        R_ids = ids_t[reason_mask[ids_t]]
        M_ids = ids_t[mem_mask[ids_t]]

        if R_ids.numel() == 0 or M_ids.numel() == 0:
            # one subgroup is empty — disable mem-diversity for this prompt
            continue

        M_feat = feat[M_ids]                              # [n_M, H]
        R_feat = feat[R_ids]                              # [n_R, H]
        R_r    = scalar_rewards[R_ids]                    # [n_R]

        sim2 = (M_feat @ R_feat.T).pow(2)                 # [n_M, n_R]

        # reward weighting on the R side; M_i not in R, no self-similarity
        w = torch.exp(R_r).unsqueeze(0).expand(M_ids.numel(), -1)   # [n_M, n_R]
        w = w / (w.sum(dim=-1, keepdim=True) + eps)

        explained = (w * sim2).sum(dim=-1)                # [n_M]
        nu_mem_raw[M_ids] = torch.sqrt(torch.clamp(1.0 - explained, min=0.0))

    return nu_mem_raw


def compute_reason_aware_nu(
    seq_feat: torch.Tensor,
    scalar_rewards: torch.Tensor,
    index: np.ndarray,
    reason_scores: torch.Tensor,
    eps: float = 1e-6,
):
    """
    reason4_match returns:
        nu_all        : [B], full-group ν (kept for metric compatibility)
        nu_reason_raw : [B], R-subgroup ν (only valid on reason positions)
        nu_mem_raw    : [B], M-vs-R cross-subgroup ν (only valid on mem positions) <- new
        reason_mask   : [B] bool
        mem_mask      : [B] bool
    """
    nu_all = compute_g2rl_group_scores(
        seq_feat=seq_feat,
        scalar_rewards=scalar_rewards,
        index=index,
        eps=eps,
    )

    reason_mask, mem_mask = build_reason_mem_masks(
        reason_scores=reason_scores,
        index=index,
    )

    nu_reason_raw = compute_nu_in_reason_subgroup(
        seq_feat=seq_feat,
        scalar_rewards=scalar_rewards,
        index=index,
        reason_mask=reason_mask,
        eps=eps,
    )

    nu_mem_raw = compute_nu_M_against_R(
        seq_feat=seq_feat,
        scalar_rewards=scalar_rewards,
        index=index,
        reason_mask=reason_mask,
        mem_mask=mem_mask,
        eps=eps,
    )

    return nu_all, nu_reason_raw, nu_mem_raw, reason_mask, mem_mask


def shape_reason_aware_rewards(
    scalar_rewards: torch.Tensor,
    nu_all: torch.Tensor,            # kept for metric compat, not used in shaping
    nu_reason_raw: torch.Tensor,
    nu_mem_raw: torch.Tensor,        # new for reason4_match
    index: np.ndarray,
    reason_mask: torch.Tensor,
    mem_mask: torch.Tensor,
    lambda_pos: float = 0.5,
    lambda_neg: float = 0.5,
    reward_clip: float = 3.0,
    eps: float = 1e-6,
):
    """
    reason4_match shaping:
      reason: ν_R (R-subgroup-internal), min-max within R subgroup     [unchanged]
      mem   : ν_M (M-vs-R cross-subgroup), min-max within M subgroup   [changed]

    vs reason1_match:
      reason1_match used nu_all for M and normalized over the WHOLE group,
      which (a) injected M-M similarity noise into ν, and (b) compressed
      ν̄_M when R had outlier ν values. reason4_match: ν is computed M-vs-R
      only (pure cross-subgroup), and min-max is performed within M only.
    """
    shaped = scalar_rewards.clone()

    # final raw nu actually used by each sample
    nu_final = torch.zeros_like(scalar_rewards)
    nu_final[reason_mask] = nu_reason_raw[reason_mask]
    nu_final[mem_mask] = nu_mem_raw[mem_mask]

    # final normalized nu used in shaping
    nu_bar = torch.zeros_like(nu_final)

    groups = _group_indices(index)
    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=scalar_rewards.device, dtype=torch.long)

        # ---- mem: normalize WITHIN mem subgroup using nu_mem_raw ----
        g_mem_ids = ids_t[mem_mask[ids_t]]
        if g_mem_ids.numel() > 1:
            g_mem_nu = nu_mem_raw[g_mem_ids]
            m_min = g_mem_nu.min()
            m_max = g_mem_nu.max()
            if torch.allclose(m_max, m_min):
                nu_bar[g_mem_ids] = 0.0
            else:
                nu_bar[g_mem_ids] = (g_mem_nu - m_min) / (m_max - m_min + eps)
        # else: mem subgroup size <= 1, nu_bar stays 0

        # ---- reason: normalize within reason subgroup using nu_reason_raw ----
        g_reason_ids = ids_t[reason_mask[ids_t]]
        if g_reason_ids.numel() > 0:
            g_reason_nu = nu_reason_raw[g_reason_ids]
            r_min = g_reason_nu.min()
            r_max = g_reason_nu.max()

            if torch.allclose(r_max, r_min):
                nu_bar[g_reason_ids] = 0.0
            else:
                nu_bar[g_reason_ids] = (g_reason_nu - r_min) / (r_max - r_min + eps)

    pos_mask = scalar_rewards > 0
    lambda_tensor = torch.full_like(scalar_rewards, lambda_neg)
    lambda_tensor[pos_mask] = lambda_pos

    shaped = scalar_rewards * (1.0 + lambda_tensor * scalar_rewards * nu_bar)
    shaped = shaped.clamp(min=-reward_clip, max=reward_clip)

    return shaped, nu_final, nu_bar


def _compute_group_reason_ratio(reason_mask: torch.Tensor, index: np.ndarray) -> float:
    groups = _group_indices(index)
    ratios = []
    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=reason_mask.device, dtype=torch.long)
        ratios.append(reason_mask[ids_t].float().mean())
    if len(ratios) == 0:
        return 0.0
    return torch.stack(ratios).mean().item()


def _compute_group_mem_ratio(mem_mask: torch.Tensor, index: np.ndarray) -> float:
    groups = _group_indices(index)
    ratios = []
    for _, ids in groups.items():
        ids_t = torch.tensor(ids, device=mem_mask.device, dtype=torch.long)
        ratios.append(mem_mask[ids_t].float().mean())
    if len(ratios) == 0:
        return 0.0
    return torch.stack(ratios).mean().item()


def apply_g2rl_reward_shaping(
    token_level_rewards: torch.Tensor,   # [B, T]
    response_mask: torch.Tensor,         # [B, T]
    seq_feat: torch.Tensor,              # [B, H]
    index: np.ndarray,
    reason_hs_feat: torch.Tensor | None = None,
    reason_direction: torch.Tensor | None = None,
    use_reason_subgroup: bool = False,
    lambda_pos: float = 0.5,
    lambda_neg: float = 0.5,
    reward_clip: float = 3.0,
    eps: float = 1e-6,
):
    """
    reason4_match: same external API as reason1_match. Internal change:
    when use_reason_subgroup=True, ν̄_M comes from M-vs-R cross-subgroup ν
    (instead of reason1_match's whole-group nu_all + whole-group min-max).
    """
    scalar_rewards = token_level_rewards.sum(dim=-1)   # [B]

    print("[DEBUG] use_reason_subgroup =", use_reason_subgroup)
    print("[DEBUG] reason_hs_feat is None =", reason_hs_feat is None)
    print("[DEBUG] reason_direction is None =", reason_direction is None)

    if use_reason_subgroup:
        assert reason_hs_feat is not None, (
            "reason_hs_feat is required when use_reason_subgroup=True"
        )
        assert reason_direction is not None, (
            "reason_direction is required when use_reason_subgroup=True"
        )

        reason_scores = project_reason_scores(
            reason_hs_feat=reason_hs_feat,
            direction=reason_direction,
            eps=eps,
        )

        nu_all, nu_reason_raw, nu_mem_raw, reason_mask, mem_mask = compute_reason_aware_nu(
            seq_feat=seq_feat,
            scalar_rewards=scalar_rewards,
            index=index,
            reason_scores=reason_scores,
            eps=eps,
        )

        shaped_scalar, nu_final, nu_bar = shape_reason_aware_rewards(
            scalar_rewards=scalar_rewards,
            nu_all=nu_all,
            nu_reason_raw=nu_reason_raw,
            nu_mem_raw=nu_mem_raw,
            index=index,
            reason_mask=reason_mask,
            mem_mask=mem_mask,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            reward_clip=reward_clip,
            eps=eps,
        )

        nu_for_debug = nu_final

        print("[REASON4 G2RL] reason_ratio_batch =", reason_mask.float().mean().item())
        print("[REASON4 G2RL] reason_ratio_group =", _compute_group_reason_ratio(reason_mask, index))
        print("[REASON4 G2RL] mem_ratio_batch =", mem_mask.float().mean().item())
        print("[REASON4 G2RL] mem_ratio_group =", _compute_group_mem_ratio(mem_mask, index))
        print("[REASON4 G2RL] reason_score_mean =", reason_scores.mean().item())
        print("[REASON4 G2RL] nu_all_mean =", nu_all.mean().item())
        print(
            "[REASON4 G2RL] nu_reason_mean =",
            nu_reason_raw[reason_mask].mean().item() if reason_mask.any() else 0.0,
        )
        print(
            "[REASON4 G2RL] nu_mem_vsR_mean =",
            nu_mem_raw[mem_mask].mean().item() if mem_mask.any() else 0.0,
        )
        print("[REASON4 G2RL] nu_final_mean =", nu_final.mean().item())

    else:
        nu = compute_g2rl_group_scores(
            seq_feat=seq_feat,
            scalar_rewards=scalar_rewards,
            index=index,
            eps=eps,
        )

        shaped_scalar, nu_bar = shape_g2rl_rewards(
            scalar_rewards=scalar_rewards,
            nu=nu,
            index=index,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            reward_clip=reward_clip,
            eps=eps,
        )

        nu_for_debug = nu

    debug_k = min(8, scalar_rewards.size(0))
    delta = shaped_scalar - scalar_rewards
    print("[G2RL sample debug] uid:", list(index[:debug_k]))
    print("[G2RL sample debug] raw:", scalar_rewards[:debug_k].detach().cpu().tolist())
    print("[G2RL sample debug] nu:", nu_for_debug[:debug_k].detach().cpu().tolist())
    print("[G2RL sample debug] nu_bar:", nu_bar[:debug_k].detach().cpu().tolist())
    print("[G2RL sample debug] shaped:", shaped_scalar[:debug_k].detach().cpu().tolist())
    print("[G2RL sample debug] delta:", delta[:debug_k].detach().cpu().tolist())

    new_rewards = torch.zeros_like(token_level_rewards)
    last_idx = response_mask.long().sum(dim=-1).clamp(min=1) - 1
    batch_idx = torch.arange(token_level_rewards.size(0), device=token_level_rewards.device)
    new_rewards[batch_idx, last_idx] = shaped_scalar

    if use_reason_subgroup:
        metrics = {
            "g2rl/reward_raw_mean": scalar_rewards.mean().item(),
            "g2rl/reward_shaped_mean": shaped_scalar.mean().item(),
            "g2rl/nu_all_mean": nu_all.mean().item(),
            "g2rl/nu_final_mean": nu_final.mean().item(),
            "g2rl/nu_bar_mean": nu_bar.mean().item(),
            "g2rl/reason_ratio_batch": reason_mask.float().mean().item(),
            "g2rl/mem_ratio_batch": mem_mask.float().mean().item(),
            "g2rl/reason_ratio_group": _compute_group_reason_ratio(reason_mask, index),
            "g2rl/mem_ratio_group": _compute_group_mem_ratio(mem_mask, index),
            "g2rl/num_reason": reason_mask.sum().item(),
            "g2rl/num_mem": mem_mask.sum().item(),
            "g2rl/reason_score_mean": reason_scores.mean().item(),
            "g2rl/nu_reason_mean": (
                nu_reason_raw[reason_mask].mean().item() if reason_mask.any() else 0.0
            ),
            # reason4_match: replaces reason1_match's "nu_mem_mean" (which was
            # nu_all[mem_mask]) with the cross-subgroup quantity actually used
            "g2rl/nu_mem_vsR_mean": (
                nu_mem_raw[mem_mask].mean().item() if mem_mask.any() else 0.0
            ),
        }
    else:
        metrics = {
            "g2rl/reward_raw_mean": scalar_rewards.mean().item(),
            "g2rl/reward_shaped_mean": shaped_scalar.mean().item(),
            "g2rl/nu_mean": nu.mean().item(),
            "g2rl/nu_bar_mean": nu_bar.mean().item(),
        }

    return new_rewards, metrics
