"""
grpo_utils.py

Data structures and group-relative advantage computation for GRPO (critic-free RL) fine-tuning
of OpenVLA on LIBERO. See PROJECT_PLAN.md Phase 2 / the approved GRPO plan for the full design
rationale (why trajectory-level advantage, why no value head, why sparse binary reward).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels) -- same convention as prismatic/vla/datasets/datasets.py
IGNORE_INDEX = -100


@dataclass
class Step:
    """One queried (VLA-inference) env step within a rollout, with enough info to recompute log-probs later."""

    input_ids: torch.Tensor  # (prompt_len,) -- prompt token ids, including the trailing empty-string token (29871)
    pixel_values: torch.Tensor  # (C, H, W) -- image tensor as produced by the processor's image transform
    action_token_ids: torch.Tensor  # (7,) -- sampled action token ids
    old_logp: torch.Tensor  # (7,) -- log-prob of each sampled action token under the policy that sampled it


@dataclass
class Trajectory:
    """One full episode rollout."""

    task_id: int
    task_description: str
    init_state_idx: int
    success: bool
    steps: List[Step] = field(default_factory=list)
    advantage: Optional[float] = None  # filled in by compute_group_advantages


def compute_group_advantages(rewards: List[float], eps: float = 1e-4) -> Optional[List[float]]:
    """
    Group-relative advantage from sparse binary episode rewards (GRPO, no value head/critic).

    Returns None if the group is degenerate (all-success or all-fail -- zero variance means zero
    gradient signal, this is expected under sparse reward and should be logged/skipped, not treated
    as an error).
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    std = rewards_arr.std()
    if std < eps:
        return None
    mean = rewards_arr.mean()
    return ((rewards_arr - mean) / (std + eps)).tolist()


def assign_advantages(trajectories: List[Trajectory]) -> List[Trajectory]:
    """Computes group advantage over `trajectories` (assumed to be one group, e.g. same task+init_state) and
    assigns `.advantage` in place. Returns the same list for convenience; degenerate groups get advantage=None
    on every trajectory (caller should skip these when building the training batch)."""
    rewards = [1.0 if t.success else 0.0 for t in trajectories]
    advantages = compute_group_advantages(rewards)
    for i, t in enumerate(trajectories):
        t.advantage = None if advantages is None else advantages[i]
    return trajectories
