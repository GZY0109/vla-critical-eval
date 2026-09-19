"""Phase 4 收尾：把 Phase 2 的 GRPOEVAL 原始 jsonl 变成可复现的对比数据 + 图表。

输入: openvla/experiments/logs/GRPOEVAL-*--sft_baseline_recheck.jsonl /
      openvla/experiments/logs/GRPOEVAL-*--grpo_real_run_v1_checkpoint.jsonl
输出: results/phase2_grpo_vs_sft.json, results/phase2_grpo_vs_sft.png

跟 PROGRESS.md 里手算的数字独立核对一遍，不是照抄。
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"

SFT_LOG = LOG_DIR / "GRPOEVAL-libero_spatial-2026_09_18-12_29_00--sft_baseline_recheck.jsonl"
GRPO_LOG = LOG_DIR / "GRPOEVAL-libero_spatial-2026_09_18-14_42_40--grpo_real_run_v1_checkpoint.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def episode_mean_entropy(ep):
    per_step = ep["per_step_entropy"]
    step_means = [statistics.mean(step) for step in per_step]
    return statistics.mean(step_means)


def summarize(episodes, label):
    n = len(episodes)
    n_success = sum(1 for e in episodes if e["success"])
    entropies = [episode_mean_entropy(e) for e in episodes]
    steps = [e["num_steps"] for e in episodes]
    return {
        "label": label,
        "n": n,
        "n_success": n_success,
        "success_rate": n_success / n,
        "entropy_mean": statistics.mean(entropies),
        "entropy_median": statistics.median(entropies),
        "num_steps_mean": statistics.mean(steps),
    }


def pair_key(ep):
    return (ep["task_id"], ep["init_state_idx"])


def paired_flip_analysis(sft_eps, grpo_eps):
    sft_by_key = {pair_key(e): e for e in sft_eps}
    grpo_by_key = {pair_key(e): e for e in grpo_eps}
    common_keys = sft_by_key.keys() & grpo_by_key.keys()
    both_success = both_fail = sft_only = grpo_only = 0
    for k in common_keys:
        s = sft_by_key[k]["success"]
        g = grpo_by_key[k]["success"]
        if s and g:
            both_success += 1
        elif not s and not g:
            both_fail += 1
        elif s and not g:
            sft_only += 1
        else:
            grpo_only += 1
    return {
        "n_paired": len(common_keys),
        "both_success": both_success,
        "both_fail": both_fail,
        "sft_success_grpo_fail": sft_only,
        "grpo_success_sft_fail": grpo_only,
        "n_flipped": sft_only + grpo_only,
    }


def main():
    sft_eps = load(SFT_LOG)
    grpo_eps = load(GRPO_LOG)

    sft_summary = summarize(sft_eps, "SFT-only")
    grpo_summary = summarize(grpo_eps, "SFT+GRPO")
    flips = paired_flip_analysis(sft_eps, grpo_eps)

    result = {
        "sft": sft_summary,
        "grpo": grpo_summary,
        "paired_flip_analysis": flips,
        "source_logs": {
            "sft": str(SFT_LOG.name),
            "grpo": str(GRPO_LOG.name),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "phase2_grpo_vs_sft.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_json}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    labels = ["SFT-only", "SFT+GRPO"]
    success_rates = [sft_summary["success_rate"] * 100, grpo_summary["success_rate"] * 100]
    axes[0].bar(labels, success_rates, color=["#4c72b0", "#dd8452"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_title(f"success rate (n={sft_summary['n']} each)")
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center")

    entropy_means = [sft_summary["entropy_mean"], grpo_summary["entropy_mean"]]
    axes[1].bar(labels, entropy_means, color=["#4c72b0", "#dd8452"])
    axes[1].set_ylabel("mean per-step action entropy")
    axes[1].set_title("action entropy (episode-mean of step-mean)")
    for i, v in enumerate(entropy_means):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center")

    flip_labels = ["both\nsuccess", "both\nfail", "SFT wins\n(GRPO fails)", "GRPO wins\n(SFT fails)"]
    flip_values = [
        flips["both_success"],
        flips["both_fail"],
        flips["sft_success_grpo_fail"],
        flips["grpo_success_sft_fail"],
    ]
    axes[2].bar(flip_labels, flip_values, color=["#55a868", "#c44e52", "#4c72b0", "#dd8452"])
    axes[2].set_ylabel("# episodes")
    axes[2].set_title(f"paired outcome (n={flips['n_paired']} pairs)")
    for i, v in enumerate(flip_values):
        axes[2].text(i, v + 0.5, str(v), ha="center")
    plt.setp(axes[2].get_xticklabels(), fontsize=8)

    fig.suptitle("Phase 2: SFT-only vs SFT+GRPO (libero_spatial, n=120 each)")
    fig.tight_layout()
    out_png = RESULTS_DIR / "phase2_grpo_vs_sft.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
