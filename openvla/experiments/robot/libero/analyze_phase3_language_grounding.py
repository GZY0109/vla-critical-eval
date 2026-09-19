"""Phase 4 收尾：把 Phase 3 轴一（语言接地反事实测试）的原始 jsonl 变成对比数据 + 图表。

输入: openvla/experiments/logs/LANGGROUND-*--sft_language_grounding.jsonl
      openvla/experiments/logs/LANGGROUND-*--grpo_language_grounding.jsonl
输出: results/phase3_language_grounding.json, results/phase3_language_grounding.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"

SFT_LOG = LOG_DIR / "LANGGROUND-libero_spatial-2026_09_18-21_33_02--sft_language_grounding.jsonl"
GRPO_LOG = LOG_DIR / "LANGGROUND-libero_spatial-2026_09_18-21_44_31--grpo_language_grounding.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def pair_key(ep):
    return (ep["scene_task_id"], ep["scene_init_idx"], ep["instr_task_id"])


def main():
    sft_eps = load(SFT_LOG)
    grpo_eps = load(GRPO_LOG)

    sft_correct = sum(1 for e in sft_eps if e["proxy_correct"])
    grpo_correct = sum(1 for e in grpo_eps if e["proxy_correct"])
    sft_native_success = sum(1 for e in sft_eps if e["native_env_success"])
    grpo_native_success = sum(1 for e in grpo_eps if e["native_env_success"])

    sft_by_key = {pair_key(e): e for e in sft_eps}
    grpo_by_key = {pair_key(e): e for e in grpo_eps}
    common_keys = sft_by_key.keys() & grpo_by_key.keys()

    per_pair = []
    n_agree = 0
    for k in sorted(common_keys):
        s = sft_by_key[k]
        g = grpo_by_key[k]
        agree = s["proxy_correct"] == g["proxy_correct"]
        n_agree += agree
        per_pair.append({
            "scene_task_id": k[0],
            "scene_init_idx": k[1],
            "instr_task_id": k[2],
            "sft_proxy_correct": s["proxy_correct"],
            "grpo_proxy_correct": g["proxy_correct"],
            "agree": agree,
        })

    result = {
        "sft": {
            "n": len(sft_eps),
            "n_correct": sft_correct,
            "proxy_accuracy": sft_correct / len(sft_eps),
            "n_native_env_success": sft_native_success,
        },
        "grpo": {
            "n": len(grpo_eps),
            "n_correct": grpo_correct,
            "proxy_accuracy": grpo_correct / len(grpo_eps),
            "n_native_env_success": grpo_native_success,
        },
        "n_paired": len(common_keys),
        "n_agree": n_agree,
        "n_disagree": len(common_keys) - n_agree,
        "per_pair": per_pair,
        "source_logs": {"sft": SFT_LOG.name, "grpo": GRPO_LOG.name},
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "phase3_language_grounding.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_json}")
    print(json.dumps({k: v for k, v in result.items() if k != "per_pair"}, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    idx = list(range(len(per_pair)))
    sft_vals = [1 if p["sft_proxy_correct"] else 0 for p in per_pair]
    grpo_vals = [1 if p["grpo_proxy_correct"] else 0 for p in per_pair]
    width = 0.35
    ax.bar([i - width / 2 for i in idx], sft_vals, width, label="SFT-only", color="#4c72b0")
    ax.bar([i + width / 2 for i in idx], grpo_vals, width, label="SFT+GRPO", color="#dd8452")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"pair {i}" for i in idx], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wrong", "correct"])
    ax.set_title(
        f"Phase 3 axis 1: language grounding counterfactual pairs (n={len(per_pair)})\n"
        f"SFT acc={result['sft']['proxy_accuracy']*100:.0f}%, "
        f"GRPO acc={result['grpo']['proxy_accuracy']*100:.0f}%, "
        f"agree on {n_agree}/{len(common_keys)} pairs"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_png = RESULTS_DIR / "phase3_language_grounding.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
