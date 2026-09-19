"""Phase 4 收尾：把 Phase 3 轴二（动态场景鲁棒性）的原始 jsonl 变成对比数据 + 图表。

v=0 复用 Phase 2 的 GRPOEVAL 静态基线（PROGRESS.md 里就是这么定义的，不重新编数字）；
v=0.05/0.1/0.2 来自 DYNPERTURB jsonl（单文件内 `velocity` 字段区分三档）。

输入: openvla/experiments/logs/GRPOEVAL-*.jsonl（v=0 基线）
      openvla/experiments/logs/DYNPERTURB-*--sft_dynamic_perturb.jsonl
      openvla/experiments/logs/DYNPERTURB-*--grpo_dynamic_perturb.jsonl
输出: results/phase3_dynamic_perturb.json, results/phase3_dynamic_perturb.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
RESULTS_DIR = Path(__file__).resolve().parents[4] / "results"

SFT_BASELINE_LOG = LOG_DIR / "GRPOEVAL-libero_spatial-2026_09_18-12_29_00--sft_baseline_recheck.jsonl"
GRPO_BASELINE_LOG = LOG_DIR / "GRPOEVAL-libero_spatial-2026_09_18-14_42_40--grpo_real_run_v1_checkpoint.jsonl"
SFT_PERTURB_LOG = LOG_DIR / "DYNPERTURB-libero_spatial-2026_09_18-17_10_34--sft_dynamic_perturb.jsonl"
GRPO_PERTURB_LOG = LOG_DIR / "DYNPERTURB-libero_spatial-2026_09_18-19_00_17--grpo_dynamic_perturb.jsonl"


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def success_rate(episodes):
    n = len(episodes)
    n_success = sum(1 for e in episodes if e["success"])
    return {"n": n, "n_success": n_success, "success_rate": n_success / n}


def by_velocity(episodes):
    velocities = sorted({e["velocity"] for e in episodes})
    return {v: success_rate([e for e in episodes if e["velocity"] == v]) for v in velocities}


def main():
    sft_baseline = load(SFT_BASELINE_LOG)
    grpo_baseline = load(GRPO_BASELINE_LOG)
    sft_perturb = load(SFT_PERTURB_LOG)
    grpo_perturb = load(GRPO_PERTURB_LOG)

    sft_curve = {"0.0": success_rate(sft_baseline)}
    sft_curve.update({str(k): v for k, v in by_velocity(sft_perturb).items()})
    grpo_curve = {"0.0": success_rate(grpo_baseline)}
    grpo_curve.update({str(k): v for k, v in by_velocity(grpo_perturb).items()})

    result = {
        "sft": sft_curve,
        "grpo": grpo_curve,
        "note": "v=0.0 复用 Phase2 GRPOEVAL 静态基线 (n=120)，v>0 各档 n=50 (10 task x 5 trial)",
        "source_logs": {
            "sft_baseline": SFT_BASELINE_LOG.name,
            "grpo_baseline": GRPO_BASELINE_LOG.name,
            "sft_perturb": SFT_PERTURB_LOG.name,
            "grpo_perturb": GRPO_PERTURB_LOG.name,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "phase3_dynamic_perturb.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_json}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    velocities = sorted(sft_curve.keys(), key=float)
    sft_rates = [sft_curve[v]["success_rate"] * 100 for v in velocities]
    grpo_rates = [grpo_curve[v]["success_rate"] * 100 for v in velocities]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = [float(v) for v in velocities]
    ax.plot(x, sft_rates, "o-", label="SFT-only", color="#4c72b0")
    ax.plot(x, grpo_rates, "s-", label="SFT+GRPO", color="#dd8452")
    for xi, yi in zip(x, sft_rates):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", color="#4c72b0")
    for xi, yi in zip(x, grpo_rates):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, -14), ha="center", color="#dd8452")
    ax.set_xlabel("perturbation velocity (m/s)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Phase 3 axis 2: dynamic perturbation robustness (libero_spatial)\nv=0 n=120, v>0 n=50/point")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = RESULTS_DIR / "phase3_dynamic_perturb.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
