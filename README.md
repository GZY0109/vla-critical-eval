# VLA Imitation vs RL Fine-tuning: Language Grounding & Dynamic Robustness Diagnostics

诊断 RL 微调对 Vision-Language-Action（VLA）模型到底改善了什么、又可能在哪些维度上牺牲了什么——不是
"跑通一个操作demo"，是拿 LoRA 模仿学习和 LoRA+GRPO 强化学习微调两个策略，在语言接地和动态场景鲁棒性
两条诊断轴上做架构级对比。

对应求职方向：多模态具身智能 / 强化学习相关岗位。完整技术方案见 [`PROJECT_PLAN.md`](./PROJECT_PLAN.md)，
实时进度见 [`PROGRESS.md`](./PROGRESS.md)。这是继 [`quadruped-wbc-mpc`](https://github.com/GZY0109/quadruped-wbc-mpc)
（经典控制 vs RL 在扰动下的鲁棒性对比）之后的第二个项目，同一套方法论从控制层搬到了决策层。

## 结构

```
openvla/                                    vendored OpenVLA 仓库（LoRA 微调基座），本项目的自定义代码都加在里面：
  vla-scripts/grpo_finetune.py                GRPO 训练主循环（LoRA r=32，沿用官方 finetune.py 配方）
  prismatic/vla/grpo_utils.py                 GRPO trajectory / advantage 计算
  experiments/robot/libero/                   自定义评测 harness + 分析脚本
    grpo_rollout.py                             GRPO 训练用的 rollout 采样
    grpo_eval_libero.py                         Phase 1/2 评测（成功率 + per-step entropy 记录）
    dynamic_perturb_eval.py                     Phase 3 轴二：动态场景扰动评测
    probe_language_pairs.py / language_grounding_eval.py   Phase 3 轴一：语言接地反事实场景配对 + 评测
    analyze_phase2_grpo.py                      Phase 2 对比数据/图表生成（读 experiments/logs/ 下的 jsonl）
    analyze_phase3_dynamic_perturb.py           Phase 3 轴二 对比数据/图表生成
    analyze_phase3_language_grounding.py        Phase 3 轴一 对比数据/图表生成
  experiments/logs/                           原始评测日志（.jsonl 逐 episode 结构化数据 + .txt 摘要）
results/                                     上面三个 analyze_*.py 脚本跑出的聚合数据（.json）+ 图表（.png）
```

## 环境

LIBERO + OpenVLA-7B（bf16），单卡 3090 24GB。安装过程记录在 `install_torch.log` /
`install_libero.log` / `install_openvla.log` / `install_flashattn.log` / `install_libero_reqs.log`
（Phase 1 装环境时的真实过程日志，含踩过的坑）。GRPO 训练/评测复用同一套 Python 环境，见
`openvla/vla-scripts/grpo_finetune.py` 和 `openvla/experiments/robot/libero/` 下各脚本的依赖。

## 复现结果摘要

**项目结论（Phase 3 首轮之后确定）**：这个项目最终交付的不是"RL 微调让 VLA 变得更好或更差"的故事，而是
一次诚实的空结果——`kl_coef=0`、2 任务、36 episodes/轮、25 轮的验证性 GRPO run，在成功率、动态鲁棒性、
语言接地三个维度上都没有产生可测量的系统性差异。跟 `quadruped-wbc-mpc` 的方法论一致：数字必须是真实跑
出来的，不好看的结果也如实报告，不为了叙事去凑。以下三张图/三份 JSON 都由
`openvla/experiments/robot/libero/analyze_*.py` 从原始评测日志（`openvla/experiments/logs/*.jsonl`）
重新计算得到，可直接复现。

**Phase 1 基线复现**（`openvla-7b-finetuned-libero-spatial` 官方 checkpoint，libero_spatial，
12 trials/task，n=120）：**82.5%**（99/120），跟官方论文数字（84.7%±0.9%，n=1500）差 2.2pp，量级上
跟社区复现（50 trials/task 时差 1.1pp）一致，样本量只有官方 1/4，方差本该更大——环境和流程验证通过。

**Phase 2：SFT-only vs SFT+GRPO 成功率与机制对比**（`results/phase2_grpo_vs_sft.json` /
`results/phase2_grpo_vs_sft.png`，各 n=120）：

| | 成功率 | 平均动作熵（7维logits，episode均值） | 平均步数 |
|---|---|---|---|
| SFT-only | 82.5% (99/120) | 0.3864 | 129.2 |
| SFT+GRPO | 82.5% (99/120) | 0.3835 | 128.4 |

总分打平，但**不是同一批 episode 在赢**：按 `(task_id, init_state_idx)` 逐条配对，120 条里 90 条两边都
成功、12 条两边都失败，另外 **18 条发生翻转**（9 条 SFT 赢/GRPO 输，9 条反过来，方向刚好互相抵消）。熵
几乎不变。说明这次训练让策略发生了实质性漂移，只是漂移对成功率是中性的，不是"什么都没学到"。

**Phase 3 轴二：动态场景鲁棒性**（`dynamic_perturb_eval.py`，持续给场景内可动黑碗注入水平速度，
`results/phase3_dynamic_perturb.json` / `.png`，v=0 复用 Phase 2 的 n=120 静态基线，v>0 各档 n=50）：

| policy | v=0 | v=0.05 | v=0.1 | v=0.2 |
|---|---|---|---|---|
| SFT-only | 82.5% (99/120) | 74.0% (37/50) | 82.0% (41/50) | 60.0% (30/50) |
| SFT+GRPO | 82.5% (99/120) | 84.0% (42/50) | 78.0% (39/50) | 64.0% (32/50) |

两条曲线都非单调（v=0.1 比 v=0.05 还高，n=50/点噪声本来就大），v=0.2 时两个策略都明显跌破静态基线，说明
动态扰动确实伤成功率；但同一档位上两个策略的差异都落在 n=50 的噪声范围内，**没有信号支持"GRPO 更/更不
鲁棒"这个结论**。

**Phase 3 轴一：语言接地反事实测试**（固定场景只换指令，`results/phase3_language_grounding.json` /
`.png`，10 组陪衬碗-目标碗位置重合 <0.15cm 的干净配对）：

| policy | proxy accuracy |
|---|---|
| SFT-only | 70% (7/10) |
| SFT+GRPO | 70% (7/10) |

不仅总分相同，**10 条配对里两个策略的对错模式 100% 一致**（逐条比对，无一条分歧）。两个策略在全部 20 条
episode 里 `native_env_success` 都是 `False`，排除了"策略压根没理会新指令、纯粹靠肌肉记忆走完原任务"这种
最坏情况。这次 GRPO 微调没有改变策略在场景/指令切换下"瞄准哪个物体"这个决策机制，无论好坏方向。

**如实的局限性说明**：以上三个维度的"零差异"结论建立在一次小规模验证性 GRPO run（无 KL 约束、2 个任务、
训练量级明显小于同类论文如 TGRPO 消融里 4-8pp 提升对应的训练规模）之上，样本量也不大（n=120/50/10）。
这更可能反映"这次训练配置本身没有触及决策边界"而不是"GRPO 微调对 VLA 决策毫无影响"这一更强的普适结论——
是否要放大训练规模验证这个结论的稳健性，留作后续方向，本次不再重跑。
