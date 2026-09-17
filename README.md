# VLA Imitation vs RL Fine-tuning: Language Grounding & Dynamic Robustness Diagnostics

诊断 RL 微调对 Vision-Language-Action（VLA）模型到底改善了什么、又可能在哪些维度上牺牲了什么——不是
"跑通一个操作demo"，是拿 LoRA 模仿学习和 LoRA+PPO 强化学习微调两个策略，在语言接地和动态场景鲁棒性
两条诊断轴上做架构级对比。

对应求职方向：多模态具身智能 / 强化学习相关岗位。完整技术方案见 [`PROJECT_PLAN.md`](./PROJECT_PLAN.md)，
实时进度见 [`PROGRESS.md`](./PROGRESS.md)。这是继 [`quadruped-wbc-mpc`](https://github.com/GZY0109/quadruped-wbc-mpc)
（经典控制 vs RL 在扰动下的鲁棒性对比）之后的第二个项目，同一套方法论从控制层搬到了决策层。

## 结构

```
src/            核心代码（LoRA微调、PPO训练、诊断评测harness）
scripts/        可运行脚本
experiments/    实验配置
results/        实验数据、图表
```

（以上目录随 Phase 1 开工陆续建立，当前仓库只有方案/进度文档。）

## 环境

（Phase 1 完成后补：LIBERO + OpenVLA 安装步骤。）

## 复现结果摘要

（暂无 —— 数字必须是真实跑出来的，Phase 1 完成后开始填。）
