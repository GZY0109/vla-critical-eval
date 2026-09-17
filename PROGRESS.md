# 进度追踪

按 `PROJECT_PLAN.md` 的 Phase 顺序记录。每完成一个里程碑就在这里加一条，commit 时带上这次改动对应的记录。

## 使用方式（维护约定）

- 每次开始工作前，先读这份文件的"当前状态"和最近几条记录，再看 `git log --oneline -20` 确认实际代码
  进度和这里记的一致。
- 每完成一个可验证的里程碑（环境跑通/复现基线/RL微调收敛/评测跑完），追加一条记录，然后 commit + push。
- 遇到需要我自己决策的问题（比如某个库版本选型、要不要追某个新方向），记录在"待决策"里，不自己瞎猜
  定重大方向。

## 当前状态

Phase 1 完成，定案数字是 12 trials/task（n=120）跑出的 82.5%，没有重跑官方默认的 50 trials/task
版本——GPU 跟 Phase 2 共用，不想在开 PPO 微调之前再占几个小时显存，小样本这个差距量级已经能说明
环境装对了，不需要更大样本去证明。下一步：Phase 2，PPO 强化学习微调（LoRA-SFT checkpoint 基础上
加 value head）。

## 重开 Pod / 新会话恢复工作的步骤（重要）

跟 `quadruped-wbc-mpc` 是同一个 pod，没有独立的持久化存储——apt/pip 依赖在 pod 重启后大概率会丢。
新会话先做：

```bash
cd ~/vla-critical-eval   # 实际路径视 clone 位置而定
bash scripts/setup_env.sh   # Phase 1 完成后才会有这个脚本，目前还没有
git log --oneline -10       # 核对代码进度与本文件一致，对不上以 git log 为准并更正本文件
```

然后读本文件"当前状态"，从上次断点接着做。方案定稿过程（含排除掉的方向和理由）见 `PROJECT_PLAN.md`
"待决策（已解决）"，不用重新讨论。

## 待决策

（当前无待决策条目）

## 记录

<!-- 格式：### Phase X - 一句话摘要 \n 具体做了什么、结果如何、下一步是什么（尽量不写日期，git 已有时间戳） -->

### Phase 0 - 方案定稿：从"纯评测"到"LoRA微调+PPO+双轴诊断"，排除了两个更激进的方向

最早的想法是纯评测（SimplerEnv + OpenVLA，只做语言接地反事实测试，不训练）。中途因为我想让项目命中
OpenVLA/LIBERO/微调/PPO/LoRA这几个JD关键词，改成了现在这个"LoRA模仿学习 + PPO强化学习微调"的对比
结构，语言接地测试从"唯一premise"降级成"两个诊断轴之一"——原来的方案是单一假设的赌注，赌"语言接地
是假的"这一个实验结果够不够戏剧化，现在有 Phase1(基线)/Phase2(RL微调对比)/Phase3(双轴诊断) 三层
交付物，就算某一层结果不够亮眼，其他层的真实数字也站得住。

中途还讨论过两个更激进的方向，都排除了，理由记在 `PROJECT_PLAN.md` 的"待决策（已解决）"里：
- 追"人形全身loco-manipulation"（WholebodyVLA, ICLR 2026）——核实过官方仓库，没有代码没有checkpoint，
  只是一个论文引用列表，追这个方向等于追一个不存在的东西。
- 从零训练一个"内化预测式VLA新架构"去对标AHEAD论文——工程量和不确定性比"评测/微调已有方法"高一个
  量级，且"我提出的新方法没跑赢已发表方法"这种结果对求职是偏弱的故事，风险收益比不划算。保留了
  AHEAD论文揭示的"动态场景鲁棒性衰减"这个评测维度（真实、低风险），舍弃了"训练新架构"这部分。

**下一步**：Phase 1 开工，装 LIBERO + OpenVLA，复现官方 LoRA 微调基线。

### Phase 1 - 环境装好，LIBERO-Spatial 基线复现出 82.5%（n=120，小样本版本，定案不追加大样本）

装好了 LIBERO + OpenVLA（`install_torch.log`/`install_libero.log`/`install_openvla.log`/
`install_flashattn.log`/`install_libero_reqs.log` 是装环境的过程记录），跑了官方 checkpoint
`openvla/openvla-7b-finetuned-libero-spatial` 在 `libero_spatial` 套件上的
评测：`--num_trials_per_task 12`（10 任务 × 12 trials = 120 episodes），跑了 78 分钟，最终成功率
82.5%（99/120）。

核实过官方数字：论文报的是 84.7% ± 0.9%（n=1500，3 seed × 500 trials，A100）；社区有人用官方默认
50 trials/task 复现出 83.6%，跟官方差 1.1pp。我们这次 82.5% 跟官方差 2.2pp，量级上跟社区那次的
1.1pp 差距一致（我们样本量只有 1/4，方差本来就该更大），说明环境装对了，不是复现失败。

一开始想按官方默认的 50 trials/task（500 episodes，预估 5-6 小时）重跑一版更贴官方数字的结果，
后来决定不跑了——GPU 要留给 Phase 2 的 PPO 微调，不想在开工前先占几个小时显存，而且 82.5% 这个
小样本数字加上跟社区复现差距的对比说明，已经足够证明环境和流程是对的，没必要为了让数字更好看
去多跑一次。中途还讨论过在简历上把这次的结果写成"迭代了500次"——否掉了，因为项目本身的卖点就是
"数字必须是真实跑出来的、如实报告"，编数字会让这条卖点本身变成假的，而且没必要：82.5% vs 84.7%±0.9%
这个对比本身（含样本量差异的解释）就是一个站得住、显得懂行的复现结果。

**下一步**：Phase 2，PPO 强化学习微调——在这次跑通的 LoRA-SFT checkpoint 基础上冻结主干、加
value head，做 LoRA-SFT vs LoRA-SFT+PPO 的成功率对比和机制分析。
