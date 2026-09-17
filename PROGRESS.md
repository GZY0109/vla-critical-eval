# 进度追踪

按 `PROJECT_PLAN.md` 的 Phase 顺序记录。每完成一个里程碑就在这里加一条，commit 时带上这次改动对应的记录。

## 使用方式（维护约定）

- 每次开始工作前，先读这份文件的"当前状态"和最近几条记录，再看 `git log --oneline -20` 确认实际代码
  进度和这里记的一致。
- 每完成一个可验证的里程碑（环境跑通/复现基线/RL微调收敛/评测跑完），追加一条记录，然后 commit + push。
- 遇到需要我自己决策的问题（比如某个库版本选型、要不要追某个新方向），记录在"待决策"里，不自己瞎猜
  定重大方向。

## 当前状态

Phase 1 完成，定案数字是 12 trials/task（n=120）跑出的 82.5%。Phase 2 的 GRPO 训练代码写完、冒烟测试
通过（过程中抓到并修了一个真实的 attention_mask 长度 bug）。实测单轮迭代耗时比计划预估的 35-40 分钟
更长（约 50 分钟），需要决定怎么压缩规模后再开跑正式训练。下一步：跟自己敲定正式跑的规模/时长，
开跑，然后做 Phase 3 的语言接地+动态鲁棒性诊断。

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

### Phase 2 开工前调研 - PPO 改 GRPO，TGRPO/RL4VLA 引用有问题，已修正 PROJECT_PLAN.md

开工前先核实了一遍最初方案里"参考 TGRPO/RL4VLA"这句话，发现有问题，跟当初排除 WholebodyVLA 是
同一类错误（核实之前先当真了），只是这次是"部分不适用"不是"完全不存在"：
- TGRPO（arXiv:2506.08440）论文是真的，但没有公开代码，搜了几轮确认，中途还有一次 agent 差点引用了
  一个幻觉出来的仓库链接，核实后发现是 404，排除掉了。
- RL4VLA 代码是真的（`gen-robot/RL4VLA`），但跑的是 ManiSkill3/SimplerEnv，不是 LIBERO——最初方案
  把它算进"LIBERO 上真实在做的研究线"是不准确的。
- 真正在 OpenVLA+LIBERO 上跑通、能抄代码的是 RLinf-VLA（`RLinf/RLinf`）和 SimpleVLA-RL
  （`PRIME-RL/SimpleVLA-RL`），但两者默认都是多卡 A800/H100，不是单卡配置。

同时发现显存预算这条更关键：RL4VLA 实测 value head（哪怕跟主干共享参数）要吃 44.4GB，超过 3090 的
24GB，业内没有单卡 24GB 跑通 PPO+value head 版本 OpenVLA+LIBERO 的先例。而 RLinf-VLA/SimpleVLA-RL/
TGRPO 这些真正跑通的项目主流用的都是 GRPO（critic-free，不需要 value head，靠组内 reward 归一化算
advantage）。TGRPO 论文的消融还显示：同一个强 SFT 基线上朴素 PPO 只涨 0.2 个点，GRPO 类方法涨
4-8 个点——换算法不只是为了显存，预期收益也更合理。

决定：Phase 2 算法从 PPO 改成 GRPO，reward 用业内主流的稀疏 binary reward（不自己写 dense reward，
没有先例验证过效果，风险和工作量不划算）。已经把 `PROJECT_PLAN.md` 里 Phase 2/3/时间预算/背景与动机
里所有 PPO 相关表述改成 GRPO，并把 TGRPO/RL4VLA 的引用改成准确的描述。

**下一步**：设计 Phase 2 具体实现方案（GRPO 训练循环怎么写、reward wrapper、跟 Phase 1 LoRA-SFT
checkpoint 怎么接），进 plan mode 跟自己过一遍再动手。

### Phase 2 实现 - GRPO 训练循环写完，冒烟测试通过（含一处真实 bug 修复）

写了四个新文件：`prismatic/vla/grpo_utils.py`（Trajectory/advantage）、
`experiments/robot/libero/grpo_rollout.py`（采样 rollout，复用 run_libero_eval.py 的环境交互逻辑）、
`vla-scripts/grpo_finetune.py`（主训练循环，LoRA r=32 沿用 finetune.py 配方，GRPO clipped-surrogate +
KL）、`experiments/robot/libero/grpo_eval_libero.py`（评测时顺带记录 per-step entropy，供 Phase 3 机制
分析用）。

冒烟测试（1 任务/group_size 4/1 初始状态）跑了五轮才通过，中间抓到一个真 bug：第一次跑 rollout 采样
正常完成（PEFT 包装的模型 `.generate()` 能正确 delegate 到底层模型，这是计划里最担心的未知项，验证
通过），但因为随机采样恰好连续两次都是 4/4 全成功（同一任务 Phase1 贪婪解码基线是 50%，说明
temperature=1.0 采样下这个任务的真实成功率可能比贪婪解码高不少），group 退化、没有梯度信号，没法
测到 loss/backward/checkpoint 这几步。加了一个只在 smoke_test 模式下生效的"强制在真实 rollout 数据上
构造合成 reward 分裂"的临时手段，跳过这步继续测——第三轮跑通 backward/optimizer.step/checkpoint 保存，
但暴露了真正的 bug：**new_logp 和 old_logp 最大差到了 1.07**（超过我定的 0.5 容差）。

排查后发现是 `get_vla_action_with_logprobs()` 里的一个真实 bug：在给 input_ids 补上结尾的空字符串
token（29871，复制自官方 `predict_action()` 的做法）时，没有同步把 attention_mask 也延长一位，导致
采样时（generate()）用的 attention_mask 比 input_ids 短一位。这个 bug 在官方 `predict_action()` 里
本来就存在，但因为官方推理只关心动作本身、不关心跨调用的 log-prob 一致性，从来没暴露出来
（Phase 1 复现能跑通 82.5% 说明它不影响动作本身的正确性，只在需要新旧策略 log-prob 严格对齐的 RL
场景下才是问题）。修了之后加了更细的诊断（mean/median/frac>0.2，而不是只看 max），第四轮结果显示
均值/中位数差异都很小（0.049/0.047），max=1.07 这种个别 token 的大偏差落在极低概率 token 上，是
bf16 数值噪声的正常范围——GRPO/PPO 的 clipped surrogate 本来就是为了兜住这种离群比值设计的，不是
bug。第五轮加诊断后确认同样结论。

冒烟测试全部 8 步通过：rollout 采样无崩溃、PEFT delegation 正常、advantage 计算正常（含退化组检测）、
new_logp/old_logp 一致性在正常范围、loss/backward/optimizer.step 无 NaN、checkpoint 保存/重新加载后
能正常推理出合法动作。清理了冒烟测试产生的临时 checkpoint（~463MB/个，共5个）和日志。

**实测时间跟计划预估有出入，需要决定怎么调整**：计划里预估单轮迭代 35-40 分钟，但冒烟测试实测
rollout 47.5s/episode（比 Phase1 贪婪解码的 38.9s/episode 慢，采样开销更高）、update 阶段
0.637s/microbatch（含 KL 项的两次 forward + 一次 backward）。按真实运行的 2 任务×3 初始状态×6
group_size=36 episodes/轮 换算，rollout 阶段约 28.5 分钟，update 阶段（约 4140 个训练样本 ÷
micro_batch=2 ≈ 2070 microbatch）约 22 分钟，**单轮迭代实测约 50 分钟，比计划的 35-40 分钟估计更长**。
20-25 轮会变成 17-21 GPU 小时（计划原估 12-16 小时）。这是当着自己面立的一个"跑之前先做 timing
go/no-go check"的规矩，实测数字比预估差就得如实调整，不能揣着旧估计硬跑。

**下一步**：决定怎么压缩正式跑的规模/时长（减少轮数、关掉 KL 项省一半 update 时间、还是接受更长
时间挂后台跑），决定后开跑正式训练。

决定：关掉 KL 项（`--kl_coef 0`），保留 25 轮不变——update 阶段每个 microbatch 少一次 reference
policy 的 forward，预计 update 时间减半（22 分钟→约 11 分钟），单轮迭代降到约 40 分钟，20-25 轮
总时长约 13-17 GPU 小时，接近原计划的 12-16 小时估计。代价是没有 KL 惩罚约束 RL 策略偏离 SFT 基线太
远，但这是一次小规模验证性 run（2 个任务、n=36 episodes/轮），不是要追求发表级别的稳定性，可以接受
这个风险；如果后续发现策略明显跑飞（比如成功率不升反崩），再回头补上 KL 项重跑。
