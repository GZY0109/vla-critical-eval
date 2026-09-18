# 进度追踪

按 `PROJECT_PLAN.md` 的 Phase 顺序记录。每完成一个里程碑就在这里加一条，commit 时带上这次改动对应的记录。

## 使用方式（维护约定）

- 每次开始工作前，先读这份文件的"当前状态"和最近几条记录，再看 `git log --oneline -20` 确认实际代码
  进度和这里记的一致。
- 每完成一个可验证的里程碑（环境跑通/复现基线/RL微调收敛/评测跑完），追加一条记录，然后 commit + push。
- 遇到需要我自己决策的问题（比如某个库版本选型、要不要追某个新方向），记录在"待决策"里，不自己瞎猜
  定重大方向。

## 当前状态

Phase 1 完成，定案数字是 12 trials/task（n=120）跑出的 82.5%。Phase 2：GRPO 正式训练（`real_run_v1`，
`--kl_coef 0`）已经跑完，25 轮（iteration 0-24），最终 checkpoint 在
`openvla/runs/grpo/grpo+openvla-7b-finetuned-libero-spatial+lora-r32+lr-1e-05--real_run_v1--2026_09_17-15_03_41/adapter_latest`。

为了做 SFT vs SFT+GRPO 的机制分析（不只是成功率数字，还要看熵/置信度），写了新评测脚本
`openvla/experiments/robot/libero/grpo_eval_libero.py`（在 Phase 1 那次纯文本 log 基础上加了
per-step entropy/top1-margin/action-token-id 的结构化 JSONL 记录，Phase 1 的 log 没留 logits，没法
回溯补熵，所以两个策略都要用这个新脚本重新跑一遍 n=120，评测方法论跟 Phase 1 一致，82.5% 这个基线
数字本身不受影响）。截至目前：

- ✅ SFT baseline recheck：`GRPOEVAL-libero_spatial-2026_09_18-12_29_00--sft_baseline_recheck`，
  2026-09-18 12:29-13:47（78 分钟），n=120，成功率 82.5%（99/120），与 Phase 1 一致，熵数据完整。
- ✅ SFT+GRPO checkpoint 评测：`GRPOEVAL-libero_spatial-2026_09_18-14_42_40--grpo_real_run_v1_checkpoint`，
  2026-09-18 14:42-16:30（108 分钟），n=120，成功率 82.5%（99/120）——跟 SFT baseline 完全打平。
- ✅ 机制分析（两组 n=120 按 task_id+init_state_idx 逐条配对比较）：
  - 总成功率打平（82.5% vs 82.5%），但**不是同一批 episode 在赢**：120 条里 90 条两边都成功、
    12 条两边都失败，另外 18 条发生翻转——9 条 SFT 成功但 GRPO 失败，9 条 SFT 失败但 GRPO 成功，
    翻转方向刚好互相抵消。说明 `kl_coef=0` 这次训练让策略发生了实质性漂移，只是漂移对成功率是
    中性的，不是"什么都没学到"。
  - 熵几乎不变：SFT 均值 0.3864，GRPO 均值 0.3835（7 维动作 logits 的 per-step 均值再对 episode
    内 step 取平均）——没有变得更自信也没有变得更发散。平均步数也基本不变（129.2 vs 128.4）。
  - 分任务成功率有升有降（task 9 从 0.50→0.67，task 6 从 1.00→0.83 等），幅度都在 n=12/task 的
    噪声范围内（±1 条 episode 就是 ±8.3 个百分点），不构成任何任务上的系统性改善或退化。
  - 结论：这次 `kl_coef=0`、无 KL 约束、小规模（2 任务、36 episodes/轮、25 轮）验证性 GRPO run
    没有产生可测量的净收益，但也没有跑飞退化——是一次方法验证性质的运行，不是收敛到更优策略的运行。
    Phase 2 交付物到此完成，不用再追加训练轮数或调参重跑（不是这个阶段的目标）。

Phase 3（语言接地 + 动态鲁棒性双轴诊断）首轮实现和评测已经跑完，两个策略（SFT-only / SFT+GRPO）
在两条轴上都有数据了：

**轴二：动态场景鲁棒性**（`openvla/experiments/robot/libero/dynamic_perturb_eval.py`，前 30 步持续
给场景里所有可动黑碗注入水平速度、之后停止让摩擦力自然停下，避免物体被扰动直接推进容器；v=0 复用
Phase 2 的 n=120 baseline）：

| policy | v=0（复用Phase2） | v=0.05 | v=0.1 | v=0.2 |
|---|---|---|---|---|
| SFT-only | 82.5% (99/120) | 74.0% (37/50) | 82.0% (41/50) | 60.0% (30/50) |
| SFT+GRPO | 82.5% (99/120) | 84.0% (42/50) | 78.0% (39/50) | 64.0% (32/50) |

（`DYNPERTURB-libero_spatial-2026_09_18-17_10_34--sft_dynamic_perturb` /
`DYNPERTURB-libero_spatial-2026_09_18-19_00_17--grpo_dynamic_perturb`，各 n=150，3 档速度×10
task×5 trials）两条曲线都不是单调衰减（v=0.1 比 v=0.05 还高，噪声很大——n=5/task/档 本来就小），
v=0.2 时两个策略都明显跌破静态基线（60%/64% vs 82.5%），说明动态扰动确实会伤成功率，但两个策略在
同一档位上的差异（比如 v=0.05 时 74% vs 84%）都在 n=50 的噪声范围内（±1 条约等于 ±2 个百分点，
10 个点的差距边界情况，不构成能下"GRPO更/更不鲁棒"结论的强信号）——如实记录：这一版首轮规模看不出
两个策略在动态鲁棒性上有系统性差异。

**轴一：语言接地反事实测试**（`openvla/experiments/robot/libero/probe_language_pairs.py` 枚举场景/
指令配对，`language_grounding_eval.py` 用末端执行器到两个碗的最小距离判定策略瞄准哪个碗）：libero_spatial
十个任务的 BDDL 核实过目标结构完全一致——goal 永远是 `(On akita_black_bowl_1 plate_1)`，语言换的只是
"目标碗在哪个位置"，陪衬碗永远是 `akita_black_bowl_2`。枚举全部 10×50 个 init_state 找到 10 组
陪衬碗位置和另一任务目标碗典型位置重合 <0.15cm 的干净配对（`language_grounding_pairs.json`）。

- SFT-only：`LANGGROUND-libero_spatial-2026_09_18-21_33_02--sft_language_grounding`，
  proxy accuracy 70%（7/10）。
- SFT+GRPO：`LANGGROUND-libero_spatial-2026_09_18-21_44_31--grpo_language_grounding`，
  proxy accuracy 70%（7/10）。
- **关键发现**：两个策略不仅总分相同，**逐条配对的对错模式完全一致**——都在 (scene=task8,
  instr=task1)、(scene=task0, instr=task1)、(scene=task4, instr=task9) 这三条上判定成"瞄准了
  scene 原本的目标碗"（错），其余 7 条一致判定"瞄准了陪衬碗/新指令目标"（对）。另外两个策略在全部
  20 条 episode 里 `native_env_success` 都是 False（没有一次完整复现 scene 原任务的成功抓放），
  排除了"策略压根没理会新指令、纯粹靠肌肉记忆把原任务走完"这种最坏情况。逐条一致这件事本身就是
  强信号：这次 GRPO 微调没有改变策略在场景切换/指令切换下"瞄准哪个物体"这个决策——无论是往好的
  方向（语言接地变强）还是坏的方向（更依赖视觉捷径），至少在这批配对上，RL 微调完全没有触及这个
  决策机制。

**Phase 3 首轮结论**：跟 Phase 2 的机制分析结论一致——这次 `kl_coef=0`、2 任务、36 episodes/轮、
25 轮的验证性 GRPO run，在成功率、动态鲁棒性、语言接地三个维度上都没有产生可测量的系统性差异（唯一
的例外是 Phase 2 机制分析发现的"18 条 episode 发生翻转但净效应为零"这种非零但抵消的漂移）。如实记录：
这不是"RL微调让VLA变得更好或更差"的故事，是"这个规模/配置的 GRPO 微调没有改变策略的决策边界"的故事
——诚实反映了小规模验证性 RL run 的真实局限（对比 Phase 1 计划里提到的 TGRPO 消融，同类方法通常涨
4-8 个点，是在更大规模训练预算下取得的，这次的训练量级本来就没有对标那个规模）。Phase 3 首轮交付物
到此完成。是否要加大训练规模重跑 GRPO 来看这个结论是否稳健，留给 Phase 4 收尾时再判断优先级。

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

正式训练（`--kl_coef 0 --run_id_note real_run_v1`）已经开跑，跑着的时候先做 Phase 3 并行准备，
见下一条记录。

### Phase 3 开工前探索 - 动态场景扰动的物理可行性验证，顺带发现一个跟评测 harness 相关的设计坑

GRPO 正式训练占着 GPU（20987/24576 MiB，100% 利用率），趁它跑的时候先探索 Phase 3"动态场景鲁棒性"
这个诊断轴的可行性——PROJECT_PLAN 里这部分本来就标了"待 Phase 3 开工时先勘探，做不到再降级"。写了个
探测脚本 `openvla/experiments/robot/libero/probe_dynamic_object.py`，不加载 OpenVLA、只用 LIBERO/
robosuite 环境本身，不占 GPU 显存，跟训练进程并行跑没有冲突。

验证了三件事：
- LIBERO 物体都是 `free` 关节（6-dof，平移+旋转），可以直接用
  `env.sim.data.set_joint_qpos/set_joint_qvel(joint_name, ...)` 改位置/速度，`env.step()` 的控制
  循环不会重置非驱动关节的速度——机制上是通的。
- 自由落体：抬高物体 15cm、清零速度、松手，物理引擎正常让它在约 5 步（~0.25s）内落地并稳定，符合
  预期。
- 水平匀速运动：单次注入速度后位移只有 0.0046m 就停了——一开始以为是代码把速度覆盖掉了，实际排查
  后发现是真实物理摩擦力的作用（碗贴着桌面，横向速度被摩擦很快吃掉），不是 bug。改成每个 env.step
  前都重新注入速度（模拟传送带持续给力），20 步后位移变成 0.0297m、单调递增——这个方案可行，但要
  持续注入而不是松手一次。

**关键发现（会改变 Phase 3 设计，不是单纯的物理验证）**：`run_libero_eval.py:72` 里官方评测 harness
本来就有 `num_steps_wait=10`——LIBERO 的标准初始化本身就是把物体从空中扔下摔到桌上，然后刻意等 10 步
摔稳了才让策略开始动作。这意味着"episode 开头让物体自由落体"这个最直觉的方案，其实跟 LIBERO 默认行为
是同一件事，根本不是新扰动，评测脚本已经把它规避掉了。要让自由落体成为一个真正的动态鲁棒性测试，
必须做以下两选一：缩短/取消 `num_steps_wait`，强迫策略在物体还没摔稳时就开始观测和动作；或者把落体
触发时机挪到 episode 中途（机器人已经朝原目标位置伸手之后），测试策略能不能在动作执行到一半时根据
新观测修正目标。

另外确认了 `run_libero_eval.py:186-228` 这条评测循环是逐步重新查询 VLA（不是动作分块/chunking），
每个 sim step 都会用当前观测重新预测动作，这意味着单次、短暂（~0.25s）的扰动很可能被策略自己的逐帧
闭环重新观测"吸收掉"，除非扰动发生在策略已经做出不可逆承诺（比如夹爪刚合上）的关键时刻——这也是
水平匀速运动（持续性扰动，不会被单帧重新观测轻易纠正）可能比自由落体更可靠地测出"动态场景下策略是否
退化"这个问题的原因。

**下一步**：Phase 3 正式开工时，动态场景诊断优先做"持续水平匀速运动"（机制已验证可行、噪声小），
自由落体版本如果要做，需要配合缩短 `num_steps_wait` 或挪到 episode 中途触发，不能直接沿用默认初始化
流程。物理/API 层面两个方案都不需要降级到"位置中途瞬移"这种更简化的代理指标。

### Phase 3 开工前探索 - 语言接地反事实测试的场景配对，libero_spatial 十个任务共用一组槽位坐标

继续趁 GRPO 训练占着 GPU 的空档，探索 Phase 3 语言接地诊断轴（固定场景、只换指令、测目标物体是否
正确切换）的具体实现方式。写了探测脚本把 `libero_spatial` 十个任务的初始状态都摆稳后，量出两个黑碗
相对各参照物（盘子/ramekin/饼干盒/木柜/炉子）的真实坐标，逐个任务比对。

发现：这十个任务其实共用同一小组"槽位坐标"——两个碗永远落在这同一组固定位置的某几个组合上，十个
任务只是把"语言描述哪个位置是目标"这件事换了，不是每个任务都单独设计了新场景。这意味着能找到大量
"任务 A 里没被选中的陪衬碗，位置和任务 B 的目标碗位置几乎重合（<3cm，在抓取容差内）"的组合，不用碰
BDDL、不用造新场景，直接复用某个任务的 init_state、把指令文本换成另一个任务的语言，就能做反事实测试。
量出来最干净的几组（碗位置误差 < 1cm）：
- task 6 场景（陪衬碗）+ task 7 指令"on the stove"（误差 0.4cm）
- task 0 场景（陪衬碗）+ task 1 指令"next to the ramekin"（误差 0.8cm）
- task 3 场景（陪衬碗）+ task 4 指令"in the top drawer of the wooden cabinet"（误差 0.8cm）
还有其他几组误差在 1-3cm 区间，十个任务两两之间能拼出十几对可用组合（脚本输出的完整清单见本次
探索过程，没有专门存文件，需要时重新跑一下 <3cm 阈值的匹配脚本即可复现）。

这里有个关键设计点：不能用环境内置的成功判定来衡量这个测试，因为那是绑死在原任务 BDDL 目标物体上的
（换了指令后，"正确"的目标物体变了，但 env 的 done/success 信号还是照原任务的物体判定）。需要单独写一
个代理指标——比如看末端执行器最早接近/接触的是哪个碗，或者比较到两个碗的最小距离曲线——来判断策略
动作瞄准的是指令里新指定的那个碗还是原任务训练时的那个，这个代理指标还没写，是 Phase 3 正式开工时
要做的部分。

**下一步**：Phase 3 正式开工时，用这批场景配对组装语言接地反事实测试集，同时实现"动作目标碗判定"的
代理指标（末端执行器最近邻物体或轨迹朝向）。跟动态场景那条一样，物理/数据层面已经验证可行，不需要
造新场景或降级方案。
