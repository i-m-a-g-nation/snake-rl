# Snake 贪吃蛇强化学习项目

## 1. 项目简介

本项目实现了一个经典的贪吃蛇（Snake）游戏，并使用深度强化学习（Deep Q-Network, DQN）训练智能体（Agent）自动玩游戏。项目提供两种训练方式：

1. **手写 DQN** (`train.py`)：纯 PyTorch 实现，便于理解 RL 算法原理
2. **SB3 DQN** (`train_sb3_dqn.py`)：使用 Stable-Baselines3 成熟库，训练更稳定

**核心特性：**
- 手写 DQN / Double DQN（学习算法原理）
- Stable-Baselines3 DQN（成熟库版本）
- 两种状态模式：basic17（推荐）/ reachable23（实验）
- Gymnasium API 兼容
- 经验回放 + Warmup 预热
- 定期评估 + Best Model 自动保存
- 死亡原因统计
- CUDA 加速训练
- 终端实时刷新游玩

## 2. 当前环境检测结果

| 项目 | 值 |
|------|-----|
| Python 版本 | 3.10.20 |
| Conda 环境 | nlp-env |
| numpy | 2.2.6 ✅ |
| torch | 2.6.0+cu124 ✅ |
| pandas | 2.3.3 ✅ |
| CUDA | RTX 4060 Laptop GPU ✅ |
| pygame | 未安装 ❌ |
| gymnasium | 未安装 ❌ |

## 3. 实际使用的库

| 库 | 用途 | 是否必须 |
|----|------|----------|
| numpy | 状态表示、数值计算 | 是 |
| torch | DQN 网络、训练 | 是（有 torch 用 DQN，无则用 numpy Q-learning） |
| pandas | 训练日志 CSV 处理 | 否（可用 csv 模块替代） |

**未使用 pygame / gymnasium / stable_baselines3**，遵循"已有库优先"原则。

## 4. 为什么优先复用现有库

1. **避免环境污染**：不随意创建新 Conda 环境，不破坏已有环境
2. **减少依赖冲突**：新安装的库可能与已有项目冲突
3. **提高可复现性**：依赖越少，环境配置越简单
4. **符合课程设计要求**：展示对底层算法的理解，而非调包

## 5. 项目结构

```
snake_rl/
├── README.md              # 项目说明文档
├── requirements.txt       # 实际依赖列表
├── environment_report.txt # 环境检测报告
├── main.py                # 加载模型观看 Agent 玩
├── play_human.py          # 人工游玩入口（终端实时刷新）
├── train.py               # 手写 DQN 训练入口
├── train_sb3_dqn.py       # Stable-Baselines3 DQN 训练入口
├── evaluate_deaths.py     # 手写 DQN 死亡原因评估
├── evaluate_sb3.py        # SB3 DQN 评估
├── agent.py               # DQN Agent 实现
├── snake_env.py           # Snake RL 环境 (Gymnasium API)
├── snake_game.py          # Snake 游戏纯逻辑
├── terminal_input.py      # 终端输入模块
├── replay_buffer.py       # 经验回放缓冲区
├── models.py              # DQN 神经网络模型
├── utils.py               # 工具函数
├── checkpoints/           # 模型保存目录
└── logs/                  # 训练日志目录
```

## 6. 如何人工游玩

### 终端实时版（无需 pygame，推荐）
```bash
conda activate nlp-env
cd snake_rl
python play_human.py
```

**控制方式：**
- `↑` `↓` `←` `→` 方向键移动（无需回车，实时响应）
- `空格` 暂停/继续
- `R` 重新开始
- `Q` 退出

**说明：** 终端版使用 `msvcrt` 实现非阻塞键盘输入，蛇按固定 FPS 自动前进，方向键控制绝对方向（上/下/左/右），内部自动转换为 DQN 的相对动作（直行/左转/右转）。

### 图形界面版（需安装 pygame）
```bash
pip install pygame
python play_human.py
```
使用方向键控制。

## 7. 如何训练

### 方式一：手写 DQN（学习算法原理）

```bash
# 默认使用 basic17 状态
python train.py --episodes 3000 --double-dqn

# 使用 reachable23 状态（实验性）
python train.py --episodes 3000 --state-mode reachable23
```

### 方式二：Stable-Baselines3 DQN（更稳定）

```bash
# 安装 SB3（如未安装）
pip install stable-baselines3

# 快速接口测试 (10000 timesteps)
python train_sb3_dqn.py --timesteps 10000 --state-mode basic17

# 正式训练 (200000 timesteps)
python train_sb3_dqn.py --timesteps 200000 --state-mode basic17

# 更长训练 (500000 timesteps)
python train_sb3_dqn.py --timesteps 500000 --state-mode basic17
```

**说明：**
- 10000 timesteps 只用于测试接口，不能判断效果
- 判断效果要看 200000 或 500000 timesteps
- Best model 自动保存在 `checkpoints/sb3_best/best_model.zip`
- Final model 保存在 `checkpoints/sb3_dqn_snake.zip`
- 使用 best_model 进行评估和观看

### 命令行参数

**train.py（手写 DQN）：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 3000 | 训练轮数 |
| `--state-mode` | basic17 | 状态模式 (basic17 / reachable23) |
| `--double-dqn` | True | 使用 Double DQN |
| `--lr` | 5e-4 | 学习率 |
| `--batch-size` | 128 | 批量大小 |
| `--eval-interval` | 100 | 评估间隔 |

**train_sb3_dqn.py（SB3 DQN）：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--timesteps` | 200000 | 训练步数 |
| `--state-mode` | basic17 | 状态模式 |
| `--lr` | 1e-4 | 学习率 |
| `--buffer-size` | 100000 | 回放缓冲区大小 |

## 8. 如何观看 Agent

### 观看手写 DQN 模型

```bash
# 观看最佳模型
python main.py --model checkpoints/best_model.pt --episodes 5 --fps 10 --terminal-render

# 观看最终模型
python main.py --model checkpoints/final_model.pt --episodes 5 --fps 10 --terminal-render
```

### 观看 SB3 DQN 模型

```bash
# 观看 SB3 模型
python main.py --model checkpoints/sb3_dqn_snake.zip --model-type sb3 --episodes 5 --fps 10 --terminal-render --state-mode basic17
```

### 通用参数

```bash
# 终端摘要模式（仅 episode 结束后输出统计）
python main.py --model checkpoints/best_model.pt --episodes 5

# 图形界面模式 (需 pygame)
python main.py --model checkpoints/best_model.pt --episodes 10 --fps 15
```

**命令行参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | checkpoints/dqn_snake.pt | 模型路径 |
| `--model-type` | auto | 模型类型 (auto/torch/sb3) |
| `--episodes` | 5 | 观看局数 |
| `--grid-size` | 20 | 网格大小 |
| `--fps` | 10 | 帧率 |
| `--terminal-render` | False | 终端逐帧渲染 |
| `--state-mode` | basic17 | 状态模式 |

**说明：**
- `--model-type auto` 会根据文件扩展名自动识别：`.pt` 用手写 DQN，`.zip` 用 SB3
- 使用 `--terminal-render` 时，每一步都显示棋盘，按 `Q` 可退出
- 不使用 `--terminal-render` 时，仅在 episode 结束后输出统计信息

## 9. 状态空间设计

项目提供两种状态模式，通过 `--state-mode` 参数切换：

### basic17（默认推荐）

17 维状态，包含危险感知、方向、食物位置、归一化特征和 Flood Fill 可达空间：

| 索引 | 特征 | 含义 |
|------|------|------|
| 0-2 | danger_straight/left/right | 危险检测 |
| 3-6 | direction_up/down/left/right | 方向 one-hot |
| 7-10 | food_left/right/up/down | 食物相对位置 |
| 11 | distance_to_food_normalized | 距食物距离归一化 |
| 12 | snake_length_normalized | 蛇身长度归一化 |
| 13 | steps_since_food_normalized | 未吃食物步数归一化 |
| 14-16 | free_space_straight/left/right | Flood Fill 可达空间 |

### reachable23（实验性）

23 维状态，在 basic17 基础上增加 Tail/Food 可达性：

| 索引 | 特征 | 含义 |
|------|------|------|
| 17-19 | tail_reachable_straight/left/right | 动作后能否到达蛇尾 |
| 20-22 | food_reachable_straight/left/right | 动作后能否到达食物 |

**推荐使用 basic17**，reachable23 训练效果可能更差（特征过多导致学习困难）。

## 10. 动作空间设计

### DQN 动作空间（相对动作）

| 动作编号 | 含义 | 说明 |
|----------|------|------|
| 0 | 直行 | 保持当前方向 |
| 1 | 左转 | 相对当前方向左转90° |
| 2 | 右转 | 相对当前方向右转90° |

**相对动作 vs 绝对动作：**
- DQN 使用相对动作（直行/左转/右转）而非绝对动作（上/下/左/右）
- 避免蛇直接反向移动导致立即死亡
- 状态空间更简洁（11维 vs 需要更多特征）

### 人工控制（绝对方向）

人工游玩使用方向键控制绝对方向（上/下/左/右），通过 `absolute_direction_to_relative_action()` 函数自动转换为相对动作：

```python
def absolute_direction_to_relative_action(current_direction, target_direction):
    diff = (target_direction - current_direction) % 4
    if diff == 0:   return 0  # 直行
    elif diff == 1: return 2  # 右转 90°
    elif diff == 3: return 1  # 左转 90°
    else:           return 0  # 反方向，保持直行（不允许）
```

**设计优势：**
- 人工玩家使用直觉化的方向键控制
- DQN 训练使用相对动作，状态空间更紧凑
- 转换函数处理了反向移动的边界情况

## 11. 奖励函数设计

| 事件 | 奖励 | 说明 |
|------|------|------|
| 吃到食物 | +10 | 正向激励 |
| 撞墙/撞自己 | -10 | 负向惩罚 |
| 普通移动 | -0.01 | 鼓励尽快找到食物 |
| 离食物更近 | +0.02 | 轻微引导（降低以避免短视） |
| 离食物更远 | -0.02 | 轻微惩罚 |
| 重复状态（绕圈） | -0.2 | 检测到同一状态出现≥3次 |
| 进入死胡同 | -0.5 | 可达空间 < 5% |
| Tail 不可达 | -0.2 | 动作后无法到达蛇尾 |
| Food 不可达 | -0.1 | 动作后无法到达食物 |
| 超过动态步数限制 | -10 | max(100, 蛇长×20) 步未吃到食物 |

**为什么降低距离奖励：**
- 原 ±0.1 过强，agent 会"追着食物跑"但不会规划路径
- 降低到 ±0.02 后，agent 更依赖 Flood Fill 特征做长期规划
- 避免"局部最优"：只学会不死，没学会吃食物

**为什么加入重复状态检测：**
- 原版 agent 容易陷入"安全绕圈"模式
- 检测 state_key = (head_pos, direction, food_pos) 的重复频率
- 最近 100 步内同一状态出现 ≥3 次即判定为绕圈

**为什么动态步数限制：**
- 固定 1000 步对短蛇太宽松，对长蛇太严格
- max(100, 蛇长×20) 让长蛇有更多时间找食物

**为什么加入 Tail/Food 可达性惩罚：**
- SelfDeath 高达 85%~100%，说明蛇变长后路径规划不足
- Tail 不可达意味着可能被困住，给予 -0.2 惩罚
- Food 不可达意味着可能封死路径，给予 -0.1 惩罚
- 惩罚较轻，避免 agent 过于保守

## 12. DQN 原理简述

### 12.1 核心思想

DQN（Deep Q-Network）是深度强化学习的经典算法，由 DeepMind 在 2013 年提出。核心思想是用深度神经网络近似 Q 值函数 $Q(s, a)$，即在状态 $s$ 下执行动作 $a$ 的预期累积回报。

### 12.2 关键组件

1. **Policy Network（策略网络）**：用于选择动作和计算 Q 值
2. **Target Network（目标网络）**：用于计算目标 Q 值，定期从 Policy Network 同步
3. **Experience Replay（经验回放）**：打破数据相关性，提高样本效率
4. **Epsilon-Greedy（ε-贪心）**：平衡探索与利用
5. **Double DQN**：减少 Q 值过估计

### 12.3 DQN vs Double DQN

**标准 DQN 的问题：**
```
target = r + γ · max_a' Q(s', a'; θ⁻)
```
使用 target_network 选择并评估动作，容易高估 Q 值（过估计），导致训练不稳定。

**Double DQN 的改进：**
```
a* = argmax_a' Q(s', a'; θ)        # policy_net 选动作
target = r + γ · Q(s', a*; θ⁻)     # target_net 评估
```
用 policy_net 选择最优动作，用 target_net 评估该动作的 Q 值，分离"选择"和"评估"，减少过估计。

### 12.4 训练流程

```
初始化 Policy Network θ, Target Network θ⁻ = θ
For each episode:
    s = env.reset()
    While not done:
        a = ε-greedy(Q(s,·; θ))     # 选择动作
        s', r, done = env.step(a)    # 执行动作
        Store (s,a,r,s',done) in Buffer  # 存储经验
        If len(Buffer) >= warmup_steps:  # 预热后开始训练
            Sample mini-batch from Buffer
            a* = argmax Q(s',·; θ)      # Double DQN: policy_net 选动作
            y = r + γ · Q(s',a*;θ⁻)     # target_net 评估
            Loss = Huber(Q(s,a;θ), y)
            Update θ by gradient descent
        Every C steps: θ⁻ = θ           # 同步目标网络
    Every eval_interval: evaluate & save best
```

### 12.5 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| γ (gamma) | 0.99 | 折扣因子 |
| lr | 5e-4 | 学习率 |
| batch_size | 128 | 批量大小 |
| replay_size | 100000 | 回放缓冲区大小 |
| target_update | 500 | 目标网络更新间隔 |
| ε_start | 1.0 | 初始探索率 |
| ε_end | 0.02 | 最终探索率 |
| ε_decay_steps | 50000 | 探索率衰减步数 |
| warmup_steps | 1000 | 预热步数 |

### 12.6 为什么 Agent 会转圈（局部最优）

1. **状态信息不足**：原 11 维状态缺少蛇身长度、可达空间等信息，agent 只能"看到眼前"
2. **距离奖励过强**：±0.1 的距离奖励让 agent 追着食物跑，但不会规划路径
3. **缺乏死胡同感知**：没有 Flood Fill，agent 无法预判"走进去出不来"
4. **绕圈没有惩罚**：agent 发现"原地转圈不会死"后，会一直转圈
5. **Q 值过估计**：标准 DQN 容易高估某些动作的 Q 值，导致策略固化

**改进方案：**
- 扩展状态到 17 维（含 Flood Fill）
- 降低距离奖励到 ±0.02
- 加入重复状态检测和惩罚
- 使用 Double DQN 减少过估计
- 定期评估 + 保存 best model

## 13. 可选安装（增强功能）

```bash
# 激活环境
conda activate nlp-env

# 安装 pygame (图形界面人工玩 + Agent 可视化)
pip install pygame

# 安装 matplotlib (训练曲线绘图)
pip install matplotlib

# 安装 gymnasium (标准 RL 接口，本项目已自实现)
pip install gymnasium

# 安装 stable_baselines3 (可选 SB3 版本，主线仍用手写 DQN)
pip install stable-baselines3
```

## 14. 已实现的优化

1. **Double DQN** ✅：减少 Q 值过估计，默认启用
2. **Flood Fill 可达空间** ✅：检测死胡同，新增 3 维状态特征
3. **两种状态模式** ✅：basic17（推荐）/ reachable23（实验）
4. **重复状态检测** ✅：检测绕圈并给予惩罚
5. **动态步数限制** ✅：max(100, 蛇长×20)
6. **Best Model 保存** ✅：定期评估，保存最优模型
7. **Warmup 预热** ✅：攒够经验后再训练
8. **奖励函数优化** ✅：降低距离奖励，增加绕圈/死胡同惩罚
9. **死亡原因统计** ✅：记录每局结束原因，评估时输出死亡比例
10. **Stable-Baselines3 DQN** ✅：成熟库版本，训练更稳定
11. **Gymnasium API 兼容** ✅：支持 gymnasium 环境检查

## 15. 死亡原因分析

每局游戏结束后，系统会记录死亡原因：

| 死亡原因 | 含义 | 诊断 |
|----------|------|------|
| `wall_collision` | 撞墙 | 基础避障能力不足，agent 未学会感知边界 |
| `self_collision` | 撞自己 | 蛇变长后空间规划不足，路径规划能力弱 |
| `no_food_timeout` | 超时未吃到食物 | 绕圈/找不到食物，策略过于保守 |
| `manual_quit` | 手动退出 | 用户主动退出 |

**评估输出示例：**
```
Episode  100 | Score:   2 | Avg50:   1.2 | Eps: 0.9800 | EvalAvg:   1.5 | EvalMax:   3 | Self:  45% | Wall:  30% | Timeout:  25%
```

**如何解读：**
- **Self% 高**：蛇变长后容易撞自己 → 需要更好的路径规划
- **Wall% 高**：基础避障差 → 需要更多训练或更好的危险感知
- **Timeout% 高**：绕圈严重 → 需要更强的绕圈惩罚或更好的状态表示

**训练日志字段：**
- `death_reason`：本局死亡原因
- `wall_collision_count`：累计撞墙次数
- `self_collision_count`：累计撞自己次数
- `no_food_timeout_count`：累计超时次数
- `eval_wall_deaths`：评估阶段撞墙次数
- `eval_self_deaths`：评估阶段撞自己次数
- `eval_timeout_deaths`：评估阶段超时次数

## 15. SB3 DQN 训练结果

### 200000 timesteps 训练结果

| 指标 | 值 |
|------|-----|
| 平均得分 | 3.95 |
| 最高得分 | 18 |
| 最低得分 | 1 |
| 平均步数 | 123.4 |
| 撞自己 | 37% |
| 撞墙 | 33% |
| 超时 | 30% |

**分析：**
- 相比 10000 timesteps（平均分 0.45，超时 90%），效果显著提升
- 死亡原因分布更均匀，说明 agent 学会了吃食物
- 建议继续训练到 500000 timesteps 以获得更好效果

### 推荐训练命令

```bash
# 正式训练
python train_sb3_dqn.py --timesteps 200000 --state-mode basic17

# 评估 best model
python evaluate_sb3.py --model checkpoints/sb3_best/best_model.zip --episodes 100 --state-mode basic17

# 观看 best model
python main.py --model checkpoints/sb3_best/best_model.zip --model-type sb3 --episodes 5 --fps 10 --terminal-render --state-mode basic17
```

## 16. SB3 timesteps 与手写 DQN episodes 的区别

| 概念 | 含义 | 说明 |
|------|------|------|
| episode | 一整局游戏 | 从开始到死亡 |
| timestep | 一次 env.step() | 一次动作执行 |

**关系：**
- 一个 episode 包含几十到几百个 timesteps
- 200000 timesteps 大约是几百到几千局
- 不能把 200000 timesteps 理解为 200000 局

**公平对比需要看：**
- total_timesteps（总步数）
- episode_count（总局数）
- avg_episode_length（平均每局步数）
- eval_avg_score（评估平均分）

## 17. 训练结果对比

| 方法 | 训练量 | 平均分 | 最高分 | 撞墙% | 撞自己% | 超时% | 备注 |
|------|--------|--------|--------|-------|---------|-------|------|
| hand_dqn_basic17_best | 3000 episodes | 1.38 | 13 | 63.3 | 29.6 | 7.1 | 手写 Double DQN |
| sb3_dqn_basic17_200k | 200000 timesteps | 3.95 | 18 | 33.0 | 37.0 | 30.0 | **推荐 SB3 模型** |
| sb3_dqn_basic17_500k_continue | 500000 timesteps | 1.54 | 4 | 71.0 | 29.0 | 0.0 | degraded |

**分析：**
- SB3 200k 效果最好，平均分 3.95，最高分 18
- 500k 继续训练反而变差（灾难性遗忘）
- 手写 DQN 3000 episodes 效果一般，但超时率低

**结论：**
- 200000 timesteps 是当前最佳训练量
- 推荐使用 `checkpoints/sb3_best/best_model.zip`

## 18. 为什么 500k Continue Training 变差

继续训练（Continue Training）可能导致效果下降，原因包括：

1. **Replay Buffer 丢失**：best_model.zip 不包含 replay buffer，继续训练时经验为空
2. **探索分布改变**：新的 exploration 参数可能破坏已学到的策略
3. **灾难性遗忘**：新数据覆盖旧知识
4. **RL 训练非单调**：强化学习不是每一步都提升，可能震荡
5. **超参数不匹配**：继续训练可能需要不同的学习率/探索率

**建议：**
- 继续训练时使用 `--load-replay-buffer` 加载 replay buffer
- 或者从零开始训练，不要 continue training
- 保存 best model，用独立 eval 判断结果

## 19. 当前推荐模型

### 推荐 SB3 模型
```
checkpoints/sb3_best/best_model.zip
avg_score = 3.95
max_score = 18
```

### 推荐观看命令
```bash
python main.py --model checkpoints/sb3_best/best_model.zip --model-type sb3 --episodes 5 --fps 10 --terminal-render --state-mode basic17
```

### 推荐评估命令
```bash
python evaluate_sb3.py --model checkpoints/sb3_best/best_model.zip --episodes 100 --state-mode basic17
```

## 20. 从零训练 500k 的推荐命令

如果需要更长训练，建议从零开始：

```bash
# 从零训练 500k
python train_sb3_dqn.py ^
  --timesteps 500000 ^
  --state-mode basic17 ^
  --run-name sb3_dqn_basic17_500k_scratch ^
  --learning-starts 10000 ^
  --exploration-fraction 0.5 ^
  --exploration-final-eps 0.05 ^
  --eval-freq 10000 ^
  --eval-episodes 30 ^
  --checkpoint-freq 50000 ^
  --save-replay-buffer

# 评估
python evaluate_sb3.py ^
  --model checkpoints/sb3_runs/sb3_dqn_basic17_500k_scratch/best_model.zip ^
  --episodes 100 ^
  --state-mode basic17 ^
  --save-path logs/sb3_runs/sb3_dqn_basic17_500k_scratch/sb3_eval.csv

# 观看
python main.py ^
  --model checkpoints/sb3_runs/sb3_dqn_basic17_500k_scratch/best_model.zip ^
  --model-type sb3 ^
  --episodes 5 ^
  --fps 10 ^
  --terminal-render ^
  --state-mode basic17
```

## 21. 继续训练 SB3 DQN（不推荐）

```bash
# 从 best model 继续训练（不推荐，可能退化）
python train_sb3_dqn.py ^
  --timesteps 500000 ^
  --state-mode basic17 ^
  --load-model checkpoints/sb3_best/best_model.zip ^
  --continue-training ^
  --run-name sb3_dqn_basic17_500k_continue ^
  --load-replay-buffer checkpoints/sb3_runs/sb3_dqn_basic17_200k/replay_buffer.pkl
```

**注意：** 继续训练可能导致灾难性遗忘，效果不一定更好。建议从零训练。
  --state-mode basic17

# 观看
python main.py ^
  --model checkpoints/sb3_runs/sb3_dqn_basic17_500k_continue/best_model/best_model.zip ^
  --model-type sb3 ^
  --episodes 5 ^
  --fps 10 ^
  --terminal-render ^
  --state-mode basic17
```

**注意：** 继续训练可能导致灾难性遗忘，效果不一定更好。建议先用 200k 模型。

## 19. 后续优化方向

1. **算法升级**：
   - Dueling DQN：分离状态价值和动作优势
   - Prioritized Experience Replay：优先回放重要经验
   - NoisyNet：用噪声网络替代 ε-greedy

2. **状态表示**：
   - 使用 CNN 处理游戏画面图像输入
   - 增加蛇尾位置、路径规划等特征

3. **训练技巧**：
   - 学习率调度（Learning Rate Scheduling）
   - 更长的训练轮数（5000+ episodes）
   - 多种子评估取平均

4. **可视化**：
   - Matplotlib 绘制训练曲线
   - TensorBoard 日志

5. **环境增强**：
   - 可变网格大小
   - 多食物模式
   - 障碍物模式

## 20. 学习路线

建议按以下顺序学习本项目：

### 第一步：理解 SnakeEnv
- 阅读 `snake_env.py`，理解状态空间、动作空间、奖励函数
- 运行 `python play_human.py`，手动玩几局，感受游戏机制

### 第二步：理解手写 DQN
- 阅读 `models.py`，理解 Q 网络结构
- 阅读 `replay_buffer.py`，理解经验回放
- 阅读 `agent.py`，理解 DQN 算法（policy network、target network、epsilon-greedy）
- 运行 `python train.py --episodes 100`，观察训练过程

### 第三步：对比 SB3 DQN
- 安装 `pip install stable-baselines3`
- 运行 `python train_sb3_dqn.py --timesteps 50000`
- 对比手写 DQN 和 SB3 DQN 的训练效果

### 第四步：深入理解
- 阅读 [CleanRL 的 dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)，理解更标准的单文件实现
- 尝试修改超参数，观察对训练效果的影响
- 尝试修改奖励函数，理解奖励工程
