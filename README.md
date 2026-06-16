# Snake 贪吃蛇强化学习项目

## 1. 项目简介

本项目实现了一个经典的贪吃蛇（Snake）游戏，并使用深度强化学习（Deep Q-Network, DQN）训练智能体（Agent）自动玩游戏。项目采用纯 PyTorch 手写 DQN 算法，不依赖 stable_baselines3 等第三方 RL 库，便于理解算法原理，适合课程设计报告。

**核心特性：**
- 纯 PyTorch 手写 DQN（Policy Network + Target Network）
- 经验回放（Experience Replay）
- Epsilon-Greedy 探索策略
- Gymnasium 风格环境接口（自实现，不强制依赖 gymnasium）
- 低维状态输入（11维特征向量）
- CUDA 加速训练（自动检测 GPU）

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
├── play_human.py          # 人工游玩入口
├── train.py               # 训练入口
├── agent.py               # DQN Agent 实现
├── snake_env.py           # Snake RL 环境 (Gymnasium 风格)
├── snake_game.py          # Snake 游戏纯逻辑
├── replay_buffer.py       # 经验回放缓冲区
├── models.py              # DQN 神经网络模型
├── utils.py               # 工具函数
├── checkpoints/           # 模型保存目录
└── logs/                  # 训练日志目录
```

## 6. 如何人工游玩

### 终端版（无需 pygame）
```bash
conda activate nlp-env
cd snake_rl
python play_human.py
```

控制方式：
- `w` / `i` = 左转
- `o` = 直行
- `p` = 右转
- `q` = 退出
- `r` = 重新开始

### 图形界面版（需安装 pygame）
```bash
pip install pygame
python play_human.py
```
使用方向键或 WASD 控制。

## 7. 如何训练

```bash
conda activate nlp-env
cd snake_rl

# 快速测试 (50 episodes)
python train.py --episodes 50

# 标准训练 (1000 episodes)
python train.py --episodes 1000

# 自定义参数
python train.py --episodes 2000 --seed 123 --save-path checkpoints/my_model.pt
```

**命令行参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 1000 | 训练轮数 |
| `--render` | False | 是否渲染 |
| `--save-path` | checkpoints/dqn_snake.pt | 模型保存路径 |
| `--seed` | 42 | 随机种子 |

**训练输出示例：**
```
Episode    10 | Score:   0 | Reward:   -10.01 | Eps: 0.9990 | Best:   0 | Avg50: 0.0
Episode   100 | Score:   2 | Reward:    15.48 | Eps: 0.9900 | Best:   3 | Avg50: 1.2
Episode   500 | Score:   8 | Reward:    72.31 | Eps: 0.9500 | Best:  12 | Avg50: 5.6
```

## 8. 如何观看 Agent

```bash
# 终端模式
python main.py --model checkpoints/dqn_snake.pt --episodes 5

# 图形界面模式 (需 pygame)
python main.py --model checkpoints/dqn_snake.pt --episodes 10 --fps 15
```

## 9. 状态空间设计

状态为 **11 维二值特征向量**，无需图像输入，训练高效：

| 索引 | 特征 | 含义 |
|------|------|------|
| 0 | danger_straight | 前方是否有危险（墙/自身） |
| 1 | danger_left | 左侧是否有危险 |
| 2 | danger_right | 右侧是否有危险 |
| 3 | direction_up | 当前方向是否朝上 |
| 4 | direction_down | 当前方向是否朝下 |
| 5 | direction_left | 当前方向是否朝左 |
| 6 | direction_right | 当前方向是否朝右 |
| 7 | food_left | 食物是否在左边 |
| 8 | food_right | 食物是否在右边 |
| 9 | food_up | 食物是否在上方 |
| 10 | food_down | 食物是否在下方 |

**设计理由：**
- 低维状态，MLP 即可处理，无需 CNN
- 特征工程合理，包含危险感知、方向、食物位置
- 训练收敛快，适合课程设计演示

## 10. 动作空间设计

| 动作编号 | 含义 | 说明 |
|----------|------|------|
| 0 | 直行 | 保持当前方向 |
| 1 | 左转 | 相对当前方向左转90° |
| 2 | 右转 | 相对当前方向右转90° |

**相对动作 vs 绝对动作：**
- 使用相对动作（直行/左转/右转）而非绝对动作（上/下/左/右）
- 避免蛇直接反向移动导致立即死亡
- 更符合人类直觉

## 11. 奖励函数设计

| 事件 | 奖励 | 说明 |
|------|------|------|
| 吃到食物 | +10 | 正向激励 |
| 撞墙/撞自己 | -10 | 负向惩罚 |
| 普通移动 | -0.01 | 鼓励尽快找到食物 |
| 离食物更近 | +0.1 | 引导向食物移动 |
| 离食物更远 | -0.1 | 惩罚远离食物 |
| 超过最大步数 | -5 | 防止无限绕圈 |

## 12. DQN 原理简述

### 12.1 核心思想

DQN（Deep Q-Network）是深度强化学习的经典算法，由 DeepMind 在 2013 年提出。核心思想是用深度神经网络近似 Q 值函数 $Q(s, a)$，即在状态 $s$ 下执行动作 $a$ 的预期累积回报。

### 12.2 关键组件

1. **Policy Network（策略网络）**：用于选择动作和计算 Q 值
2. **Target Network（目标网络）**：用于计算目标 Q 值，定期从 Policy Network 同步
3. **Experience Replay（经验回放）**：打破数据相关性，提高样本效率
4. **Epsilon-Greedy（ε-贪心）**：平衡探索与利用

### 12.3 训练流程

```
初始化 Policy Network θ, Target Network θ⁻ = θ
For each episode:
    s = env.reset()
    While not done:
        a = ε-greedy(Q(s,·; θ))     # 选择动作
        s', r, done = env.step(a)    # 执行动作
        Store (s,a,r,s',done) in Buffer  # 存储经验
        Sample mini-batch from Buffer    # 采样
        y = r + γ·max_a' Q(s',a';θ⁻)   # 计算目标
        Loss = Huber(Q(s,a;θ), y)       # 计算损失
        Update θ by gradient descent     # 更新网络
        Every C steps: θ⁻ = θ           # 同步目标网络
```

### 12.4 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| γ (gamma) | 0.99 | 折扣因子 |
| lr | 1e-3 | 学习率 |
| batch_size | 64 | 批量大小 |
| replay_size | 50000 | 回放缓冲区大小 |
| target_update | 1000 | 目标网络更新间隔 |
| ε_start | 1.0 | 初始探索率 |
| ε_end | 0.05 | 最终探索率 |
| ε_decay_steps | 10000 | 探索率衰减步数 |

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

## 14. 后续优化方向

1. **算法升级**：
   - Double DQN：减少 Q 值过估计
   - Dueling DQN：分离状态价值和动作优势
   - Prioritized Experience Replay：优先回放重要经验
   - NoisyNet：用噪声网络替代 ε-greedy

2. **状态表示**：
   - 使用 CNN 处理游戏画面图像输入
   - 增加蛇身长度、食物距离等特征

3. **奖励工程**：
   - 蛇身长度奖励
   - 存活时间奖励
   - 曲线路径惩罚

4. **训练技巧**：
   - 学习率调度（Learning Rate Scheduling）
   - 梯度裁剪调优
   - 更长的训练轮数（5000+ episodes）

5. **可视化**：
   - Matplotlib 绘制训练曲线
   - TensorBoard 日志
   - Pygame 实时可视化

6. **环境增强**：
   - 可变网格大小
   - 多食物模式
   - 障碍物模式
