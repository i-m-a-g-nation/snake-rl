# Snake RL 项目最终总结

## 1. 项目目标

实现一个 Snake 贪吃蛇游戏，并用强化学习训练 agent 自动玩游戏。比较手写 DQN 和 Stable-Baselines3 DQN 的效果。

## 2. 环境与依赖

| 项目 | 值 |
|------|-----|
| Python | 3.10.20 |
| Conda 环境 | nlp-env |
| PyTorch | 2.6.0+cu124 |
| CUDA | RTX 4060 Laptop GPU |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.3.0 |

## 3. 最终推荐模型

### 最强 Agent：手写 Double DQN
```
checkpoints/best_model_basic17.pt
avg_score = 23.93
max_score = 57
state_mode = basic17 (17 维)
```

观看命令：
```bash
python main.py --model checkpoints/best_model_basic17.pt --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode basic17
```

### 标准库 Baseline：SB3 DQN
```
checkpoints/sb3_best/best_model.zip
avg_score = 4.40
max_score = 14
state_mode = basic17
```

## 4. 手写 DQN vs SB3 DQN 对比表

| 方法 | 训练量 | 平均分 | 最高分 | 撞墙% | 撞自己% | 超时% | 备注 |
|------|--------|--------|--------|-------|---------|-------|------|
| **手写 Double DQN** | 500 episodes | **23.93** | **57** | 22.0 | 78.0 | 0.0 | 最强 agent |
| SB3 vanilla DQN | 200k timesteps | 4.40 | 14 | 25.0 | 60.0 | 15.0 | 标准库 baseline |
| SB3 500k continue | 500k timesteps | 1.54 | 4 | 71.0 | 29.0 | 0.0 | 退化实验 |

## 5. 为什么手写 DQN 更强

1. **Double DQN**：分离动作选择和评估，减少 Q 值过估计
2. **更好的状态表示**：17 维特征包含 flood fill 可达空间
3. **定制奖励函数**：绕圈惩罚、死胡同检测、动态步数限制
4. **更长训练**：500 episodes vs SB3 的 200k timesteps（约 1000+ episodes）

## 6. 为什么 23 维 reachable 特征失败

1. **特征过多**：23 维相比 17 维增加 6 个特征
2. **学习困难**：更多特征需要更多训练数据
3. **过拟合风险**：特征过多可能导致过拟合
4. **计算开销**：flood fill 计算较慢

结论：basic17 是更好的默认选择。

## 7. 为什么 500k Continue 失败

1. **灾难性遗忘**：继续训练覆盖已有知识
2. **Replay Buffer 丢失**：best_model 不含 replay buffer
3. **探索分布改变**：新超参数破坏策略
4. **RL 非单调**：强化学习不是每一步都提升

教训：继续训练需要更谨慎，建议从零训练。

## 8. 学到的 RL 关键概念

1. **Q-learning**：用 Q 值函数评估状态-动作对的价值
2. **Deep Q-Network**：用神经网络近似 Q 值函数
3. **Double DQN**：分离动作选择和评估，减少过估计
4. **Experience Replay**：打破数据相关性，提高样本效率
5. **Target Network**：稳定训练，定期同步
6. **Epsilon-Greedy**：平衡探索与利用
7. **Reward Engineering**：设计合适的奖励函数
8. **State Engineering**：设计合适的特征表示

## 9. 后续如果继续改

可以尝试：

1. **Dueling DQN**：分离状态价值和动作优势
2. **Prioritized Experience Replay**：优先回放重要经验
3. **QR-DQN**：分位数回归 DQN
4. **Rainbow DQN**：结合多种改进
5. **更长训练**：1000+ episodes
6. **超参数调优**：学习率、探索率衰减

**注意：** 这些不是当前必要任务，当前手写 Double DQN 已经足够好。

## 10. 项目文件结构

```
snake_rl/
├── README.md              # 项目说明
├── FINAL_SUMMARY.md       # 最终总结（本文件）
├── requirements.txt       # 依赖列表
├── environment_report.txt # 环境检测报告
├── snake_game.py          # 游戏逻辑
├── snake_env.py           # RL 环境
├── agent.py               # DQN Agent
├── models.py              # Q 网络
├── replay_buffer.py       # 经验回放
├── train.py               # 手写 DQN 训练
├── train_sb3_dqn.py       # SB3 DQN 训练
├── evaluate_torch.py      # 手写 DQN 评估
├── evaluate_sb3.py        # SB3 DQN 评估
├── compare_results.py     # 结果对比
├── recommend_model.py     # 模型推荐
├── main.py                # 观看 Agent
├── play_human.py          # 人工游玩
├── terminal_input.py      # 终端输入
├── utils.py               # 工具函数
├── checkpoints/           # 模型保存
├── logs/                  # 训练日志
└── outputs/               # 对比结果
```
