# Snake RL 项目最终总结

## 1. 项目目标

实现一个 Snake 贪吃蛇游戏，并用强化学习训练 agent 自动玩游戏。比较手写 DQN 和 Stable-Baselines3 DQN 的效果，探索不同 DQN 变体的效果。

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

## 3. 最终推荐模型

**Dueling Double DQN basic17**
```
checkpoints/best_model_dueling_basic17.pt
avg_score = 35.40 ± 0.76 (多 seed)
max_score = 67
```

观看命令：
```bash
python main.py --model checkpoints/best_model_dueling_basic17.pt --model-type torch --episodes 5 --fps 10 --terminal-render --state-mode basic17
```

## 4. 核心发现

### Dueling DQN 有效
- Dueling 架构通过分离 V(s) 和 A(s,a) 改善状态价值估计
- dueling_only: avg=35.40, max=67
- baseline_current: avg=30.27, max=66
- 提升约 17%

### Action Mask 有害
- mask_only: avg=2.89（远低于 baseline 30.27）
- 可能原因：Snake 只有 3 个动作，mask 改变训练分布，减少 agent 从死亡负反馈中学习的机会
- 不再推荐 action mask

### SB3 DQN 是 baseline
- SB3 200k: avg=4.40, max=14
- 不是最强，但作为标准库对照

### 继续训练不一定提升
- SB3 500k continue: avg=1.54（退化）
- 说明 RL 训练不是单调提升

## 5. Ablation 结果

| 方法 | 平均分 | 最高分 | 备注 |
|------|--------|--------|------|
| **dueling_only** | **35.40** | **67** | 当前最强 |
| old_best_basic17 | 35.36 | 55 | 旧模型 |
| baseline_current | 30.27 | 66 | 纯 Double DQN |
| mask_dueling | 16.14 | 36 | Mask 混合 |
| SB3 200k | 4.40 | 14 | SB3 baseline |
| mask_only | 2.89 | 11 | Action Mask 有害 |
| SB3 500k continue | 1.54 | 4 | 退化实验 |

## 6. 后续实验方向

如果继续改进，应单独实验：
- dueling + PER
- dueling + n-step
- dueling + PER + n-step

**不要再使用 action mask。**

## 7. 项目文件结构

```
snake_rl/
├── README.md
├── FINAL_SUMMARY.md
├── requirements.txt
├── environment_report.txt
├── snake_game.py
├── snake_env.py
├── agent.py
├── models.py
├── replay_buffer.py
├── prioritized_replay_buffer.py
├── n_step_buffer.py
├── train.py
├── train_sb3_dqn.py
├── evaluate_torch.py
├── evaluate_sb3.py
├── compare_results.py
├── recommend_model.py
├── main.py
├── play_human.py
├── terminal_input.py
├── utils.py
├── scripts/run_ablation.py
├── checkpoints/
│   ├── best_model_dueling_basic17.pt  # 当前推荐
│   ├── best_model_basic17.pt          # 旧模型
│   └── ablation/                      # ablation 实验
├── logs/
│   ├── ablation/
│   └── sb3_runs/
└── outputs/
    └── compare_results.csv
```
