# Snake RL 项目最终总结

## 1. 项目目标

实现一个 Snake 贪吃蛇游戏，并用强化学习训练 agent 自动玩游戏。比较手写 DQN 和 Stable-Baselines3 DQN 的效果，探索不同 DQN 变体的效果。

## 2. 最终结论

在本项目中，最有效的方法是 **Dueling Double DQN basic17**。复杂增强并未稳定带来提升，部分方法甚至明显退化。最终推荐模型为 `checkpoints/best_model_dueling_basic17.pt`。

## 3. 最终推荐模型

**Dueling Double DQN basic17**
```
checkpoints/best_model_dueling_basic17.pt
avg_score = 34.51 ± 1.25 (多 seed 复评)
max_score = 64.0
历史最佳: avg=35.40, max=67
```

## 4. 最终对比表

| 方法 | Avg Score | Std | Max Score | 状态 |
|------|-----------|-----|-----------|------|
| **Dueling Double DQN basic17** | **34.51** | **1.25** | **64.0** | **推荐** |
| config_A_seed_1_3000 | 32.95 | 0.92 | 70.6 | max更高但avg更低 |
| SB3 vanilla DQN 200k | 4.40 | - | 14 | baseline |
| Action Mask | 2.89 | - | 11 | 有害 |
| Safety Filter | 3.31 | 0.22 | 3.6 | 过度保守 |
| Grid CNN | 0.22 | - | 1 | 未学起来 |
| PER | 28.47 | - | 48 | 未超过dueling |
| n-step | 3.29 | - | 15 | 明显失败 |
| NoisyNet | 27.50 | - | 42 | 未超过dueling |
| Strong Teacher | 13.12 | 10.56 | 53 | 弱于DQN |

## 5. 失败路线总结

1. **Action Mask** (avg=2.89): 显著有害，改变训练分布
2. **Safety Filter** (avg=3.31): 过度保守，timeout接近100%
3. **Grid CNN** (avg=0.22): 当前训练规模下未学起来
4. **Rainbow-lite**: PER/n-step/NoisyNet均未超过Dueling-only
5. **DQfD-lite**: planner avg=13.12，teacher弱于DQN
6. **Dueling Sweep**: config_A未超过current best

## 6. 后续研究方向

如果继续研究，可以考虑：
- 更严谨的统计检验（如Mann-Whitney U检验）
- 更大规模seed sweep（50+ seeds）
- 从零设计更强planner
- Full-grid observation + 更长训练
- 但这些不属于当前项目必要范围
