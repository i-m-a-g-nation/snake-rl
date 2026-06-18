# Snake RL 项目最终决策

## 1. 最终推荐模型

```
checkpoints/best_model_dueling_basic17.pt
Dueling Double DQN basic17
avg_score = 34.51 ± 1.25
max_score = 64.0
```

## 2. 为什么选择 Dueling Double DQN basic17

1. **最高稳定平均分**：34.51 ± 1.25，是所有实验中最高的
2. **一致性好**：5 seed 评估标准差仅 1.25
3. **训练成熟**：已经过充分训练和验证
4. **简单高效**：没有复杂增强，反而效果最好

## 3. 为什么不选择 config_A_seed_1_3000

虽然 config_A_seed_1_3000 的 max_score=70.6 高于 current best 的 64.0，但：
- avg_score=32.95 低于 current best 的 34.51
- 平均分比最高分更能代表策略稳定性
- 不能因为单次高分就选择平均分更低的模型

## 4. 为什么不再继续训练

1. 已尝试多种增强路线，均未稳定超过 baseline
2. 超参数 sweep 未找到更优配置
3. 更长训练（如 10000 episodes）可能有提升，但收益递减
4. 当前模型已经足够好，满足项目目标

## 5. 失败路线总结

| 路线 | 结果 | 原因 |
|------|------|------|
| Action Mask | avg=2.89 | 改变训练分布，显著有害 |
| Safety Filter | avg=3.31 | 过度保守，timeout接近100% |
| Grid CNN | avg=0.22 | 当前训练规模下未学起来 |
| PER | avg=28.47 | 未超过Dueling-only |
| n-step | avg=3.29 | 明显失败 |
| NoisyNet | avg=27.50 | 未超过Dueling-only |
| DQfD-lite | teacher avg=13.12 | teacher弱于DQN |
| Dueling Sweep | avg=32.95 | 未超过current best |

## 6. 后续研究方向

如果继续研究，可以考虑：
1. 更严谨的统计检验（如 Mann-Whitney U 检验）
2. 更大规模 seed sweep（50+ seeds）
3. 从零设计更强 planner
4. Full-grid observation + 更长训练
5. 但这些不属于当前项目必要范围
