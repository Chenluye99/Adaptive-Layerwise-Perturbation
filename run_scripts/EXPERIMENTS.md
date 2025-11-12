# 实验脚本完整指南

## 📋 脚本列表

### 1. run_exp_baseline.sh - GRPO基线
```bash
bash scripts/run_exp_baseline.sh
```

### 2. run_exp_bypass.sh - GRPO + Bypass
```bash
bash scripts/run_exp_bypass.sh
```

### 3. run_exp_tis.sh - GRPO + TIS（支持全部高级参数）
```bash
# 基础用法
bash scripts/run_exp_tis.sh [level] [mode] [threshold] [veto] [geometric]

# 示例
bash scripts/run_exp_tis.sh sequence              # 默认：Sequence TIS
bash scripts/run_exp_tis.sh token                 # Token TIS
bash scripts/run_exp_tis.sh cum-token             # Cumulative token TIS
bash scripts/run_exp_tis.sh cum-turn              # Cumulative turn TIS
bash scripts/run_exp_tis.sh sequence mask 3.0     # Mask mode (CIS)
bash scripts/run_exp_tis.sh token truncate 5.0 0.001  # With veto
bash scripts/run_exp_tis.sh cum-token truncate 5.0 0.0 true  # Geometric
```

### 4. run_exp_perturbation.sh - Perturbation（支持4种模式）
```bash
# 基础用法
bash scripts/run_exp_perturbation.sh [loss_mode] [std] [geometric]

# 示例
bash scripts/run_exp_perturbation.sh              # 默认：sequence, 0.02
bash scripts/run_exp_perturbation.sh token        # Token-level
bash scripts/run_exp_perturbation.sh sequence     # Sequence-level
bash scripts/run_exp_perturbation.sh cum-token    # Cumulative token
bash scripts/run_exp_perturbation.sh cum-turn     # Cumulative turn
bash scripts/run_exp_perturbation.sh sequence 0.05  # 自定义std
bash scripts/run_exp_perturbation.sh sequence 0.02 true  # Geometric
```

---

## 🎯 参数详解

### TIS参数（run_exp_tis.sh）

| 参数 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| **level** | token/sequence/cum-token/cum-turn | sequence | IS聚合级别 |
| **mode** | truncate/mask | truncate | TIS或CIS模式 |
| **threshold** | 数值 | 5.0 | IS权重上限 |
| **veto** | 数值 | 0.0 | Veto阈值（0=禁用） |
| **geometric** | true/false | false | 几何平均 |

### Perturbation参数（run_exp_perturbation.sh）

| 参数 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| **loss_mode** | token/sequence/cum-token/cum-turn | sequence | 损失计算模式 |
| **perturb_std** | 数值 | 0.02 | 扰动标准差 |
| **geometric** | true/false | false | 几何平均 |

---

## 📊 实验组合示例

```bash
# === 基础实验 ===
bash scripts/run_exp_baseline.sh
bash scripts/run_exp_bypass.sh

# === TIS实验（4种level × 2种mode） ===
# Truncate mode (TIS)
bash scripts/run_exp_tis.sh token truncate 5.0
bash scripts/run_exp_tis.sh sequence truncate 5.0
bash scripts/run_exp_tis.sh cum-token truncate 5.0
bash scripts/run_exp_tis.sh cum-turn truncate 5.0

# Mask mode (CIS)
bash scripts/run_exp_tis.sh token mask 3.0
bash scripts/run_exp_tis.sh sequence mask 3.0

# === Perturbation实验（4种mode × 3种std） ===
# Sequence-level with different std
bash scripts/run_exp_perturbation.sh sequence 0.01
bash scripts/run_exp_perturbation.sh sequence 0.02
bash scripts/run_exp_perturbation.sh sequence 0.05

# All modes with std=0.02
bash scripts/run_exp_perturbation.sh token 0.02
bash scripts/run_exp_perturbation.sh sequence 0.02
bash scripts/run_exp_perturbation.sh cum-token 0.02
bash scripts/run_exp_perturbation.sh cum-turn 0.02
```

---

## 🚀 后台批量运行

```bash
# 运行核心对比实验
screen -dmS exp1 bash scripts/run_exp_baseline.sh
screen -dmS exp2 bash scripts/run_exp_bypass.sh
screen -dmS exp3 bash scripts/run_exp_tis.sh token
screen -dmS exp4 bash scripts/run_exp_tis.sh sequence
screen -dmS exp5 bash scripts/run_exp_perturbation.sh sequence

# 查看状态
screen -ls
```

---

## 📈 WandB监控

**Project**: mismatch_rl_research  
**Entity**: mismatch-perturbation

**关键指标**:
- `actor/is_ratio/mean` - IS权重均值
- `rollout_corr/rollout_is_eff_sample_size` - 有效样本大小
- `train/reward_mean` - 训练奖励
