# 实验脚本使用指南

## 📋 脚本列表

### 1. run_exp_baseline.sh
GRPO基线实验，无任何mismatch修正方法
```bash
bash scripts/run_exp_baseline.sh
```

### 2. run_exp_bypass.sh
GRPO + Bypass，使用rollout logprob作为参考
```bash
bash scripts/run_exp_bypass.sh
```

### 3. run_exp_tis.sh
GRPO + 截断重要性采样（TIS），支持token/sequence两种level
```bash
# Token-level TIS
bash scripts/run_exp_tis.sh token

# Sequence-level TIS（默认）
bash scripts/run_exp_tis.sh sequence
```

### 4. run_exp_perturbation.sh
GRPO + Bypass + Perturbation，支持4种loss模式
```bash
# 使用默认参数（sequence, std=0.02）
bash scripts/run_exp_perturbation.sh

# 指定loss模式
bash scripts/run_exp_perturbation.sh token        # Token-level
bash scripts/run_exp_perturbation.sh sequence     # Sequence-level
bash scripts/run_exp_perturbation.sh cum-token    # Cumulative token
bash scripts/run_exp_perturbation.sh cum-turn     # Cumulative turn

# 自定义扰动强度
bash scripts/run_exp_perturbation.sh sequence 0.05
```

## 🚀 后台运行

```bash
screen -dmS exp1 bash scripts/run_exp_baseline.sh
screen -dmS exp2 bash scripts/run_exp_bypass.sh
screen -dmS exp3 bash scripts/run_exp_tis.sh token
screen -dmS exp4 bash scripts/run_exp_tis.sh sequence
screen -dmS exp5 bash scripts/run_exp_perturbation.sh sequence
```

## 📊 WandB监控
- **Project**: mismatch_rl_research
- **Entity**: mismatch-perturbation
