# Run 2 Training Plan — multiHRI-matched

## Execution Schedule

### Step 0: BC Training (~25 min, CPU)

```bash
for SEED in 0 1 2 3 4; do
    python train_bc.py --seed $SEED --bc-epochs 100 --num-bc-agents 3
done
```

- 2×64 MLP, 3 agents × 3 layouts × 5 seeds = 45 BC models
- Eval: BC self-play (seeds 3,4)

### Step 1: SP Population (5 seeds × 16 agents = 80 agents vmapped)

```bash
python train_sp_population.py --seeds 0,1,2,3,4 \
    --total-timesteps 100000000 --pop-size 16 --num-agents 2 --wandb
```

- 100M timesteps per agent, 952 PPO updates each
- 35 checkpoints per agent = 560 total
- Eval: SP Pop self-play (seeds 3,4)

### Step 2: SP Ego (3 seeds vmapped)

```bash
python train_sp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --wandb
```

- 100M timesteps, 952 PPO updates
- Eval: SP ego self-play

### Step 3: MEP-16 Pop + Ego (5+3 seeds vmapped)

```bash
python train_mep.py --seeds 0,1,2,3,4 --ego-seeds 0,1,2 \
    --pop-timesteps 100000000 --ego-timesteps 100000000 \
    --pop-size 16 --epoch-updates 1 --num-agents 2 --num-unseen 1 --wandb
```

- Pop: 952 rounds, 1 PPO update/round, ~59 rounds/agent
- Ego: 952 PPO updates with prioritized partner sampling
- Eval: MEP-16 Pop self-play (seeds 3,4), MEP-16 ego self-play

### Step 4: MEP-8 Pop + Ego

Same as Step 3 with `--pop-size 8`. ~119 rounds/agent.

### Step 5: BCP Ego (3 seeds vmapped)

```bash
python train_bcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 \
    --bc-anneal-horizon 8000 --wandb
```

- 3 layouts only, 100M timesteps
- Eval: BCP ego self-play

### Step 6: FCP Ego (3 seeds vmapped)

```bash
python train_fcp.py --seeds 0,1,2 \
    --total-timesteps 100000000 --num-agents 2 --num-unseen 1 --wandb
```

- Pool: 560 checkpoints (16 agents × 35 ckpts), H/M/L uniform
- 100M timesteps
- Eval: FCP ego self-play

### Step 7: Cross-play evaluation

All ego agents (seeds 0,1,2) vs all unseen partners (seeds 3,4):
- vs 32 SP partners (16×2 seeds)
- vs 32 MEP-16 partners (16×2)
- vs 16 MEP-8 partners (8×2)
- vs 18 BC partners (3×3×2)

Print final tables.

## Key Differences from Run 1

| What changed | Run 1 | Run 2 |
|-------------|-------|-------|
| Network | 3×64 (180K params) | **2×256 (400K params)** |
| Data per PPO update | 5,120 transitions | **105,000 transitions** |
| Total timesteps | 60-120M | **100M all methods** |
| Reward | sparse + shaped | **Originaler ratio blend** |
| Position diversity | None | **Slot shuffle each update** |
| MEP training | 20 updates/round | **1 update/round** |
| Checkpoints | 15 | **35** |
