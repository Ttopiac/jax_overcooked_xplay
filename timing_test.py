"""Timing test: FCP ego with IMPALA shallow CNN + dual encoder.

6 seeds, 1M steps, 8×8 tiles palette-indexed.
Ego: pixel CNN. Partners: egocentric MLP (Run 3's SP Pop).
Measures post-compile per-update time to estimate full 100M run.
"""
import os
import time
import pickle
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import jax
import jax.numpy as jnp
from overcooked_env import Overcooked, overcooked_layouts
from networks import ActorCritic, IMPALACNNActorCritic
from obs_encoding import ObsEncoder
from ppo import create_train_state
from train_fcp import make_fcp_train, get_config, LAYOUT_NAMES
from evaluate import categorize_by_skill_per_layout
from reward_shaping import RewardShaper

class Args:
    seeds = '0,1,2,3,4,5'
    total_timesteps = 1_000_000
    num_agents = 2
    random_start_pos = False
    num_unseen = 1
    xplay_mode = 'hetero'
    num_envs_per_layout = 42
    num_steps = 500
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    vf_coef = 0.3
    ent_coef = 0.01
    num_epochs = 4
    minibatch_size = 512
    hidden_dim = 256
    num_layers = 2
    obs_encoding = 'pixel'
    reward_mode = 'originaler'
    reward_shaping_factor = 1.0
    reward_shaping_horizon = float('inf')

args = Args()
seeds = [int(s) for s in args.seeds.split(',')]
config = get_config(args)
n_updates = config['num_updates']

print(f'=== FCP Ego Timing Test ===')
print(f'Seeds: {seeds}, num_updates: {n_updates}')
print(f'8×8 tiles, palette-indexed, IMPALA shallow CNN')
print(f'Ego: pixel CNN, Partners: egocentric MLP')
print()

# Set up encoder and networks
encoder = ObsEncoder(mode='pixel')
network = IMPALACNNActorCritic(action_dim=6, palette=encoder.atlas.palette)
partner_encoder = ObsEncoder(mode='egocentric')

# Load SP Pop partner pool (from Run 3)
with open('checkpoints/sp_pop_seed0.pkl', 'rb') as f:
    sp_data = pickle.load(f)
partner_dims = sp_data['config'].get('hidden_dims', (256, 256))
partner_network = ActorCritic(action_dim=6, hidden_dims=partner_dims)

reward_shaper = RewardShaper(
    mode=config['reward_mode'],
    reward_shaping_factor=config['reward_shaping_factor'],
    reward_shaping_horizon=config['reward_shaping_horizon'],
)

# Build per-seed partner pool (same as FCP main)
# SP Pop only has seeds 0-4, so cycle for seeds >= 5
all_pool_params = []
for seed in seeds:
    pop_seed = seed % 5
    pop_path = f'checkpoints/sp_pop_seed{pop_seed}.pkl'
    scores_path = f'checkpoints/sp_pop_seed{pop_seed}_layout_scores.pkl'
    with open(pop_path, 'rb') as f:
        pop_data = pickle.load(f)
    with open(scores_path, 'rb') as f:
        scores_data = pickle.load(f)
    ckpt_params = pop_data['ckpt_params']
    layout_scores = scores_data['layout_scores']
    flat_params = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), ckpt_params)
    pool = categorize_by_skill_per_layout(
        jax.device_put(flat_params), jnp.array(layout_scores), LAYOUT_NAMES
    )
    all_pool_params.append(pool)
stacked_pool = jax.tree.map(lambda *xs: jnp.stack(xs), *all_pool_params)

# Build training function
fcp_train_fn = make_fcp_train(config, network, encoder, reward_shaper,
                               partner_network=partner_network, partner_encoder=partner_encoder)
multi_fcp_fn = jax.jit(jax.vmap(fcp_train_fn))

rngs = jnp.stack([jax.random.PRNGKey(s + 2000) for s in seeds])

# Run 1: compile + execute
print(f'Compiling + running {len(seeds)} seeds...')
t0 = time.time()
results = multi_fcp_fn(rngs, stacked_pool)
jax.block_until_ready(results)
t1 = time.time()
compile_plus_run = t1 - t0
print(f'Compile + run: {compile_plus_run:.1f}s ({compile_plus_run/60:.1f} min)')

# Run 2: pure execution (no compile)
rngs2 = jnp.stack([jax.random.PRNGKey(s + 5000) for s in seeds])
print(f'Running again (no compile)...')
t2 = time.time()
results2 = multi_fcp_fn(rngs2, stacked_pool)
jax.block_until_ready(results2)
t3 = time.time()
scan_time = t3 - t2

compile_time = compile_plus_run - scan_time
per_update = scan_time / n_updates
per_agent_per_update = per_update / len(seeds)

# Extrapolate to 100M steps
real_updates = 952  # 100M / (500 * 210)
est_full = per_update * real_updates
est_per_agent = per_agent_per_update * real_updates

for i, seed in enumerate(seeds):
    r = float(results2['metrics']['mean_reward'][i, -1])
    print(f'  seed {seed}: reward={r:.2f}')

print()
print(f'=== Results ===')
print(f'JIT compile: {compile_time:.1f}s ({compile_time/60:.1f} min)')
print(f'Scan time ({n_updates} updates, {len(seeds)} seeds): {scan_time:.1f}s')
print(f'Per update: {per_update:.2f}s')
print(f'Per agent per update: {per_agent_per_update:.3f}s')
print()
print(f'=== Extrapolation to 100M steps ===')
print(f'952 updates, {len(seeds)} seeds: {est_full/3600:.1f}h')
print(f'Per agent: {est_per_agent/3600:.1f}h')

# Write report
with open('reports/timing_test.md', 'w') as f:
    f.write('# Timing Test: FCP Ego with IMPALA Shallow CNN\n\n')
    f.write(f'Architecture: IMPALA shallow (2 conv layers, 185K params)\n')
    f.write(f'Obs: 8×8 tiles, palette-indexed (1 byte/pixel)\n')
    f.write(f'Ego: pixel CNN, Partners: egocentric MLP\n')
    f.write(f'Seeds: {len(seeds)}, envs/layout: 42, steps: 500\n\n')
    f.write(f'| Metric | Value |\n')
    f.write(f'|--------|-------|\n')
    f.write(f'| JIT compile | {compile_time:.1f}s |\n')
    f.write(f'| Scan time ({n_updates} updates) | {scan_time:.1f}s |\n')
    f.write(f'| Per update | {per_update:.2f}s |\n')
    f.write(f'| Per agent per update | {per_agent_per_update:.3f}s |\n')
    f.write(f'| Est. 100M steps ({len(seeds)} seeds) | {est_full/3600:.1f}h |\n')
    f.write(f'| Est. per agent (100M) | {est_per_agent/3600:.1f}h |\n')

print(f'\nReport saved to reports/timing_test.md')
