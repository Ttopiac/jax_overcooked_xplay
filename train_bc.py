"""Behavioral Cloning (BC) model training for Overcooked.

Trains BC models on human demonstration data from Carroll et al. (2019).
Per-layout training with 64×64 MLP (matching Carroll's architecture).

Trains N BC agents per layout (with different seeds) to create a small
BC population for BCP training.

Usage:
  python train_bc.py --seed 0 --num-bc-agents 3
"""

import argparse
import ast
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training.train_state import TrainState

from networks import ActorCritic
from obs_encoding import EGO_FLAT_DIM, EGO_SIZE, EGO_CHANNELS

OVERCOOKED_AI_PATH = "/home/nathan/overcooked_ai_carroll/src"

# Layouts with human data available
BC_LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring"]

ACTION_MAP = {
    (0, -1): 0,  (0, 1): 1,  (1, 0): 2,  (-1, 0): 3,  (0, 0): 4,
}
INTERACT_IDX = 5

DIR_TO_IDX = {(0, -1): 0, (0, 1): 1, (1, 0): 2, (-1, 0): 3}
ROT_K = {0: 2, 1: 0, 2: 3, 3: 1}

# Carroll's BC architecture: 64×64 (NOT 256×256)
BC_HIDDEN_DIMS = (64, 64)


def parse_action(action):
    if isinstance(action, str) and action == "INTERACT":
        return INTERACT_IDX
    return ACTION_MAP[tuple(action)]


def egocentric_obs_numpy(obs_grid, player_pos, player_orientation):
    C = obs_grid.shape[2]
    pad = EGO_SIZE // 2
    obs_chw = obs_grid.transpose(2, 0, 1).astype(np.float32)
    padded = np.pad(obs_chw, ((0, 0), (pad, pad), (pad, pad)))
    x, y = player_pos
    window = padded[:, x:x + EGO_SIZE, y:y + EGO_SIZE]
    dir_idx = DIR_TO_IDX[tuple(player_orientation)]
    k = ROT_K[dir_idx]
    if k > 0:
        window = np.rot90(window, k=k, axes=(1, 2))
    return window.flatten()


def load_and_process_data(data_path, layouts=None):
    sys.path.insert(0, OVERCOOKED_AI_PATH)
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

    print(f"Loading data from {data_path}...")
    df = pd.read_pickle(data_path)
    if layouts:
        df = df[df["layout_name"].isin(layouts)]

    result = {}
    for layout_name in df["layout_name"].unique():
        layout_df = df[df["layout_name"] == layout_name]
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        all_obs, all_actions = [], []

        for _, row in layout_df.iterrows():
            state_raw = row["state"]
            state_dict = json.loads(state_raw) if isinstance(state_raw, str) else state_raw
            state = OvercookedState.from_dict(state_dict)
            ja_raw = row["joint_action"]
            ja = ast.literal_eval(ja_raw) if isinstance(ja_raw, str) else ja_raw
            obs_both = mdp.lossless_state_encoding(state, horizon=400)

            for p_idx in range(2):
                obs_grid = np.array(obs_both[p_idx])
                player = state.players[p_idx]
                ego_obs = egocentric_obs_numpy(obs_grid, player.position, player.orientation)
                all_obs.append(ego_obs)
                all_actions.append(parse_action(ja[p_idx]))

        obs_array = np.stack(all_obs)
        action_array = np.array(all_actions, dtype=np.int32)
        result[layout_name] = (obs_array, action_array)
        print(f"  {layout_name}: {len(obs_array)} samples, "
              f"action dist: {np.bincount(action_array, minlength=6)}")
    return result


def train_bc(obs, actions, config, rng):
    """Train one BC model with cross-entropy loss."""
    network = ActorCritic(action_dim=6, hidden_dims=BC_HIDDEN_DIMS)

    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros((EGO_FLAT_DIM,)))
    tx = optax.adam(config["bc_lr"])
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    n = len(obs)
    n_val = int(n * config["val_split"])
    rng, split_rng = jax.random.split(rng)
    perm = jax.random.permutation(split_rng, n)
    obs, actions = obs[perm], actions[perm]
    val_obs, val_actions = jnp.array(obs[:n_val]), jnp.array(actions[:n_val])
    train_obs, train_actions = jnp.array(obs[n_val:]), jnp.array(actions[n_val:])
    n_train = len(train_obs)

    @jax.jit
    def train_step(state, batch_obs, batch_actions):
        def loss_fn(params):
            pi, _ = state.apply_fn(params, batch_obs)
            return -pi.log_prob(batch_actions).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_step(params, obs, actions):
        pi, _ = network.apply(params, obs)
        acc = (pi.logits.argmax(axis=-1) == actions).mean()
        return -pi.log_prob(actions).mean(), acc

    batch_size = config["bc_batch_size"]
    best_val_loss = float("inf")
    best_params = train_state.params
    patience_counter = 0

    for epoch in range(config["bc_epochs"]):
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, n_train)
        train_obs_s = train_obs[perm]
        train_act_s = train_actions[perm]

        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_train - batch_size + 1, batch_size):
            train_state, loss = train_step(
                train_state, train_obs_s[i:i+batch_size], train_act_s[i:i+batch_size]
            )
            epoch_loss += float(loss)
            n_batches += 1

        val_loss, val_acc = eval_step(train_state.params, val_obs, val_actions)
        val_loss, val_acc = float(val_loss), float(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = jax.device_get(train_state.params)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config['bc_epochs']}: "
                  f"train_loss={epoch_loss/max(n_batches,1):.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # Early stopping (patience=20, matching Carroll)
        if patience_counter >= 20:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    return best_params, {"val_loss": best_val_loss}


def main():
    parser = argparse.ArgumentParser(description="Train BC models on human data")
    parser.add_argument("--data-path", type=str,
                        default="/home/nathan/overcooked_ai_carroll/src/human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-bc-agents", type=int, default=3,
                        help="Number of BC agents per layout (different seeds)")
    args = parser.parse_args()

    config = {
        "bc_lr": args.bc_lr,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "val_split": args.val_split,
    }

    t0 = time.time()
    layout_data = load_and_process_data(args.data_path, layouts=BC_LAYOUTS)
    print(f"Data loading took {time.time() - t0:.1f}s")

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs("checkpoints", exist_ok=True)

    # Train N BC agents per layout (different seeds for diversity)
    # bc_params[layout_name] = list of N param dicts
    all_bc_params = {}
    for layout_name, (obs, actions) in layout_data.items():
        all_bc_params[layout_name] = []
        for agent_idx in range(args.num_bc_agents):
            print(f"\n[BC] {layout_name} agent {agent_idx}/{args.num_bc_agents} "
                  f"({len(obs)} samples)")
            rng, bc_rng = jax.random.split(rng)
            bc_params, metrics = train_bc(obs, actions, config, bc_rng)
            all_bc_params[layout_name].append(bc_params)
            print(f"[BC]   val_loss={metrics['val_loss']:.4f}")

    ckpt = {
        "bc_params": {k: [jax.device_get(p) for p in v]
                      for k, v in all_bc_params.items()},
        "config": config,
        "layouts": list(layout_data.keys()),
        "num_bc_agents": args.num_bc_agents,
        "bc_hidden_dims": BC_HIDDEN_DIMS,
    }
    ckpt_path = f"checkpoints/bc_seed{args.seed}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"\n[BC] Saved {args.num_bc_agents} agents × {len(layout_data)} layouts to {ckpt_path}")
    print(f"[BC] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
