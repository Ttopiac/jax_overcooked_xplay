"""Behavioral Cloning (BC) model training for Overcooked.

Trains a BC model on human demonstration data from Carroll et al. (2019).
The BC model maps egocentric observations to action distributions.

Data pipeline:
  1. Load human-human trajectories from overcooked_ai pickle files
  2. Parse states using overcooked_ai library
  3. Convert to 26-channel grid observations via lossless_state_encoding
  4. Apply egocentric encoding (7x7 centered on agent, rotated to face SOUTH)
  5. Train action classifier with cross-entropy loss

Usage:
  python train_bc.py --data-path /path/to/2019_hh_trials_all.pickle
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

# Add overcooked_ai to path
OVERCOOKED_AI_PATH = "/home/nathan/overcooked_ai_carroll/src"

# Direction vector -> JaxMARL action index
ACTION_MAP = {
    (0, -1): 0,   # UP/NORTH
    (0, 1): 1,    # DOWN/SOUTH
    (1, 0): 2,    # RIGHT/EAST
    (-1, 0): 3,   # LEFT/WEST
    (0, 0): 4,    # STAY
}
INTERACT_IDX = 5

# Direction index for egocentric rotation
# Original overcooked_ai: NORTH=(0,-1), SOUTH=(0,1), EAST=(1,0), WEST=(-1,0)
DIR_TO_IDX = {
    (0, -1): 0,  # NORTH
    (0, 1): 1,   # SOUTH
    (1, 0): 2,   # EAST
    (-1, 0): 3,  # WEST
}

# Rotation amounts to make agent face SOUTH (matching obs_encoding.py)
ROT_K = {0: 2, 1: 0, 2: 3, 3: 1}  # indexed by dir_idx


def parse_action(action):
    """Convert action from human data format to index."""
    if isinstance(action, str) and action == "INTERACT":
        return INTERACT_IDX
    return ACTION_MAP[tuple(action)]


def egocentric_obs_numpy(obs_grid, player_pos, player_orientation):
    """Convert (H, W, C) grid obs to egocentric (C, 7, 7) using numpy.

    Args:
        obs_grid: (H, W, C) numpy array, per-player lossless encoding.
        player_pos: (x, y) tuple.
        player_orientation: (dx, dy) tuple.

    Returns:
        (EGO_FLAT_DIM,) flattened float32 array.
    """
    C = obs_grid.shape[2]
    pad = EGO_SIZE // 2

    # Transpose to (C, H, W) — but obs_grid is (W, H, C) from overcooked_ai
    # overcooked_ai uses (width, height, channels) indexing
    obs_chw = obs_grid.transpose(2, 0, 1).astype(np.float32)

    # Pad spatial dims
    padded = np.pad(obs_chw, ((0, 0), (pad, pad), (pad, pad)))

    # Extract 7x7 window centered on player
    # player_pos is (x, y) in overcooked_ai = (col, row) in grid coords
    x, y = player_pos
    window = padded[:, x:x + EGO_SIZE, y:y + EGO_SIZE]

    # Rotate so agent always faces SOUTH
    dir_idx = DIR_TO_IDX[tuple(player_orientation)]
    k = ROT_K[dir_idx]
    if k > 0:
        window = np.rot90(window, k=k, axes=(1, 2))

    return window.flatten()


def load_and_process_data(data_path, layouts=None):
    """Load human data and convert to (obs, action) pairs.

    Args:
        data_path: Path to pickle file.
        layouts: List of layout names to include. None = all.

    Returns:
        Dict mapping layout_name -> (obs_array, action_array)
    """
    sys.path.insert(0, OVERCOOKED_AI_PATH)
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

    print(f"Loading data from {data_path}...")
    df = pd.read_pickle(data_path)

    if layouts:
        df = df[df["layout_name"].isin(layouts)]

    available_layouts = df["layout_name"].unique()
    print(f"Available layouts: {list(available_layouts)}")

    result = {}
    for layout_name in available_layouts:
        layout_df = df[df["layout_name"] == layout_name]
        mdp = OvercookedGridworld.from_layout_name(layout_name)

        all_obs = []
        all_actions = []

        for _, row in layout_df.iterrows():
            # Parse state
            state_raw = row["state"]
            state_dict = json.loads(state_raw) if isinstance(state_raw, str) else state_raw
            state = OvercookedState.from_dict(state_dict)

            # Parse actions
            ja_raw = row["joint_action"]
            ja = ast.literal_eval(ja_raw) if isinstance(ja_raw, str) else ja_raw

            # Get observations for both players
            obs_both = mdp.lossless_state_encoding(state, horizon=400)
            # obs_both shape: (2, W, H, C)

            for p_idx in range(2):
                obs_grid = np.array(obs_both[p_idx])  # (W, H, C)
                player = state.players[p_idx]

                ego_obs = egocentric_obs_numpy(
                    obs_grid, player.position, player.orientation
                )
                action_idx = parse_action(ja[p_idx])

                all_obs.append(ego_obs)
                all_actions.append(action_idx)

        obs_array = np.stack(all_obs)
        action_array = np.array(all_actions, dtype=np.int32)
        result[layout_name] = (obs_array, action_array)
        print(f"  {layout_name}: {len(obs_array)} samples, "
              f"action dist: {np.bincount(action_array, minlength=6)}")

    return result


def train_bc(obs, actions, config, rng):
    """Train BC model with cross-entropy loss.

    Args:
        obs: (N, EGO_FLAT_DIM) float32 observations.
        actions: (N,) int32 action labels.
        config: dict with training hyperparameters.

    Returns:
        Trained params, metrics.
    """
    network = ActorCritic(action_dim=6, hidden_dims=config["hidden_dims"])

    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros((EGO_FLAT_DIM,)))
    tx = optax.adam(config["bc_lr"])
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # Train/val split
    n = len(obs)
    n_val = int(n * config["val_split"])
    perm = jax.random.permutation(rng, n)
    obs, actions = obs[perm], actions[perm]
    val_obs, val_actions = jnp.array(obs[:n_val]), jnp.array(actions[:n_val])
    train_obs, train_actions = jnp.array(obs[n_val:]), jnp.array(actions[n_val:])
    n_train = len(train_obs)

    @jax.jit
    def train_step(state, batch_obs, batch_actions):
        def loss_fn(params):
            pi, _ = state.apply_fn(params, batch_obs)
            log_probs = pi.log_prob(batch_actions)
            return -log_probs.mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(params, obs, actions):
        pi, _ = network.apply(params, obs)
        log_probs = pi.log_prob(actions)
        preds = pi.logits.argmax(axis=-1)
        accuracy = (preds == actions).mean()
        return -log_probs.mean(), accuracy

    batch_size = config["bc_batch_size"]
    best_val_loss = float("inf")
    best_params = train_state.params

    for epoch in range(config["bc_epochs"]):
        # Shuffle training data
        rng, shuffle_rng = jax.random.split(rng)
        perm = jax.random.permutation(shuffle_rng, n_train)
        train_obs_shuffled = train_obs[perm]
        train_actions_shuffled = train_actions[perm]

        # Mini-batch training
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train - batch_size + 1, batch_size):
            batch_o = train_obs_shuffled[i : i + batch_size]
            batch_a = train_actions_shuffled[i : i + batch_size]
            train_state, loss = train_step(train_state, batch_o, batch_a)
            epoch_loss += float(loss)
            n_batches += 1

        # Validation
        val_loss, val_acc = eval_step(train_state.params, val_obs, val_actions)
        val_loss, val_acc = float(val_loss), float(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = jax.device_get(train_state.params)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['bc_epochs']}: "
                  f"train_loss={epoch_loss/max(n_batches,1):.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    return best_params, {"val_loss": best_val_loss}


def main():
    parser = argparse.ArgumentParser(description="Train BC model on human data")
    parser.add_argument("--data-path", type=str,
                        default="/home/nathan/overcooked_ai_carroll/src/human_aware_rl/static/human_data/cleaned/2019_hh_trials_all.pickle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    config = {
        "bc_lr": args.bc_lr,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "val_split": args.val_split,
        "hidden_dims": tuple([args.hidden_dim] * args.num_layers),
    }

    # Layouts available in JaxMARL
    target_layouts = ["cramped_room", "asymmetric_advantages", "coordination_ring"]

    t0 = time.time()
    layout_data = load_and_process_data(args.data_path, layouts=target_layouts)
    print(f"Data loading took {time.time() - t0:.1f}s")

    rng = jax.random.PRNGKey(args.seed)
    os.makedirs("checkpoints", exist_ok=True)

    # Train per-layout BC models (matching Carroll)
    all_bc_params = {}
    for layout_name, (obs, actions) in layout_data.items():
        print(f"\n[BC] Training BC for {layout_name} ({len(obs)} samples)")
        rng, bc_rng = jax.random.split(rng)
        bc_params, metrics = train_bc(obs, actions, config, bc_rng)
        all_bc_params[layout_name] = bc_params
        print(f"[BC] {layout_name}: best val_loss={metrics['val_loss']:.4f}")

    # Also train a combined BC across all layouts
    print(f"\n[BC] Training combined BC across all layouts")
    all_obs = np.concatenate([d[0] for d in layout_data.values()])
    all_actions = np.concatenate([d[1] for d in layout_data.values()])
    rng, bc_rng = jax.random.split(rng)
    combined_params, combined_metrics = train_bc(all_obs, all_actions, config, bc_rng)
    all_bc_params["combined"] = combined_params
    print(f"[BC] combined: best val_loss={combined_metrics['val_loss']:.4f}")

    # Save
    ckpt = {
        "bc_params": {k: jax.device_get(v) for k, v in all_bc_params.items()},
        "config": config,
        "layouts": list(layout_data.keys()),
    }
    ckpt_path = f"checkpoints/bc_seed{args.seed}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"\n[BC] Saved to {ckpt_path}")
    print(f"[BC] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
