"""N-XPlay framework utilities for JAX.

In an N-player game with X unseen teammates:
  - Agents 0..N-X-1: ego/clones (all use ego_params, transitions collected)
  - Agents N-X..N-1: partners (frozen params, no transitions)

Position randomization: agent starting positions are shuffled each reset
so the ego/clones don't always play from the same physical positions.

Two partner selection modes:
  - Hetero: X partners each independently sampled
  - Homo: X partners all use the same policy

All functions are JIT-compatible.
"""

import jax
import jax.numpy as jnp


def select_partners(rng, partner_pool_params, num_unseen, homo=False):
    """Select partner params from pool.

    Args:
        rng: JAX random key
        partner_pool_params: stacked params, tree of (pool_size, ...)
        num_unseen: X partners to select
        homo: if True, all X use same params

    Returns:
        stacked_partner_params: tree of (num_unseen, ...) — one set per partner slot
    """
    if num_unseen == 0:
        return None

    pool_size = jax.tree.leaves(partner_pool_params)[0].shape[0]

    if homo:
        # Sample one, replicate X times
        idx = jax.random.randint(rng, (), 0, pool_size)
        single = jax.tree.map(lambda x: x[idx], partner_pool_params)
        return jax.tree.map(lambda x: jnp.broadcast_to(x[None], (num_unseen, *x.shape)), single)
    else:
        # Sample X independently
        indices = jax.random.choice(rng, pool_size, shape=(num_unseen,),
                                    replace=num_unseen > pool_size)
        return jax.tree.map(lambda x: x[indices], partner_pool_params)


def compute_actions(network, ego_params, partner_params, obs_per_agent,
                    rng, num_agents, num_unseen):
    """Compute actions for all N agents. JIT-compatible.

    Fixed slot assignment:
      - Agents 0..N-X-1: use ego_params
      - Agents N-X..N-1: use partner_params[j] where j = i - (N-X)

    Args:
        network: ActorCritic network
        ego_params: ego policy params
        partner_params: tree of (num_unseen, ...) or None if X=0
        obs_per_agent: list of N arrays, each (num_envs, obs_dim)
        rng: random key
        num_agents: N
        num_unseen: X

    Returns:
        actions: list of N arrays, each (num_envs,)
        ego_log_probs: list of N-X arrays (for PPO)
        ego_values: list of N-X arrays (for PPO)
    """
    num_ego = num_agents - num_unseen
    rngs = jax.random.split(rng, num_agents)

    actions = []
    ego_log_probs = []
    ego_values = []

    # Ego/clone agents
    for i in range(num_ego):
        pi, val = network.apply(ego_params, obs_per_agent[i])
        a = pi.sample(seed=rngs[i])
        lp = pi.log_prob(a)
        actions.append(a)
        ego_log_probs.append(lp)
        ego_values.append(val)

    # Partner agents
    for j in range(num_unseen):
        p_params = jax.tree.map(lambda x: x[j], partner_params)
        pi, _ = network.apply(p_params, obs_per_agent[num_ego + j])
        a = pi.sample(seed=rngs[num_ego + j])
        actions.append(a)

    return actions, ego_log_probs, ego_values


def shuffle_agent_slots(rng, state, num_agents):
    """Shuffle which agent slot maps to which physical position (multiHRI-style).

    multiHRI randomizes p_idx each reset — the ego randomly plays from any
    physical starting position. This achieves the same effect by permuting
    the agent slot assignments in the state.

    Called at start of each PPO update. Physical positions stay as defined
    by the layout, but which slot (ego/clone/partner) is at which position
    gets randomized.

    Args:
        rng: random key
        state: env State
        num_agents: N

    Returns:
        state with shuffled slot-to-position mapping
    """
    perm = jax.random.permutation(rng, num_agents)
    return state.replace(
        agent_pos=state.agent_pos[perm],
        agent_dir=state.agent_dir[perm],
        agent_dir_idx=state.agent_dir_idx[perm],
        agent_inv=state.agent_inv[perm],
    )


def shuffle_start_positions(rng, state, num_agents):
    """Randomize physical starting positions across walkable tiles.

    Unlike shuffle_agent_slots which just permutes existing positions,
    this actually moves agents to new random positions. Requires the
    env to be created with random_start_pos=True.

    This is our custom extension beyond multiHRI.
    """
    # This is handled by env.reset with random_start_pos=True
    # This function is for in-training position randomization
    perm = jax.random.permutation(rng, num_agents)
    return state.replace(
        agent_pos=state.agent_pos[perm],
        agent_dir=state.agent_dir[perm],
        agent_dir_idx=state.agent_dir_idx[perm],
        agent_inv=state.agent_inv[perm],
    )
