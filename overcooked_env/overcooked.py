"""N-agent Overcooked environment in JAX.

Extended from JaxMARL's 2-agent Overcooked to support N agents.
Key changes from original:
  - Constructor takes num_agents parameter
  - Collision resolution handles all pairwise agent interactions
  - Observations keep 26 channels: ego + overlaid teammates (multiHRI approach)
  - Layouts define max agent spawn positions, subset used based on num_agents
"""

from collections import OrderedDict
from enum import IntEnum
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict

from .common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map)
from .layouts import overcooked_layouts as layouts


BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "PLATE_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

class Actions(IntEnum):
    up = 0
    down = 1
    right = 2
    left = 3
    stay = 4
    interact = 5
    done = 6


@struct.dataclass
class State:
    agent_pos: chex.Array      # (num_agents, 2)
    agent_dir: chex.Array      # (num_agents, 2)
    agent_dir_idx: chex.Array  # (num_agents,)
    agent_inv: chex.Array      # (num_agents,)
    goal_pos: chex.Array
    pot_pos: chex.Array
    wall_map: chex.Array
    maze_map: chex.Array
    time: int
    terminal: bool


POT_EMPTY_STATUS = 23
POT_FULL_STATUS = 20
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3

URGENCY_CUTOFF = 40
DELIVERY_REWARD = 20


class Overcooked(MultiAgentEnv):
    """N-agent Overcooked environment."""
    def __init__(
            self,
            layout=FrozenDict(layouts["cramped_room"]),
            num_agents: int = 2,
            random_reset: bool = False,
            max_steps: int = 400,
    ):
        super().__init__(num_agents=num_agents)

        self.height = layout["height"]
        self.width = layout["width"]
        # Keep 26 channels regardless of num_agents (multiHRI approach)
        self.obs_shape = (self.width, self.height, 26)

        self.agent_view_size = 5
        self.layout = layout
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        self.action_set = jnp.array([
            Actions.up, Actions.down, Actions.right,
            Actions.left, Actions.stay, Actions.interact,
        ])

        self.random_reset = random_reset
        self.max_steps = max_steps

        self.observation_spaces = {
            a: spaces.Box(0, 255, self.obs_shape) for a in self.agents
        }
        self.action_spaces = {
            a: spaces.Discrete(len(self.action_set), dtype=jnp.uint32) for a in self.agents
        }

    def step_env(
            self,
            key: chex.PRNGKey,
            state: State,
            actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:

        acts = self.action_set.take(indices=jnp.array(
            [actions[self.agents[i]] for i in range(self.num_agents)]
        ))

        state, reward, shaped_rewards = self.step_agents(key, state, acts)
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {self.agents[i]: reward for i in range(self.num_agents)}
        shaped_reward_dict = {self.agents[i]: shaped_rewards[i] for i in range(self.num_agents)}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {'shaped_reward': shaped_reward_dict},
        )

    def reset(
            self,
            key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        random_reset = self.random_reset
        layout = self.layout

        h = self.height
        w = self.width
        num_agents = self.num_agents
        all_pos = np.arange(np.prod([h, w]), dtype=jnp.uint32)

        wall_idx = layout.get("wall_idx")
        occupied_mask = jnp.zeros_like(all_pos)
        occupied_mask = occupied_mask.at[wall_idx].set(1)
        wall_map = occupied_mask.reshape(h, w).astype(jnp.bool_)

        # Reset agent positions
        key, subkey = jax.random.split(key)
        agent_idx = jax.random.choice(subkey, all_pos, shape=(num_agents,),
                                      p=(~occupied_mask.astype(jnp.bool_)).astype(jnp.float32), replace=False)

        # Use layout-defined positions
        # For N <= len(agent_idx): use first N default positions
        # For N > len(agent_idx): use all_walkable_idx (first N walkable tiles)
        default_agent_idx = layout.get("agent_idx", agent_idx)
        all_walkable = layout.get("all_walkable_idx", default_agent_idx)

        # Select positions: prefer default positions, fill rest from walkable
        if num_agents <= default_agent_idx.shape[0]:
            layout_agent_idx = default_agent_idx[:num_agents]
        else:
            # Use all defaults + additional from walkable
            layout_agent_idx = all_walkable[:num_agents]

        agent_idx = random_reset * agent_idx + (1 - random_reset) * layout_agent_idx
        agent_pos = jnp.array([agent_idx % w, agent_idx // w], dtype=jnp.uint32).transpose()
        occupied_mask = occupied_mask.at[agent_idx].set(1)

        key, subkey = jax.random.split(key)
        agent_dir_idx = jax.random.choice(subkey, jnp.arange(len(DIR_TO_VEC), dtype=jnp.int32), shape=(num_agents,))
        agent_dir = DIR_TO_VEC.at[agent_dir_idx].get()

        # Layout features
        empty_table_mask = jnp.zeros_like(all_pos)
        empty_table_mask = empty_table_mask.at[wall_idx].set(1)

        goal_idx = layout.get("goal_idx")
        goal_pos = jnp.array([goal_idx % w, goal_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[goal_idx].set(0)

        onion_pile_idx = layout.get("onion_pile_idx")
        onion_pile_pos = jnp.array([onion_pile_idx % w, onion_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[onion_pile_idx].set(0)

        plate_pile_idx = layout.get("plate_pile_idx")
        plate_pile_pos = jnp.array([plate_pile_idx % w, plate_pile_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[plate_pile_idx].set(0)

        pot_idx = layout.get("pot_idx")
        pot_pos = jnp.array([pot_idx % w, pot_idx // w], dtype=jnp.uint32).transpose()
        empty_table_mask = empty_table_mask.at[pot_idx].set(0)

        key, subkey = jax.random.split(key)
        pot_status = jax.random.randint(subkey, (pot_idx.shape[0],), 0, 24)
        pot_status = pot_status * random_reset + (1 - random_reset) * jnp.ones((pot_idx.shape[0],)) * 23

        onion_pos = jnp.array([])
        plate_pos = jnp.array([])
        dish_pos = jnp.array([])

        maze_map = make_overcooked_map(
            wall_map, goal_pos, agent_pos, agent_dir_idx,
            plate_pile_pos, onion_pile_pos, pot_pos, pot_status,
            onion_pos, plate_pos, dish_pos,
            pad_obs=True, num_agents=num_agents,
            agent_view_size=self.agent_view_size
        )

        # Agent inventory
        key, subkey = jax.random.split(key)
        possible_items = jnp.array([OBJECT_TO_INDEX['empty'], OBJECT_TO_INDEX['onion'],
                                    OBJECT_TO_INDEX['plate'], OBJECT_TO_INDEX['dish']])
        random_agent_inv = jax.random.choice(subkey, possible_items, shape=(num_agents,), replace=True)
        agent_inv = random_reset * random_agent_inv + \
                    (1 - random_reset) * jnp.ones(num_agents, dtype=jnp.int32) * OBJECT_TO_INDEX['empty']

        state = State(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            agent_dir_idx=agent_dir_idx,
            agent_inv=agent_inv,
            goal_pos=goal_pos,
            pot_pos=pot_pos,
            wall_map=wall_map.astype(jnp.bool_),
            maze_map=maze_map,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)
        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Return observations for all agents. 26 channels per agent.

        For each agent i:
          Channel 0: ego position (agent i)
          Channel 1: all other agents' positions (overlaid)
          Channels 2-5: ego orientation (4 one-hot direction channels)
          Channels 6-9: all other agents' orientations (overlaid)
          Channels 10-25: environment layers (same for all agents)
        """
        width = self.obs_shape[0]
        height = self.obs_shape[1]
        n_channels = 26
        padding = (state.maze_map.shape[0] - height) // 2
        num_agents = self.num_agents

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]
        soup_loc = jnp.array(maze_map == OBJECT_TO_INDEX["dish"], dtype=jnp.uint8)

        pot_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        pot_status = state.maze_map[padding:-padding, padding:-padding, 2] * pot_loc_layer
        onions_in_pot_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status >= POT_FULL_STATUS)
        onions_in_soup_layer = jnp.minimum(POT_EMPTY_STATUS - pot_status, MAX_ONIONS_IN_POT) * (pot_status < POT_FULL_STATUS) \
                               * pot_loc_layer + MAX_ONIONS_IN_POT * soup_loc
        pot_cooking_time_layer = pot_status * (pot_status < POT_FULL_STATUS)
        soup_ready_layer = pot_loc_layer * (pot_status == POT_READY_STATUS) + soup_loc
        urgency_layer = jnp.ones(maze_map.shape, dtype=jnp.uint8) * ((self.max_steps - state.time) < URGENCY_CUTOFF)

        # Per-agent position layers: (num_agents, H, W)
        agent_pos_layers = jnp.zeros((num_agents, height, width), dtype=jnp.uint8)
        for k in range(num_agents):
            agent_pos_layers = agent_pos_layers.at[k, state.agent_pos[k, 1], state.agent_pos[k, 0]].set(1)

        # Agent inventory overlay on maze_map for obs
        agent_inv_items = jnp.expand_dims(state.agent_inv, (1, 2)) * agent_pos_layers
        all_agent_mask = jnp.sum(agent_pos_layers, 0)
        maze_map_obs = jnp.where(all_agent_mask, agent_inv_items.sum(0), maze_map)

        soup_ready_layer = soup_ready_layer + \
            (jnp.sum(agent_inv_items, 0) == OBJECT_TO_INDEX["dish"]) * all_agent_mask
        onions_in_soup_layer = onions_in_soup_layer + \
            (jnp.sum(agent_inv_items, 0) == OBJECT_TO_INDEX["dish"]) * 3 * all_agent_mask

        env_layers = jnp.stack([
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8),
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["wall"], dtype=jnp.uint8),
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["onion_pile"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomato pile
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["plate_pile"], dtype=jnp.uint8),
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["goal"], dtype=jnp.uint8),
            jnp.array(onions_in_pot_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes in pot
            jnp.array(onions_in_soup_layer, dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes in soup
            jnp.array(pot_cooking_time_layer, dtype=jnp.uint8),
            jnp.array(soup_ready_layer, dtype=jnp.uint8),
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8),
            jnp.array(maze_map_obs == OBJECT_TO_INDEX["onion"], dtype=jnp.uint8),
            jnp.zeros(maze_map.shape, dtype=jnp.uint8),  # tomatoes
            urgency_layer,
        ])  # (16, H, W)

        # Per-agent direction layers: (num_agents, 4, H, W)
        # For each agent, 4 direction channels (one-hot)
        all_dir_layers = jnp.zeros((num_agents, 4, height, width), dtype=jnp.uint8)
        for k in range(num_agents):
            all_dir_layers = all_dir_layers.at[k, state.agent_dir_idx[k], :, :].set(agent_pos_layers[k])

        # Build per-agent obs
        obs_dict = {}
        for i in range(num_agents):
            obs = jnp.zeros((n_channels, height, width), dtype=jnp.uint8)

            # Channel 0: ego position
            obs = obs.at[0].set(agent_pos_layers[i])

            # Channel 1: all other agents' positions (overlaid)
            others_mask = jnp.ones(num_agents, dtype=jnp.bool_).at[i].set(False)
            teammates_pos = (agent_pos_layers * others_mask[:, None, None]).sum(axis=0)
            obs = obs.at[1].set(jnp.minimum(teammates_pos, 1))  # clip to 0/1

            # Channels 2-5: ego orientation
            obs = obs.at[2:6].set(all_dir_layers[i])

            # Channels 6-9: all other agents' orientations (overlaid)
            teammates_dir = (all_dir_layers * others_mask[:, None, None, None]).sum(axis=0)
            obs = obs.at[6:10].set(jnp.minimum(teammates_dir, 1))

            # Channels 10-25: environment layers
            obs = obs.at[10:].set(env_layers)

            obs_dict[self.agents[i]] = jnp.transpose(obs, (1, 2, 0))  # (H, W, 26)

        return obs_dict

    def step_agents(
            self, key: chex.PRNGKey, state: State, action: chex.Array,
    ) -> Tuple[State, float, chex.Array]:
        num_agents = self.num_agents

        # Update agent position (forward action)
        is_move_action = jnp.logical_and(action != Actions.stay, action != Actions.interact)
        is_move_action_t = jnp.expand_dims(is_move_action, 0).transpose()

        fwd_pos = jnp.minimum(
            jnp.maximum(state.agent_pos + is_move_action_t * DIR_TO_VEC[jnp.minimum(action, 3)]
                        + ~is_move_action_t * state.agent_dir, 0),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32)
        )

        # Wall/goal blocking
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            fwd_goal = jax.vmap(lambda pos, goal: jnp.logical_and(pos[0] == goal[0], pos[1] == goal[1]),
                               in_axes=(None, 0))(fwd_position, goal_pos)
            return fwd_wall, jnp.any(fwd_goal)

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(
            _wall_or_goal, in_axes=(0, None, None)
        )(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape((num_agents, 1))
        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_t)

        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)

        # N-agent collision resolution
        # For each pair of agents: if they would occupy the same cell, both stay
        # Also prevent swapping (passing through each other)
        resolved_pos = jnp.array(fwd_pos)

        def _resolve_collisions(resolved_pos):
            """Iterative collision resolution for N agents."""
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    # Same destination collision
                    same_dest = jnp.all(resolved_pos[i] == resolved_pos[j])
                    resolved_pos = resolved_pos.at[i].set(
                        jnp.where(same_dest, state.agent_pos[i], resolved_pos[i]))
                    resolved_pos = resolved_pos.at[j].set(
                        jnp.where(same_dest, state.agent_pos[j], resolved_pos[j]))

                    # Swap collision (agents passing through each other)
                    swap = jnp.logical_and(
                        jnp.all(fwd_pos[i] == state.agent_pos[j]),
                        jnp.all(fwd_pos[j] == state.agent_pos[i]))
                    resolved_pos = resolved_pos.at[i].set(
                        jnp.where(swap, state.agent_pos[i], resolved_pos[i]))
                    resolved_pos = resolved_pos.at[j].set(
                        jnp.where(swap, state.agent_pos[j], resolved_pos[j]))
            return resolved_pos

        agent_pos = _resolve_collisions(resolved_pos).astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interactions — process each agent sequentially
        fwd_pos_interact = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = (action == Actions.interact)

        agent_inv = jnp.array(state.agent_inv)
        total_reward = jnp.float32(0.0)
        shaped_rewards = jnp.zeros(num_agents)

        for k in range(num_agents):
            candidate_map, new_inv, rew, shaped_rew = self.process_interact(
                maze_map, state.wall_map, fwd_pos_interact, agent_inv, k
            )
            do_interact = is_interact_action[k]
            maze_map = jax.lax.select(do_interact, candidate_map, maze_map)
            agent_inv = agent_inv.at[k].set(
                jax.lax.select(do_interact, new_inv, agent_inv[k]))
            total_reward = total_reward + jax.lax.select(do_interact, rew, 0.)
            shaped_rewards = shaped_rewards.at[k].set(
                jax.lax.select(do_interact, shaped_rew, 0.))

        # Update agent visuals in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array([OBJECT_TO_INDEX['agent'], agent_idx % 9, agent_dir_idx], dtype=jnp.uint8)
            return agent_pos[0], agent_pos[1], agent_pos_prev[0], agent_pos_prev[1], agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(
            agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(num_agents)
        )
        empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)

        height = self.obs_shape[1]
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(empty)
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Cook pots
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= POT_FULL_STATUS)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = is_cooking * not_done * (pot_status - 1) + (~is_cooking) * pot_status
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False),
            total_reward,
            shaped_rewards,
        )

    def process_interact(
            self,
            maze_map: chex.Array,
            wall_map: chex.Array,
            fwd_pos_all: chex.Array,
            inventory_all: chex.Array,
            player_idx: int):
        """Process interact action for one agent. Unchanged from original."""
        fwd_pos = fwd_pos_all[player_idx]
        inventory = inventory_all[player_idx]
        shaped_reward = 0.

        height = self.obs_shape[1]
        padding = (maze_map.shape[0] - height) // 2

        maze_object_on_table = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0]].get()
        object_on_table = maze_object_on_table[0]

        object_is_pile = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate_pile"],
                                        object_on_table == OBJECT_TO_INDEX["onion_pile"])
        object_is_pot = jnp.array(object_on_table == OBJECT_TO_INDEX["pot"])
        object_is_goal = jnp.array(object_on_table == OBJECT_TO_INDEX["goal"])
        object_is_agent = jnp.array(object_on_table == OBJECT_TO_INDEX["agent"])
        object_is_pickable = jnp.logical_or(
            jnp.logical_or(object_on_table == OBJECT_TO_INDEX["plate"],
                           object_on_table == OBJECT_TO_INDEX["onion"]),
            object_on_table == OBJECT_TO_INDEX["dish"])
        is_table = jnp.logical_and(wall_map.at[fwd_pos[1], fwd_pos[0]].get(), ~object_is_pot)
        table_is_empty = jnp.logical_or(object_on_table == OBJECT_TO_INDEX["wall"],
                                        object_on_table == OBJECT_TO_INDEX["empty"])

        pot_status = maze_object_on_table[-1]

        inv_is_empty = jnp.array(inventory == OBJECT_TO_INDEX["empty"])
        object_in_inv = inventory
        holding_onion = jnp.array(object_in_inv == OBJECT_TO_INDEX["onion"])
        holding_plate = jnp.array(object_in_inv == OBJECT_TO_INDEX["plate"])
        holding_dish = jnp.array(object_in_inv == OBJECT_TO_INDEX["dish"])

        case_1 = (pot_status > POT_FULL_STATUS) * holding_onion * object_is_pot
        case_2 = (pot_status == POT_READY_STATUS) * holding_plate * object_is_pot
        case_3 = (pot_status > POT_READY_STATUS) * (pot_status <= POT_FULL_STATUS) * object_is_pot
        else_case = ~case_1 * ~case_2 * ~case_3

        shaped_reward += case_1 * BASE_REW_SHAPING_PARAMS["PLACEMENT_IN_POT_REW"]
        shaped_reward += case_2 * BASE_REW_SHAPING_PARAMS["SOUP_PICKUP_REWARD"]

        new_pot_status = case_1 * (pot_status - 1) + case_2 * POT_EMPTY_STATUS + case_3 * pot_status + else_case * pot_status
        new_object_in_inv = case_1 * OBJECT_TO_INDEX["empty"] + case_2 * OBJECT_TO_INDEX["dish"] + case_3 * object_in_inv + else_case * object_in_inv

        successful_pickup = is_table * ~table_is_empty * inv_is_empty * jnp.logical_or(object_is_pile, object_is_pickable)
        successful_drop = is_table * table_is_empty * ~inv_is_empty
        successful_delivery = is_table * object_is_goal * holding_dish
        no_effect = jnp.logical_and(jnp.logical_and(~successful_pickup, ~successful_drop), ~successful_delivery)

        new_object_on_table = (
            no_effect * object_on_table
            + successful_delivery * object_on_table
            + successful_pickup * object_is_pile * object_on_table
            + successful_pickup * object_is_pickable * OBJECT_TO_INDEX["wall"]
            + successful_drop * object_in_inv
        )

        new_object_in_inv = (
            no_effect * new_object_in_inv
            + successful_delivery * OBJECT_TO_INDEX["empty"]
            + successful_pickup * object_is_pickable * object_on_table
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["plate_pile"]) * OBJECT_TO_INDEX["plate"]
            + successful_pickup * (object_on_table == OBJECT_TO_INDEX["onion_pile"]) * OBJECT_TO_INDEX["onion"]
            + successful_drop * OBJECT_TO_INDEX["empty"]
        )

        has_picked_up_plate = successful_pickup * (new_object_in_inv == OBJECT_TO_INDEX["plate"])
        num_plates_in_inv = jnp.sum(inventory_all == OBJECT_TO_INDEX["plate"])
        pot_loc_layer = jnp.array(maze_map[padding:-padding, padding:-padding, 0] == OBJECT_TO_INDEX["pot"], dtype=jnp.uint8)
        padded_map = maze_map[padding:-padding, padding:-padding, 2]
        num_notempty_pots = jnp.sum((padded_map != POT_EMPTY_STATUS) * pot_loc_layer)
        is_dish_pickup_useful = num_plates_in_inv < num_notempty_pots
        plate_loc_layer = jnp.array(maze_map == OBJECT_TO_INDEX["plate"], dtype=jnp.uint8)
        no_plates_on_counters = jnp.sum(plate_loc_layer) == 0

        shaped_reward += no_plates_on_counters * has_picked_up_plate * is_dish_pickup_useful * BASE_REW_SHAPING_PARAMS["PLATE_PICKUP_REWARD"]

        inventory = new_object_in_inv

        new_maze_object_on_table = (
            object_is_pot * OBJECT_INDEX_TO_VEC[new_object_on_table].at[-1].set(new_pot_status)
            + ~object_is_pot * ~object_is_agent * OBJECT_INDEX_TO_VEC[new_object_on_table]
            + object_is_agent * maze_object_on_table
        )

        maze_map = maze_map.at[padding + fwd_pos[1], padding + fwd_pos[0], :].set(new_maze_object_on_table)

        reward = jnp.array(successful_delivery, dtype=float) * DELIVERY_REWARD
        return maze_map, inventory, reward, shaped_reward

    def is_terminal(self, state: State) -> bool:
        return state.time >= self.max_steps

    @property
    def name(self) -> str:
        return "Overcooked"

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str = None):
        if agent is None:
            return self.action_spaces[self.agents[0]]
        return self.action_spaces[agent]
