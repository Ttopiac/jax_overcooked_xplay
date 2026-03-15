import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

# Each layout defines:
#   agent_idx: default 2-agent spawn positions (backward compatible)
#   all_walkable_idx: all walkable tiles for N-agent spawning
#   max_agents: maximum number of agents for this layout

cramped_room = {
    "height" : 4,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4, 5,9, 10,14, 15,16,17,18,19]),
    "agent_idx" : jnp.array([6, 8]),
    "all_walkable_idx" : jnp.array([6, 7, 8, 11, 12, 13]),
    "max_agents" : 6,
    # Single connected region — no spawn constraint
    "spawn_zones" : None,
    "goal_idx" : jnp.array([18]),
    "plate_pile_idx" : jnp.array([16]),
    "onion_pile_idx" : jnp.array([5,9]),
    "pot_idx" : jnp.array([2])
}

asymm_advantages = {
    "height" : 5,
    "width" : 9,
    "wall_idx" : jnp.array([0,1,2,3,4,5,6,7,8,
                            9,11,12,13,14,15,17,
                            18,22,26,
                            27,31,35,
                            36,37,38,39,40,41,42,43,44]),
    "agent_idx" : jnp.array([29, 32]),
    "all_walkable_idx" : jnp.array([10, 16, 19, 20, 21, 23, 24, 25, 28, 29, 30, 32, 33, 34]),
    "max_agents" : 14,
    # Two separated rooms — must have ≥1 agent in each
    "spawn_zones" : [jnp.array([10, 19, 20, 21, 28, 29, 30]),   # left room
                     jnp.array([16, 23, 24, 25, 32, 33, 34])],  # right room
    "goal_idx" : jnp.array([12,17]),
    "plate_pile_idx" : jnp.array([39,41]),
    "onion_pile_idx" : jnp.array([9,14]),
    "pot_idx" : jnp.array([22,31])
}

coord_ring = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4, 5,9, 10,12,14, 15,19, 20,21,22,23,24]),
    "agent_idx" : jnp.array([7, 11]),
    "all_walkable_idx" : jnp.array([6, 7, 8, 11, 13, 16, 17, 18]),
    "max_agents" : 8,
    "spawn_zones" : None,  # single connected region
    "goal_idx" : jnp.array([22]),
    "plate_pile_idx" : jnp.array([10]),
    "onion_pile_idx" : jnp.array([15,21]),
    "pot_idx" : jnp.array([3,9])
}

forced_coord = {
    "height" : 5,
    "width" : 5,
    "wall_idx" : jnp.array([0,1,2,3,4, 5,7,9, 10,12,14, 15,17,19, 20,21,22,23,24]),
    "agent_idx" : jnp.array([11,8]),
    "all_walkable_idx" : jnp.array([6, 8, 11, 13, 16, 18]),
    "max_agents" : 6,
    # Two separated corridors — must have ≥1 agent in each
    "spawn_zones" : [jnp.array([6, 11, 16]),   # left corridor
                     jnp.array([8, 13, 18])],   # right corridor
    "goal_idx" : jnp.array([23]),
    "onion_pile_idx" : jnp.array([5,10]),
    "plate_pile_idx" : jnp.array([15]),
    "pot_idx" : jnp.array([3,9])
}

counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

def layout_grid_to_dict(grid):
    """Convert grid string to layout dict. Adds all_walkable_idx and max_agents."""
    rows = grid.split('\n')
    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    keys = ["wall_idx", "agent_idx", "goal_idx", "plate_pile_idx", "onion_pile_idx", "pot_idx"]
    symbol_to_key = {"W": "wall_idx", "A": "agent_idx", "X": "goal_idx",
                     "B": "plate_pile_idx", "O": "onion_pile_idx", "P": "pot_idx"}

    layout_dict = {key: [] for key in keys}
    layout_dict["height"] = len(rows)
    layout_dict["width"] = len(rows[0])
    width = len(rows[0])
    walkable = []

    for i, row in enumerate(rows):
        for j, obj in enumerate(row):
            idx = width * i + j
            if obj in symbol_to_key.keys():
                layout_dict[symbol_to_key[obj]].append(idx)
            if obj in ["X", "B", "O", "P"]:
                layout_dict["wall_idx"].append(idx)
            elif obj == " " or obj == "A":
                walkable.append(idx)

    for key in symbol_to_key.values():
        layout_dict[key] = jnp.array(layout_dict[key])

    layout_dict["all_walkable_idx"] = jnp.array(walkable)
    layout_dict["max_agents"] = len(walkable)
    layout_dict["spawn_zones"] = None  # single connected region for grid-defined layouts

    return FrozenDict(layout_dict)


overcooked_layouts = {
    "cramped_room" : FrozenDict(cramped_room),
    "asymm_advantages" : FrozenDict(asymm_advantages),
    "coord_ring" : FrozenDict(coord_ring),
    "forced_coord" : FrozenDict(forced_coord),
    "counter_circuit" : layout_grid_to_dict(counter_circuit_grid)
}
