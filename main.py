import re
import imageio
import numpy as np
import gymnasium as gym
from high_level_policy import CubePlanOracle
import mujoco


# red, blue, yellow, green

ACTION_LINES = [
    "(unstack b2 b3 robot)",
    "(stack b2 b1 robot)",
    "(pick-up b3 robot)",
    "(stack b3 b2 robot)",
    "(pick-up b4 robot)",
    "(stack b4 b3 robot)",
]


def parse_action_line(line: str):
    # parser
    line = line.strip()
    m = re.match(r"^\(([^\s\)]+)(?:\s+([^\s\)]+))?(?:\s+([^\s\)]+))?(?:\s+([^\s\)]+))?\)$", line)
    verb = m.group(1)
    args = [a for a in m.groups()[1:] if a is not None]
    return verb, args


def discover_block_ids(info: dict):
    # gets block ids from info
    ids = []
    pat = re.compile(r"^privileged/block_(\d+)_pos$")
    for k in info.keys():
        mm = pat.match(k)
        if mm:
            ids.append(int(mm.group(1)))
    return sorted(set(ids))


def map_block_symbol(sym: str, available_ids):
    """Map 'b1'/'b2'/... onto environment block ids.
    Assumptions for simplification:
      - available_ids are contiguous 0-based integers (e.g., [0, 1, ..., N-1]).
      - sym is 'b' (or 'B') followed by a 1-based integer (e.g., 'b1', 'b2').
    """
    #TODO this has to take the actions into account. the mapping is most likely different
    m = re.match(r"^[bB](\d+)$", sym)
    k = int(m.group(1))
   # return k - 1

    # mapping b2 = 1, b3 = 0, b1 = 2, b4 = 3
    mapping = {1: 2, 2: 1, 3: 0, 4: 3}
    return mapping.get(k, None)


def set_start_state(env, startconfiguration):
    unwrapped_env = env.unwrapped
    for i in range(unwrapped_env._num_cubes):
        # Get the specific joint for the cube
        joint = unwrapped_env.data.joint(f'object_joint_{i}')
        
        # Set its position (the first 3 values of qpos)
        joint.qpos[:3] = startconfiguration[i]
        
        # Reset its velocity to zero for stability
        joint.qvel[:] = 0

    # CRITICAL: Re-compute the simulation state after changing qpos
    mujoco.mj_forward(unwrapped_env.model, unwrapped_env.data)
    unwrapped_env.post_step()



# MAIN

env = gym.make('cube-quadruple-v0', terminate_at_goal=True, mode='data_collection', permute_blocks=False)
ob, info = env.reset()
# red, blue, yellow, green
set_start_state(env, [[0.5, -0.1, 0], [0.5, -0.1, 0.041], [0.3, -0.2, 0], [0.5, 0.2, 0]])
unwrapped_env = env.unwrapped
obs = unwrapped_env.compute_observation()
info = unwrapped_env.get_reset_info()

agent = CubePlanOracle(env=env, noise=0, noise_smoothing=0)
agent.reset(ob, info)

available_ids = discover_block_ids(info)
print("Available block ids (from info):", available_ids)
parsed_actions = [parse_action_line(s) for s in ACTION_LINES]

# Build a symbol->id mapping on demand and echo decisions each time

def start_high_level_action(verb: str, args: list, info: dict):
    verb_norm = verb.lower().replace('-', '_')

    # robot not needed
    filtered_args = [a for a in args if a.lower() != 'robot']

    # Map symbols 
    if verb_norm in {"pick_up", "pickup"}:
        b_sym = filtered_args[0]
        b_id = map_block_symbol(b_sym, available_ids)
        print(f"[plan] (pick-up {b_sym}) -> block_id={b_id}")
        agent.start_task(info, 'pick_up', block_id=b_id)
        return {"verb": "pick_up", "block": b_id}

    elif verb_norm == "unstack":
        # We accept (unstack top base) but only top is required by the policy
        top_sym = filtered_args[0]
        base_sym = filtered_args[1] if len(filtered_args) >= 2 else None
        top_id = map_block_symbol(top_sym, available_ids)
        base_id = map_block_symbol(base_sym, available_ids) if base_sym else None
        print(f"[plan] (unstack {top_sym} {base_sym}) -> top_id={top_id}, base_id={base_id}")
        agent.start_task(info, 'unstack', top_block_id=top_id)
        return {"verb": "unstack", "top": top_id, "base": base_id}

    elif verb_norm == "stack":
        if len(filtered_args) == 1:
            base_sym = filtered_args[0]
            held_sym = None
        else:
            held_sym, base_sym = filtered_args[0], filtered_args[1]
        base_id = map_block_symbol(base_sym, available_ids)
        print(f"[plan] (stack {held_sym} {base_sym}) -> base_id={base_id} (held cube assumed)")
        agent.start_task(info, 'stack', base_block_id=base_id)
        return {"verb": "stack", "base": base_id, "held_label": held_sym}

    elif verb_norm in {"put_down", "putdown"}:
        print("[plan] (put-down) -> table placement under current gripper")
        agent.start_task(info, 'put_down')
        return {"verb": "put_down"}

    else:
        print("wrong action")

# Execute the action sequence

frames = []

for verb, args in parsed_actions:
    meta = start_high_level_action(verb, args, info)

    step_count = 0
    last_done = getattr(agent, '_done', True)
    while True:
        a = agent.select_action(ob, info)
        ob, reward, terminated, truncated, info = env.step(a)
        step_count += 1

        # Frame capture
        frame = getattr(env.unwrapped, 'get_pixel_observation', None)
        if callable(frame):
            img = env.unwrapped.get_pixel_observation()
            if img is not None:
                frames.append(img)

        plan_done = getattr(agent, '_done', True)
        if plan_done and not last_done:
            print(f"[plan] Finished {meta} in {step_count} env steps.")
            break
        last_done = plan_done

        if terminated or truncated:
            break

    if terminated or truncated:
        break


save_path = 'test_policy.mp4'
try:
    env_dt = float(getattr(env.unwrapped, '_control_timestep', 0.03))
    fps = int(round(1.0 / env_dt)) if env_dt > 0 else 30
except Exception:
    fps = 30

with imageio.get_writer(save_path, fps=fps) as writer:
    for f in frames:
        writer.append_data(f)
print('Saved video to', save_path)

