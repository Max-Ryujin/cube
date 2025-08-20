import numpy as np

from ogbench.manipspace import lie
from ogbench.manipspace.oracles.plan.plan_oracle import PlanOracle


class CubePlanOracle(PlanOracle):
    """
    High-level actions:
        - pick_up(block_id): pick a given cube and hold it.
        - stack(base_block_id): place the cube in the gripper on top of another cube.
        - unstack(top_block_id): pick a cube a from another cube b and hold it.
        - put_down(): place the cube in the gripper on the table.
    """

    def __init__(self, *args, block_size=0.04, approach_height=0.12, clearance_xy=0.10, **kwargs):
        super().__init__(*args, **kwargs)
        self._block_size = float(block_size) # Size of the cube to be manipulated.
        self._approach_h = float(approach_height) # Height to approach the cube
        self._clearance_xy = float(clearance_xy) # Clearance in the XY plane

        self._plan = None # Current plan being executed
        self._t_init = 0.0 # Time at which the plan was initialized
        self._t_max = 0.0 # Maximum time for the plan
        self._done = True # Whether the plan has been completed

    def pick_up(self, info, block_id):
        plan_input = {
            'effector_initial': self.to_pose(   # Initial pose of the effector
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'block_initial': self.to_pose(      # Initial pose of the block to be picked up
                pos=info[f'privileged/block_{block_id}_pos'],
                yaw=info[f'privileged/block_{block_id}_yaw'][0],
            ),
        }
        times, poses, grasps = self._compute_keyframes_pick_up(plan_input) # Compute keyframes for the pick-up action
        self._install_plan(times, poses, grasps, info)

    def unstack(self, info, top_block_id):
        plan_input = {
            'effector_initial': self.to_pose(
                pos=info['proprio/effector_pos'],
                yaw=info['proprio/effector_yaw'][0],
            ),
            'block_initial': self.to_pose(
                pos=info[f'privileged/block_{top_block_id}_pos'],
                yaw=info[f'privileged/block_{top_block_id}_yaw'][0],
            ),
        }
        times, poses, grasps = self._compute_keyframes_pick_up(plan_input, unstack_bias=True) # unstack bias for more clearance to the top
        self._install_plan(times, poses, grasps, info)

    def stack(self, info, base_block_id):
        eff_initial = self.to_pose(
            pos=info['proprio/effector_pos'],
            yaw=info['proprio/effector_yaw'][0],
        )
        base_pose = self.to_pose(
            pos=info[f'privileged/block_{base_block_id}_pos'],
            yaw=info[f'privileged/block_{base_block_id}_yaw'][0],
        )
        place_translation = base_pose.translation().copy()
        place_translation[-1] += self._block_size  # move goal up over the base block
        block_goal = lie.SE3.from_rotation_and_translation(  # Goal pose for the block to be stacked
            rotation=base_pose.rotation(),
            translation=place_translation,
        )
        plan_input = {
            'effector_initial': eff_initial,
            'place_target': block_goal,
        }
        times, poses, grasps = self._compute_keyframes_stack(plan_input)
        self._install_plan(times, poses, grasps, info)

    def put_down(self, info):
        # place cube on the table below the current position
        # TODO Check if another cube is in the way.
        eff_initial = self.to_pose(
            pos=info['proprio/effector_pos'],
            yaw=info['proprio/effector_yaw'][0],
        )
        place_translation = eff_initial.translation().copy()
        place_translation[-1] = self._block_size / 2.0 # place the block on the table
        block_goal = lie.SE3.from_rotation_and_translation( # precise goal pose for the block to be placed
            rotation=eff_initial.rotation(),
            translation=place_translation,
        )
        plan_input = {
            'effector_initial': eff_initial,
            'place_target': block_goal,
        }
        times, poses, grasps = self._compute_keyframes_stack(plan_input)
        self._install_plan(times, poses, grasps, info)

    def start_task(self, info, action: str, **kwargs):
        # start selected action
        action = action.lower().replace('-', '_')
        if action == 'pick_up':
            return self.pick_up(info, **kwargs)
        if action == 'stack':
            return self.stack(info, **kwargs)
        if action == 'unstack':
            return self.unstack(info, **kwargs)
        if action == 'put_down':
            return self.put_down(info, **kwargs)

    def _install_plan(self, times_dict, poses_dict, grasps_dict, info):
        # Install the computed plan into the oracle
        pose_seq = [poses_dict[name] for name in times_dict.keys()] # Extract pose sequence
        grasp_seq = [grasps_dict[name] for name in times_dict.keys()] # Extract grasp sequence
        time_seq = list(times_dict.values()) # Extract time sequence
        self._t_init = float(info['time'][0]) # Extract initial time
        self._t_max = float(time_seq[-1]) # Extract final time
        self._done = False
        self._plan = self.compute_plan(time_seq, pose_seq, grasp_seq) # Install the plan

    def _compute_keyframes_pick_up(self, plan_input, unstack_bias: bool = False):
        # Compute keyframes for the pick-up or unstack action
        poses = {}
        block_translation = plan_input['block_initial'].translation().copy()
        if unstack_bias:
            block_translation[-1] += self._block_size / 2.0 # needs to grab a bit higher
        block_contact = self.shortest_yaw(         # Align the effector with the block to be picked up
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(plan_input['block_initial']),
            translation=block_translation,
        )
        poses['initial'] = plan_input['effector_initial'] # Initial effector pose
        approach_h = self._approach_h * (1.3 if unstack_bias else 1.0) # add higher approach for stacks
        poses['approach'] = self.above(block_contact, approach_h) # Above block contact
        poses['pick_start'] = block_contact # Start of pick motion
        poses['pick_end'] = block_contact # End of pick motion
        poses['retreat'] = poses['approach']  # move back
        midway = poses['retreat']
        poses['clearance'] = lie.SE3.from_rotation_and_translation(
            rotation=poses['retreat'].rotation(),
            translation=np.array([
                midway.translation()[0],
                midway.translation()[1],
                poses['initial'].translation()[-1],
            ]),
        )
        dt = self._dt
        times = {  # Time for each keyframe
            'initial': 0.0,
            'approach': 2 * dt,
            'pick_start': dt * 2.5,
            'pick_end': dt * 3.5,
            'retreat': dt * 4.5,
            'clearance': dt * 5.5,
        }
        grasps = {}
        g = 0.0  # Grasping force
        for name in times.keys(): # Iterate through each keyframe
            if name == 'pick_end':  # End of pick motion
                g = 1.0  # Grasping force needs to be applied
            grasps[name] = g
        return times, poses, grasps  # return keyframes

    def _compute_keyframes_stack(self, plan_input):
        # Compute keyframes for the stack and place action
        poses = {}
        place_target = plan_input['place_target']
        aligned_place = self.shortest_yaw(
            eff_yaw=self.get_yaw(plan_input['effector_initial']),
            obj_yaw=self.get_yaw(place_target),
            translation=place_target.translation(),
        )
        poses['initial'] = plan_input['effector_initial']
        poses['approach'] = self.above(aligned_place, self._approach_h)
        poses['place_start'] = aligned_place
        poses['place_end'] = aligned_place
        poses['retreat'] = poses['approach']
        midway = poses['retreat']
        poses['clearance'] = lie.SE3.from_rotation_and_translation(
            rotation=midway.rotation(),
            translation=np.array([
                midway.translation()[0],
                midway.translation()[1],
                poses['initial'].translation()[-1] + 0.05,
            ]),
        )
        dt = self._dt
        times = {
            'initial': 0.0,
            'approach': 2 * dt,
            'place_start': dt * 2.5,
            'place_end': dt * 3.5,
            'retreat': dt * 4.5,
            'clearance': dt * 5.5,
        }
        grasps = {}
        g = 1.0
        for name in times.keys():
            if name == 'place_end':
                g = 0.0   # release grasp
            grasps[name] = g
        return times, poses, grasps

    def reset(self, ob, info):
        # Reset the oracle state
        self._plan = None
        self._done = True
        self._t_init = float(info['time'][0]) if 'time' in info else 0.0
        self._t_max = 0.0
        return None

