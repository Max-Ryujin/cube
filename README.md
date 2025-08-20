# cube


## requirements 
- PlanOracle from ogbench.manipspace.oracles.plan.plan_oracle 
- lie from ogbench.manipspace
- numpy
- gymnasium
- imageio
- re
- mujoco


## Usage
Main.py:
Line 77: create env.
Line 80: set initial state positions for cubes.
Line 85: Create Policy
Line 90: Parse actions
Line 142: Loop over actions and call start_high_level_action for each one
Line 146: Inner loop that performs low level action until high level action is done.
170+: Save to video.

