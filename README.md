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
call run_actions in main with args:
- initial_configuration: The starting configuration as a list of 4 cube positions.
- action_sequence: A list of actions to perform as a list of strings.
- optional: env: The environment to use (if not provided, a new one will be created).
- optional: output_path: The path to save the output video.
