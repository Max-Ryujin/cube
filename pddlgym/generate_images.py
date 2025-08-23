import pddlgym
import imageio
from pathlib import Path

# This is SLOW!! 
# Since I only have to generate the dataset once it does not matter, but it is strange.


NUM_INSTANCES = 9
NUM_RANDOM_STEPS = 4
OUTPUT_DIR = Path("generated_images")
ENVIRONMENTS_TO_PROCESS = ["PDDLEnvSokoban-v0", "PDDLEnvGripper-v0"]

def generate_images_for_env(env_name: str, num_instances: int, num_steps: int, base_output_dir: Path):
    env_output_dir = base_output_dir / env_name
    env_output_dir.mkdir(parents=True, exist_ok=True)

    env = pddlgym.make(env_name)

    print(type(env))
    print("has render:", hasattr(env, "render"))
    print("metadata:", getattr(env, "metadata", None))
    print("methods:", [m for m in dir(env) if not m.startswith("_")])
    # show source of the class if possible:
    import inspect
    print(inspect.getsource(env.__class__))


    for i in range(num_instances):
        problem_idx = i

        obs, _ = env.reset()
      

        img = env.render()

        file_path = env_output_dir / f"{env_name}_problem{problem_idx}_step0.png"
        if img is not None:
            imageio.imwrite(file_path, img)

        for step in range(1, num_steps + 1):
            action = env.action_space.sample(obs)
            obs, reward, done, truncated, info = env.step(action)
            img = env.render()
            file_path = env_output_dir / f"{env_name}_problem{problem_idx}_step{step}.png"
            if img is not None:
                imageio.imwrite(file_path, img)
            if done:
                break
    env.close()


def main():
    for env_name in ENVIRONMENTS_TO_PROCESS:
        generate_images_for_env(
            env_name=env_name,
            num_instances=NUM_INSTANCES,
            num_steps=NUM_RANDOM_STEPS,
            base_output_dir=OUTPUT_DIR
        )
if __name__ == "__main__":
    main()