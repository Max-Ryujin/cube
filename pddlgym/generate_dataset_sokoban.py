import pddlgym
import imageio
from pathlib import Path
from pddlgym import sokoban
import numpy as np
import json

# This is SLOW!! 
# Since I only have to generate the dataset once it does not matter, but it is strange.


NUM_INSTANCES = 9
NUM_RANDOM_STEPS = 5
OUTPUT_DIR = Path("generated_images")
ENVIRONMENTS_TO_PROCESS = ["PDDLEnvSokoban-v0"]


def extract_sokoban_positions(obs, img_size, grid_size):
    """
    Extracts object locations from a sokoban observation and computes both
    scaled positions (pixel centers) and exact bounding boxes.

    Returns:
        (positions, bounding_boxes)
        positions: dict[obj_type -> [(x,y), ...]]   # pixel centers
        bounding_boxes: dict[obj_type -> [((x1,y1),(x2,y2)), ...]]
    """
    H, W = img_size[:2]
    rows, cols = grid_size
    cell_h, cell_w = H // rows, W // cols

    positions = {
        'player': [],
        'stone': [],
        'goal': [],
        'wall': [],
        'empty': []
    }
    bounding_boxes = {k: [] for k in positions.keys()}

    def add_obj(key, r, c):
        # center
        cy = c * cell_w + cell_w // 2
        cx = r * cell_h + cell_h // 2
        positions[key].append((cx, cy))

        # bounding box
        y1, x1 = (c-1) * cell_w, (r-1) * cell_h
        y2, x2 = c * cell_w, r * cell_h
        bounding_boxes[key].append(((x1, y1), (x2, y2)))

    # predicates with coords
    direct_map = {'player': 'player', 'stone': 'stone'}
    for pred, key in direct_map.items():
        for (r, c) in sokoban.get_locations(obs.literals, pred):
            add_obj(key, r, c)

    # predicates with loc_str
    value_map = {'clear': 'empty', 'is-nongoal': 'wall', 'is-goal': 'goal'}
    for pred, key in value_map.items():
        for v in sokoban.get_values(obs.literals, pred):
            r, c = sokoban.loc_str_to_loc(v[0])
            add_obj(key, r, c)

    return positions, bounding_boxes


def draw_bounding_boxes(img, bounding_boxes, color=(0.3, 0.3, 0), thickness=1):
    import cv2
    img_with_boxes = img.copy()
    for boxes in bounding_boxes.values():
        for (top_left, bottom_right) in boxes:
            cv2.rectangle(img_with_boxes, top_left, bottom_right, color, thickness)
    img_with_boxes = (img_with_boxes * 255).astype(np.uint8)
    #save image with bounding boxes for debugging
    imageio.imwrite("debug_bounding_boxes.png", img_with_boxes)
    return img_with_boxes

def serialize_positions_bboxes(positions, bounding_boxes):
	serial_positions = {k: [list(p) for p in v] for k, v in positions.items()}
	serial_bboxes = {
		k: [[list(top_left), list(bottom_right)] for (top_left, bottom_right) in v]
		for k, v in bounding_boxes.items()
	}
	return serial_positions, serial_bboxes


def generate_images_for_env(env_name: str, num_instances: int, num_steps: int, base_output_dir: Path):
	env_output_dir = base_output_dir / env_name
	env_output_dir.mkdir(parents=True, exist_ok=True)

	env = pddlgym.make(env_name)

	# collect dataset entries for this environment
	dataset_entries = []

	for i in range(num_instances):
		problem_idx = i

		obs, _ = env.reset()

		img = env.render(mode="human_crisp")

		layout = sokoban.build_layout(obs.literals)
		grid_height, grid_width = layout.shape
		grid_size = (grid_height, grid_width)

		positions, bounding_boxes = extract_sokoban_positions(obs, img_size=img.shape, grid_size=grid_size)

		# step 0 image
		filename = f"{env_name}_problem{problem_idx}_step0.png"
		file_path = env_output_dir / filename
		if img is not None:
			img = (img * 255).astype(np.uint8)
			imageio.imwrite(file_path, img)

		# add entry for step 0
		serial_pos, serial_boxes = serialize_positions_bboxes(positions, bounding_boxes)
		dataset_entries.append({
			"problem_idx": int(problem_idx),
			"step": 0,
			"image_path": str(Path(env_name) / filename),  # relative to base_output_dir
			"positions": serial_pos,
			"bounding_boxes": serial_boxes
		})

		# random steps
		for step in range(1, num_steps + 1):
			action = env.action_space.sample(obs)
			obs, reward, done, truncated, info = env.step(action)
			img = env.render()
			filename = f"{env_name}_problem{problem_idx}_step{step}.png"
			file_path = env_output_dir / filename
			if img is not None:
				imageio.imwrite(file_path, img)

			positions, bounding_boxes = extract_sokoban_positions(obs, img_size=img.shape, grid_size=grid_size)
			serial_pos, serial_boxes = serialize_positions_bboxes(positions, bounding_boxes)
			dataset_entries.append({
				"problem_idx": int(problem_idx),
				"step": int(step),
				"image_path": str(Path(env_name) / filename),  # relative to base_output_dir
				"positions": serial_pos,
				"bounding_boxes": serial_boxes
			})

			if done:
				break
	env.close()

	# write per-environment JSON file
	json_path = env_output_dir / f"{env_name}_dataset.json"
	with json_path.open("w") as f:
		json.dump({"entries": dataset_entries}, f, indent=2)


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