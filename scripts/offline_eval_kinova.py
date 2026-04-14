"""Offline evaluation for OpenPI pi05 policies on LeRobot Kinova datasets."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.transforms as _transforms


def _to_numpy(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _tree_to_numpy(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _tree_to_numpy(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_tree_to_numpy(v) for v in data)
    return _to_numpy(data)


def _create_lerobot_dataset(repo_id: str, action_horizon: int, dataset_root: str | None):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    try:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root) if dataset_root else None
    except TypeError:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id) if dataset_root is not None else None

    if meta is None:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    delta_timestamps = {"action": [t / meta.fps for t in range(action_horizon)]}

    try:
        dataset = lerobot_dataset.LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    except TypeError:
        dataset = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    return dataset


def _strip_action_from_repack(group: _transforms.Group) -> _transforms.Group:
    new_inputs = []
    for transform in group.inputs:
        if isinstance(transform, _transforms.RepackTransform):
            structure = transform.structure
            if isinstance(structure, dict) and "actions" in structure:
                new_structure = {k: v for k, v in structure.items() if k != "actions"}
                new_inputs.append(dataclasses.replace(transform, structure=new_structure))
                continue
        new_inputs.append(transform)
    return _transforms.Group(inputs=tuple(new_inputs), outputs=group.outputs)


def _extract_action(sample: dict) -> tuple[str, np.ndarray]:
    if "action" in sample:
        return "action", _to_numpy(sample["action"])
    if "actions" in sample:
        return "actions", _to_numpy(sample["actions"])
    raise KeyError("No action key found in sample. Expected 'action' or 'actions'.")


def _extract_state(sample: dict) -> np.ndarray | None:
    if "observation.state" in sample:
        return _to_numpy(sample["observation.state"])
    if "observation/state" in sample:
        return _to_numpy(sample["observation/state"])
    if "observation" in sample and isinstance(sample["observation"], dict):
        obs = sample["observation"]
        if "state" in obs:
            return _to_numpy(obs["state"])
    return None


def _build_action_chunk(dataset, action_key: str, start_idx: int, n_action_steps: int) -> np.ndarray | None:
    first = _to_numpy(dataset[start_idx][action_key])
    if first.ndim == 1:
        chunk = []
        for offset in range(n_action_steps):
            try:
                action = _to_numpy(dataset[start_idx + offset][action_key])
            except IndexError:
                return None
            chunk.append(action)
        return np.stack(chunk, axis=0)
    if first.ndim == 2:
        if first.shape[0] < n_action_steps:
            return None
        return first[:n_action_steps]
    if first.ndim == 3 and first.shape[0] == 1:
        first = first.squeeze(0)
        if first.shape[0] < n_action_steps:
            return None
        return first[:n_action_steps]
    raise ValueError(f"Unexpected action shape: {first.shape}")


@dataclasses.dataclass
class Args:
    # the dataset_root must match the config in config.py!
    config_name: str = "pi05_kinova"
    checkpoint_dir: str = "./checkpoints/pi05_kinova/20260402_0010_goal_image_pi05/29999"
    dataset_root: str =  "./dataset/20260402_T00-00-01-00_merge_goal_image"
    episode_index: int = 145
    max_frames: int | None = 1000
    default_prompt: str | None =  "<control_mode> end effector </control_mode> Assemble to match the goal image."
    # "<control_mode> end effector </control_mode> Assemble the currently grasped LEGO brick onto the existing structure on the green baseplate, matching the configuration shown in the goal image."
    # "<control_mode> end effector </control_mode> Assemble to match the goal image."
    plot: bool = True


class OfflinePolicyEvaluator:
    def __init__(self, args: Args):
        self.args = args
        self.config = _config.get_config(args.config_name)

        if self.config.data is None:
            raise ValueError("Config is missing data settings.")
        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        if data_config.repo_id is None:
            raise ValueError("Config data repo_id is None; cannot load dataset.")

        self._repack_transforms = _strip_action_from_repack(data_config.repack_transforms)

        self.action_horizon = self.config.model.action_horizon

        self.policy = _policy_config.create_trained_policy(
            self.config,
            args.checkpoint_dir,
            repack_transforms=self._repack_transforms,
            default_prompt=args.default_prompt,
        )

        self.dataset = _create_lerobot_dataset(data_config.repo_id, self.action_horizon, args.dataset_root)

    def _get_episode_bounds(self, episode_index: int) -> tuple[int, int]:
        if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "episodes"):
            episodes = self.dataset.meta.episodes
            if episode_index < 0 or episode_index >= len(episodes):
                raise ValueError(f"Episode {episode_index} out of range (0..{len(episodes) - 1}).")
            info = episodes[episode_index]
            if "dataset_from_index" in info:
                return info["dataset_from_index"], info["length"]
            if "from_index" in info:
                return info["from_index"], info["length"]
            # LeRobot v2.1 episodes may only include length; compute offset.
            offset = 0
            for idx in range(episode_index):
                offset += episodes[idx]["length"]
            return offset, info["length"]

        # Fallback: treat the whole dataset as a single episode
        if episode_index != 0:
            raise ValueError("Dataset does not contain episodes; only episode_index=0 is valid.")
        return 0, len(self.dataset)

    def evaluate_episode(self) -> dict[int, dict[str, Any]]:
        episode_start, episode_length = self._get_episode_bounds(self.args.episode_index)

        if hasattr(self.policy, "reset"):
            self.policy.reset()

        max_frames = self.args.max_frames or episode_length
        max_frames = min(max_frames, episode_length)

        results: dict[int, dict[str, Any]] = {}

        for frame_idx in range(max_frames):
            sample_idx = episode_start + frame_idx
            sample = self.dataset[sample_idx]

            action_key, _ = _extract_action(sample)
            obs = {k: v for k, v in sample.items() if k not in {"action", "actions"}}
            obs = _tree_to_numpy(obs)
            current_state = _extract_state(sample)

            # Skip frames where we cannot build a full action chunk.
            gt_chunk = _build_action_chunk(self.dataset, action_key, sample_idx, self.action_horizon)
            if gt_chunk is None:
                continue

            pred = self.policy.infer(obs)
            pred_chunk = _to_numpy(pred["actions"])

            if pred_chunk.ndim == 1:
                pred_chunk = pred_chunk[None, ...]

            # Align lengths defensively.
            min_len = min(pred_chunk.shape[0], gt_chunk.shape[0])
            pred_chunk = pred_chunk[:min_len]
            gt_chunk = gt_chunk[:min_len]

            mse = float(np.mean((pred_chunk - gt_chunk) ** 2))
            pos_error = float(np.mean(np.linalg.norm(pred_chunk[:, :3] - gt_chunk[:, :3], axis=1)))

            results[frame_idx] = {
                "prediction_mse": mse,
                "mean_position_error": pos_error,
                "predicted_action_chunk": pred_chunk,
                "ground_truth_action_chunk": gt_chunk,
                "observation": obs,
                "current_state": current_state,
            }

        return results


def _plot_metrics(results: dict[int, dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    frames = sorted(results.keys())
    mse = [results[idx]["prediction_mse"] for idx in frames]
    pos_err = [results[idx]["mean_position_error"] for idx in frames]

    if not frames:
        print("No frames to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(frames, mse, "b-o", linewidth=2, markersize=4, alpha=0.7)
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Overall MSE")
    ax1.set_title("Action MSE")
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames, pos_err, "g-o", linewidth=2, markersize=4, alpha=0.7)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("Mean Position Error (m)")
    ax2.set_title("Position Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


class InteractiveEpisodeVisualizer:
    def __init__(
        self,
        results: dict[int, dict[str, Any]],
        image_keys: tuple[str, str] = ("observation.images.fixed_camera", "observation.images.wrist_camera"),
    ) -> None:
        self.results = results
        self.image_keys = image_keys
        self.valid_frames = sorted([
            idx for idx, data in results.items() if data.get("predicted_action_chunk") is not None
        ])
        if not self.valid_frames:
            raise ValueError("No frames with predictions found in results")

        self.current_idx = 0
        self.current_frame = self.valid_frames[self.current_idx]

        self.fig = None
        self.gs = None
        self._setup_figure()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._update_plot()

        print("\n" + "=" * 60)
        print("Interactive Episode Visualizer")
        print("=" * 60)
        print(f"Total frames with predictions: {len(self.valid_frames)}")
        print(f"Frame range: {self.valid_frames[0]} - {self.valid_frames[-1]}")
        print("\nControls:")
        print("  Left Arrow  : Previous frame")
        print("  Right Arrow : Next frame")
        print("  Home        : First frame")
        print("  End         : Last frame")
        print("  Q / Esc     : Quit")
        print("=" * 60 + "\n")

    def _setup_figure(self) -> None:
        import matplotlib.pyplot as plt

        self.fig = plt.figure(figsize=(14, 8))
        self.gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 2], hspace=0.2, wspace=0.3)

    def _on_key_press(self, event) -> None:
        if event.key == "right":
            if self.current_idx < len(self.valid_frames) - 1:
                self.current_idx += 1
                self.current_frame = self.valid_frames[self.current_idx]
                self._update_plot()
            else:
                print("Already at last frame")
        elif event.key == "left":
            if self.current_idx > 0:
                self.current_idx -= 1
                self.current_frame = self.valid_frames[self.current_idx]
                self._update_plot()
            else:
                print("Already at first frame")
        elif event.key == "home":
            self.current_idx = 0
            self.current_frame = self.valid_frames[self.current_idx]
            self._update_plot()
        elif event.key == "end":
            self.current_idx = len(self.valid_frames) - 1
            self.current_frame = self.valid_frames[self.current_idx]
            self._update_plot()
        elif event.key in ["q", "escape"]:
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _plot_images(self, observation: dict) -> None:
        import matplotlib.pyplot as plt

        for row, key in enumerate(self.image_keys):
            ax = self.fig.add_subplot(self.gs[row, 0])
            img = observation.get(key)
            if img is None:
                ax.set_title(f"Missing: {key}")
                ax.axis("off")
                continue

            img = _to_numpy(img)
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] != 3:
                img = np.transpose(img, (1, 2, 0))
            if np.issubdtype(img.dtype, np.floating) and (img.min() < 0 or img.max() > 1.5):
                img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(key.split(".")[-1].replace("_", " ").title())
            ax.axis("off")

    def _plot_trajectory(self, frame_idx: int, frame_data: dict) -> None:
        ax = self.fig.add_subplot(self.gs[:, 1], projection="3d")

        pred_chunk = frame_data["predicted_action_chunk"]
        gt_chunk = frame_data["ground_truth_action_chunk"]
        current_state = frame_data.get("current_state")

        pred_positions = pred_chunk[:, :3]
        gt_positions = gt_chunk[:, :3]

        if current_state is not None and current_state.shape[0] >= 3:
            pos = current_state[:3]
            ax.scatter(pos[0], pos[1], pos[2], color="gold", s=200, marker="*", edgecolors="black", label="Current")

        ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], "b-o", label="Ground Truth")
        ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], "r--^", label="Predicted")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Predicted vs GT Actions (Frame {frame_idx})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _update_plot(self) -> None:
        self.fig.clear()
        frame_data = self.results[self.current_frame]
        observation = frame_data.get("observation", {})

        self.gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 2], hspace=0.2, wspace=0.3)
        self._plot_images(observation)
        self._plot_trajectory(self.current_frame, frame_data)

        self.fig.suptitle(
            f"Frame {self.current_frame} ({self.current_idx + 1} of {len(self.valid_frames)})",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        self.fig.canvas.draw()

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def main(args: Args) -> None:
    evaluator = OfflinePolicyEvaluator(args)
    results = evaluator.evaluate_episode()

    if not results:
        print("No results computed. Check dataset length and action_horizon.")
        return

    mse_values = [v["prediction_mse"] for v in results.values()]
    pos_values = [v["mean_position_error"] for v in results.values()]

    print("Evaluation summary")
    print(f"  Frames evaluated: {len(results)}")
    print(f"  Mean MSE: {np.mean(mse_values):.6f}  (std: {np.std(mse_values):.6f})")
    print(f"  Mean position error: {np.mean(pos_values):.6f}m  (std: {np.std(pos_values):.6f})")

    if args.plot:
        _plot_metrics(results)

    print("\nLaunching interactive visualizer...")
    visualizer = InteractiveEpisodeVisualizer(results)
    visualizer.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
