import dataclasses
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_IMPORT_PATHS = [
    REPO_ROOT / "src",
    REPO_ROOT / "packages" / "openpi-client" / "src",
    REPO_ROOT,
]
for path in reversed(LOCAL_IMPORT_PATHS):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openpi_client import action_chunk_broker
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import tyro

from examples.kinova_real import env as _env


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_kinova_zoom_goalimage_lastframe"
    checkpoint_dir: str = "./checkpoints/pi05_kinova/20260410_T02-00-00-00_merge_zoom_goalimage_lastframe_pi05/49999"

    action_horizon: int = 4

    num_episodes: int = 1
    max_episode_steps: int = 1000
    max_hz: float = 10.0

    render_height: int = 224
    render_width: int = 224
    wait_timeout_sec: float = 10.0

    goal_image_path: str = ""
    prompt: str = "<control_mode> end effector </control_mode> Assemble the currently grasped LEGO brick onto the existing structure on the green baseplate, matching the configuration shown in the goal image."
    # "<control_mode> end effector </control_mode> Assemble to match the goal image.",


def main(args: Args) -> None:
    if not args.goal_image_path:
        raise ValueError("--goal_image_path is required for Kinova goal-image policy.")

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    logging.info("Resolved config_name=%s", args.config_name)
    logging.info("Resolved data repo_id=%s", data_config.repo_id)
    logging.info("Resolved data asset_id=%s", data_config.asset_id)

    policy: _policy.Policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
    )
    logging.info("Loaded local policy from %s (config=%s)", args.checkpoint_dir, args.config_name)

    logging.info("Creating Kinova environment...")
    environment = _env.KinovaEnvironment(
        render_height=args.render_height,
        render_width=args.render_width,
        goal_image_path=args.goal_image_path,
        prompt=args.prompt,
        wait_timeout_sec=args.wait_timeout_sec,
    )
    logging.info("Kinova environment created.")

    logging.info("Creating policy agent...")
    agent = _policy_agent.PolicyAgent(
        policy=action_chunk_broker.ActionChunkBroker(
            policy=policy,
            action_horizon=args.action_horizon,
        )
    )
    logging.info("Policy agent created.")

    logging.info("Creating runtime...")
    runtime = _runtime.Runtime(
        environment=environment,
        agent=agent,
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    logging.info("Runtime created. Starting run loop...")

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
