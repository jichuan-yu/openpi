import dataclasses
import logging

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
    config_name: str = "pi0_kinova"
    checkpoint_dir: str = "./checkpoints/pi0_kinova/20260326_0010/30000"

    action_horizon: int = 16

    num_episodes: int = 1
    max_episode_steps: int = 1000
    max_hz: float = 5.0

    render_height: int = 224
    render_width: int = 224
    wait_timeout_sec: float = 10.0

    goal_image_path: str = ""
    prompt: str = "Assemble to match the goal image."


def main(args: Args) -> None:
    if not args.goal_image_path:
        raise ValueError("--goal_image_path is required for Kinova goal-image policy.")

    policy: _policy.Policy = _policy_config.create_trained_policy(
        _config.get_config(args.config_name),
        args.checkpoint_dir,
    )
    logging.info("Loaded local policy from %s (config=%s)", args.checkpoint_dir, args.config_name)

    runtime = _runtime.Runtime(
        environment=_env.KinovaEnvironment(
            render_height=args.render_height,
            render_width=args.render_width,
            goal_image_path=args.goal_image_path,
            prompt=args.prompt,
            wait_timeout_sec=args.wait_timeout_sec,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
