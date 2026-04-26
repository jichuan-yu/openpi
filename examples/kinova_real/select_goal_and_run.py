from __future__ import annotations

"""
Example:
python /home/lego/openpi/examples/kinova_real/select_goal_and_run.py \
  --args.config-name pi05_kinova \
  --args.checkpoint-dir /home/lego/openpi/checkpoints/pi05_kinova/20260402_0010_goal_image_pi05/29999 \
  --args.video-path "/home/lego/Jichuan/DexBrick/data/file-000(7)_h264.mp4" \
  --args.prompt "<control_mode> end effector </control_mode> Assemble to match the goal image."

python /home/lego/openpi/examples/kinova_real/select_goal_and_run.py \
  --args.config-name pi05_kinova \
  --args.checkpoint-dir /home/lego/openpi/checkpoints/pi05_kinova/20260402_0010_goal_image_pi05/29999 \
  --args.video-path /home/lego/openpi/dataset/20260402_T00-00-01-00_merge_goal_image/videos/observation.images.goal_image/chunk-000/file-000_h264.mp4
  --args.prompt "<control_mode> end effector </control_mode> Assemble to match the goal image."
  
python /home/lego/openpi/examples/kinova_real/select_goal_and_run.py \
  --args.config-name pi05_kinova_zoom_goalimage_lastframe \
  --args.checkpoint-dir /home/lego/openpi/checkpoints/pi05_kinova/20260410_T02-00-00-00_merge_zoom_goalimage_lastframe_pi05/49999 \
  --args.video-path /home/lego/openpi/dataset/20260410_T02-00-00-00_merge_zoom_goalimage_lastframe/videos/observation.images.goal_image/chunk-000/file-000_h264.mp4


"""
                                 
import dataclasses
from pathlib import Path
import sys

import cv2
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

@dataclasses.dataclass
class Args:
    video_path: str = (
        "./dataset/20260410_T02-00-00-00_merge_zoom_goalimage_lastframe/"
        "videos/observation.images.goal_image/chunk-000/file-000_h264.mp4"
    )
    output_path: str = "./goal.jpg"
    window_name: str = "Select Goal And Run"
    checkpoint_dir: str = "./checkpoints/pi05_kinova/20260410_T02-00-00-00_merge_zoom_goalimage_lastframe_pi05/49999"
    config_name: str = "pi05_kinova_zoom_goalimage_lastframe"
    action_horizon: int = 4
    num_episodes: int = 1
    max_episode_steps: int = 1000
    max_hz: float = 10.0
    render_height: int = 224
    render_width: int = 224
    wait_timeout_sec: float = 10.0
    prompt: str = "Assemble to match the goal image."


def _run_policy(args: Args) -> None:
    from examples.kinova_real import main as kinova_main

    policy_args = kinova_main.Args(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        action_horizon=args.action_horizon,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        max_hz=args.max_hz,
        render_height=args.render_height,
        render_width=args.render_width,
        wait_timeout_sec=args.wait_timeout_sec,
        goal_image_path=args.output_path,
        prompt=args.prompt,
    )
    kinova_main.main(policy_args)


def _save_frame(frame_bgr, output_path: str | Path, frame_index: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError(f"Failed to save image to: {output_path}")
    print(f"Saved frame {frame_index} to {output_path}")


def _read_frame(cap: cv2.VideoCapture, frame_index: int, video_path: Path):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from video: {video_path}")
    return frame_bgr


def _run_terminal_fallback(cap: cv2.VideoCapture, total_frames: int, video_path: Path, args: Args) -> None:
    print("OpenCV GUI is unavailable. Falling back to terminal mode.")
    print("Type 's' to start selecting a frame, then enter a frame number, then press Enter again to save+run.")
    print("Type 'q' at any prompt to quit.")

    capture_armed = False
    current_index = 0

    while True:
        if not capture_armed:
            command = input("Command [s/q]: ").strip().lower()
            if command == "q":
                print("Quit without running policy.")
                return
            if command != "s":
                print("Please type 's' to start or 'q' to quit.")
                continue
            capture_armed = True
            print(f"Capture armed. Choose a frame index from 0 to {total_frames - 1}.")

        frame_input = input(f"Frame index [current={current_index}]: ").strip().lower()
        if frame_input == "q":
            print("Quit without running policy.")
            return
        if frame_input:
            current_index = int(frame_input)
        if current_index < 0 or current_index >= total_frames:
            print(f"Frame index out of range. Please choose 0 to {total_frames - 1}.")
            continue

        frame_bgr = _read_frame(cap, current_index, video_path)
        preview_path = Path(args.output_path).with_name(f"{Path(args.output_path).stem}_preview_{current_index}.jpg")
        _save_frame(frame_bgr, preview_path, current_index)
        confirm = input("Press Enter to use this frame and run policy, or type another frame index/q: ").strip().lower()
        if confirm == "":
            _save_frame(frame_bgr, args.output_path, current_index)
            _run_policy(args)
            return
        if confirm == "q":
            print("Quit without running policy.")
            return
        current_index = int(confirm)


def main(args: Args) -> None:
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Failed to get frame count from video: {video_path}")

    window_name = args.window_name
    max_index = total_frames - 1
    current_index = 0
    rendered_index = -1
    frame_bgr = None
    capture_armed = False

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.createTrackbar("frame", window_name, 0, max_index, lambda _x: None)
    except cv2.error:
        _run_terminal_fallback(cap, total_frames, video_path, args)
        cap.release()
        cv2.destroyAllWindows()
        return

    print(f"Opened video: {video_path}")
    print(f"Total frames: {total_frames}")
    print("Controls:")
    print("  Drag the trackbar to choose a frame.")
    print("  Press 's' to start capture.")
    print("  Press Enter to save the current frame as goal image and run policy.")
    print("  Press 'q' to quit.")

    try:
        while True:
            current_index = cv2.getTrackbarPos("frame", window_name)
            if current_index != rendered_index:
                frame_bgr = _read_frame(cap, current_index, video_path)
                rendered_index = current_index

            preview = frame_bgr.copy()
            cv2.putText(
                preview,
                f"frame {current_index}/{max_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "s = start capture, Enter = save+run, q = quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if capture_armed:
                cv2.putText(
                    preview,
                    "CAPTURE ARMED",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 128, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                capture_armed = True
                print(f"Capture armed at frame {current_index}. Press Enter to save and run policy.")
            elif key in (13, 10):
                if not capture_armed:
                    print("Press 's' first to arm capture.")
                    continue
                if frame_bgr is None:
                    raise RuntimeError("No frame available to save.")
                _save_frame(frame_bgr, args.output_path, current_index)
                cv2.destroyAllWindows()
                cap.release()
                _run_policy(args)
                return
            elif key in (ord("q"), 27):
                print("Quit without running policy.")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
