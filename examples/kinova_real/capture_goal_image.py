from __future__ import annotations

import dataclasses
from pathlib import Path

import cv2
import tyro


@dataclasses.dataclass
class Args:
    video_path: str
    output_path: str = "./goal.jpg"
    window_name: str = "Capture Goal Image"


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

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.createTrackbar("frame", window_name, 0, max_index, lambda _x: None)

    print(f"Opened video: {video_path}")
    print(f"Total frames: {total_frames}")
    print("Controls:")
    print("  Drag the trackbar to choose a frame.")
    print("  Press 's' to save the current frame.")
    print("  Press 'q' to quit.")

    try:
        while True:
            current_index = cv2.getTrackbarPos("frame", window_name)
            if current_index != rendered_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError(f"Failed to read frame {current_index} from video: {video_path}")

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
                    "s = save, q = quit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                rendered_index = current_index

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                output_path = Path(args.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if frame_bgr is None:
                    raise RuntimeError("No frame available to save.")
                if not cv2.imwrite(str(output_path), frame_bgr):
                    raise RuntimeError(f"Failed to save image to: {output_path}")
                print(f"Saved frame {current_index} to {output_path}")
            elif key in (ord("q"), 27):
                print("Quit frame capture.")
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
