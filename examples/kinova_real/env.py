from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, WrenchStamped
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from sensor_msgs.msg import CompressedImage, JointState
from typing_extensions import override


class KinovaEnvironment(_environment.Environment):
    """Real-time Kinova environment backed by ROS topics."""

    def __init__(
        self,
        *,
        fixed_camera_topic: str = "/fixed_cam/camera/color/image_raw/compressed",
        wrist_camera_topic: str = "/wrist_cam/camera/color/image_raw/compressed",
        ee_pose_topic: str = "/kinova_ros_control/feedback_cartesian_fk",
        gripper_state_topic: str = "/kinova_ros_control/feedback_gripper_state",
        external_wrench_topic: str = "/kinova_ros_control/feedback_external_wrench",
        ee_action_topic: str = "/kinova_ros_control/cartesian_target",
        gripper_action_topic: str = "/kinova_ros_control/gripper_target",
        render_height: int = 224,
        render_width: int = 224,
        force_scale_factor: float = 0.05,
        goal_image_path: str | None = None,
        prompt: str | None = "Assemble to match the goal image.",
        wait_timeout_sec: float = 10.0,
        init_node: bool = True,
    ) -> None:
        self._render_height = render_height
        self._render_width = render_width
        self._force_scale_factor = force_scale_factor
        self._prompt = prompt
        self._wait_timeout_sec = wait_timeout_sec

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._images: dict[str, Optional[np.ndarray]] = {
            "fixed_camera": None,
            "wrist_camera": None,
            "goal_image": None,
        }
        self._ee_pose_9d: Optional[np.ndarray] = None
        self._gripper_position: Optional[np.ndarray] = None
        self._external_force: Optional[np.ndarray] = None

        if init_node and not rospy.core.is_initialized():
            rospy.init_node("openpi_kinova_runtime", anonymous=True)

        self._fixed_camera_sub = rospy.Subscriber(
            fixed_camera_topic,
            CompressedImage,
            lambda msg: self._camera_callback(msg, "fixed_camera"),
            queue_size=3,
        )
        self._wrist_camera_sub = rospy.Subscriber(
            wrist_camera_topic,
            CompressedImage,
            lambda msg: self._camera_callback(msg, "wrist_camera"),
            queue_size=3,
        )
        self._ee_pose_sub = rospy.Subscriber(
            ee_pose_topic,
            PoseStamped,
            self._pose_callback,
            queue_size=10,
        )
        self._gripper_state_sub = rospy.Subscriber(
            gripper_state_topic,
            JointState,
            self._gripper_callback,
            queue_size=10,
        )
        self._external_wrench_sub = rospy.Subscriber(
            external_wrench_topic,
            WrenchStamped,
            self._wrench_callback,
            queue_size=10,
        )

        self._ee_pose_action_pub = rospy.Publisher(
            ee_action_topic,
            PoseStamped,
            queue_size=10,
        )
        self._gripper_action_pub = rospy.Publisher(
            gripper_action_topic,
            JointState,
            queue_size=10,
        )

        if goal_image_path is not None:
            self.set_goal_image_from_path(goal_image_path)

    @override
    def reset(self) -> None:
        self._wait_until_ready(self._wait_timeout_sec)

    @override
    def is_episode_complete(self) -> bool:
        # Real robot execution is externally controlled.
        return False

    @override
    def get_observation(self) -> dict:
        with self._lock:
            if self._images["fixed_camera"] is None:
                raise RuntimeError("Missing fixed camera image.")
            if self._images["wrist_camera"] is None:
                raise RuntimeError("Missing wrist camera image.")
            if self._images["goal_image"] is None:
                raise RuntimeError("Missing goal image. Set --goal-image-path.")
            if self._ee_pose_9d is None:
                raise RuntimeError("Missing ee pose feedback.")
            if self._gripper_position is None:
                raise RuntimeError("Missing gripper state feedback.")
            if self._external_force is None:
                raise RuntimeError("Missing external wrench feedback.")

            state = np.concatenate(
                [
                    self._ee_pose_9d,
                    self._gripper_position,
                    self._external_force * self._force_scale_factor,
                ],
                axis=0,
            ).astype(np.float32)

            obs = {
                "observation/fixed_camera": self._images["fixed_camera"],
                "observation/wrist_camera": self._images["wrist_camera"],
                "observation/goal_image": self._images["goal_image"],
                "observation/state": state,
            }

            if self._prompt is not None:
                obs["prompt"] = self._prompt

            return obs

    @override
    def apply_action(self, action: dict) -> None:
        if "actions" not in action:
            raise KeyError("Expected action dict with key 'actions'.")

        action_vec = np.asarray(action["actions"], dtype=np.float32)
        if action_vec.shape != (10,):
            raise ValueError(f"Expected action shape (10,), got {action_vec.shape}")

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.get_rostime()
        pose_msg.pose.position.x = float(action_vec[0])
        pose_msg.pose.position.y = float(action_vec[1])
        pose_msg.pose.position.z = float(action_vec[2])

        quat_xyzw = self._rot6d_to_quat(action_vec[3:9])
        pose_msg.pose.orientation.x = float(quat_xyzw[0])
        pose_msg.pose.orientation.y = float(quat_xyzw[1])
        pose_msg.pose.orientation.z = float(quat_xyzw[2])
        pose_msg.pose.orientation.w = float(quat_xyzw[3])
        self._ee_pose_action_pub.publish(pose_msg)

        gripper_position = float(np.clip(action_vec[9], 0.0, 1.0))
        gripper_msg = JointState()
        gripper_msg.header.stamp = rospy.get_rostime()
        gripper_msg.position = [gripper_position]
        self._gripper_action_pub.publish(gripper_msg)

    def set_goal_image_from_path(self, path: str) -> None:
        image_path = Path(path)
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read goal image from {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.set_goal_image(img_rgb)

    def set_goal_image(self, image_rgb: np.ndarray) -> None:
        processed = self._preprocess_image(image_rgb)
        with self._lock:
            self._images["goal_image"] = processed

    def _wait_until_ready(self, timeout_sec: float) -> None:
        start = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            with self._lock:
                ready = (
                    self._images["fixed_camera"] is not None
                    and self._images["wrist_camera"] is not None
                    and self._images["goal_image"] is not None
                    and self._ee_pose_9d is not None
                    and self._gripper_position is not None
                    and self._external_force is not None
                )
            if ready:
                return

            now = rospy.Time.now().to_sec()
            if timeout_sec > 0 and (now - start) >= timeout_sec:
                raise TimeoutError(
                    "Timed out waiting for Kinova observations. "
                    "Check camera/state topics and goal image." 
                )
            rospy.sleep(0.02)

    def _camera_callback(self, msg: CompressedImage, cam_name: str) -> None:
        img_bgr = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = self._preprocess_image(img_rgb)
        with self._lock:
            self._images[cam_name] = img

    def _pose_callback(self, msg: PoseStamped) -> None:
        q = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        rot6d = self._quat_to_rot6d(q)
        pos = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ],
            dtype=np.float32,
        )
        ee_pose_9d = np.concatenate([pos, rot6d], axis=0)
        with self._lock:
            self._ee_pose_9d = ee_pose_9d

    def _gripper_callback(self, msg: JointState) -> None:
        if not msg.position:
            return
        with self._lock:
            self._gripper_position = np.array([msg.position[0]], dtype=np.float32)

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        force = msg.wrench.force
        with self._lock:
            self._external_force = np.array([force.x, force.y, force.z], dtype=np.float32)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self._render_height, self._render_width)
        )

    @staticmethod
    def _quat_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
        x, y, z, w = quat_xyzw
        # Quaternion to rotation matrix.
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r02 = 2 * (x * z + y * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - x * w)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        r22 = 1 - 2 * (x * x + y * y)
        rot = np.array(
            [
                [r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22],
            ],
            dtype=np.float32,
        )
        # 6D representation = first two columns.
        return np.concatenate([rot[:, 0], rot[:, 1]], axis=0)

    @staticmethod
    def _rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
        a1 = rot6d[:3]
        a2 = rot6d[3:6]

        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        a2_proj = a2 - np.dot(b1, a2) * b1
        b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
        b3 = np.cross(b1, b2)

        rot = np.stack([b1, b2, b3], axis=1)
        return KinovaEnvironment._rotmat_to_quat(rot)

    @staticmethod
    def _rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
        # Returns quaternion in [x, y, z, w].
        t = np.trace(rot)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            w = 0.25 * s
            x = (rot[2, 1] - rot[1, 2]) / s
            y = (rot[0, 2] - rot[2, 0]) / s
            z = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s

        quat = np.array([x, y, z, w], dtype=np.float32)
        quat /= np.linalg.norm(quat) + 1e-8
        return quat
