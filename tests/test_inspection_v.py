import numpy as np
import asyncio
import random
import cv2
import math
from datetime import datetime
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(ROOT_DIR, "media")

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, quaternion_to_rpy, unpack_image
from projectairsim.drone import YawControlMode
from projectairsim.types import ImageType
from projectairsim.image_utils import ImageDisplay

# ========== Config ==========
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = os.path.join(ROOT_DIR, "media")
DEFAULT_Z = -6.0
VIDEO_FPS = 30
MOVE_VELOCITY = 1.0

# Image display settings
SUBWIN_WIDTH = 640
SUBWIN_HEIGHT = 360


class Intersection:
    id: str
    x: float
    y: float
    z: float

    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    @property
    def coords(self):
        return (self.x, self.y, self.z)

class RoadNetwork:
    def __init__(self, default_z = DEFAULT_Z):
        self._nodes = {}
        self._adj = {}
        self.default_z = default_z

    def add_intersection(self, id, x, y, z = None):
        z = self.default_z if z is None else z
        self._nodes[id] = Intersection(id, x, y, z)
        self._adj.setdefault(id, [])

    def add_connection(self, a, b, bidirectional = True):
        if a not in self._nodes or b not in self._nodes:
            return
        if b not in self._adj[a]:
            self._adj[a].append(b)
        if bidirectional and a not in self._adj[b]:
            self._adj[b].append(a)

    def neighbors(self, id):
        return list(self._adj.get(id, []))

    def random_neighbor(self, id: str):
        neigh = self.neighbors(id)
        if not neigh:
            return None
        return np.random.choice(neigh)

    def coords(self, id):
        node = self._nodes.get(id)
        return None if node is None else node.coords

    def create_road_network(self) -> None:
        coords = {
            "A": (129.2, -225.8), "B": (129.2, -160.0), "C": (129.3, -89.8), "D": (128.2, -9.6),
            "E": (133.1, 72.2), "F": (139.6, 160.6), "G": (3.6, -229.5), "H": (3.6, -157.9),
            "I": (4.8, -92.4), "J": (6.0, -9.6), "K": (5.0, 72.2), "L": (0.4, 160.6),
            "M": (-125.0, -229.5), "N": (-125.4, -156.1), "O": (-125.4, -88.3), "P": (-125.4, -7.6),
            "Q": (-123.4, 72.2), "R": (-130.8, 160.6), "S": (-252.7, -229.5), "T": (-253.3, -162.3),
            "U": (-253.3, -93.2), "V": (-253.3, -11.4), "W": (-253.3, 72.6), "X": (-253.3, 159.4)
        }
        for k, (x, y) in coords.items():
            self.add_intersection(k, x, y)

        edges = [
            ("A","B"),("A","G"),("B","C"),("B","H"),("C","D"),("C","I"),("D","E"),("D","J"),
            ("E","F"),("E","K"),("F","L"),("G","H"),("G","M"),("H","I"),("H","N"),("I","J"),
            ("I","O"),("J","K"),("J","P"),("K","L"),("K","Q"),("L","R"),("M","N"),("M","S"),
            ("N","O"),("N","T"),("O","P"),("O","U"),("P","Q"),("P","V"),("Q","R"),("Q","W"),
            ("R","X"),("S","T"),("T","U"),("U","V"),("V","W"),("W","X")
        ]
        for a, b in edges:
            self.add_connection(a, b)

class CollisionState:
    def __init__(self):
        self.collision = False

    def collision_callback(self, value = True):
        projectairsim_log().info("Exit because of collision.")
        self.collision = value

class ImageHandler:
    def __init__(self, client, display_enabled = True, save_enabled = False):
        self.client = client
        self.display_enabled = display_enabled
        self.save_enabled = save_enabled

        self.image_display = ImageDisplay(
            num_subwin=3,
            screen_res_x=2560,
            screen_res_y=1440,
            subwin_width=SUBWIN_WIDTH,
            subwin_height=SUBWIN_HEIGHT
        )
        self.video_writers = {}
        self.topics = {}

    def start(self) -> None:
        if self.display_enabled:
            self.image_display.start()

    def stop(self) -> None:
        if self.display_enabled:
            self.image_display.stop()
        self._close_writers()

    def register_window(self, name, idx) -> None:
        self.image_display.add_image(name, subwin_idx=idx, resize_x=SUBWIN_WIDTH, resize_y=SUBWIN_HEIGHT)

    def subscribe_camera(self, sensor_topic, window_name: str) -> None:
        self.client.subscribe(sensor_topic, lambda topic, msg: self._on_image_received(msg, window_name))

    def _on_image_received(self, image_msg, topic_name: str) -> None:
        if self.display_enabled:
            self.image_display.receive(image_msg, topic_name)

        if self.save_enabled:
            frame = unpack_image(image_msg)
            if frame is None:
                return
            writer = self._get_or_create_writer(topic_name, frame)
            bgr_frame = self._frame_to_bgr(frame)
            if bgr_frame is not None:
                writer.write(bgr_frame)

    def _frame_to_bgr(self, frame: np.ndarray):
        if frame is None:
            return None
        if frame.ndim == 2:
            depth_map = np.nan_to_num(frame)
            if depth_map.dtype != np.uint8:
                depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_uint8 = depth_norm.astype(np.uint8)
            else:
                depth_uint8 = depth_map
            return cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _get_or_create_writer(self, topic_name: str, sample_frame: np.ndarray) -> cv2.VideoWriter:
        if topic_name in self.video_writers:
            return self.video_writers[topic_name]

        os.makedirs(SAVE_PATH, exist_ok=True)
        height, width = sample_frame.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in str(topic_name) if c.isalnum() or c in ('_', '-')).rstrip()
        video_path = os.path.join(SAVE_PATH, f"{safe_name}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))
        self.video_writers[topic_name] = writer
        projectairsim_log().info(f"Created video writer for {topic_name}: {video_path}")
        return writer

    def _close_writers(self) -> None:
        for name, writer in list(self.video_writers.items()):
            if writer is not None:
                writer.release()
        self.video_writers.clear()

class DroneController:
    def __init__(self, client, world_scene, drone_name = "Drone1"):
        self.client = client
        self.client.connect()
        self.world = World(self.client, world_scene, delay_after_load_sec=2)
        self.drone = Drone(self.client, self.world, drone_name)
        self.drone.enable_api_control()
        self.drone.arm()

    def disconnect(self) -> None:
        self.drone.disarm()
        self.drone.disable_api_control()
        self.client.disconnect()

    async def takeoff_async(self):
        return await self.drone.takeoff_async()

    async def land_async(self):
        return await self.drone.land_async()

    async def move_by_velocity_z_async(self, v_north, v_east, z, duration, yaw_control_mode=YawControlMode.ForwardOnly, yaw_is_rate=False, yaw=0):
        return await self.drone.move_by_velocity_z_async(v_north, v_east, z, duration, yaw_control_mode=yaw_control_mode, yaw_is_rate=yaw_is_rate, yaw=yaw)

    def get_ground_truth_kinematics(self):
        return self.drone.get_ground_truth_kinematics()

    def get_images(self, camera_name, types):
        return self.drone.get_images(camera_name, types)


class InspectionManager:
    def __init__(self,
                 scene_file = "scene_drone_classic.jsonc",
                 start_intersection = "J",
                 display_images = True,
                 save_images = False):
        self.client = ProjectAirSimClient()
        self.drone_ctrl = DroneController(self.client, scene_file, drone_name="Drone1")
        self.image_handler = ImageHandler(self.client, display_enabled=display_images, save_enabled=save_images)
        self.collision_state = CollisionState()

        # road network and navigation
        self.roadnetwork = RoadNetwork()
        self.roadnetwork.create_road_network()
        self.target_distance_threshold = 5.0  # (m)
        self.change_direction_distance_threshold = 10000 # (mm)

        # smoothing state
        self.prev_v_north = 0.0
        self.prev_v_east = 0.0
        self.smoothing_factor = 0.3

        # positions and targets
        self._init_positions(start_intersection)

        # setup subscriptions and displays
        self._setup_subscriptions()

    def _init_positions(self, start_intersection):
        # initial position update
        self.update_cur_position()
        self.from_position = self.cur_position
        # self.target_id = self.roadnetwork.random_neighbor(start_intersection)
        self.target_id = "D"
        if self.target_id is None:
            raise RuntimeError(f"No neighbor found for start node {start_intersection}")
        coords = self.roadnetwork.coords(self.target_id)
        if coords is None:
            raise RuntimeError(f"Coordinates for {self.target_id} not found")
        self.to_position = coords

    def _setup_subscriptions(self) -> None:
        # windows
        chase = "ChaseCam"
        front = "FrontCam"
        depth = "FrontDepthImage"
        self.image_handler.register_window(chase, 0)
        self.image_handler.register_window(front, 1)
        self.image_handler.register_window(depth, 2)

        # subscribe cameras
        self.client.subscribe(self.drone_ctrl.drone.sensors["Chase"]["scene_camera"],
                              lambda topic, msg: self.image_handler._on_image_received(msg, chase))
        self.client.subscribe(self.drone_ctrl.drone.sensors["front_center"]["scene_camera"],
                              lambda topic, msg: self.image_handler._on_image_received(msg, front))
        self.client.subscribe(self.drone_ctrl.drone.sensors["front_center"]["depth_planar_camera"],
                              lambda topic, msg: self.image_handler._on_image_received(msg, depth))

        # collision topic
        self.client.subscribe(self.drone_ctrl.drone.robot_info["collision_info"],
                              lambda topic, msg: self.collision_state.collision_callback(True))

        # start the image display if needed
        self.image_handler.start()

    def update_cur_position(self):
        kin = self.drone_ctrl.get_ground_truth_kinematics()
        pos = kin['pose']['position']
        self.cur_position = (pos['x'], pos['y'], pos['z'])

    def update_from_and_to_position(self):
        self.from_position = self.cur_position
        next_target = self.roadnetwork.random_neighbor(self.target_id)
        if next_target is None:
            projectairsim_log().warning("No further neighbors from current target; staying at current target.")
            return
        self.target_id = next_target
        self.to_position = self.roadnetwork.coords(self.target_id)

    def has_arrived(self):
        kin = self.drone_ctrl.get_ground_truth_kinematics()
        pos = kin['pose']['position']
        dist = np.linalg.norm(np.array([pos['x'], pos['y'], pos['z']]) - np.array(self.to_position))
        return dist < self.target_distance_threshold

    def _compute_obstacle_avoidance(self):
        # read planar depth image from front_center
        images = self.drone_ctrl.get_images("front_center", [ImageType.DEPTH_PLANAR])
        depth_image = unpack_image(images[ImageType.DEPTH_PLANAR])
        if depth_image is None:
            projectairsim_log().warning("No depth image available; defaulting to forward small movement.")
            # fallback: small forward velocity
            return (0.1, 0.0, self.to_position[2])

        # split middle strip and bands similar to original
        middle = np.vsplit(depth_image, 3)[1]
        num_bands = 5
        bands = np.hsplit(middle, np.linspace(1, middle.shape[1]-1, num_bands+1, dtype=int)[1:-1]) if middle.shape[1] >= num_bands else [middle]
        band_depths = [band.min() for band in bands]
        center_index = len(bands) // 2
        projectairsim_log().info(f"Band depths (mm): {band_depths}")

        # obstacle detection
        front_min = band_depths[center_index]
        if front_min < self.change_direction_distance_threshold:
            projectairsim_log().info("Obstacle detected ahead.")
            safe_index = int(np.argmax(band_depths))
            projectairsim_log().info(f"Steering toward band index: {safe_index}")

            total_bands = len(bands)
            angle_per_band = (math.pi / 2) / total_bands
            angle_offset = (safe_index - center_index) * angle_per_band

            kinematics = self.drone_ctrl.drone.get_ground_truth_kinematics()
            orientation = kinematics['pose']['orientation']
            roll, pitch, yaw = quaternion_to_rpy(orientation["w"], orientation["x"], orientation["y"], orientation["z"])
            abs_yaw = yaw + angle_offset
            v_north = MOVE_VELOCITY * math.cos(abs_yaw)
            v_east = MOVE_VELOCITY * math.sin(abs_yaw)
            z = self.to_position[2]
        else:
            # go toward target
            self.update_cur_position()
            cur_x, cur_y, cur_z = self.cur_position
            to_x, to_y, to_z = self.to_position
            dx = to_x - cur_x
            dy = to_y - cur_y
            distance = math.hypot(dx, dy)
            unit_dx = dx / (distance + 1e-6)
            unit_dy = dy / (distance + 1e-6)
            v_north = unit_dx * MOVE_VELOCITY
            v_east = unit_dy * MOVE_VELOCITY
            z = to_z

        # smoothing
        v_north = self.prev_v_north * (1 - self.smoothing_factor) + v_north * self.smoothing_factor
        v_east = self.prev_v_east * (1 - self.smoothing_factor) + v_east * self.smoothing_factor
        self.prev_v_north, self.prev_v_east = v_north, v_east
        projectairsim_log().info(f"Computed velocities - North: {v_north:.2f}, East: {v_east:.2f}, Z: {z:.2f}")

        return v_north, v_east, z

    async def step(self) -> None:
        v_north, v_east, z = self._compute_obstacle_avoidance()
        task = await self.drone_ctrl.move_by_velocity_z_async(v_north, v_east, z, 2 / MOVE_VELOCITY, yaw_control_mode=YawControlMode.ForwardOnly, yaw_is_rate=False, yaw=0)
        await task
        self.update_cur_position()

    async def run(self) -> None:
        try:
            projectairsim_log().info("Starting takeoff...")
            await self.drone_ctrl.takeoff_async()
            self.update_cur_position()

            while not self.collision_state.collision:
                await self.step()
                if self.has_arrived():
                    self.update_from_and_to_position()
                    projectairsim_log().info(f"New target intersection ID: {self.target_id}, coordinates: {self.to_position}")
                await asyncio.sleep(0)
        finally:
            projectairsim_log().info("Landing and cleaning up...")
            await self.drone_ctrl.land_async()
            self.shutdown()

    def shutdown(self) -> None:
        """Clean up resources: stop image handler and disconnect drone client."""
        self.image_handler.stop()
        self.image_handler._close_writers()
        self.drone_ctrl.disconnect()


# ========== Main Entrypoint ==========
def main():
    manager = InspectionManager(scene_file="scene_drone_classic.jsonc", start_intersection="J", display_images=True, save_images=True)
    try:
        asyncio.run(manager.run())
    except KeyboardInterrupt:
        projectairsim_log().info("Interrupted by user, shutting down.")
        manager.shutdown()


if __name__ == "__main__":
    main()