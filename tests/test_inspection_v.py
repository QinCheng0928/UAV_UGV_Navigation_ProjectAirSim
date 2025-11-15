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

class Intersection:
    def __init__(self, id, x, y, z):
        self.id = id
        self.coords = (x, y, z) 
        self.connections = []  
    
    def add_connection(self, target_id):
        if target_id not in self.connections:
            self.connections.append(target_id)

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
    """Simple adjacency-list road network."""
    def __init__(self, default_z = -6):
        self._nodes = {}
        self._adj = {}
        self.default_z = default_z

    def add_intersection(self, id, x, y, z = None) -> None:
        z = self.default_z if z is None else z
        self._nodes[id] = Intersection(id, x, y, z)
        self._adj.setdefault(id, [])

    def add_connection(self, a, b, bidirectional = True) -> None:
        if a not in self._nodes or b not in self._nodes:
            return
        if b not in self._adj[a]:
            self._adj[a].append(b)
        if bidirectional and a not in self._adj[b]:
            self._adj[b].append(a)

    def neighbors(self, id):
        return list(self._adj.get(id, []))

    def random_neighbor(self, id):
        neigh = self.neighbors(id)
        return None if not neigh else np.random.choice(neigh).item()

    def coords(self, id):
        node = self._nodes.get(id)
        return None if node is None else node.coords

    def create_road_network(self) -> None:
        # intersections (copied coordinates)
        self.add_intersection("A",  129.2, -225.8)
        self.add_intersection("B",  129.2, -160.0)
        self.add_intersection("C",  129.3, -89.8 )
        self.add_intersection("D",  128.2, -9.6  )
        self.add_intersection("E",  133.1,  72.2 )
        self.add_intersection("F",  139.6,  160.6)
        self.add_intersection("G",  3.6  , -229.5)
        self.add_intersection("H",  3.6  , -157.9)
        self.add_intersection("I",  4.8  , -92.4 )
        self.add_intersection("J",  6.0  , -9.6  )
        self.add_intersection("K",  5.0  ,  72.2 )
        self.add_intersection("L",  0.4  ,  160.6)
        self.add_intersection("M", -125.0, -229.5)
        self.add_intersection("N", -125.4, -156.1)
        self.add_intersection("O", -125.4, -88.3 )
        self.add_intersection("P", -125.4, -7.6  )
        self.add_intersection("Q", -123.4,  72.2 )
        self.add_intersection("R", -130.8,  160.6)
        self.add_intersection("S", -252.7, -229.5)
        self.add_intersection("T", -253.3, -162.3)
        self.add_intersection("U", -253.3, -93.2 )
        self.add_intersection("V", -253.3, -11.4 )
        self.add_intersection("W", -253.3,  72.6 )
        self.add_intersection("X", -253.3,  159.4)

        # connections (bidirectional default)
        edges = [
            ("A","B"),("A","G"),("B","C"),("B","H"),("C","D"),("C","I"),("D","E"),("D","J"),
            ("E","F"),("E","K"),("F","L"),("G","H"),("G","M"),("H","I"),("H","N"),("I","J"),
            ("I","O"),("J","K"),("J","P"),("K","L"),("K","Q"),("L","R"),("M","N"),("M","S"),
            ("N","O"),("N","T"),("O","P"),("O","U"),("P","Q"),("P","V"),("Q","R"),("Q","W"),
            ("R","X"),("S","T"),("T","U"),("U","V"),("V","W"),("W","X")
        ]
        for a,b in edges:
            self.add_connection(a,b)

class CollisionState:
    def __init__(self):
        self.collision = False     

    def collision_callback(self, value):
        projectairsim_log().info("Exit because of collision.")
        self.collision = value

class Inspection:
    subwin_width, subwin_height = 640, 360
    options={
        "display": True, 
        "save": False
        }    
    
    def __init__(self):
        # init simulation env and drone
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, "scene_drone_classic.jsonc", delay_after_load_sec=2)
        self.drone = Drone(self.client, self.world, "Drone1")
        self.drone.enable_api_control()
        self.drone.arm()

        # subscribe image and collision topic
        self.image_display = ImageDisplay(
            num_subwin=3,
            screen_res_x=2560,
            screen_res_y=1440,
            subwin_width=self.subwin_width,
            subwin_height=self.subwin_height
            )
        self.chase_cam_window = "ChaseCam"
        self.image_display.add_image(self.chase_cam_window, subwin_idx=0, resize_x=self.subwin_width, resize_y=self.subwin_height) 
        self.client.subscribe(
            self.drone.sensors["Chase"]["scene_camera"],
            lambda topic, msg: self.save_and_display_image(msg, self.chase_cam_window),
        )
        self.front_cam_window = "FrontCam"
        self.image_display.add_image(self.front_cam_window,  subwin_idx=1, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.client.subscribe(
            self.drone.sensors["front_center"]["scene_camera"],
            lambda topic, msg: self.save_and_display_image(msg, self.front_cam_window),
        )
        self.front_depth_name = "FrontDepthImage"
        self.image_display.add_image(self.front_depth_name, subwin_idx=2, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.client.subscribe(
            self.drone.sensors["front_center"]["depth_planar_camera"],
            lambda topic, msg: self.save_and_display_image(msg, self.front_depth_name),
        )
        self.image_display.start()

        self._collison = CollisionState()
        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            lambda topic, msg: self._collison.collision_callback(True),
        )

        # save video writer
        self.video_writers = {} 

        # help determine whether the destination has beed reached
        self.target_distance_threshold = 5.0

        # help determine whether the drone needs to change direction
        self.change_direction_distance_threshold = 5.0

        # create road network
        self.roadnetwork = RoadNetwork()
        self.roadnetwork.create_road_network()

        # other states
        self.update_cur_position()
        self.from_position = self.cur_position
        self.target_id = self.roadnetwork.random_neighbor("J")
        self.to_position = self.roadnetwork.coords(self.target_id)

        self.prev_v_north = 0.0
        self.prev_v_east = 0.0
        self.smoothing_factor = 0.3

    async def step(self):
        v_north, v_east, z = self._action()
        move_task = await self.drone.move_by_velocity_z_async(v_north, v_east, z , 1, yaw_control_mode=YawControlMode.ForwardOnly, yaw_is_rate=False, yaw=0)
        await move_task
        self.update_cur_position()

    def _action(self):
        move_velocity = 1.0

        result = self.drone.get_images("front_center", [ImageType.DEPTH_PLANAR])
        depth_image = unpack_image(result[ImageType.DEPTH_PLANAR])

        middle = np.vsplit(depth_image, 3)[1]
        bands = np.hsplit(middle, [200,400,600,800,1000,1200])
        front_min = bands[len(bands) // 2].min() 
        
        if front_min < self.change_direction_distance_threshold:
            projectairsim_log().info("Has obstacle.")
            band_depths = [band.min() for band in bands]
            
            center_index = len(bands) // 2
            left_index = center_index - 1
            right_index = center_index + 1
            
            found_safe_direction = None
            while left_index >= 0 or right_index < len(bands):
                if right_index < len(bands) and band_depths[right_index] >= self.change_direction_distance_threshold:
                    found_safe_direction = "right"
                    safe_band_index = right_index
                    break
                
                if left_index >= 0 and band_depths[left_index] >= self.change_direction_distance_threshold:
                    found_safe_direction = "left"
                    safe_band_index = left_index
                    break
                
                right_index += 1
                left_index -= 1
            
            if found_safe_direction:
                projectairsim_log().info(f"Safe direction found: {found_safe_direction}, band index: {safe_band_index}")
                total_bands = len(bands)
                angle_per_band = (np.pi / 2) / total_bands
                angle_offset = (safe_band_index - center_index) * angle_per_band
                
                yaw = quaternion_to_rpy(self.drone.get_ground_truth_kinematics()['pose']['orientation'])[-1]
                abs_yaw = yaw + angle_offset
                v_north = move_velocity * math.cos(abs_yaw)
                v_east = move_velocity * math.sin(abs_yaw)              
            else:
                projectairsim_log().warning("No safe direction found! Executing backup strategy.")
                safest_band_index = np.argmax(band_depths)
                projectairsim_log().info(f"Choosing safest band: {safest_band_index} with depth: {band_depths[safest_band_index]:.2f}")
                total_bands = len(bands)
                angle_per_band = (np.pi / 2) / total_bands
                angle_offset = (safest_band_index - center_index) * angle_per_band
                
                yaw = quaternion_to_rpy(self.drone.get_ground_truth_kinematics()['pose']['orientation'])[-1]
                abs_yaw = yaw + angle_offset
                v_north = move_velocity * math.cos(abs_yaw)  
                v_east = move_velocity * math.sin(abs_yaw)

        else:
            projectairsim_log().info("Without obstacle.")
            cur_x, cur_y, cur_z = self.cur_position
            to_x, to_y, to_z = self.to_position
            dx = to_x - cur_x
            dy = to_y - cur_y
            distance = math.sqrt(dx**2 + dy**2)
            unit_dx = dx / distance
            unit_dy = dy / distance
                 
            v_north = unit_dx * move_velocity
            v_east = unit_dy * move_velocity
        
        # smooth the velocity
        v_north = self.prev_v_north * (1 - self.smoothing_factor) + v_north * self.smoothing_factor
        v_east = self.prev_v_east * (1 - self.smoothing_factor) + v_east * self.smoothing_factor
        self.prev_v_north = v_north
        self.prev_v_east = v_east

        return (v_north, v_east, to_z)

    def update_cur_position(self):
        kinematics = self.drone.get_ground_truth_kinematics()
        cur_position_dict = kinematics['pose']['position']
        self.cur_position = (cur_position_dict['x'], cur_position_dict['y'], cur_position_dict['z'])
        
    def update_from_and_to_position(self):
        self.from_position = self.cur_position
        self.target_id = self.roadnetwork.random_neighbor(self.target_id)
        self.to_position = self.roadnetwork.coords(self.target_id)
        
    def has_arrived(self):
        kinematics = self.drone.get_ground_truth_kinematics()
        dist_to_target = np.linalg.norm(
            np.array([kinematics['pose']['position']['x'], kinematics['pose']['position']['y'], kinematics['pose']['position']['z']]) - 
            np.array([self.to_position[0], self.to_position[1], self.to_position[2]])
        )
        return dist_to_target < self.target_distance_threshold

    def save_and_display_image(self, image, topic):
        if self.options["display"]:
            self.image_display.receive(image, topic)
                
        if self.options["save"]:
            np_image = unpack_image(image)
            if topic not in self.video_writers:
                self.create_video_writer_for_topic(np_image, topic)

            writer = self.video_writers.get(topic)
                
            if len(np_image.shape) == 2:
                depth_map = np.array(np_image)
                depth_map = np.nan_to_num(depth_map)
        
                if depth_map.dtype != np.uint8:
                    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                    depth_uint8 = depth_normalized.astype(np.uint8)
                else:
                    depth_uint8 = depth_map

                frame_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
            elif len(np_image.shape) == 3 and np_image.shape[2] == 3:
                frame_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                    
            writer.write(frame_bgr)

    def create_video_writer_for_topic(self, np_image, topic):
        height, width = np_image.shape[:2]
        os.makedirs(SAVE_PATH, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in str(topic) if c.isalnum() or c in ('_', '-')).rstrip()
        video_path = os.path.join(SAVE_PATH, f"{safe_topic}_{timestamp}.mp4")
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        self.video_writers[topic] = writer
            
    def close_videos(self):
        for topic, writer in self.video_writers.items():
            if writer is not None:
                writer.release()
        self.video_writers.clear()    

    def close_projectairsim(self):
        self.drone.disarm()
        self.drone.disable_api_control()

        self.image_display.stop()
        self.client.disconnect()
        
    def close_all(self):
        self.close_videos()
        self.close_projectairsim()

    async def run(self):
        try:
            takeoff_task = await self.drone.takeoff_async()
            await takeoff_task
            self.update_cur_position()

            while self._collison.collision == False:
                await self.step()
                if self.has_arrived():
                    self.update_from_and_to_position()
                    projectairsim_log().info(f"New target intersection ID: {self.target_id}, coordinates: {self.to_position}")
        finally:
            land_task = await self.drone.land_async()
            await land_task
            self.close_all()

if __name__ == "__main__":
    inspection_tast = Inspection()
    asyncio.run(inspection_tast.run())   
