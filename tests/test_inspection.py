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

class RoadNetwork:
    z = -6

    def __init__(self):
        self.intersections = {}
    
    def add_intersection(self, id, x, y, z):
        self.intersections[id] = Intersection(id, x, y, z)
    
    def add_connection(self, from_id, to_id, bidirectional=True):
        if from_id in self.intersections and to_id in self.intersections:
            self.intersections[from_id].add_connection(to_id)
            if bidirectional:
                self.intersections[to_id].add_connection(from_id)
    
    def get_reachable_intersections_id(self, from_id):
        return self.intersections[from_id].connections

    def get_random_reachable_intersection_id(self, from_id):
        reachable = self.intersections[from_id].connections
        if not reachable:
            return None
        return random.choice(reachable)

    def get_coordinates(self, id):
        if id in self.intersections:
            return self.intersections[id].coords
        return None

    def create_road_network(self):
        # set the intersections and their coordinates
        self.add_intersection("A",  129.2, -225.8, self.z)
        self.add_intersection("B",  129.2, -160.0, self.z)
        self.add_intersection("C",  129.3, -89.8 , self.z)
        self.add_intersection("D",  128.2, -9.6  , self.z)
        self.add_intersection("E",  133.1,  72.2 , self.z)
        self.add_intersection("F",  139.6,  160.6, self.z)
        self.add_intersection("G",  3.6  , -229.5, self.z)
        self.add_intersection("H",  3.6  , -157.9, self.z)
        self.add_intersection("I",  4.8  , -92.4 , self.z)
        self.add_intersection("J",  6.0  , -9.6  , self.z)
        self.add_intersection("K",  5.0  ,  72.2 , self.z)
        self.add_intersection("L",  0.4  ,  160.6, self.z)
        self.add_intersection("M", -125.0, -229.5, self.z)
        self.add_intersection("N", -125.4, -156.1, self.z)
        self.add_intersection("O", -125.4, -88.3 , self.z)
        self.add_intersection("P", -125.4, -7.6  , self.z)
        self.add_intersection("Q", -123.4,  72.2 , self.z)
        self.add_intersection("R", -130.8,  160.6, self.z)
        self.add_intersection("S", -252.7, -229.5, self.z)
        self.add_intersection("T", -253.3, -162.3, self.z)
        self.add_intersection("U", -253.3, -93.2 , self.z)
        self.add_intersection("V", -253.3, -11.4 , self.z)
        self.add_intersection("W", -253.3,  72.6 , self.z)
        self.add_intersection("X", -253.3,  159.4, self.z)

        # add connections between intersections
        self.add_connection("A", "B")
        self.add_connection("A", "G")
        self.add_connection("B", "A")
        self.add_connection("B", "C")
        self.add_connection("B", "H")
        self.add_connection("C", "B")
        self.add_connection("C", "D")
        self.add_connection("C", "I")
        self.add_connection("D", "C")
        self.add_connection("D", "E")
        self.add_connection("D", "J")
        self.add_connection("E", "D")
        self.add_connection("E", "F")
        self.add_connection("E", "K")
        self.add_connection("F", "E")
        self.add_connection("F", "L")

        self.add_connection("G", "A")
        self.add_connection("G", "H")
        self.add_connection("G", "M")
        self.add_connection("H", "B")
        self.add_connection("H", "G")
        self.add_connection("H", "I")
        self.add_connection("H", "N")
        self.add_connection("I", "C")
        self.add_connection("I", "H")
        self.add_connection("I", "J")
        self.add_connection("I", "O")
        self.add_connection("J", "D")
        self.add_connection("J", "I")
        self.add_connection("J", "K")
        self.add_connection("J", "P")
        self.add_connection("K", "E")
        self.add_connection("K", "J")
        self.add_connection("K", "L")
        self.add_connection("K", "Q")
        self.add_connection("L", "F")
        self.add_connection("L", "K")
        self.add_connection("L", "R")

        self.add_connection("M", "G")
        self.add_connection("M", "N")
        self.add_connection("M", "S")
        self.add_connection("N", "H")
        self.add_connection("N", "M")
        self.add_connection("N", "O")
        self.add_connection("N", "T")
        self.add_connection("O", "I")
        self.add_connection("O", "N")
        self.add_connection("O", "P")
        self.add_connection("O", "U")
        self.add_connection("P", "J")
        self.add_connection("P", "O")
        self.add_connection("P", "Q")
        self.add_connection("P", "V")
        self.add_connection("Q", "K")
        self.add_connection("Q", "P")
        self.add_connection("Q", "R")
        self.add_connection("Q", "W")
        self.add_connection("R", "L")
        self.add_connection("R", "Q")
        self.add_connection("R", "X")

        self.add_connection("S", "M")
        self.add_connection("S", "T")
        self.add_connection("T", "N")
        self.add_connection("T", "S")
        self.add_connection("T", "U")
        self.add_connection("U", "O")
        self.add_connection("U", "T")
        self.add_connection("U", "V")
        self.add_connection("V", "P")
        self.add_connection("V", "U")
        self.add_connection("V", "W")
        self.add_connection("W", "Q")
        self.add_connection("W", "V")
        self.add_connection("W", "X")
        self.add_connection("X", "R")
        self.add_connection("X", "W")

class CollisonState:
    def __init__(self):
        self.collision = False

    def set_collision(self, value):
        self.collision = value

    def collision_callback(self, value):
        projectairsim_log().info("Exit because of collision.")
        self.set_collision(value)

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

        self._collison = CollisonState()
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
        self.target_id = self.roadnetwork.get_random_reachable_intersection_id("J")
        self.to_position = self.roadnetwork.get_coordinates(self.target_id)

        self.prev_v_north = 0.0
        self.prev_v_east = 0.0
        self.smoothing_factor = 0.3

    async def step(self):
        v_north, v_east = self._action()
        move_task = await self.drone.move_by_velocity_z_async(v_north, v_east, self.roadnetwork.z , 0.1, yaw_control_mode=YawControlMode.ForwardOnly, yaw_is_rate=False, yaw=0)
        await move_task
        self.update_cur_position()

    def _action(self):
        if self.has_obstacles_ahead():
            pass
        else:
            cur_x, cur_y, cur_z = self.cur_position
            to_x, to_y, to_z = self.to_position
            dx = to_x - cur_x
            dy = to_y - cur_y
            distance = math.sqrt(dx**2 + dy**2)
            unit_dx = dx / distance
            unit_dy = dy / distance
                
            move_velocity = 1.0
            v_north = unit_dx * move_velocity
            v_east = unit_dy * move_velocity
        
        # smooth the velocity
        v_north = self.prev_v_north * (1 - self.smoothing_factor) + v_north * self.smoothing_factor
        v_east = self.prev_v_east * (1 - self.smoothing_factor) + v_east * self.smoothing_factor
        self.prev_v_north = v_north
        self.prev_v_east = v_east

        return (v_north, v_east)

    async def take_off(self):
        takeoff_task = await self.drone.takeoff_async()
        await takeoff_task
        self.update_cur_position()
        projectairsim_log().info("Take off completed.")

    async def land(self):
        land_task = await self.drone.land_async()
        await land_task
        projectairsim_log().info("Land completed.")

    def update_cur_position(self):
        kinematics = self.drone.get_ground_truth_kinematics()
        cur_position_dict = kinematics['pose']['position']
        self.cur_position = (cur_position_dict['x'], cur_position_dict['y'], cur_position_dict['z'])
        
    def update_from_and_to_position(self):
        self.from_position = self.cur_position
        self.target_id = self.roadnetwork.get_random_reachable_intersection_id(self.target_id)
        self.to_position = self.roadnetwork.get_coordinates(self.target_id)

    def has_obstacles_ahead(self):
        # this will return png width= 1280, height= 720
        result = self.drone.get_images("front_center", [ImageType.DEPTH_PLANAR])
        depth_image = unpack_image(result[ImageType.DEPTH_PLANAR])

        middle = np.vsplit(depth_image, 3)[1]
        bands = np.hsplit(middle, [100,200,300,400,500,600,700,800,900,1000,1100,1200])
        assert len(bands) == 13
        return bands[len(bands) // 2].min() < self.change_direction_distance_threshold
        
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
            await self.take_off()

            while self._collison.collision == False:
                await self.step()
                if self.has_arrived():
                    self.update_from_and_to_position()
                    projectairsim_log().info(f"New target intersection ID: {self.target_id}, coordinates: {self.to_position}")
        finally:
            await self.land()
            self.close_all()

if __name__ == "__main__":
    inspection_tast = Inspection()
    asyncio.run(inspection_tast.run())   

