import numpy as np
import asyncio
import random
import cv2
import math
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

class State:
    def __init__(self):
        self.collision = False

    def set_collision(self, value):
        self.collision = value

    def collision_callback(self, value):
        projectairsim_log().info("Exit because of collision.")
        self.set_collision(value)

z = -6.0

def create_road_network(roadnetwork):
    # set the intersections and their coordinates
    roadnetwork.add_intersection("A",  129.2, -225.8, z)
    roadnetwork.add_intersection("B",  129.2, -160.0, z)
    roadnetwork.add_intersection("C",  129.3, -89.8 , z)
    roadnetwork.add_intersection("D",  128.2, -9.6  , z)
    roadnetwork.add_intersection("E",  133.1,  72.2 , z)
    roadnetwork.add_intersection("F",  139.6,  160.6, z)
    roadnetwork.add_intersection("G",  3.6  , -229.5, z)
    roadnetwork.add_intersection("H",  3.6  , -157.9, z)
    roadnetwork.add_intersection("I",  4.8  , -92.4 , z)
    roadnetwork.add_intersection("J",  6.0  , -9.6  , z)
    roadnetwork.add_intersection("K",  5.0  ,  72.2 , z)
    roadnetwork.add_intersection("L",  0.4  ,  160.6, z)
    roadnetwork.add_intersection("M", -125.0, -229.5, z)
    roadnetwork.add_intersection("N", -125.4, -156.1, z)
    roadnetwork.add_intersection("O", -125.4, -88.3 , z)
    roadnetwork.add_intersection("P", -125.4, -7.6  , z)
    roadnetwork.add_intersection("Q", -123.4,  72.2 , z)
    roadnetwork.add_intersection("R", -130.8,  160.6, z)
    roadnetwork.add_intersection("S", -252.7, -229.5, z)
    roadnetwork.add_intersection("T", -253.3, -162.3, z)
    roadnetwork.add_intersection("U", -253.3, -93.2 , z)
    roadnetwork.add_intersection("V", -253.3, -11.4 , z)
    roadnetwork.add_intersection("W", -253.3,  72.6 , z)
    roadnetwork.add_intersection("X", -253.3,  159.4, z)

    # add connections between intersections
    roadnetwork.add_connection("A", "B")
    roadnetwork.add_connection("A", "G")
    roadnetwork.add_connection("B", "A")
    roadnetwork.add_connection("B", "C")
    roadnetwork.add_connection("B", "H")
    roadnetwork.add_connection("C", "B")
    roadnetwork.add_connection("C", "D")
    roadnetwork.add_connection("C", "I")
    roadnetwork.add_connection("D", "C")
    roadnetwork.add_connection("D", "E")
    roadnetwork.add_connection("D", "J")
    roadnetwork.add_connection("E", "D")
    roadnetwork.add_connection("E", "F")
    roadnetwork.add_connection("E", "K")
    roadnetwork.add_connection("F", "E")
    roadnetwork.add_connection("F", "L")

    roadnetwork.add_connection("G", "A")
    roadnetwork.add_connection("G", "H")
    roadnetwork.add_connection("G", "M")
    roadnetwork.add_connection("H", "B")
    roadnetwork.add_connection("H", "G")
    roadnetwork.add_connection("H", "I")
    roadnetwork.add_connection("H", "N")
    roadnetwork.add_connection("I", "C")
    roadnetwork.add_connection("I", "H")
    roadnetwork.add_connection("I", "J")
    roadnetwork.add_connection("I", "O")
    roadnetwork.add_connection("J", "D")
    roadnetwork.add_connection("J", "I")
    roadnetwork.add_connection("J", "K")
    roadnetwork.add_connection("J", "P")
    roadnetwork.add_connection("K", "E")
    roadnetwork.add_connection("K", "J")
    roadnetwork.add_connection("K", "L")
    roadnetwork.add_connection("K", "Q")
    roadnetwork.add_connection("L", "F")
    roadnetwork.add_connection("L", "K")
    roadnetwork.add_connection("L", "R")

    roadnetwork.add_connection("M", "G")
    roadnetwork.add_connection("M", "N")
    roadnetwork.add_connection("M", "S")
    roadnetwork.add_connection("N", "H")
    roadnetwork.add_connection("N", "M")
    roadnetwork.add_connection("N", "O")
    roadnetwork.add_connection("N", "T")
    roadnetwork.add_connection("O", "I")
    roadnetwork.add_connection("O", "N")
    roadnetwork.add_connection("O", "P")
    roadnetwork.add_connection("O", "U")
    roadnetwork.add_connection("P", "J")
    roadnetwork.add_connection("P", "O")
    roadnetwork.add_connection("P", "Q")
    roadnetwork.add_connection("P", "V")
    roadnetwork.add_connection("Q", "K")
    roadnetwork.add_connection("Q", "P")
    roadnetwork.add_connection("Q", "R")
    roadnetwork.add_connection("Q", "W")
    roadnetwork.add_connection("R", "L")
    roadnetwork.add_connection("R", "Q")
    roadnetwork.add_connection("R", "X")

    roadnetwork.add_connection("S", "M")
    roadnetwork.add_connection("S", "T")
    roadnetwork.add_connection("T", "N")
    roadnetwork.add_connection("T", "S")
    roadnetwork.add_connection("T", "U")
    roadnetwork.add_connection("U", "O")
    roadnetwork.add_connection("U", "T")
    roadnetwork.add_connection("U", "V")
    roadnetwork.add_connection("V", "P")
    roadnetwork.add_connection("V", "U")
    roadnetwork.add_connection("V", "W")
    roadnetwork.add_connection("W", "Q")
    roadnetwork.add_connection("W", "V")
    roadnetwork.add_connection("W", "X")
    roadnetwork.add_connection("X", "R")
    roadnetwork.add_connection("X", "W")

def check_keyboard():
    key = cv2.waitKey(1) & 0xFF
    return key

async def main():
    # create road network
    roadnetwork = RoadNetwork()
    create_road_network(roadnetwork)

    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, "scene_drone_classic.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        drone.enable_api_control()
        drone.arm()

        cv2.namedWindow("Drone Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone Control", 300, 100)

        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        client.subscribe(
            drone.robot_info["collision_info"],
            lambda topic, msg: state.collision_callback(True),
        )

        target_id = roadnetwork.get_random_reachable_intersection_id("J")
        to_position = roadnetwork.get_coordinates(target_id)

        state = State()
        target_distance_threshold = 5.0
        while not state.collision:
            # exit
            key = check_keyboard()
            if key == ord('q') or key == ord('Q'):
                projectairsim_log().info("Exit because of user input.")
                break

            # step
            kinematics = drone.get_ground_truth_kinematics()
            cur_position = kinematics['pose']['position']

            cur_x, cur_y, cur_z = cur_position["x"], cur_position["y"], cur_position["z"]
            to_x, to_y, to_z = to_position
            dx = to_x - cur_x
            dy = to_y - cur_y
            distance = math.sqrt(dx**2 + dy**2)
            unit_dx = dx / distance
            unit_dy = dy / distance
            
            move_distance = 5.0
            north = cur_x + unit_dx * move_distance
            east = cur_y + unit_dy * move_distance

            move_task = await drone.move_to_position_async(north, east, z, 1.0,  yaw_control_mode=YawControlMode.ForwardOnly, yaw_is_rate=False, yaw=0)
            await move_task
            
            # has reached target
            # True: move towards others target position
            kinematics = drone.get_ground_truth_kinematics()
            cur_position = kinematics['pose']['position']
            dist_to_target = np.linalg.norm(
                np.array([cur_position['x'], cur_position['y'], cur_position['z']]) - 
                np.array([to_position[0], to_position[1], to_position[2]])
            )
            if dist_to_target < target_distance_threshold:
                target_id = roadnetwork.get_random_reachable_intersection_id(target_id)
                to_position = roadnetwork.get_coordinates(target_id)

        land_task = await drone.land_async()
        await land_task
        drone.disarm()
        drone.disable_api_control()

    finally:
        cv2.destroyAllWindows()
        client.disconnect()


    
if __name__ == "__main__":
    asyncio.run(main())
    

