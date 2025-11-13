import time
import math
import sys
import numpy as np
import asyncio
import cv2
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.utils import projectairsim_log, quaternion_to_rpy, unpack_image
from projectairsim.drone import YawControlMode
from projectairsim.types import ImageType
from projectairsim.image_utils import ImageDisplay

def check_keyboard():
    key = cv2.waitKey(1) & 0xFF
    return key

async def main():
    client = ProjectAirSimClient()
    try:
        client.connect()
        world = World(client, "scene_drone_classic.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        drone.enable_api_control()
        drone.arm()

        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        velocity = 3.0 
        vx, vy = 0, 0
        
        cv2.namedWindow("Drone Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drone Control", 300, 100)
        
        flying = True
        while flying:
            key = check_keyboard()
            
            if key == ord('w') or key == ord('W'):
                vx, vy = velocity, 0 
            elif key == ord('s') or key == ord('S'):
                vx, vy = -velocity, 0  
            elif key == ord('a') or key == ord('A'):
                vx, vy = 0, -velocity  
            elif key == ord('d') or key == ord('D'):
                vx, vy = 0, velocity   
            elif key == ord('q') or key == ord('Q'):
                flying = False
                vx, vy = 0, 0 
            elif key == ord('b') or key == ord('B'):
                kinematics = drone.get_ground_truth_kinematics()
                print(f"Current kinematics: {kinematics}")
            else:
                vx, vy = 0, 0
            
            await drone.move_by_velocity_z_async(
                vx, vy, -6, 0.1, 
                YawControlMode.ForwardOnly, yaw=0, yaw_is_rate=False
            )
            
            await asyncio.sleep(0.05)

        land_task = await drone.land_async()
        await land_task
        drone.disarm()
        drone.disable_api_control()
        cv2.destroyAllWindows()

    except Exception as err:
        cv2.destroyAllWindows()

    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())