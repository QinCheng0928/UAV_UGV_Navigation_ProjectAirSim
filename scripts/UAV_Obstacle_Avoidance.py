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


def display_and_save_callback(image_display, msg, chase_cam_window, writer, writer_size):
    image_display.receive(msg, chase_cam_window)


    img_np = unpack_image(msg)

    if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
        if img_np.dtype != np.uint8:
            img_np = np.nan_to_num(img_np)
            max_val = np.max(img_np) if np.max(img_np) > 0 else 1
            img_np = (img_np / max_val * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    if (img_np.shape[1], img_np.shape[0]) != writer_size:
        img_np = cv2.resize(img_np, writer_size)

    writer.write(img_np)


async def main():
    client = ProjectAirSimClient()

    subwin_width, subwin_height = 640, 360
    image_display = ImageDisplay(
        num_subwin=3,
        screen_res_x=2560,
        screen_res_y=1440,
        subwin_width=subwin_width,
        subwin_height=subwin_height
    )
    
    try:
        client.connect()
        world = World(client, "scene_drone_classic.jsonc", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        writer_size = (subwin_width, subwin_height)
        os.makedirs(os.path.join(ROOT_DIR, "media"), exist_ok=True)

        chase_cam_window = "ChaseCam"
        chase_cam_writer = cv2.VideoWriter(os.path.join(ROOT_DIR, "media", f"{chase_cam_window}.mp4"), fourcc, fps, writer_size)
        image_display.add_image(chase_cam_window, subwin_idx=0, resize_x=subwin_width, resize_y=subwin_height) 
        client.subscribe(
            drone.sensors["Chase"]["scene_camera"],
            lambda _, msg: display_and_save_callback(image_display, msg, chase_cam_window, chase_cam_writer, writer_size),
        )

        front_cam_window = "FrontCam"
        front_cam_writer = cv2.VideoWriter(os.path.join(ROOT_DIR, "media", f"{front_cam_window}.mp4"), fourcc, fps, writer_size)
        image_display.add_image(front_cam_window,  subwin_idx=1, resize_x=subwin_width, resize_y=subwin_height)
        client.subscribe(
            drone.sensors["FrontCamera"]["scene_camera"],
            lambda _, msg: display_and_save_callback(image_display, msg, front_cam_window, front_cam_writer, writer_size),
        )

        depth_name = "DepthImage"
        depth_writer = cv2.VideoWriter(os.path.join(ROOT_DIR, "media", f"{depth_name}.mp4"), fourcc, fps, writer_size)
        image_display.add_image(depth_name, subwin_idx=2, resize_x=subwin_width, resize_y=subwin_height)
        client.subscribe(
            drone.sensors["front_center"]["depth_planar_camera"],
            lambda _, msg: display_and_save_callback(image_display, msg, depth_name, depth_writer, writer_size),
        )

        image_display.start()

        drone.enable_api_control()
        drone.arm()

        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        yaw = 0
        pi = 3.14159265483
        vx = 0
        vy = 0

        num_segments = 8
        segment_angle = pi / (2 * num_segments)

        while True:
            # this will return png width= 256, height= 144
            result = drone.get_images("front_center", [ImageType.DEPTH_PLANAR])
            depth_image = unpack_image(result[ImageType.DEPTH_PLANAR])

            # slice the image so we only check what we are headed into (and not what is down on the ground below us).

            middle = np.vsplit(depth_image, 3)[1]

            bands = np.hsplit(middle, num_segments)
            mins = [np.min(x) for x in bands]
            projectairsim_log().info(mins)
            max = np.argmax(mins)    
            distance = mins[max]
        
            orientation = drone.get_ground_truth_kinematics()["pose"]["orientation"]
            pitch, roll, yaw = quaternion_to_rpy(orientation["w"], orientation["x"], orientation["y"], orientation["z"])
            
            # we have a 90 degree field of view (pi/2), we've sliced that into 5 chunks, each chunk then represents
            # an angular delta of the following pi/10.
            change = (max - (num_segments // 2)) * segment_angle
        
            yaw = (yaw + change)
            vx = math.cos(yaw)
            vy = math.sin(yaw)
            projectairsim_log().info("switching angle yaw=%f vx=%f vy=%f max=%f distance=%d", math.degrees(yaw), vx, vy, max, distance)
        
            if (vx == 0 and vy == 0):
                vx = math.cos(yaw)
                vy = math.sin(yaw)

            move_task = await drone.move_by_velocity_z_async(vx, vy,-6, 1, YawControlMode.ForwardOnly, yaw=0, yaw_is_rate=False)
            await move_task

    finally:
        image_display.stop()
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
