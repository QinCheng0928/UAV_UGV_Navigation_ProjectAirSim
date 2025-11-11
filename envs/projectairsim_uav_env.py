import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import asyncio
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import ImageDisplay, unpack_image
from projectairsim.types import ImageType
from projectairsim.drone import YawControlMode
from projectairsim.utils import (
    quaternion_to_rpy, 
    projectairsim_log,
    )
from envs.utils.type import State

class ProjectAirSimSmallCityEnv(gym.Env):
    subwin_width, subwin_height = 320, 180

    def __init__(self):
        super().__init__()
        self.sim_config_filename = "scene_drone_classic.jsonc"

        self.dt = 0.5
        self.maxv = 5
        self.max_episode_steps = 500
        self.target_point = self.random_target_point()
        self.goal_distance_threshold = 10.0

        random.seed(42)
        np.random.seed(42)

        self.loop = asyncio.get_event_loop()

        self.client = ProjectAirSimClient()
        self.client.connect()

        self.image_display = ImageDisplay(
            num_subwin=7,
            screen_res_x=2560,
            screen_res_y=1440,
            subwin_width=self.subwin_width,
            subwin_height=self.subwin_height
        )
        self.image_display.start()

        self.chase_cam_window = "ChaseCam"
        self.image_display.add_image(self.chase_cam_window, subwin_idx=0, resize_x=self.subwin_width, resize_y=self.subwin_height) 
        self.front_cam_window = "FrontCam"
        self.image_display.add_image(self.front_cam_window,  subwin_idx=1, resize_x=self.subwin_width, resize_y=self.subwin_height)

        self.front_depth_name = "FrontDepthImage"
        self.image_display.add_image(self.front_depth_name, subwin_idx=2, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.front_right_depth_image = "FrontRightDepthImage"
        self.image_display.add_image(self.front_right_depth_image, subwin_idx=3, resize_x=self.subwin_width, resize_y=self.subwin_height) 
        self.front_left_depth_image = "FrontLeftDepthImage"
        self.image_display.add_image(self.front_left_depth_image,  subwin_idx=4, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.bottom_depth_image = "BottomDepthImage"
        self.image_display.add_image(self.bottom_depth_image, subwin_idx=5, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.back_depth_image = "BackDepthImage"
        self.image_display.add_image(self.back_depth_image, subwin_idx=6, resize_x=self.subwin_width, resize_y=self.subwin_height)

        # 0 = vx
        # 1 = vy
        # 2 = vz
        # 3 = dis_x
        # 4 = dis_y 
        # 5 = dis_z
        # 6 = relative_yaw       (relative_yaw = yaw − goal_yaw)
        # 7 = angular_velocity
        # 8 = collision
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False]
        assert len(self.state) == 9

        # roll_rate, pitch_rate, yaw_rate
        max_rate = 1.0  # rad/s, 可根据需要调大或调小
        self.action_space = spaces.Box(low=np.array([-max_rate, -max_rate, -max_rate], dtype=np.float32),
                                    high=np.array([ max_rate,  max_rate,  max_rate], dtype=np.float32),
                                    dtype=np.float32)

        # We will first divide the image vertically into three parts, top, bottom, and middle, and take out every part
        # Divide the image horizontally into eight parts and take the maximum value of each part as the feature vector of the observation space 
        # The shape is (1, 24)
        # We have four perspectives: front, rear, bottom, left, and right.
        # The shape is (1, 24 * 5)
        # 
        # We also choose velocity in 3D space (vx vy, vz), distance from the endpoint (dis_x, dis_y, dis_z), 
        # relative heading angle, and angular velocity as features of the state
        # The shape is (1, 8)
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(24 * 5 + 8,), dtype=np.float32)
        

    """
        The reset() method must return a tuple (obs, info) 
    """
    def reset(self, *, seed=None, options=None):
        # reset the sim world and drone
        self.world = World(self.client, self.sim_config_filename)
        self.drone = Drone(self.client, self.world, "Drone1")     

        self.client.unsubscribe_all()
        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            self._collision_callback,
        )

        # Init the image display windows
        self.client.subscribe(
            self.drone.sensors["Chase"]["scene_camera"],
            lambda _, msg: self.image_display.receive(msg, self.chase_cam_window),
        )
        self.client.subscribe(
            self.drone.sensors["front_center"]["scene_camera"],
            lambda _, msg: self.image_display.receive(msg, self.front_cam_window),
        )
        self.client.subscribe(
            self.drone.sensors["front_center"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, self.front_depth_name),
        )
        self.client.subscribe(
            self.drone.sensors["front_right"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, self.front_right_depth_image),
        )
        self.client.subscribe(
            self.drone.sensors["front_left"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, self.front_left_depth_image),
        )
        self.client.subscribe(
            self.drone.sensors["bottom_center"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, self.bottom_depth_image),
        )
        self.client.subscribe(
            self.drone.sensors["back_center"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, self.back_depth_image),
        )

        # reset state variables
        self.current_step = 0 
        self.target_point = self.random_target_point()

        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False]
        self.update_state()
        self.distance_x = self.state[State.dis_x]
        self.distance_y = self.state[State.dis_y]
        self.distance_z = self.state[State.dis_z]
        self.total_dis = math.sqrt(self.distance_x **2 + self.distance_y **2 + self.distance_z **2)
        self.previous_distance = self.total_dis
        self.cur_distance = self.total_dis

        self.drone.enable_api_control()
        self.drone.arm()

        obs = self._get_obs()
        info = {}
        return (obs, info)

    """
        The step() method must return five values: (obs, reward, terminated, truncated, info).  
    """
    def step(self, action):
        self.current_step += 1
        
        self.loop.run_until_complete(self._simulate(action))
        self.previous_distance = self.cur_distance
        self.cur_distance = math.sqrt(self.state[State.dis_x] **2 + self.state[State.dis_y] **2 + self.state[State.dis_z] **2)
        
        obs = self._get_obs()
        reward = self._rewards()
        done = self._is_terminal()
        truncated = self._is_truncated()
        info = {}
        projectairsim_log().info(f"obs: {obs}")
        projectairsim_log().info(f"reward: {reward}, done: {done}, truncated: {truncated}, info: {info}")
        return (obs, reward, done, truncated, info)
    
    async def _simulate(self, action):
        projectairsim_log().info(f"Action taken: {action}")

        roll = action[0]
        pitch = action[1]
        yaw = action[2]

        vx = self.maxv * math.cos(yaw) * math.cos(pitch)
        vy = self.maxv * math.sin(yaw) * math.cos(pitch)
        vz = -self.maxv * math.sin(pitch)

        vx = np.clip(vx, -self.maxv, self.maxv)
        vy = np.clip(vy, -self.maxv, self.maxv)
        vz = np.clip(vz, -self.maxv, self.maxv)

        if self._has_arrived():
            vx = 0.0
            vy = 0.0
            vz = 0.0
        
        projectairsim_log().info(f"Velocity command: vx={vx}, vy={vy}, vz={vz}")
        
        # send velocity command to the drone
        move_task = await self.drone.move_by_velocity_async(v_north=vx, v_east=vy, v_down=vz, duration=0.5, yaw_control_mode = YawControlMode.ForwardOnly, yaw=0, yaw_is_rate=False)
        await move_task     

        self.update_state()


    def _get_obs(self):
        # Remove collision tag and Normalizing [low=0, high=255]
        state_feature = self.normalize_state()

        # Obtain and ormalizing depth image
        camera_names = ["front_center", "front_right", "front_left", "bottom_center", "back_center"]
        result_list = [self.drone.get_images(camera_name, [ImageType.DEPTH_PLANAR]) for camera_name in camera_names]
        depth_image_list = [unpack_image(result[ImageType.DEPTH_PLANAR]) for result in result_list]
        image_feature_list = []
        for depth_image in depth_image_list:
            if depth_image.ndim == 3:
                depth_image = depth_image[:, :, 0]
            image_feature_list.append(self.normalize_image(depth_image))

        image_feature = np.hstack(image_feature_list)
        image_feature = image_feature.astype(np.float32)

        assert state_feature.shape == (1, 8)
        assert image_feature.shape == (1, 24 * len(camera_names))

        combined_row = np.hstack([image_feature, state_feature])  # (1,128)
        combined = combined_row.flatten().astype(np.float32)     # (128,)
        return combined
    
    def _rewards(self):
        # TODO Optimize reward function
        arrive_reward = 10.0
        crash_penalty = -20.0

        k_d = 5.0    # distance progress coef
        k_y = 0.5    # yaw error coef
        k_a = 0.1    # action/ang vel coef
        k_o = 2.0    # obstacle proximity coef

        # positive if moved closer
        reward_dist = k_d * (self.previous_distance - self.cur_distance) / max(self.total_dis, 1e-6)

        # yaw error cost
        yaw_err = self.state[State.relative_yaw]
        reward_yaw = - k_y * (abs(yaw_err) / math.pi)  # in [-k_y, 0]

        # angular velocity cost
        ang_vel = self.state[State.angular_velocity]
        reward_act = - k_a * (abs(ang_vel) / math.pi)


        # aggregate (dense part)
        reward = reward_dist + reward_yaw + reward_act

        # collision flag
        if self.state[State.collision]:
            return crash_penalty

        # arrived
        if self._has_arrived():
            return arrive_reward

        return float(reward)


    def update_state(self):
        state = self.drone.get_ground_truth_kinematics()

        self.state[State.vx] = state["twist"]["linear"]["x"]
        self.state[State.vy] = state["twist"]["linear"]["y"]
        self.state[State.vz] = state["twist"]["linear"]["z"]
        self.state[State.dis_x] = abs(state["pose"]["position"]["x"] - self.target_point[0])
        self.state[State.dis_y] = abs(state["pose"]["position"]["y"] - self.target_point[1])
        self.state[State.dis_z] = abs(state["pose"]["position"]["z"] - self.target_point[2])
        projectairsim_log().info(f"Position: x={state['pose']['position']['x']}, y={state['pose']['position']['y']}, z={state['pose']['position']['z']}")
        projectairsim_log().info(f"Dis: dis_x={self.state[State.dis_x]}, dis_y={self.state[State.dis_y]}, dis_z={self.state[State.dis_z]}")

        # relative_yaw = yaw − goal_yaw
        # [-pi, pi)
        orientation = state["pose"]["orientation"]
        roll, pitch, yaw = quaternion_to_rpy(orientation["w"], orientation["x"], orientation["y"], orientation["z"])
        goal_yaw = math.atan2(self.target_point[1] - state["pose"]["position"]["y"], self.target_point[0] - state["pose"]["position"]["x"])
        angle = yaw - goal_yaw
        self.state[State.relative_yaw] = (angle + math.pi) % (2*math.pi) - math.pi
        
        self.state[State.angular_velocity] = state["twist"]["angular"]["z"]

    # ==================================================
    # utils functions
    # ==================================================

    def normalize_state(self):
        vx_norm = (self.state[State.vx] / self.maxv / 2 + 0.5) * 255
        vy_norm = (self.state[State.vy] / self.maxv / 2 + 0.5) * 255
        vz_norm = (self.state[State.vz] / self.maxv / 2 + 0.5) * 255

        dis_x_norm = (self.state[State.dis_x] / max(self.distance_x, 1e-6) / 2 + 0.5) * 255
        dis_y_norm = (self.state[State.dis_y] / max(self.distance_y, 1e-6) / 2 + 0.5) * 255
        dis_z_norm = (self.state[State.dis_z] / max(self.distance_z, 1e-6) / 2 + 0.5) * 255

        relative_yaw_norm = (self.state[State.relative_yaw] / math.pi / 2 + 0.5) * 255

        angular_velocity_norm = (self.state[State.angular_velocity] / math.pi / 2 + 0.5) * 255

        return np.array([[vx_norm, vy_norm, vz_norm, dis_x_norm, dis_y_norm, dis_z_norm, relative_yaw_norm, angular_velocity_norm]])

    def normalize_image(self, depth_image):
        image_scaled = np.clip(depth_image, 0, 50000) / 50000 * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)
        # Extract features [low=0, high=255]
        rows = np.vsplit(image_uint8, 3)
        split_image = [band.max() for row in rows for band in np.hsplit(row, 8)]
        return np.array([split_image], dtype=np.float32)

    def random_target_point(self):
        # TODO Determine several feasible endpoints
        target_point_set = [
            [50.0, -30.0, -10.0],
            [125.0, 45.0, -5.0],
            [-80.0, -120.0, -15.0],
        ]
        target = random.choice(target_point_set)
        projectairsim_log().info(f"Target point: {target}")
        return target

    def _has_arrived(self):
        return math.sqrt(self.state[State.dis_x] ** 2 + self.state[State.dis_y] ** 2 + self.state[State.dis_z] ** 2) < self.goal_distance_threshold


    def _is_terminal(self):
        projectairsim_log().info(f"Collison: {self.state[State.collision]}, arrived: {self._has_arrived()}")
        return bool((self.state[State.collision] or self._has_arrived()))

    def _is_truncated(self):
        return self.current_step >= self.max_episode_steps    

    def _collision_callback(self, topic=None, msg=None):
        projectairsim_log().info("collision")
        self.state[State.collision] = True

    def distance_3d(self, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return np.linalg.norm(p1 - p2)
    
    def close(self):
        projectairsim_log().info(f"{self.__class__.__module__}.{self.__class__.__name__} is del")
        self.drone.disarm()
        self.drone.disable_api_control()
        self.client.disconnect()
        self.image_display.stop()
