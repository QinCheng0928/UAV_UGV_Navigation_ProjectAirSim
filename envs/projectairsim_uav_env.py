import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import asyncio
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import ImageDisplay, unpack_image
from projectairsim.types import ImageType
from projectairsim.utils import (
    load_scene_config_as_dict, 
    quaternion_to_rpy, 
    )
from envs.utils.type import (
    ActionType,
    State,
    )

class ProjectAirSimSmallCityEnv(gym.Env):
    subwin_width, subwin_height = 640, 360

    def __init__(self):
        super().__init__()
        self.sim_config_filename = "scene_drone_classic.jsonc"

        self.dv = 0.5
        self.maxv = 5
        self.current_step = 0 
        self.max_episode_steps = 500
        self.target_point = self.random_target_point()
        self.goal_distance_threshold = 10.0

        self.loop = asyncio.get_event_loop()

        # init the sim world and drone
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, self.sim_config_filename)
        self.drone = Drone(self.client, self.world, "Drone1")

        self.client.subscribe(
            self.drone.robot_info["collision_info"],
            lambda topic, msg: self._collision_callback,
        )

        # Init the image display windows
        self.image_display = ImageDisplay(
            num_subwin=3,
            screen_res_x=2560,
            screen_res_y=1440,
            subwin_width=self.subwin_width,
            subwin_height=self.subwin_height
        )
        chase_cam_window = "ChaseCam"
        self.image_display.add_image(chase_cam_window, subwin_idx=0, resize_x=self.subwin_width, resize_y=self.subwin_height) 
        self.client.subscribe(
            self.drone.sensors["Chase"]["scene_camera"],
            lambda _, msg: self.image_display.receive(msg, chase_cam_window),
        )

        front_cam_window = "FrontCam"
        self.image_display.add_image(front_cam_window,  subwin_idx=1, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["scene_camera"],
            lambda _, msg: self.image_display.receive(msg, chase_cam_window),
        )

        depth_name = "DepthImage"
        self.image_display.add_image(depth_name, subwin_idx=2, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.client.subscribe(
            self.drone.sensors["front_center"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, chase_cam_window),
        )

        # 
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
        self.update_state()

        self.distance_x = self.state[State.dis_x]
        self.distance_y = self.state[State.dis_y]
        self.distance_z = self.state[State.dis_z]

        # The speed variation of North East Down South West Up and BRAKE
        self.action_space = gym.spaces.Discrete(7)  

        # We will first divide the image vertically into three parts, top, bottom, and middle, and take out the middle part
        # Divide the middle image horizontally into eight parts and take the maximum value of each part as the feature vector of the observation space 
        # Tth shape is (1, 8)
        # 
        # We also choose velocity in 3D space (vx vy, vz), distance from the endpoint (dis_x, dis_y, dis_z), 
        # relative heading angle, and angular velocity as features of the state
        # The shape is (1, 8)
        self.observation_space = spaces.Box(
            low=0, 
            high=1,
            shape=(1, 8 + 8),
            dtype=np.float32
            )
        
        self.drone.enable_api_control()
        self.drone.arm()

    """
        The reset() method must return a tuple (obs, info) 
    """
    def reset(self, *, seed=None, options=None):
        # reset the world and drone
        config_loaded, _ = load_scene_config_as_dict(
            self.sim_config_filename, 
            sim_config_path="sim_config/", 
            sim_instance_idx=-1
        )
        self.world.load_scene(config_loaded, delay_after_load_sec=0)
        self.drone = Drone(self.client, self.world, "Drone1")                

        # resubscribe image display windows
        chase_cam_window = "ChaseCam"
        self.image_display.add_image(chase_cam_window, subwin_idx=0, resize_x=self.subwin_width, resize_y=self.subwin_height) 
        self.client.subscribe(
            self.drone.sensors["Chase"]["scene_camera"],
            lambda _, msg: self.image_display.receive(msg, chase_cam_window),
        )

        front_cam_window = "FrontCam"
        self.image_display.add_image(front_cam_window,  subwin_idx=1, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.client.subscribe(
            self.drone.sensors["FrontCamera"]["scene_camera"],
            lambda _, msg: self.image_display.receive(msg, chase_cam_window),
        )

        depth_name = "DepthImage"
        self.image_display.add_image(depth_name, subwin_idx=2, resize_x=self.subwin_width, resize_y=self.subwin_height)
        self.client.subscribe(
            self.drone.sensors["front_center"]["depth_planar_camera"],
            lambda _, msg: self.image_display.receive(msg, chase_cam_window),
        )

        # reset state variables
        self.current_step = 0 
        self.target_point = self.random_target_point()
        self.update_state()
        self.distance_x = self.state[State.dis_x]
        self.distance_y = self.state[State.dis_y]
        self.distance_z = self.state[State.dis_z]

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
        
        obs = self._get_obs()
        reward = self._rewards()
        done = self._is_terminal()
        truncated = self._is_truncated()
        info = {}
        return (obs, reward, done, truncated, info)
    
    async def _simulate(self, action):
        action = int(action)
        print(f"Action taken: {ActionType.NUM2NAME[action]}")
        
        vx = float(self.state[State.vx]) * 0.9
        vy = float(self.state[State.vy]) * 0.9
        vz = float(self.state[State.vz]) * 0.9
        
        # update velocity based on action
        if action == ActionType.NORTH:
            vx += self.dv
        elif action == ActionType.EAST:
            vy += self.dv
        elif action == ActionType.DOWN:
            vz += self.dv
        elif action == ActionType.SOUTH:
            vx -= self.dv
        elif action == ActionType.WEST:
            vy -= self.dv
        elif action == ActionType.UP:
            vz -= self.dv
        elif action == ActionType.BRAKE:
            vx *= 0.5
            vy *= 0.5
            vz *= 0.5

        vx = np.clip(vx, -self.maxv, self.maxv)
        vy = np.clip(vy, -self.maxv, self.maxv)
        vz = np.clip(vz, -self.maxv, self.maxv)

        if self._has_arrived():
            vx = 0.0
            vy = 0.0
            vz = 0.0
        
        print(f"Velocity command: vx={vx}, vy={vy}, vz={vz}")
        
        # send velocity command to the drone
        move_up_task = await self.drone.move_by_velocity_async(v_north=vx, v_east=vy, v_down=vz, duration=0.5)
        await move_up_task     


    def _get_obs(self):
        # Remove collision tag and Normalizing [low=0, high=255]
        state_feature = self.normalize_state()

        # Obtain and ormalizing depth image
        result = self.drone.get_images("front_center", [ImageType.DEPTH_PLANAR])
        depth_image = unpack_image(result[ImageType.DEPTH_PLANAR])
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        image_feature = self.normalize_image(depth_image)

        assert state_feature.shape[0] == 1 and state_feature.shape[1] == 8 
        assert image_feature.shape[0] == 1 and image_feature.shape[1] == 8 

        return np.concatenate((image_feature, state_feature), axis=0)

    def _rewards(self):
        # TODO
        pass

    def update_state(self):
        state = self.drone.get_ground_truth_kinematics()

        self.state[State.vx] = state["twist"]["linear"]["x"]
        self.state[State.vy] = state["twist"]["linear"]["y"]
        self.state[State.vz] = state["twist"]["linear"]["z"]
        self.state[State.dis_x] = abs(state["pose"]["position"]["x"] - self.target_point[0])
        self.state[State.dis_y] = abs(state["pose"]["position"]["y"] - self.target_point[1])
        self.state[State.dis_z] = abs(state["pose"]["position"]["z"] - self.target_point[2])

        # relative_yaw = yaw − goal_yaw
        # [-pi, pi)
        orientation = state["pose"]["orientation"]
        pitch, roll, yaw = quaternion_to_rpy(orientation["w"], orientation["x"], orientation["y"], orientation["z"])
        goal_yaw = math.atan2(self.target_point[1] - state["pose"]["position"]["y"], self.target_point[0] - state["pose"]["position"]["x"])
        angle = yaw - goal_yaw
        self.state[State.relative_yaw] = (angle + math.pi) % (2*math.pi) - math.pi
        
        self.state[State.angular_velocity] = state["twist"]["linear"]["z"]

    # ==================================================
    # utils functions
    # ==================================================

    def normalize_state(self):
        vx_norm = (self.state[State.vx] / self.maxv / 2 + 0.5) * 255
        vy_norm = (self.state[State.vy] / self.maxv / 2 + 0.5) * 255
        vz_norm = (self.state[State.vz] / self.maxv / 2 + 0.5) * 255

        dis_x_norm = (self.state[State.dis_x] / self.distance_x / 2 + 0.5) * 255
        dis_y_norm = (self.state[State.dis_y] / self.distance_y / 2 + 0.5) * 255
        dis_z_norm = (self.state[State.dis_z] / self.distance_z / 2 + 0.5) * 255

        angular_velocity_norm = (self.state[State.angular_velocity] / 2 / 2 + 0.5) * 255

        relative_yaw_norm = (self.state[State.relative_yaw] / math.pi / 2 + 0.5) * 255

        return np.array([[vx_norm, vy_norm, vz_norm, dis_x_norm, dis_y_norm, dis_z_norm, relative_yaw_norm, angular_velocity_norm]])

    def normalize_image(self, depth_image):
        image_scaled = np.clip(depth_image, 0, 15) / 15 * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)
        # Extract features [low=0, high=255]
        middle = np.vsplit(image_uint8, 3)[1]
        bands = np.hsplit(middle, 8)
        split_image = []
        for i in range(8):
            split_image.append(bands[i].max())
        return np.array([split_image])

    def random_target_point(self):
        # TODO Determine several feasible endpoints
        return [7.0, 7.0, 7.0]

    def _has_arrived(self):
        return self.distance_3d(
            np.array([self.state[State.vx], self.state[State.vy], self.state[State.vz]]),
            self.target_point
        ) < self.goal_distance_threshold

    def _is_terminal(self):
        return bool((self.state[State.collision] or self._has_arrived()))

    def _is_truncated(self):
        return self.current_step >= self.max_episode_steps    

    def _collision_callback(self):
        self.state[State.collision] = True

    def distance_3d(self, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return np.linalg.norm(p1 - p2)