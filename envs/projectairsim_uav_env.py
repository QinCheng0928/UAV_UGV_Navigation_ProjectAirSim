import numpy as np
from gymnasium import gym
from gymnasium import spaces

import asyncio
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.image_utils import ImageDisplay, unpack_image
from projectairsim.utils import load_scene_config_as_dict

class ProjectAirSimSmallCityEnv(gym.Env):
    subwin_width, subwin_height = 640, 360

    def __init__(self):
        super().__init__()
        self.sim_config_filename = "scene_drone_classic.jsonc"

        self.dv = 1.0
        self.current_step = 0 
        self.max_episode_steps = 500
        self.target_point = self.random_target_point()
        self.goal_distance_threshold = 10.0

        self.loop = asyncio.get_event_loop()

        # init the sim world and drone
        self.client = ProjectAirSimClient()
        self.client.connect()
        self.world = World(self.client, self.sim_config_fname)
        self.drone = Drone(self.client, self.world, "Drone1")

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

        # relative_yaw = yaw − goal_yaw
        # vx = 0
        # vy = 1
        # vz = 2
        # dis_x = 3
        # dis_y = 4
        # dis_z = 5
        # relative_yaw = 6
        # angular_velocity = 7
        # collision = 8
        self.state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False]
        assert len(self.state) == 9
        self.update_state()

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

    """
        The reset() method must return a tuple (obs, info) 
    """
    def reset(self, *, seed=None, options=None):
        # reset the world and drone
        config_loaded, _ = load_scene_config_as_dict(
            self.sim_config_fname, 
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

        # TODO: Return to observation and info
        obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        info = {}
        return (obs, info)

    """
        The step() method must return five values: (obs, reward, terminated, truncated, info).  
    """
    def step(self, action):
        # TODO step once
        obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if action == 0:
            reward = 1
        else:
            reward = -1
        terminated = False  
        truncated = False
        info = {}
        return (obs, reward, terminated, truncated, info)
    
    # ==================================================
    # utils functions
    # ==================================================
    def _has_arrived(self):
        return self.distance_3d(np.array[self.state["x"], self.state["y"], self.state["z"]], self.target_point) < self.goal_distance_threshold

    def _is_terminal(self):
        return bool((self.collision or self._has_arrived()))

    def _is_truncated(self):
        return self.sim_step >= self.max_sim_steps    

    def update_state():
        # TODO update state
        pass

    def random_target_point(self):
        # TODO Determine several feasible endpoints
        return [0.0, 0.0, 0.0]
        

    def distance_3d(self, p1, p2):
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        return np.linalg.norm(p1 - p2)