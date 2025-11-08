# 🚘 UAV_Navigation_ProjectAirSim

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/QinCheng0928/UAV_Navigation_ProjectAirSim/blob/main/docs/README_zh.md">简体中文</a> 
    </p>
</h4>

Project AirSim is a simulation platform for drones, robots, and other autonomous systems.

Building on the previous work of **[AirSim](https://github.com/microsoft/AirSim)**, it leverages **[Unreal Engine 5](https://www.unrealengine.com/)** to provide photo-realistic visuals, while providing the simulation framework needed to integrate custom physics, controllers, actuators, and sensors to develop an autonomous system.

Project AirSim consists of three main layers:

1. **Project AirSim Sim Libs** - Base infrastructure for defining a generic robot structure and simulation scene tick loop

2. **Project AirSim Plugin** - Host package (currently an Unreal Plugin) that builds on the sim libs to connect external components (controller, physics, rendering) at runtime that are specific to each configured robot-type scenario (ex. quadrotor drones)

3. **Project AirSim Client Library** - End-user library to enable API calls to interact with the robot and simulation over a network connection

## 🚀 Project Objective

**Stage 1 **：**[DONE]** Implementation of rule-based obstacle avoidance algorithm for unmanned aerial vehicles  
**Stage 2 **：**[TODO]** Implement DRL based autonomous navigation and obstacle avoidance algorithm for UAV  
**Stage 3 **：**[TODO]** Implement DRL based autonomous navigation and obstacle avoidance algorithm for UGV    
**Stage 4 **：**[TODO]** Expand multi-agent scenarios to achieve collaborative simulation and task allocation between drones and ground vehicles 

## 🧩 Repository Structure
```
UAV_Navigation_ProjectAirSim/
├── checkpoints/        # Training model save directory (DRL model weights)
├── docs/               # Project Documents, Instructions, and Design Drawings
├── envs/               # Custom Reinforcement Learning Environment and Simulation Interface
├── media/              # Multimedia files such as demonstration images, videos, etc
├── scripts/            # Script files for startup, training, evaluation, etc
├── tests/              # Unit testing and functional verification scripts
├── requirements.txt    # Python Dependency Package List
├── README.md           # Documentation
```

## 🙏 Acknowledgments

This project is developed and improved based on the following excellent open source projects:

- [Project AirSim](https://github.com/iamaisim/ProjectAirSim.git)  

- [Microsoft AirSim](https://github.com/microsoft/AirSim.git)  

- [UAV Auto Navigation and Object Tracking based on RL](https://github.com/jzstudent/UAV-auto-navigation-and-object-tracking-based-on-RL.git)  

- [UAV Navigation DRL AirSim](https://github.com/heleidsn/UAV_Navigation_DRL_AirSim.git)  