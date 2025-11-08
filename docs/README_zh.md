# 🚘 基于ProjectAirSim的无人机(UAV)自主导航

<h4 align="center">
    <p>
        <a href="https://github.com/QinCheng0928/UAV_Navigation_ProjectAirSim/blob/main/README.md">English</a> |
        <b>简体中文</b> 
    </p>
</h4>

Project AirSim是一个用于无人机、机器人和其他自主系统的仿真平台。

在 **[AirSim](https://github.com/microsoft/AirSim)** 先前工作的基础上, 它利用了 **[Unreal Engine 5](https://www.unrealengine.com/)** 提供逼真的视觉效果，同时提供集成定制物理、控制器、执行器和传感器以开发自主系统所需的仿真框架。

AirSim项目由三个主要层组成：

1. **Project AirSim Sim Libs** - 用于定义通用机器人结构和仿真场景时钟循环的基础设施


2. **Project AirSim Plugin** - 构建在Sim Libs上的虚幻引擎插件，其在运行时连接 特定机器人场景（如四旋翼无人机）的外部组件（如控制器、物理、渲染）

3. **Project AirSim Client Library** - 终端用户库，允许通过网络调用API与机器人和仿真交互

## 🚀 项目目标

**阶段一**：**[DONE]** 实现基于规则的无人机避障算法
**阶段二**：**[TODO]** 实现基于 DRL 的无人机自主导航与避障算法  
**阶段三**：**[TODO]** 扩展多智能体场景，实现无人机与地面车辆的协同仿真与任务分配  


## 🧩 仓库结构
```
UAV_Navigation_ProjectAirSim/
├── checkpoints/        # 训练模型保存目录（DRL模型权重）
├── docs/               # 项目文档、说明及设计图
├── envs/               # 自定义强化学习环境与仿真接口
├── media/              # 演示图片、视频等多媒体文件
├── scripts/            # 启动、训练、评估等脚本文件
├── tests/              # 单元测试与功能验证脚本
├── requirements.txt    # Python依赖包列表
├── README.md           # 说明文档
```

## 🙏 致谢

本项目参考并基于以下优秀的开源项目开发与改进：

- [Project AirSim](https://github.com/iamaisim/ProjectAirSim.git)  

- [Microsoft AirSim](https://github.com/microsoft/AirSim.git)  

- [UAV Auto Navigation and Object Tracking based on RL](https://github.com/jzstudent/UAV-auto-navigation-and-object-tracking-based-on-RL.git)  

- [UAV Navigation DRL AirSim](https://github.com/heleidsn/UAV_Navigation_DRL_AirSim.git)  