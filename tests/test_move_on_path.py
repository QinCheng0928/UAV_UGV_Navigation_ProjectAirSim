import asyncio
from projectairsim import ProjectAirSimClient, Drone, World
from projectairsim.drone import YawControlMode

async def main():
    client = ProjectAirSimClient()
    client.connect()
    world = World(client, "scene_drone_classic.jsonc", delay_after_load_sec=2)
    drone = Drone(client, world, "Drone1")

    drone.enable_api_control()
    drone.arm()
    takeoff_task = await drone.takeoff_async()
    await takeoff_task

    # segment =[
    #     [0, 5, -6],
    #     [0, 10, -6],
    #     [0, 15, -6],
    # ]
    segment =[
        [0, 0, -7],
        [5, 0, -6],
    ]
    move_task = await drone.move_on_path_async(
        path=segment,
        velocity=2.0,
        yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
        lookahead=-1,
        adaptive_lookahead=1
    )
    await move_task

    land_task = await drone.land_async()
    await land_task

    drone.disarm()
    drone.disable_api_control()

    client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())