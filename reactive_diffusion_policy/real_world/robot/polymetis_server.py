import sys
sys.path.append("/home/ps/reactive_diffusion_policy")

import threading
from typing import List, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import scipy.spatial.transform as st

from reactive_diffusion_policy.common.data_models import (
    BimanualRobotStates,
    TargetTCPRequest,
    MoveGripperRequest,
)
from reactive_diffusion_policy.real_world.robot.franka_polymetis_controller import (
    FrankaPolymetisController,
)

# HOME_JOINTS = [
#     0.24886237143896478, -0.10823570083692469, -0.2110291881155267,
#     -2.6155513998496334, -0.17867161943735246, 3.024342456234826,
#     0.3109925351142883,
# ]

# HOME_JOINTS = [
#     -1.376264507362717, -0.1910906426119044, 2.1959850223273554,
#     -2.2395674360513635, -0.48146272668541595, 2.3278511639171175,
#     1.723992445891102
# ]

# wipeboard success
# HOME_JOINTS = [
#  -0.8629707813052511,
#  -1.6016379059239436,
#  1.477659173965454,
#  -2.2641962561821147,
#  1.2974666399624584,
#  1.841560215853223,
#  0.6923241043152109,
# ]

# HOME_JOINTS = [ 
#  -2.1029106945426834,
#  0.021080016961662066,
#  2.875408588585734,
#  -2.4994109044003747,
#  -1.0093096452090473,
#  2.5536308212810086,
#  2.361131541571129,
# ]

# for wipe board
# HOME_JOINTS = [
#  0.8252645933688374,
#  0.21940158814122493,
#  -0.06185944646085302,
#  -2.1303135861290823,
#  -0.5583156519267293,
#  2.3317965381541197,
#  1.9901671702127548,
# ]
# HOME_JOINTS = [
#  0.8300591464941961,
#  0.19100460045797782,
#  0.11786532278467383,
#  -2.022717607397782,
#  -0.5385859121448816,
#  2.0437041405506813,
#  2.0003034434301985,
# ]

# for drop cup
# HOME_JOINTS = [
#   0.4759580416323846,
#   0.24603609291712442,
#   0.47744604637748317,
#   -2.5759915150090267,
#   -2.169796640588177,
#   1.739133627161482,
#   2.865941947458519,
# ]


# for pickplace sponge
# HOME_JOINTS = [
#  -1.8752569154416907,
#  -0.260638446602905,
#  2.889558211184555,
#  -2.2496999623482683,
#  -0.9784168304138714,
#  1.963893310593234,
#  2.4873323632859523,
# ]
# HOME_JOINTS = [
#  0.6605581376322529,
#  0.34316640973237744,
#  0.119865093700825,
#  -2.032868146505498,
#  -0.7386827577286297,
#  2.100034594906701,
#  2.25040875132713,
# ]

# for pickplace watercup
# HOME_JOINTS = [
#  1.1421933678890368,
#  0.18658366364441195,
#  0.18827763438224793,
#  -2.3232510834435183,
#  -1.0109048912061582,
#  1.4064124296859928,
#  2.6822350312852197,
# ]

# HOME_JOINTS = [
#  0.9025927072136025,
#  0.5763052233968395,
#  0.3674577986399332,
#  -1.8714395883882207,
#  -1.1477752407774944,
#  1.3506854696704282,
#  2.6217007098181377,
# ]

# HOME_JOINTS = [
#  2.1567326648136107,
#  1.2045977136416846,
#  -1.1566396247037807,
#  -1.8325099541567247,
#  -0.2206595512866027,
#  1.7628228647267845,
#  2.067680888657774,
# ]


# for pickplace sandwich
# HOME_JOINTS = [
#  -0.3636728005764777,
#  0.33882578815698317,
#  1.140107414812847,
#  -2.5141896693581023,
#  -1.4907667926328612,
#  1.7461438401617386,
#  2.602548463209598,
# ]


# for pickegg
HOME_JOINTS = [
 -1.1880373329551597,
 -0.8829978849706593,
 1.7282171129246187,
 -2.4069795378982355,
 0.41006126869387094,
 2.534228428575752,
 1.269965081372192,
]


class PolymetisServer:
    def __init__(
        self,
        host_ip: str = "127.0.0.1",
        port: int = 8092,
        server_ip: str = "172.16.13.170",  #R3 IP 172.16.13.171
        server_port: int = 4242,
        gripper_config: Dict = {
            "mode": "single",
            "port1": "/dev/ttyUSB2",
            "motor1_id": 1,
        },
        force_sensor_port: str = "/dev/ttyUSB0",
    ) -> None:
        self.host_ip = host_ip
        self.port = port

        logger.info("Initializing FrankaPolymetisController...")
        self.controller = FrankaPolymetisController(
            server_ip=server_ip,
            server_port=server_port,
            gripper_config=gripper_config,
            force_sensor_port=force_sensor_port,
        )

        self.app = FastAPI(on_shutdown=[self.shutdown])
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=False,
            expose_headers=["*"],
            max_age=86400,
        )
        self.setup_routes()

    def shutdown(self):
        logger.info("Closing Polymetis controller...")
        self.controller.close()
        logger.info("Polymetis controller closed.")

    @staticmethod
    def _pose_7d_to_6d_rpy(pose_7d: List[float]) -> List[float]:
        """Convert [x,y,z,qw,qx,qy,qz] to [x,y,z,r,p,y]."""
        xyz = pose_7d[:3]
        qw, qx, qy, qz = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
        R = st.Rotation.from_quat([qx, qy, qz, qw])  # scipy uses [qx,qy,qz,qw]
        rpy = R.as_euler("xyz").tolist()
        return xyz + rpy

    def setup_routes(self):
        @self.app.get("/get_current_robot_states", response_model=BimanualRobotStates)
        async def get_current_robot_states() -> BimanualRobotStates:
            states = await run_in_threadpool(self.controller.get_current_robot_states)
            return BimanualRobotStates(
                leftRobotTCP=states["leftRobotTCP"],
                leftRobotTCPVel=states["leftRobotTCPVel"],
                leftRobotTCPWrench=states["leftRobotTCPWrench"],
                leftGripperState=states["leftGripperState"],
                leftRobotTCPTarget=states["leftRobotTCPTarget"],
                leftJoinStates=states["leftJoinStates"],
            )

        @self.app.post("/move_tcp/{robot_side}")
        async def move_tcp(robot_side: str, request: TargetTCPRequest) -> Dict[str, str]:
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' supported.")
            goal_rpy = self._pose_7d_to_6d_rpy(request.target_tcp)
            await run_in_threadpool(self.controller.tcp_move, goal_rpy)
            return {"message": "Left robot finished moving to target tcp."}

        @self.app.get("/get_current_tcp/{robot_side}")
        async def get_current_tcp(robot_side: str) -> List[float]:
            if robot_side != "left":
                raise HTTPException(status_code=400, detail="Only 'left' supported.")
            states = await run_in_threadpool(self.controller.get_current_robot_states)
            return states["leftRobotTCP"]

        @self.app.post("/birobot_go_home")
        async def birobot_go_home() -> Dict[str, str]:
            await run_in_threadpool(self.controller.reset_to_home, HOME_JOINTS)
            return {"message": "Robot has gone home."}

        @self.app.post("/move_gripper/{robot_side}")
        async def move_gripper(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side != "left":
                return {"message": f"Action for '{robot_side}' gripper is ignored. Only 'left' is handled."}
            await run_in_threadpool(
                self.controller.gripper_controller.move_gripper,
                request.width,
                request.force_limit,
            )
            return {"message": f"Left gripper moving to width {request.width} with force limit {request.force_limit}"}

        @self.app.post("/move_gripper_force/{robot_side}")
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side != "left":
                return {"message": f"Action for '{robot_side}' gripper is ignored. Only 'left' is handled."}
            await run_in_threadpool(
                self.controller.gripper_controller.move_gripper,
                request.width,
                request.force_limit,
            )
            return {"message": f"Left gripper grasp with force limit {request.force_limit}"}

        @self.app.post("/stop_gripper/{robot_side}")
        async def stop_gripper(robot_side: str) -> Dict[str, str]:
            if robot_side != "left":
                return {"message": f"Action for '{robot_side}' gripper is ignored. Only 'left' is handled."}
            await run_in_threadpool(self.controller.gripper_controller.stop_gripper)
            return {"message": "Left gripper stopping"}

    def run(self):
        logger.info(f"Start Polymetis Server at http://{self.host_ip}:{self.port}")
        uvicorn.run(self.app, host=self.host_ip, port=self.port, access_log=False)


if __name__ == "__main__":
    server = PolymetisServer()
    server.controller.reset_to_home(HOME_JOINTS)
    server.run()
