#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import lcm
import threading
import numpy as np
import time
from typing import List
from loguru import logger

from exlcm import control_command_t, franka_state_t
from std_msgs.msg import Float64MultiArray


class FlexivController(Node):

    def __init__(self):
        super().__init__('flexiv_controller_client')

        self.DOF = 7

        self.LCM_URL = "udpm://239.255.76.67:7667?ttl=64"
        self.lc = lcm.LCM(self.LCM_URL)
        self.LCM_COMMAND_CHANNEL = "ROBOT_COMMAND"

        self.ROS2_COMMAND_TOPIC = '/robot_command'
        self.command_subscriber = self.create_subscription(
            Float64MultiArray,
            self.ROS2_COMMAND_TOPIC,
            self.ros2_command_callback,
            10
        )
        self.get_logger().info(f"正在监听ROS2指令话题: '{self.ROS2_COMMAND_TOPIC}'")


        self.LCM_STATE_CHANNEL = "FRANKA_STATE"
        self.lc.subscribe(self.LCM_STATE_CHANNEL, self.lcm_state_handler)

        self.current_q = [0.0] * self.DOF
        self.current_q_d = [0.0] * self.DOF
        self.current_K_F_ext_hat_K = [0.0] * 6
        self.current_O_F_ext_hat_K = [0.0] * 6
        self.current_O_T_EE = [0.0] * 16
        self.current_O_T_EE_d = [0.0] * 16
        self.current_tcp = [0.0] * 6  

        self.lcm_thread = threading.Thread(target=self.lcm_loop)
        self.lcm_thread.daemon = True
        self.lcm_thread.start()

        self.get_logger().info("FlexivController (ROS2 Client) 节点已启动并正在运行...")

    def ros2_command_callback(self, msg: Float64MultiArray):
        """
        【新增】当 '/robot_command' 话题收到消息时，此函数被调用。
        """
        self.get_logger().info(f"从ROS2话题收到指令: {msg.data}")

        self.tcp_move(msg.data)

    def lcm_state_handler(self, channel, data):
        """处理从LCM收到的机器人状态消息"""
        try:
            msg = franka_state_t.decode(data)
            
            self.current_q = list(msg.q)
            self.current_q_d = list(msg.q_d)
            self.current_K_F_ext_hat_K = list(msg.K_F_ext_hat_K)
            self.current_O_F_ext_hat_K = list(msg.O_F_ext_hat_K)
            self.current_O_T_EE = list(msg.O_T_EE)
            self.current_O_T_EE_d = list(msg.O_T_EE_d)

            matrix = np.array(msg.O_T_EE).reshape((4, 4), order='F')
            position = matrix[:3, 3]
            self.current_tcp[:3] = position.tolist()

            self.get_logger().info(f"已更新机器人状态: q={self.current_q}", throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"处理LCM状态消息时出错: {e}")

    def lcm_loop(self):
        """在后台线程中运行的LCM循环"""
        while rclpy.ok():
            self.lc.handle_timeout(100)


    def clear_fault(self):
        pass

    def get_current_robot_states(self):
        pass

    def get_current_gripper_states(self):
        pass

    def get_current_gripper_force(self):
        pass

    def get_current_gripper_width(self):
        pass

    def get_current_q(self) -> List[float]:
        return self.current_q.copy()

    def get_current_q_d(self) -> List[float]:
        return self.current_q_d.copy()
        
    def get_K_F_ext_hat_K(self) -> List[float]:
        return self.current_K_F_ext_hat_K.copy()
        
    def get_O_F_ext_hat_K(self) -> List[float]:
        return self.current_O_F_ext_hat_K.copy()
        
    def get_O_T_EE(self) -> List[float]:
        return self.current_O_T_EE.copy()
        
    def get_O_T_EE_d(self) -> List[float]:
        return self.current_O_T_EE_d.copy()

    def get_current_tcp(self) -> List[float]:
        return self.current_tcp.copy()

    def move_delta(self, delta_6dof: List[float]):
        pass

    def tcp_move(self, target_tcp: List[float]):
        if len(target_tcp) != 6:
            self.get_logger().error(f"tcp_move指令长度错误，需要6个元素，但收到了{len(target_tcp)}个")
            return
            
        lcm_msg = control_command_t()
        lcm_msg.timestamp = int(time.time() * 1e9)
        lcm_msg.xyzrpy = [float(x) for x in target_tcp]
        
        self.lc.publish(self.LCM_COMMAND_CHANNEL, lcm_msg.encode())
        self.get_logger().info(f"已通过LCM发送TCP指令: {lcm_msg.xyzrpy}")

    def execute_primitive(self, primitive_command: str):
        pass

    @staticmethod
    def parse_pt_states(pt_states,parse_target):
        pass


def main(args=None):
    """
    标准的ROS2节点启动函数
    """
    rclpy.init(args=args)

    flexiv_controller_node = FlexivController()
    flexiv_controller_node.tcp_move([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    print("\n=======================================================")
    print(" Flexiv ROS2 客户端节点正在运行...")
    print(" 它现在会持续监听和发送消息。")
    print(f" 你可以在另一个终端使用 'ros2 topic pub' 命令来发送控制指令:")
    print(f" ros2 topic pub --once {flexiv_controller_node.ROS2_COMMAND_TOPIC} std_msgs/msg/Float64MultiArray \"data: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]\"")
    print(" 按下 Ctrl+C 来关闭节点。")
    print("=======================================================\n")

    try:
        rclpy.spin(flexiv_controller_node)
    except KeyboardInterrupt:
        print("节点被用户关闭。")
    finally:
        flexiv_controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()