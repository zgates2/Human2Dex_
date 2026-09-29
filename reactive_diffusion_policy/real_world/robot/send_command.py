import requests
import numpy as np
import time
import json
from spatialmath import SE3

ROBOT_SERVER_IP = "127.0.0.1"
ROBOT_SERVER_PORT = 8092

session = requests.session()

def send_command(endpoint: str, data: dict = None):
    """
    向机器人服务器发送命令。
    此函数严格遵循您在 RealRobotEnvironment 中定义的版本。
    """
    url = f"http://{ROBOT_SERVER_IP}:{ROBOT_SERVER_PORT}{endpoint}"
    print(f"--- 发送请求 ---")
    print(f"URL: {url}")
    
    response = None
    try:
        if 'get' in endpoint:
            print("方法: GET")
            response = session.get(url, timeout=10)
        else:
            print(f"方法: POST")
            if data:
                print(f"数据 (部分显示): {str(data)}...")

            if 'move' in endpoint:
                print("移动指令的特殊处理: 超时设为0.001s，并忽略超时错误")
                try:
                    response = session.post(url, json=data, timeout=0.001)
                except requests.exceptions.ReadTimeout:
                    print("捕获到ReadTimeout，按设计忽略，继续执行。")
                    response = None
            else:
                response = session.post(url, json=data, timeout=10)

        if response is not None:
            response.raise_for_status()
            print("收到成功响应 ---")

            response_json = response.json()
            # print(json.dumps(response_json, indent=2))
            return response_json
        else:
            print("无响应 (移动指令超时)，按设计返回空字典 ---")
            return dict()

    except requests.exceptions.RequestException as e:
        print(f"\n[错误] 请求失败: {e}")
        return None
    except json.JSONDecodeError:
        print("--- 收到成功响应 (但非JSON格式或无内容) ---")
        return {"status": "success", "message": "Received non-JSON success response."}

def test_get_robot_states():
    print("测试: 获取当前机器人完整状态...")
    send_command('/get_current_robot_states')

def test_go_home():
    print("\n测试: 控制机器人回归初始位置...")
    send_command('/birobot_go_home')

def test_move_tcp_incremental(robot_side: str):
    # print(f"测试: 增量移动 {robot_side} 手臂...")

    # print(f"获取 {robot_side} 手臂的当前姿态...")
    # response_data = send_command(f'/get_current_tcp/{robot_side}')
    # if response_data is None or not isinstance(response_data, list):
    #     print(f"无法获取有效的当前姿态，已跳过移动测试。")
    #     return
    
    # current_pose_list = response_data
    # current_pose_se3 = SE3(current_pose_list, check=False)

    # delta_6dpose = [0.1, -0.1, 0.1, 0, 0, np.pi/6] 
    # print(f"定义相对运动 (dx,dy,dz,dr,dp,dy): {np.round(delta_6dpose, 3).tolist()}")
    
    # delta_pose = SE3.Trans(delta_6dpose[0:3]) * SE3.RPY(delta_6dpose[3:6], order='xyz')
    # target_pose_se3 = current_pose_se3 * delta_pose

    # request_data = {
    #     "target_tcp_matrix": target_pose_se3.A.tolist(),
    #     "duration": 4.0
    # }
    
    # send_command(f'/move_tcp/{robot_side}', data=request_data)
    target_tcp = [0.5542806981908298,0.03234132931322995,0.3678113635732887,0.002992255642590003,-0.9271359218892071,0.37449327797530024,-0.012837971028709502]
    send_command(f'/move_tcp/{robot_side}', target_tcp )


if __name__ == "__main__":
    print("开始测试 Bimanual Flexiv Server API")

    test_get_robot_states()
    time.sleep(1)

    test_move_tcp_incremental('left')
    # print("等待机器人移动 (4秒)...")
    time.sleep(4.5)

    test_go_home()
    print("\n等待机器人回归初始位置...")
    time.sleep(5) 

    print("\n测试: 获取复位后的最终状态...")
    test_get_robot_states()

    print("所有测试已执行完毕。")