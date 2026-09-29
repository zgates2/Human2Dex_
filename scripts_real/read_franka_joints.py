#!/usr/bin/env python3
"""Read and print Franka joint positions once."""

import zerorpc

    
def main():
    client = zerorpc.Client(heartbeat=20, timeout=10)
    client.connect("tcp://172.16.13.171:4242")

    try:
        print("joint_positions:", client.get_joint_positions())
        print("joint_velocities:", client.get_joint_velocities())
    finally:
        client.close()


if __name__ == "__main__":
    main()