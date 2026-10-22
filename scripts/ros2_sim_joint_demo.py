"""Simulator-only ROS 2 DDS smoke test for Unitree G1.

Reads LowState and sends a small trajectory to the left elbow through LowCmd.
The integrated unitree_rl_mjlab simulator bridge currently ignores the CRC field;
this script is strictly for local simulation and is not a real-robot controller.
"""

import os
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from unitree_hg.msg import LowCmd, LowState
except ImportError as exc:
    raise SystemExit(
        "ROS 2 Humble and the unitree_hg workspace must be sourced first. "
        "See docs/run_empty_simulation.md."
    ) from exc


MOTOR_COUNT = 29
LEFT_ELBOW_INDEX = 18
MOVE_RAD = 0.10
LEG_SECONDS = 2.0
KP = 5.0
KD = 0.5


class SimJointDemo(Node):
    def __init__(self):
        super().__init__("g1_sim_joint_demo")
        self.state = None
        self.state_received_at = 0.0
        self.started_at = None
        self.publisher = self.create_publisher(LowCmd, "/lowcmd", 10)
        self.subscription = self.create_subscription(
            LowState, "lowstate", self.on_state, 10
        )
        self.timer = self.create_timer(0.01, self.send_command)
        self.get_logger().info(
            "Waiting for LowState; only left elbow will move by 0.10 rad and return."
        )

    def on_state(self, message):
        self.state = message
        self.state_received_at = time.monotonic()

    def send_command(self):
        if self.state is None:
            return
        if time.monotonic() - self.state_received_at > 0.10:
            self.get_logger().error("LowState is stale; stopping LowCmd publication.")
            self.destroy_timer(self.timer)
            rclpy.shutdown()
            return

        if self.started_at is None:
            self.started_at = time.monotonic()
            self.get_logger().info(
                f"Received LowState; mode_machine={self.state.mode_machine}. "
                "Starting the simulator-only joint test."
            )

        elapsed = time.monotonic() - self.started_at
        if elapsed < LEG_SECONDS:
            fraction = elapsed / LEG_SECONDS
            offset = MOVE_RAD * fraction
        elif elapsed < 2 * LEG_SECONDS:
            fraction = (elapsed - LEG_SECONDS) / LEG_SECONDS
            offset = MOVE_RAD * (1.0 - fraction)
        else:
            offset = 0.0

        command = LowCmd()
        command.mode_pr = 0  # PR mode, as used in the Unitree G1 sample.
        command.mode_machine = self.state.mode_machine
        for index in range(MOTOR_COUNT):
            motor = command.motor_cmd[index]
            motor.mode = 1
            motor.q = float(self.state.motor_state[index].q)
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = KP
            motor.kd = KD

        command.motor_cmd[LEFT_ELBOW_INDEX].q += float(offset)
        # The local unitree_rl_mjlab simulator bridge does not validate CRC.
        # Never use this command construction against physical hardware.
        command.crc = 0
        self.publisher.publish(command)


def main():
    domain = os.environ.get("ROS_DOMAIN_ID", "0")
    dds_uri = os.environ.get("CYCLONEDDS_URI", "")
    if domain != "0" or 'name="lo"' not in dds_uri:
        raise SystemExit(
            "Refusing to publish: this demo requires ROS_DOMAIN_ID=0 and "
            'CYCLONEDDS_URI configured for interface name="lo". '
            "This guard is for the local simulator only."
        )

    rclpy.init()
    node = SimJointDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
