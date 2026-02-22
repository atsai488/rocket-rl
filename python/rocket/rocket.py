from threading import Thread
from typing import Callable
from rocket_onnx.onnx_command_generator import JointCommand
import logging
import socket
import struct

class Rocket:
    def __init__(self, config) -> None:
        self._started_streaming = False
        self._command_stream_stopping = False
        self._state_stream_stopping = False

        self._command_thread = None
        self._state_thread = None
        self.logger = logging.getLogger("Rocket")
        self.config = config
        if config.verbose:
            logging.basicConfig(level=logging.DEBUG)
        self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state_sock.bind(("", self.config.state_port))

    def __del__(self):
        """clean up active streams and threads if spot goes out of scope or is deleted"""
        self.stop_command_stream()
        self.stop_state_stream()


    def power_on(self):
        """Turn on power to robot's motors."""
        # TODO send message to turn on the enable signal to all the motors
        pass

    def start_state_stream(self, on_state_update: Callable[[dict], None]):
        """Begin a background thread that receives UDP state packets.

        `on_state_update` will be called with whatever Python object you build
        from each packet (dict, dataclass, etc.).
        """
        self._state_thread = Thread(
            target=self._udp_state_listener, args=(on_state_update,), daemon=True
        )
        self._state_thread.start()

    def _udp_state_listener(self, on_state_update):
        """runs in a thread; blocks in recvfrom and hands packets to user callback."""
        while not self._state_stream_stopping:
            data, _ = self._state_sock.recvfrom(self.config.STATE_MSG_SIZE)
            state = self._decode_state_packet(data)
            on_state_update(state)

    def _decode_state_packet(self, data: bytes) -> dict:
        # example: joint positions + imu as a struct of floats
        # return {"joints": [..], "imu": {...}}
        fmt = "<6f9f"
        vals = struct.unpack(fmt, data)
        return {"joints": vals[:6], "imu": vals[6:]}

    def stop_state_stream(self):
        if self._state_thread is not None:
            self._state_stream_stopping = True
            self._state_thread.join()

    def start_command_stream(self, command_policy, timing_policy):
        """Create command stream to send joint level commands to the robot.

        arguments:
        command_policy -- Callable that creates one joint command
        timing_policy -- Callable that blocks until the next time a command should be sent
        """
        self._command_thread = Thread(
            target=self._run_command_stream, args=(command_policy, timing_policy), daemon=True
        )
        self._command_thread.start()
    
    def _run_command_stream(
        self, command_policy: Callable[[None], JointCommand], timing_policy: Callable[[None], None]
    ):
        """private function to be run in command stream thread.

        arguments
        command_policy -- callback supplied to start_command_stream to create commands
        timing_policy -- callback supplied to start_command_stream to control timing
        """
        try:
            self.logger.info("Starting command stream")
            self._udp_command_sender(command_policy, timing_policy)
        except Exception as e:
            self.logger.error(f"Error in command stream: {e}")
        finally:
            self.logger.info("Command stream stopped")
    
    def _udp_command_sender(self, command_policy, timing_policy):
        """Send commands over UDP to the robot."""
        cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        robot_addr = (self.config.robot_ip, self.config.command_port)

        while not self._command_stream_stopping:
            if timing_policy():
                cmd = command_policy()
                pkt = self._encode_command(cmd)
                cmd_sock.sendto(pkt, robot_addr)
                self._started_streaming = True
            else:
                self.logger.warning("timing policy timeout")
                return

    def _encode_command(self, cmd) -> bytes:
        """Convert command object to bytes for UDP."""
        # Assume cmd has joint_angles (6 floats)
        pkt = struct.pack("<6f", 
            cmd.joint_angles[0], cmd.joint_angles[1], cmd.joint_angles[2],
            cmd.joint_angles[3], cmd.joint_angles[4], cmd.joint_angles[5],
        )
        return pkt

    def stop_command_stream(self):
        """Stop sending joint commands to the robot."""
        if self._command_thread is not None:
            self._command_stream_stopping = True
            self._command_thread.join()


