import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List
from rocket.rocket_state import RocketState
from threading import Event


@dataclass
class JointCommand:
    joint_angles: List[float]



@dataclass
class RocketOnnxContext:
    """data class to hold runtime data needed by the controller"""

    event = Event()
    latest_state = RocketState()
    count = 0


class StateHandler:
    """Class to be used as callback for state stream to put state date
    into the controllers context
    """

    def __init__(self, context: RocketOnnxContext) -> None:
        self._context = context

    def __call__(self, state: dict):
        """make class a callable and handle incoming state stream when called

        arguments
        state -- proto msg from spot containing most recent data on the robots state"""
        self._context.latest_state.update_from_udp(state)
        self._context.event.set()
        
class RocketOnnxPositionController:

    def __init__(self, context: RocketOnnxContext, config, model_path, verbose=False):
        self.state = context
        self.config = config
        self.verbose = verbose

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.N = 6
        self.last_action = np.zeros(self.N)

    def __call__(self):
        # Get latest observation safely
        obs = self.state.latest_state.to_observation(self.last_action)

        model_input = np.array([obs], dtype=np.float32)
        output = self.session.run(None, {self.input_name: model_input})[0][0]

        target = output[:self.N] * self.config.action_scale

        # Optional: clamp to joint limits
        if hasattr(self.config, "joint_min"):
            target = np.maximum(target, self.config.joint_min)
        if hasattr(self.config, "joint_max"):
            target = np.minimum(target, self.config.joint_max)

        self.last_action = target

        if self.verbose:
            print("Command:", target)

        return JointCommand(joint_angles=target.tolist())
    def build_observation(self, state):
        joints = np.array(state["joints"][:self.N])
        imu = np.array(state["imu"])

        accel = imu[0:3]
        gyro = imu[3:6]
        mag = imu[6:9]

        obs = np.concatenate([
            joints,
            gyro,
            accel,
            mag,
            self.last_action
        ])

        return obs