import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List

@dataclass
class JointCommand:
    joint_angles: List[float]

class RocketOnnxPositionController:
    def __init__(self, context, config, model_path, verbose=False):
        self.context = context
        self.config = config
        self.verbose = verbose

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.N = 6
        self.last_action = np.zeros(self.N)

    def __call__(self):
        if self.context.latest_state is None:
            raise RuntimeError("No state available")

        obs = self.build_observation(self.context.latest_state)

        model_input = np.array([obs], dtype=np.float32)
        output = self.session.run(None, {self.input_name: model_input})[0][0]

        # Scale action if needed
        target = output[:self.N] * self.config.action_scale

        self.last_action = target

        if self.verbose:
            print("Target joint angles:", target)

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