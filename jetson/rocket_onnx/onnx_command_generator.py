import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from typing import List
from rocket.rocket_state import RocketState
from threading import Event


@dataclass
class JointCommand:
    joint_angles: List[float]



@dataclass
class RocketOnnxContext:
    """data class to hold runtime data needed by the controller"""

    event: Event = field(default_factory=Event)
    latest_state: RocketState = field(default_factory=RocketState)
    count: int = 0

        
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
        obs = self.state.latest_state.to_observation()
        print("Motor positions", obs[0:7])
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
        self.state.latest_state.update_servo_position({"joints": target.tolist()[:2]})
        return JointCommand(joint_angles=target.tolist())