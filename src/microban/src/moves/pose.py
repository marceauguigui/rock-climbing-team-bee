import numpy as np
from microban_model import JOINTS

class PoseMove:
    description = "One slider per joint: offset from the start pose (rad)"
    kp = 125
    options, option_docs = {}, {}
    sliders = {j: (-1.0, 1.0, 0.0) for j in JOINTS}

    def start(self, obs):
        self.q0 = np.array(obs["q"], dtype=float)
        self.last = None

    def step(self, obs):
        offsets = {j: round(float(self.slider[j]), 2) for j in JOINTS}
        if offsets != self.last:
            self.last = offsets
            print({j: v for j, v in offsets.items() if v}, end="\r\n", flush=True)
        return self.q0 + np.array([offsets[j] for j in JOINTS])

