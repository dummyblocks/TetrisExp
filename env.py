import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tetris


class SinglePlayerTetris(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None) -> None:
        super().__init__()
        self.w = 10
        self.h = 20
        self.tile_size = 20
        self.preview_num = 5
        self.game = tetris.Game(self.w, self.h*2, self.tile_size, self.preview_num, self.metadata["render_fps"])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(8)
        # 0: noop, 1: left, 2: right, 3: ccw, 4: cw, 5: softdrop, 6: harddrop, 7: hold

        self.observation_space = spaces.Dict({
            "field": spaces.Box(0, 1, (self.h, self.w), dtype=int),
            "current_mino": spaces.Box(0, 1, (4, 4), dtype=int),
            "hold_mino": spaces.Box(0, 1, (4, 4), dtype=int),
            "preview_minos": spaces.Box(0, 1, (4*self.preview_num, 4), dtype=int),
            
        })

    def step()