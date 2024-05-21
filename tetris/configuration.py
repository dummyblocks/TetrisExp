from typing import List


CW = "cw"
RCW = "rcw"
I = "I"
O = "O"
S = "S"
Z = "Z"
L = "L"
J = "J"
T = "T"

SINGLE = 1
DOUBLE = 2
TRIPLE = 3
TETRIS = 4
T_SPIN = "tsp"
T_MINI = "mtsp"
D_SOFT = "sd"
D_HARD = "hd"
B2B = "b2b"
PERFECT = "pc"

lc_table = [0, 100, 300, 500, 800]
tsp_table = [400, 800, 1200, 1600, 0]
mtsp_table = [100, 200, 0, 0, 0]
drop_table = [1, 2]
b2b_bonus = 1.5

lc_linesent = [0, 0, 1, 2, 4]
tsp_linesent = [0, 2, 4, 6, 0]
mtsp_linesent = [0, 0, 0, 0, 0]
b2b_linesent = 1
pc_linesent = 10


class Pos:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def add(self, pos):
        if isinstance(pos, tuple):
            self.x += pos[0]
            self.y += pos[1]
        elif isinstance(pos, Pos):
            self.x += pos.x
            self.y += pos.y
        elif isinstance(pos, (int, float)):
            self.x += pos
            self.y += pos

    def sub(self, pos):
        if isinstance(pos, tuple):
            self.x -= pos[0]
            self.y -= pos[1]
        elif isinstance(pos, Pos):
            self.x -= pos.x
            self.y -= pos.y
        elif isinstance(pos, (int, float)):
            self.x -= pos
            self.y -= pos

    def __add__(self, pos):
        if isinstance(pos, tuple):
            return Pos(self.x + pos[0], self.y + pos[1])
        elif isinstance(pos, Pos):
            return Pos(self.x + pos.x, self.y + pos.y)
        elif isinstance(pos, (int, float)):
            return Pos(self.x + pos, self.y + pos)

    def __sub__(self, pos):
        if isinstance(pos, tuple):
            return Pos(self.x - pos[0], self.y - pos[1])
        elif isinstance(pos, Pos):
            return Pos(self.x - pos.x, self.y - pos.y)
        elif isinstance(pos, (int, float)):
            return Pos(self.x - pos, self.y - pos)

    @property
    def tuple(self):
        return (self.x, self.y)


class Mino:
    def __init__(self, name, blocks: List[Pos], center: Pos, y_offset: float = 0):
        self.name = name
        self.blocks = blocks
        self.center = center
        self.rotation_status = 0  ## 0 -> 1 -> 2 -> 3 -> 0. rotate clockwise makes + 1
        self.y_offset = y_offset
