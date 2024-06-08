import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tetris
from collections import deque
import pygame as pg
import time

def _recur_search_pos(system, prog, last, visited, s, a):
    mino = system.get_current_mino()
    visited.append((mino.center.x, mino.center.y, mino.rotation_status))
    obs = get_obs_from_system(system)
    print((mino.center.x, mino.center.y, mino.rotation_status), prog)
    for i in range(6):
        # 0: left, 1: right, 2: hard, 3: soft, 4: CCW, 5: CW
        if i == 3:
            continue
        if (i == 1 and last == 0) or (i == 0 and last == 1) or (i == 5 and last == 4) or (i == 4 and last == 5):
            continue
        if i == 2:
            s.append(obs)
            a.append(prog + [2])
            del system, prog
            return
        new_system = system.copy()
        match i:
            case 0:
                new_system.try_move_left()
            case 1:
                new_system.try_move_right()
            case 4:
                new_system.try_rotate_rcw()
            case 5:
                new_system.try_rotate_cw()
        new_mino = new_system.get_current_mino()
        if (new_mino.center.x, new_mino.center.y, new_mino.rotation_status) in visited:
            continue
        _recur_search_pos(new_system, prog + [i], i, visited, s, a)
        
def get_obs_from_system(system):
    field_with_cur_mino = np.array(system.field[20:], dtype=bool) * 1.0 # 3
    mino = system.get_current_mino()
    
    for block in mino.blocks:
        if block.x >= 0 and block.y >= 0:
            field_with_cur_mino[block.y, block.x] = 0.7 #2

    left_disp = np.zeros((20, 5),dtype=float)
    holdblock = system.get_hold_mino()
    if holdblock:
        for block in holdblock.blocks:
            left_disp[1+block.y, 1+block.x] = 1.0 # 1

    right_disp = np.zeros((20, 5),dtype=float)
    for idx, mn in enumerate(system.get_preview_mino_list()):
        for block in mn.blocks:
            right_disp[1+idx*4+block.y, 2+block.x] = 1.0 # 1

    field_with_cur_mino = np.concatenate((left_disp, field_with_cur_mino, right_disp),axis=1)

    mino_id = system._curr_mino_num
    mino_pos = np.array([mino.center.x, mino.center.y], dtype=np.int64)
    mino_rot = mino.rotation_status % 4

    hold_id = system._hold_mino_num
    if hold_id is False:
        hold_id = 7
    preview_ids = np.array(
        (system._bag + system._next_bag)[: 5],
        dtype=np.int64,
    )

    holdpossible = 0 if system._hold_used else 1
    system.hard_drop()
    lc = system.last_line_cleared
    system.last_line_cleared = 0
    ls = system.last_lines_sent
    system.last_lines_sent = 0
    il = system.incoming_garbage_next + system.receive_queue_lines
    state = {
        "image": field_with_cur_mino.reshape((1, *field_with_cur_mino.shape)).astype(float),
        "mino_pos": mino_pos,
        "mino_rot": mino_rot,
        "mino": mino_id,
        "hold": hold_id,
        "preview": preview_ids,
        "status": np.array([
            lc,
            ls,
            il,
            holdpossible,
        ], dtype=np.int64),  # line cleared, lines sent, current line queue, hold possible
    }
    return state

def calc_height_bumpiness(field):
    count = 0
    x = list(range(field.shape[1]))
    heights = [0] * field.shape[1]
    for y in range(field.shape[0]-1,-1,-1):
        for _x in x:
            if _x >= 0 and field[y,_x] > 0:
                x[_x]=-1
                heights[_x] = y + 1
    for x in range(field.shape[1]-1):
        count += np.abs(heights[x+1] - heights[x])
    return np.sum(heights), count

def count_holes(field):
    _field = np.ones((field.shape[0]+2,field.shape[1]+2),dtype=np.int8)
    _field[-1,:] = np.zeros((field.shape[1]+2))
    _field[1:-1,1:-1] = field.astype(np.int8)
    queue = deque()
    delta = [(-1,0),(1,0),(0,1),(0,-1)]
    visit = []
    count = 2
    queue.append((0,0))
    while len(queue) > 0:
        y, x = queue.popleft()
        for dy, dx in delta:
            new = (y+dy, x+dx)
            if 0  <= new[0] < _field.shape[0] and 0 <= new[1] < _field.shape[1] \
               and _field[new[0],new[1]] == 1 and new not in visit:
                visit.append(new)
                queue.append(new)
                count += 1
    queue.append((field.shape[0]+1,0))
    while len(queue) > 0:
        y, x = queue.popleft()
        for dy, dx in delta:
            new = (y+dy, x+dx)
            if 0  <= new[0] < _field.shape[0] and 0 <= new[1] < _field.shape[1] \
               and _field[new[0],new[1]] == 0 and new not in visit:
                visit.append(new)
                queue.append(new)
                count += 1
    return (field.shape[0] + 2)*(field.shape[1] + 2) - count


class TetrisWrapper(gym.wrappers.FlattenObservation):
    def __init__(self, env):
        super().__init__(env)

    def get_all_next_hd(self):
        states, actions = self.env.get_all_next_hd()
        flattened = [self.observation(state) for state in states]
        return flattened, actions


class SinglePlayerTetris(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, fps=3, fast_soft=False, draw_ghost=True, auto_drop=False, draw_hold_next=True, enable_no_op=False) -> None:
        super().__init__()
        self.w = 10
        self.h = 20
        self.tile_size = 20
        self.screen_margin = 10
        self.preview_num = 5
        self.main_screen = None
        self.clock = None
        self.fps = fps  # three operation in 1 second
        self.fast_sd = fast_soft
        self.draw_ghost = draw_ghost
        self.auto_drop = auto_drop
        self.draw_hold_next = draw_hold_next

        self.game = tetris.Game(
            self.w,
            self.h,
            self.tile_size,
            self.preview_num,
            self.fps,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(7 + (1 if enable_no_op else 0))
        # 0: left, 1: right, 2: hard, 3: soft, 4: CCW, 5: CW, #6: noop, 7: hold

        self.observation_space = spaces.Dict(
            {
                #"field": spaces.Box(0, 3, (self.h, self.w), dtype=np.int64),
                #"field_view": spaces.Box(0, 3, (self.h, self.w), dtype=np.int64),
                #"image": spaces.Box(0, 255, (1, self.h*2, self.h*2), dtype=np.uint8),
                "image": spaces.Box(0.0, 1.0, (1, self.h, self.h), dtype=float),
                "mino_pos": spaces.Box(
                    np.array([0, 0]),
                    np.array([self.w - 1, 2 * self.h - 1]),
                    dtype=np.int64,
                ),
                "mino_rot": spaces.Discrete(4),
                "mino": spaces.Discrete(7),
                "hold": spaces.Discrete(8, start=0),
                "preview": spaces.MultiDiscrete([7] * self.preview_num),
                "status": spaces.Box(
                    np.array([0, 0, 0, 0]), np.array([4, 100, 100, 1]), dtype=np.int64
                ),  # line cleared, lines sent, current line queue, hold possible
            }  # combo, b2b, ......
        )

        self.action_fp = [
                self.game.system.try_move_left,
                self.game.system.try_move_right,
                self.game.system.hard_drop,
                self.game.system.try_soft_drop,
                self.game.system.try_rotate_rcw,
                self.game.system.try_rotate_cw,
                self.game.system.hold,
            ]
        if enable_no_op:
            self.action_fp = self.action_fp[0:6] + [(lambda *_: None)] + self.action_fp[6:]

        if self.fast_sd:
            self.action_fp[3]=self.game.system.fast_soft_drop


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward = 0
        self.game.system.init()
        state = self._get_obs_from_game()
        if self.render_mode == "human":
            self.render()
        return state, {}

    def _get_obs_from_game(self):
        #field = np.array(self.game.system.field[self.h :], dtype=bool) * 3
        field_with_cur_mino = np.array(self.game.system.field[self.h :], dtype=bool) * 1.0 # 3

        self.agg_height, self.bumpiness = calc_height_bumpiness(field_with_cur_mino)
        self.holes = count_holes(field_with_cur_mino)
        print(self.agg_height, self.holes, self.bumpiness)

        mino = self.game.system.get_current_mino()
        
        for block in mino.blocks:
            if block.x >= 0 and block.y >= 0:
                field_with_cur_mino[block.y, block.x] = 0.7 #2
        if self.draw_ghost:
            for block in self.game.system.get_ghost_mino().blocks:
                if block.x >= 0 and block.y >= 0:
                    field_with_cur_mino[block.y, block.x] = 0.3 # 1
        
        if self.draw_hold_next:
            left_disp = np.zeros((20, 5),dtype=float)
            holdblock = self.game.system.get_hold_mino()
            if holdblock:
                for block in holdblock.blocks:
                    left_disp[1+block.y, 1+block.x] = 1.0 # 1

            right_disp = np.zeros((20, 5),dtype=float)
            for idx, mn in enumerate(self.game.system.get_preview_mino_list()):
                for block in mn.blocks:
                    right_disp[1+idx*4+block.y, 2+block.x] = 1.0 # 1
            field_with_cur_mino = np.concatenate((left_disp, field_with_cur_mino, right_disp),axis=1)

        mino_id = self.game.system._curr_mino_num
        mino_pos = np.array([mino.center.x, mino.center.y], dtype=np.int64)
        mino_rot = mino.rotation_status % 4

        hold_id = self.game.system._hold_mino_num
        if hold_id is False:
            hold_id = 7
        preview_ids = np.array(
            (self.game.system._bag + self.game.system._next_bag)[: self.preview_num],
            dtype=np.int64,
        )

        lc = self.game.system.last_line_cleared
        self.last_line_cleared = lc
        self.game.system.last_line_cleared = 0
        ls = self.game.system.last_lines_sent
        self.last_lines_sent = ls
        self.game.system.last_lines_sent = 0
        il = (
            self.game.system.incoming_garbage_next
            + self.game.system.receive_queue_lines
        )
        holdpossible = 0 if self.game.system._hold_used else 1

        state = {
            #"field": field,
            #"field_view": field_with_cur_mino,
            "image": field_with_cur_mino.reshape((1, *field_with_cur_mino.shape)).astype(float),
            #"image": (255*np.kron(field_with_cur_mino.reshape((1, *field_with_cur_mino.shape)).astype(float),np.ones((2,2),float))).astype(np.uint8),
            "mino_pos": mino_pos,
            "mino_rot": mino_rot,
            "mino": mino_id,
            "hold": hold_id,
            "preview": preview_ids,
            "status": np.array([
                lc,
                ls,
                il,
                holdpossible,
            ], dtype=np.int64),  # line cleared, lines sent, current line queue, hold possible
        }

        return state

    def step(self, action):

        self.game.system.frame_check(self.fps,auto_drop=self.auto_drop)
        self.action_fp[action]()

        #line_send = self.game.system.outgoing_garbage_send()
        #self.game.system.receive_garbage(line_send)
        self.state = self._get_obs_from_game()
        self.reward = self.get_reward()
        terminated = self.game.system.is_game_over()
        if terminated:
            self.reward -= 10
        #if self.reward == 0:
        #    self.reward = -0.01
        if self.render_mode == "human":
            self.render()
        return self.state, self.reward, terminated, False, {}

    def get_reward(self):
        return 4 * (self.last_line_cleared + self.last_lines_sent) ** 1.5 # + self.game.system.outgoing_linedown_send()*0.1
        return (self.last_line_cleared + self.last_lines_sent) ** 2 #+ self.game.system.outgoing_linedown_send()*0.25
    
    def get_all_next_hd(self):
        states = []
        group_actions = []
        dx = self.w // 2
        visited = []
        for rot in range(-2, 3):
            for delta_x in range(-dx, dx+1):
                mino = self.game.system.get_current_mino()
                print(mino.center.x, mino.center.y)
                visited.append([(mino.center.x, mino.center.y, mino.rotation_status)])
        for rot in range(-2, 3):
            for delta_x in range(-dx,dx+1):  
                # 0: left, 1: right, 2: hard, 3: soft, 4: CCW, 5: CW, #6: noop, 7: hold 
                _system = self.game.system.copy()
                actions = []
                if rot < 0:
                    for _ in range(-rot):
                        _system.try_rotate_rcw()
                        actions.append(4)
                else:
                    for _ in range(rot):
                        _system.try_rotate_cw()
                        actions.append(5)
                if delta_x < 0:
                    for _ in range(-delta_x):
                        _system.try_move_left()
                        actions.append(0)
                else:
                    for _ in range(delta_x):
                        _system.try_move_right()
                        actions.append(1)
                y = _system.get_current_mino().center.y
                _system.fast_soft_drop()
                ny = _system.get_current_mino().center.y
                actions.extend([3] * int(y - ny))
                _recur_search_pos(_system, actions, None, visited, states, group_actions)

                _system = self.game.system.copy()
                _system.hold()
                actions = [6]
                if rot < 0:
                    for _ in range(-rot):
                        _system.try_rotate_rcw()
                        actions.append(4)
                else:
                    for _ in range(rot):
                        _system.try_rotate_cw()
                        actions.append(5)
                if delta_x < 0:
                    for _ in range(-delta_x):
                        _system.try_move_left()
                        actions.append(0)
                else:
                    for _ in range(delta_x):
                        _system.try_move_right()
                        actions.append(1)
                y = _system.get_current_mino().center.y
                _system.fast_soft_drop()
                ny = _system.get_current_mino().center.y
                actions.extend([3] * int(y - ny))
                _recur_search_pos(_system, actions, None, visited, states, group_actions)

        return states, group_actions
    
    def calculate_finesse(self):
        return 0


    def render(self):
        if self.render_mode is None:
            return
        if self.main_screen is None:
            pg.init()
            pg.display.init()
            self.top_margin = self.bot_margin = self.left_margin = self.right_margin = (
                4 * self.screen_margin
            )

            ## area
            self.hold_area = (5 * self.tile_size, 3 * self.tile_size)
            self.info_area = (5 * self.tile_size, 10 * self.tile_size)
            self.play_area = (self.w * self.tile_size, self.h * self.tile_size)
            self.preview_area = (
                5 * self.tile_size,
                3 * self.tile_size * self.preview_num,
            )
            _main_x_size = (
                self.left_margin
                + self.hold_area[0]
                + self.screen_margin
                + self.play_area[0]
                + self.screen_margin
                + self.preview_area[0]
                + self.right_margin
            )
            _main_y_size = self.top_margin + self.play_area[1] + self.bot_margin
            main_area = (_main_x_size, _main_y_size)
            if self.clock is None:
                self.clock = pg.time.Clock()
            self.main_screen = pg.display.set_mode(main_area)
            self.background_screen = pg.Surface(main_area)
            self.hold_screen = pg.Surface(self.hold_area)
            self.info_screen = pg.Surface(self.info_area)
            self.play_screen = pg.Surface(self.play_area)
            self.preview_screen = pg.Surface(self.preview_area)
            self.info_font = pg.font.SysFont("Cambria", self.screen_margin, bold=True)

        if self.render_mode == "human":
            self.main_screen.blit(self.background_screen, (0, 0))
            self.main_screen.blit(self.hold_screen, (self.left_margin, self.top_margin))
            self.main_screen.blit(
                self.info_screen,
                (
                    self.left_margin,
                    self.top_margin + self.hold_area[1] + self.screen_margin,
                ),
            )
            self.main_screen.blit(
                self.play_screen,
                (
                    self.left_margin + self.hold_area[0] + self.screen_margin,
                    self.top_margin,
                ),
            )
            self.main_screen.blit(
                self.preview_screen,
                (
                    self.left_margin
                    + self.hold_area[0]
                    + self.screen_margin
                    + self.play_area[0]
                    + self.screen_margin,
                    self.top_margin,
                ),
            )
            game = self.game
            # info fill
            self.info_screen.fill(pg.Color("BLACK"))
            if game.system.combo_count > 0:
                self.info_screen.blit(
                    self.info_font.render(
                        str(f"Combo : {game.system.combo_count}"), 1, pg.Color("GRAY")
                    ),
                    (self.screen_margin, self.screen_margin),
                )
            self.info_screen.blit(
                self.info_font.render(
                    f"Pieces : {game.system.used_mino_count}", 1, pg.Color("GRAY")
                ),
                (self.screen_margin, self.screen_margin * 5),
            )
            self.info_screen.blit(
                self.info_font.render(
                    f"Score : {game.system.total_score}", 1, pg.Color("GRAY")
                ),
                (self.screen_margin, self.screen_margin * 9),
            )
            self.info_screen.blit(
                self.info_font.render(
                    f"Lines Sent : {game.system.total_lines_sent}", 1, pg.Color("GRAY")
                ),
                (self.screen_margin, self.screen_margin * 10),
            )
            self.info_screen.blit(
                self.info_font.render(
                    f"Last Sent : {game.system.last_lines_sent}", 1, pg.Color("YELLOW")
                ),
                (self.screen_margin, self.screen_margin * 11),
            )

            self.info_screen.blit(
                self.info_font.render(
                    f"Garbage Queue : {game.system.receive_queue_lines}",
                    1,
                    pg.Color("ORANGE"),
                ),
                (self.screen_margin, self.screen_margin * 13),
            )
            self.info_screen.blit(
                self.info_font.render(
                    f"Garbage Next : {game.system.incoming_garbage_next}",
                    1,
                    pg.Color("ORANGE"),
                ),
                (self.screen_margin, self.screen_margin * 14),
            )
            self.info_screen.blit(
                self.info_font.render(
                    f"Reward : {self.reward}",
                    1,
                    pg.Color("YELLOW"),
                ),
                (self.screen_margin, self.screen_margin * 16),
            )
            if self.reward > 0:
                print('rew:',self.reward)
            self.background_screen.fill(pg.Color("gray20"))
            self.hold_screen.fill(pg.Color("BLACK"))
            self.play_screen.fill(pg.Color("BLACK"))
            self.preview_screen.fill(pg.Color("BLACK"))

            game.painter.draw_ghost_mino(self.play_screen, game.system.get_ghost_mino())
            game.painter.draw_current_mino(
                self.play_screen, game.system.get_current_mino()
            )
            game.painter.draw_hold_mino(self.hold_screen, game.system.get_hold_mino())
            game.painter.draw_preview_mino_list(
                self.preview_screen, game.system.get_preview_mino_list()
            )
            game.painter.draw_field(self.play_screen, game.system.get_draw_field())
            game.painter.draw_grid(self.play_screen)

            self.clock.tick(self.metadata["render_fps"])
            ## pygame display update
            pg.display.flip()

def handle_human_input(env, long_move=False, long_drop=False):
    action = 6
    for event in pg.event.get():
        if event.type == pg.QUIT:
            env.close()
            exit()
        #action = 6
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_a:
                if long_move:
                    env.game.system.turn_on_auto_move_left()
                action = 0
            elif event.key == pg.K_d:
                if long_move:
                    env.game.system.turn_on_auto_move_right()
                action = 1
            elif event.key == pg.K_w:
                action = 2
            elif event.key == pg.K_s:
                if long_drop:
                    env.game.system.turn_on_sdf()
                action = 3
            elif event.key == pg.K_q:
                action = 4
            elif event.key == pg.K_e:
                action = 5
            elif event.key == pg.K_x:
                action = 7
            elif event.key == pg.K_ESCAPE:
                env.close()
                exit()
        if event.type == pg.KEYUP:
            if event.key == pg.K_a and long_move:
                env.game.system.turn_off_auto_move_left()
            elif event.key == pg.K_d and long_move:
                env.game.system.turn_off_auto_move_right()
            elif event.key == pg.K_s and long_drop:
                env.game.system.turn_off_sdf()
        #return action
    return action


if __name__ == "__main__":

    control = True
    fps = 60
    env = SinglePlayerTetris(render_mode="human",fps=fps,fast_soft=False,draw_ghost=True,auto_drop=True, draw_hold_next=True,enable_no_op=True)
    print(env.reset())
    while True:

        if control:
            action = handle_human_input(env, long_drop=True, long_move=True)
        else:
            pg.event.pump()
            action = env.action_space.sample()
        # 0: left, 1: right, 2: hard, 3: soft, 4: CCW, 5: CW, 6: noop, 7: hold

        state, reward, term, trunc, info = env.step(action)
        #print(state['image'].astype(np.bool_)*1)
        if term:
            env.reset()
        time.sleep(1./fps)
