import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tetris
import pygame as pg
import time

class SinglePlayerTetris(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, fps=3, fast_soft=False, draw_ghost=True, auto_drop=False, draw_hold_next=True) -> None:
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

        self.action_space = spaces.Discrete(7)
        # 0: left, 1: right, 2: hard, 3: soft, 4: CCW, 5: CW, #6: noop, 7: hold

        self.observation_space = spaces.Dict(
            {
                #"field": spaces.Box(0, 3, (self.h, self.w), dtype=np.int64),
                #"field_view": spaces.Box(0, 3, (self.h, self.w), dtype=np.int64),
                #"image": spaces.Box(0, 255, (1, self.h*4, self.w*4), dtype=np.uint8),
                "image": spaces.Box(0, 255, (1, self.h*4, self.h*4), dtype=np.uint8),
                "mino_pos": spaces.Box(
                    np.array([0, 0]),
                    np.array([self.w - 1, 2 * self.h - 1]),
                    dtype=np.int64,
                ),
                "mino_rot": spaces.Box(0, 3, (1,), dtype=np.int64),
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
                # (lambda *_: None),
                self.game.system.hold,
            ]
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
        field = np.array(self.game.system.field[self.h :], dtype=bool) * 3
        field_with_cur_mino = np.array(self.game.system.field[self.h :], dtype=bool) * 255 # 3

        mino = self.game.system.get_current_mino()
        
        for block in mino.blocks:
            if block.x >= 0 and block.y >= 0:
                field_with_cur_mino[block.y, block.x] = 180 #2
        if self.draw_ghost:
            for block in self.game.system.get_ghost_mino().blocks:
                if block.x >= 0 and block.y >= 0:
                    field_with_cur_mino[block.y, block.x] = 110 # 1
        
        if self.draw_hold_next:
            left_disp = np.zeros((20, 5),dtype=np.uint8)
            holdblock = self.game.system.get_hold_mino()
            if holdblock:
                for block in holdblock.blocks:
                    if block.x >= 0 and block.y >= 0:
                        left_disp[block.y, block.x] = 255 # 1

            right_disp = np.zeros((20, 5),dtype=np.uint8)
            for idx, mn in enumerate(self.game.system.get_preview_mino_list()):
                for block in mn.blocks:
                    if block.x >= 0 and block.y >= 0:
                        right_disp[idx*4+block.y, block.x] = 255 # 1
            field_with_cur_mino = np.concatenate((left_disp, field_with_cur_mino, right_disp),axis=1)

        mino_id = self.game.system._curr_mino_num
        mino_pos = np.array([mino.center.x, mino.center.y], dtype=np.int64)
        mino_rot = np.array([mino.rotation_status % 4], dtype=np.int64)

        hold_id = self.game.system._hold_mino_num
        if hold_id == False:
            hold_id = 0
        else:
            hold_id += 1
        preview_ids = np.array(
            (self.game.system._bag + self.game.system._next_bag)[: self.preview_num],
            dtype=np.int64,
        )
        if len(self.game.system._bag + self.game.system._next_bag) < self.preview_num:
            print(self.game.system._bag, self.game.system._next_bag)
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
            "image": np.kron(field_with_cur_mino.reshape((1, *field_with_cur_mino.shape)).astype(np.uint8),np.ones((4,4),np.uint8)),
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
        return 4 * (self.last_line_cleared + self.last_lines_sent) ** 1.5 + self.game.system.outgoing_linedown_send()*0.1
        return (self.last_line_cleared + self.last_lines_sent) ** 2 #+ self.game.system.outgoing_linedown_send()*0.25

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
    env = SinglePlayerTetris(render_mode="human",fps=fps,fast_soft=False,draw_ghost=True,auto_drop=True)
    print(env.reset())
    while True:

        if control:
            action = handle_human_input(env, long_drop=True, long_move=True)
        else:
            pg.event.pump()
            action = env.action_space.sample()
        # 0: left, 1: right, 2: hard, 3: soft, 4: CCW, 5: CW, 6: noop, 7: hold

        state, reward, term, trunc, info = env.step(action)
        # print(state["field_view"])
        if term:
            env.reset()
        time.sleep(1./fps)
