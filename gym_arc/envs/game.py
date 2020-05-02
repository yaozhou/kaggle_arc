import pygame
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pdb

#05f2a901.json
#06df4c85.json
#08ed6ac7.json
#0d3d703e.json
#178fcbfb.json
#1a07d186.json

COLOR_PALETTE = [
    pygame.Color(0, 0, 0),
    pygame.Color(0, 116, 217),
    pygame.Color(255, 65, 54),
    pygame.Color(46, 204, 64),
    pygame.Color(255, 221, 0),
    pygame.Color(170, 170, 170),
    pygame.Color(240, 18, 190),
    pygame.Color(255, 133, 27),
    pygame.Color(127, 219, 255),
    pygame.Color(135, 12, 37)
]

GRID_LENGTH = 30
PAD_LENGTH = 50
MAX_SHAPLE_NUM = 3
MAX_FEATURE_NUM = 14
COLOR_EMPTY = COLOR_PALETTE[0]
COLOR_SELECTED = pygame.Color(0x99,0x66,0xff)

class Grid:
    def __init__(self, x, y, idx, color, shape):
        self.x = x
        self.y = y
        self.idx = idx
        self.color = color
        self.shape = shape

    def move_vert(self, delta):
        self.y += delta
        self.idx += self.shape.game_engine.width * delta

    def move_hori(self, delta):
        self.x += delta
        self.idx += delta

class Shape:
    def __init__(self, point_list, idx, game_engine):
        self.grid_list = []
        self.idx = idx
        self.game_engine = game_engine

class GameEngine:
    ACTION_SEL_0 = 0
    ACTION_SEL_1 = 1
    ACTION_SEL_2 = 2
    ACTION_SEL_3 = 3
    ACTION_SEL_4 = 4
    ACTION_ATTENSION_TOP = 5
    ACTION_ATTENSION_BOTTOM = 6
    ACTION_ATTENSION_LEFT = 7
    ACTION_ATTENSION_RIGHT = 8
    ACTION_MOVE_UNTIL_COLISSION = 9

    ACTION_SEL_COLOR_0 = 10
    ACTION_SEL_COLOR_1 = 11
    ACTION_SEL_COLOR_2 = 12
    ACTION_SEL_COLOR_3 = 13
    ACTION_SEL_COLOR_4 = 14
    ACTION_SEL_COLOR_5 = 15
    ACTION_SEL_COLOR_6 = 16
    ACTION_SEL_COLOR_7 = 17
    ACTION_SEL_COLOR_8 = 18
    ACTION_SEL_COLOR_9 = 19
    ACITON_CONVERT_COLOR = 20

    DIRECTION_TOP = 1
    DIRECTION_BOTTOM = 2
    DIRECTION_LEFT = 3
    DIRECTION_RIGHT = 4

    def __init__(self, input, output, need_ui):
        
        self.width = len(input[0])
        self.height = len(input)
        self.action_n = self.ACTION_MOVE_UNTIL_COLISSION + 1
        self.input = np.array(input).flatten()
        self.answer = np.array(output).flatten()
        self.finish_score = self.width * self.height
        self.need_ui = need_ui
        if (need_ui):
            pygame.init()
            self.screen = pygame.display.set_mode([self.width * GRID_LENGTH + PAD_LENGTH * 2, 
                    self.height * GRID_LENGTH + PAD_LENGTH * 2])
        
        #self.reset()

    def reset(self):
        self.shape_list = []
        self.cur_sel = -1
        self.cur_sel_color = -1
        self.cur_attension = 0
        self.cur_score = self.calc_state_score(self.input, self.answer)
        #print('total score (%s) initial score(%d) ' % (self.width * self.height, self.cur_score))

        self.init_shape_list_from_input(self.input)
        self.features = self.shape_list_2_feature()
        return self.features

    def shape_2_feature(self, shape):
        feature = np.zeros(MAX_FEATURE_NUM, dtype=np.float32)

        left = right = top = bottom = -1

        for grid in shape.grid_list:
            if (left == -1 or grid.x < left):
                left = grid.x
            
            if (right == -1 or grid.x > right):
                right = grid.x

            if (top == -1 or grid.y < top):
                top = grid.y

            if (bottom == -1 or grid.y > bottom):
                bottom = grid.y

        width = right - left + 1
        height = bottom - top + 1
        ratio = width / height

        #print(left, right, top, bottom)

        feature[0] = shape.grid_list[0].color  / 10.0
        feature[1] = len(shape.grid_list) / 10.0
        feature[2] = left / 10.0
        feature[3] = right / 10.0
        feature[4] = top / 10.0
        feature[5] = bottom / 10.0
        feature[6] = width / 10.0
        feature[7] = height / 10.0
        feature[8] = ratio
        feature[9] = (1 if self.cur_sel == shape.idx else 0)
        feature[10] = (1 if self.cur_attension == self.DIRECTION_TOP else 0)
        feature[11] = (1 if self.cur_attension == self.DIRECTION_BOTTOM else 0)
        feature[12] = (1 if self.cur_attension == self.DIRECTION_LEFT else 0)
        feature[13] = (1 if self.cur_attension == self.DIRECTION_RIGHT else 0)

        return feature


    def shape_list_2_feature(self):
        features = np.zeros((MAX_SHAPLE_NUM, MAX_FEATURE_NUM), dtype=np.float32)

        for idx, shape in enumerate(self.shape_list):
            features[idx] = self.shape_2_feature(shape)

        # if (self.cur_attension == self.DIRECTION_TOP):
        #     features[-1][0] = 1
        # elif (self.cur_attension == self.DIRECTION_BOTTOM):
        #     features[-2][0] = 1
        # elif (self.cur_attension == self.DIRECTION_LEFT):
        #     features[-3][0] = 1
        # elif (self.cur_attension == self.DIRECTION_RIGHT):
        #     features[-4][0] = 1

        return features

    def calc_state_score(self, state, answer):
        #print(state)
        return np.sum(state == answer)

    def shape_list_2_state(self):
        state = np.zeros(self.width * self.height, dtype=np.int64)
        #pdb.set_trace()
        for shape in self.shape_list:
            for grid in shape.grid_list:
                state[grid.idx] = grid.color

        return np.array(state)
    
    def calc_current_score(self):
        cur_state = self.shape_list_2_state()
        #print(cur_state)
        score = self.calc_state_score(cur_state, self.answer)
        return score

    def init_shape_list_from_input(self, input):
        input = np.array(input).flatten()
        #print(input)
        G = nx.Graph()

        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                color = input[idx]
                if (color == 0): continue
                
                up = (i - 1) * self.width + j
                down = (i + 1) * self.width + j
                left = i * self.width + j - 1
                right = i * self.width + j + 1
                up_left = up - 1
                up_right = up + 1
                down_left = down - 1
                down_right = down + 1

                if (i == 0): up = up_left = up_right = -1
                if (i == self.height - 1): down = down_left = down_right = -1
                if (j == 0): left = up_left = down_left = -1
                if (j == self.width - 1): right = up_right = down_right = -1

                #print(up, down, left, right, up_left, up_right, down_left, down_right)

                G.add_node(idx)

                if (up >= 0 and color == input[up]):
                    G.add_edge(idx, up)

                if (up_left >= 0 and color == input[up_left]):
                    G.add_edge(idx, up_left)
                
                if (up_right >= 0 and color == input[up_right]):
                    G.add_edge(idx, up_right)

                if (down >= 0 and color == input[down]):
                    G.add_edge(idx, down)

                if (down_left >= 0 and color == input[down_left]):    
                    G.add_edge(idx, down_left)

                if (down_right >= 0 and color == input[down_right]):
                    G.add_edge(idx, down_right)

                if (left >= 0 and color == input[left]):
                    G.add_edge(idx, left)

                if (right >= 0 and color == input[right]):
                    G.add_edge(idx, right)

        components = np.array(list(nx.connected_components(G)))
        for component_idx, component in enumerate(components):
            shape = Shape([], component_idx, self)

            for grid_idx in list(component):
                i = grid_idx // self.width
                j = grid_idx % self.width
                grid = Grid(j, i, grid_idx, input[grid_idx], shape)
                shape.grid_list.append(grid)

            self.shape_list.append(shape)

    def select_shape(self, idx):
        self.cur_sel = idx

    def select_direct_attension(self, direction):
        self.cur_attension = direction

    def get_shape_idx_from_xy(self, x, y):
        for shape in self.shape_list:
            for grid in shape.grid_list:
                if (grid.x == x and grid.y == y):
                    return grid.shape.idx

        return -1

    # move
    def move_vert_until_collision(self, down):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        shape = self.shape_list[self.cur_sel]
        stop = False
        delta = 0

        while True:
            for grid in shape.grid_list:
                y = grid.y + delta + (1 if down else -1)
                if (y > self.height - 1 or y < 0):
                    stop = True
                    break

                shape_idx = self.get_shape_idx_from_xy(grid.x, y)
                if (shape_idx >= 0 and shape_idx != self.cur_sel):
                    stop = True
                    break
            if (stop): break
            delta += (1 if down else -1)

        for grid in shape.grid_list:
            grid.move_vert(delta)

    # move hori
    def move_hori_until_collision(self, right):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        shape = self.shape_list[self.cur_sel]
        stop = False
        delta = 0

        while True:
            for grid in shape.grid_list:
                x = grid.x + delta + (1 if right else -1)
                if (x > self.width - 1 or x < 0):
                    stop = True
                    break

                shape_idx = self.get_shape_idx_from_xy(x, grid.y)
                if (shape_idx >= 0 and shape_idx != self.cur_sel):
                    stop = True
                    break

            if (stop): break
            delta += (1 if right else -1)

        for grid in shape.grid_list:
            grid.move_hori(delta)

    def select_color(self, color):
        self.cur_sel_color = color

    def convert_2_color(self):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return
        if (self.cur_sel_color < 0 or self.cur_sel_color > self.ACTION_SEL_COLOR_9): return

        shape = self.shape_list[self.cur_sel]

        for grid in shape.grid_list:
            grid.color = self.cur_sel_color

    def move_until_collision(self):
        #pdb.set_trace()
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        if (self.cur_attension == self.DIRECTION_BOTTOM):
            self.move_vert_until_collision(True)
        elif (self.cur_attension == self.DIRECTION_TOP):
            self.move_vert_until_collision(False)
        elif (self.cur_attension == self.DIRECTION_RIGHT):
            self.move_hori_until_collision(True)
        elif (self.cur_attension == self.DIRECTION_LEFT):
            self.move_hori_until_collision(False)

    def do_action(self, action):
        #pdb.set_trace()
        if (action == GameEngine.ACTION_SEL_0):
            self.select_shape(0)
        elif (action == GameEngine.ACTION_SEL_1):
            self.select_shape(1)
        elif (action == GameEngine.ACTION_SEL_2):
            self.select_shape(2)
        elif (action == GameEngine.ACTION_SEL_3):
            self.select_shape(3)
        elif (action == GameEngine.ACTION_SEL_4):
            self.select_shape(4)
        elif (action == GameEngine.ACTION_ATTENSION_TOP):
            self.select_direct_attension(self.DIRECTION_TOP)
        elif (action == GameEngine.ACTION_ATTENSION_BOTTOM):
            self.select_direct_attension(self.DIRECTION_BOTTOM)
        elif (action == GameEngine.ACTION_ATTENSION_LEFT):
            self.select_direct_attension(self.DIRECTION_LEFT)
        elif (action == GameEngine.ACTION_ATTENSION_RIGHT):
            self.select_direct_attension(self.DIRECTION_RIGHT)
        elif (action == GameEngine.ACTION_MOVE_UNTIL_COLISSION):
            self.move_until_collision()
        elif (action == GameEngine.ACTION_SEL_COLOR_0):
            self.select_color(0)
        elif (action == GameEngine.ACTION_SEL_COLOR_1):
            self.select_color(1)
        elif (action == GameEngine.ACTION_SEL_COLOR_2):
            self.select_color(2)
        elif (action == GameEngine.ACTION_SEL_COLOR_3):
            self.select_color(3)
        elif (action == GameEngine.ACTION_SEL_COLOR_4):
            self.select_color(4)
        elif (action == GameEngine.ACTION_SEL_COLOR_5):
            self.select_color(5)
        elif (action == GameEngine.ACTION_SEL_COLOR_6):
            self.select_color(6)
        elif (action == GameEngine.ACTION_SEL_COLOR_7):
            self.select_color(7)
        elif (action == GameEngine.ACTION_SEL_COLOR_8):
            self.select_color(8)
        elif (action == GameEngine.ACTION_SEL_COLOR_9):
            self.select_color(9)
        elif (action == GameEngine.ACITON_CONVERT_COLOR):
            self.convert_2_color()

        done = False

        new_score= self.calc_current_score()
        if (new_score == self.finish_score):
            done = True
        progress = new_score - self.cur_score
        #print(new_score, self.cur_score, progress, self.finish_score)
        self.cur_score = new_score

        #print('current socre : %d progress : %d' % (self.cur_score, progress))

        self.features = self.shape_list_2_feature()

        return self.features, progress, done, None

    def draw_game(self):
        if (not self.need_ui): return
        # walkaround to fix rl main loop problem
        pygame.event.get()

        self.screen.fill(COLOR_PALETTE[0])

        # draw lines
        for i in range(self.height + 1):
            pygame.draw.line(self.screen, (255, 255, 255), (PAD_LENGTH + 0, PAD_LENGTH +i * GRID_LENGTH), 
                            (PAD_LENGTH +GRID_LENGTH * self.width, PAD_LENGTH +i * GRID_LENGTH), 1)

        for i in range(self.width + 1):
            pygame.draw.line(self.screen, (255, 255, 255), (PAD_LENGTH + i * GRID_LENGTH, PAD_LENGTH +0),
                            (PAD_LENGTH + i * GRID_LENGTH, PAD_LENGTH + self.height * GRID_LENGTH), 1)

        # draw text
        font = pygame.font.Font('freesansbold.ttf', 16)

        for shape_idx, shape in enumerate(self.shape_list):
            for grid in shape.grid_list:
                text_surface = font.render(str(shape_idx), True, (125,125,125))
                pygame.draw.rect(self.screen, COLOR_PALETTE[grid.color], pygame.Rect(PAD_LENGTH + grid.x * GRID_LENGTH,
                                PAD_LENGTH + grid.y * GRID_LENGTH, GRID_LENGTH, GRID_LENGTH))

                if (shape_idx == self.cur_sel):
                    pygame.draw.rect(self.screen, COLOR_SELECTED, (PAD_LENGTH + grid.x * GRID_LENGTH, 
                    PAD_LENGTH + grid.y * GRID_LENGTH, GRID_LENGTH, GRID_LENGTH), 1)

                self.screen.blit(text_surface, dest=(PAD_LENGTH + (grid.x + 0.5) * GRID_LENGTH, 
                            PAD_LENGTH + (grid.y + 0.5) * GRID_LENGTH))

        # draw attension
        if (self.cur_attension == GameEngine.DIRECTION_TOP):
            pygame.draw.circle(self.screen, COLOR_SELECTED, ((self.width * GRID_LENGTH + PAD_LENGTH * 2) // 2, PAD_LENGTH // 2), 15)

        elif (self.cur_attension == GameEngine.DIRECTION_BOTTOM):
            pygame.draw.circle(self.screen, COLOR_SELECTED, ((self.width * GRID_LENGTH + PAD_LENGTH * 2) // 2, 
                                PAD_LENGTH * 2 + self.height * GRID_LENGTH - PAD_LENGTH // 2 ), 15)
        elif (self.cur_attension == GameEngine.DIRECTION_LEFT):
            pygame.draw.circle(self.screen, COLOR_SELECTED, (PAD_LENGTH // 2, (PAD_LENGTH * 2 + self.height * GRID_LENGTH) // 2), 15)

        elif (self.cur_attension == GameEngine.DIRECTION_RIGHT):
            pygame.draw.circle(self.screen, COLOR_SELECTED, (self.width * GRID_LENGTH + PAD_LENGTH * 2 - PAD_LENGTH // 2, 
                                (PAD_LENGTH * 2 + self.height * GRID_LENGTH) // 2), 15)

        pygame.display.flip()

    def process_key(self, event):
        obs = reward = done = info = None
        if event.key == pygame.K_0:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_SEL_0)
        elif event.key == pygame.K_1:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_SEL_1)
        elif event.key == pygame.K_2:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_SEL_2)
        elif event.key == pygame.K_3:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_SEL_3)
        elif event.key == pygame.K_4:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_SEL_4)
        elif event.key == pygame.K_UP:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_ATTENSION_TOP)
        elif event.key == pygame.K_DOWN:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_ATTENSION_BOTTOM)
        elif event.key == pygame.K_LEFT:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_ATTENSION_LEFT)
        elif event.key == pygame.K_RIGHT:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_ATTENSION_RIGHT)
        elif event.key == pygame.K_SPACE:
            obs, reward, done, info = self.do_action(GameEngine.ACTION_MOVE_UNTIL_COLISSION)

        return obs, reward, done, info


if __name__ == "__main__":
    INPUT_FILE = '/Users/yao/develop/ARC/data/training/08ed6ac7.json'
    with open(INPUT_FILE,'r') as f:
        puzzle = json.load(f)
    puzzle_input = puzzle['train'][0]['input']
    puzzle_output = puzzle['train'][0]['output']
    game_engine = GameEngine(puzzle_input, puzzle_output, True)
    game_engine.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                obs, reward, done, info = game_engine.process_key(event)
                print(reward)

        game_engine.draw_game()

    pygame.quit()