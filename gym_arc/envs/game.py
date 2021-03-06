import pygame
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pdb
from actions import ComAction
from actions import SingleAction
import functools

#05f2a901.json  移动形状撞向方块            *      不泛化
#06df4c85.json
#08ed6ac7.json  高度次序决定颜色            *      不泛化
#0d3d703e.json  固定规则更换shape颜色       *
#178fcbfb.json  点延伸成线(覆盖其他颜色)，直到边界        *   目前shape总数在test上刚好差一个
#1a07d186.json  


# 1caeab9d.json  竖直高度对齐                   *   不泛化
# 25d487eb.json  点按单方向延伸(跳过其他颜色)

# 1f642eb9   点向大方块移动，在边界碰撞时融入，需要考虑同一方块不同颜色的问题
# 25ff71a9   向下移动一个格子
# 2c608aff.json 选择点后，向大方块延伸

# 重复选择，选择不存在的shape,负reward 


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
MAX_SHAPLE_NUM = 6 + 1
MAX_FEATURE_NUM = 30
COLOR_EMPTY = COLOR_PALETTE[0]
COLOR_SELECTED = pygame.Color(0x99,0x66,0xff)
COLOR_SELECTED_AS_TARGET = pygame.Color(0x44, 0x33, 0x88)

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
        self.selected = False
        self.selected_as_target = False

    # fix me, 临时解决方案，shape到底要不要容纳不同颜色还没想通
    def get_color(self):
        return self.grid_list[0].color

    def change_color(self, color):
        for grid in self.grid_list:
            grid.color = color

    def get_bound(self):
        left = right = top = bottom = -1

        for grid in self.grid_list:
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
        return left, top, width, height

    def move(self, delta_x, delta_y):
        for grid in self.grid_list:
            grid.move_hori(delta_x)
            grid.move_vert(delta_y)


class GameEngine:
    # single action
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

    DIRECTION_TOP = 0
    DIRECTION_BOTTOM = 1
    DIRECTION_LEFT = 2
    DIRECTION_RIGHT = 3

    def __init__(self, input, output, need_ui, action_mode):
        
        self.width = len(input[0])
        self.height = len(input)
        self.action_n = self.ACITON_CONVERT_COLOR + 1
        self.input = np.array(input).flatten()
        self.answer = np.array(output).flatten()
        self.finish_score = self.width * self.height
        self.need_ui = need_ui
        self.action_mode = action_mode

        if (self.action_mode == 'combo'):
            self.action_n = ComAction.ACTION_COM_NUM.value - 1
        else:
            self.action_n = SingleAction.ACTION_SINGLE_NUM.value - 1

        if (need_ui):
            pygame.init()
            self.screen = pygame.display.set_mode([self.width * GRID_LENGTH + PAD_LENGTH * 2, 
                    self.height * GRID_LENGTH + PAD_LENGTH * 2])
        
        #self.reset()

    def reset(self):
        self.shape_list = []
        self.cur_sel = -1
        self.cur_sel_color = -1
        self.cur_attension = -1
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

        feature[shape.grid_list[0].color] = 1
        if (self.cur_sel_color > 0 and self.cur_sel_color <= 9):
            feature[self.cur_sel_color + 10] = 1

        #feature[0] = shape.grid_list[0].color  / 10.0
        feature[20] = len(shape.grid_list) / 10.0
        feature[21] = left / 10.0
        feature[22] = right / 10.0
        feature[23] = top / 10.0
        feature[24] = bottom / 10.0
        feature[25] = width / 10.0
        feature[26] = height / 10.0
        feature[27] = ratio
        feature[28] = float(shape.selected)
        feature[29] = float(shape.selected_as_target)

        

        # feature[10] = (1 if self.cur_attension == self.DIRECTION_TOP else 0)
        # feature[11] = (1 if self.cur_attension == self.DIRECTION_BOTTOM else 0)
        # feature[12] = (1 if self.cur_attension == self.DIRECTION_LEFT else 0)
        # feature[13] = (1 if self.cur_attension == self.DIRECTION_RIGHT else 0)

        return feature


    def shape_list_2_feature(self):
        features = np.zeros((MAX_SHAPLE_NUM, MAX_FEATURE_NUM), dtype=np.float32)

        for idx, shape in enumerate(self.shape_list):
            features[idx] = self.shape_2_feature(shape)

        # shape = features[-1]

        # if (self.cur_attension == self.DIRECTION_TOP):
        #     shape[0] = 1
        # elif (self.cur_attension == self.DIRECTION_BOTTOM):
        #     shape[1] = 1
        # elif (self.cur_attension == self.DIRECTION_LEFT):
        #     shape[2] = 1
        # elif (self.cur_attension == self.DIRECTION_RIGHT):
        #     shape[3] = 1

        # if (self.cur_sel_color >= 0 and self.cur_sel_color <= 9):
        #     shape[self.cur_sel_color + 4] = 1

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

    def select_shape_by_color(self, color):
        necessary = (not self.has_shape_selected() and self.has_color_shape(color))

        self.unselect_all_shape()

        for shape in self.shape_list:
            if (shape.get_color() == color):
                shape.selected = True

        return not necessary

    def select_target_by_color(self, color):
        necessary = self.has_shape_selected() and not self.has_shape_target_selected()

        for shape in self.shape_list:
            shape.selected_as_target = (shape.get_color() == color)

        return necessary

    def select_shape_by_size_rank(self, idx):
        # size 有大小相同的元素
        size_list = list(map(lambda shape: len(shape.grid_list), self.shape_list))
        if (len(np.unique(size_list)) < len(size_list)):
            return False

        if (idx < 0 or idx >= len(self.shape_list)):
            return False

        necessary = (not self.has_shape_selected)

        self.unselect_all_shape()

        #pdb.set_trace()

        r = sorted(self.shape_list, key=functools.cmp_to_key(lambda s1, s2: len(s1.grid_list) >= len(s2.grid_list)))

        if (idx >= 0 and idx < len(self.shape_list)):
            r[idx].selected = True

        return necessary

    def has_shape_selected(self):
        found = False

        for shape in self.shape_list:
            if (shape.selected):
                found = True

        return found

    def has_shape_target_selected(self):
        found = False

        for shape in self.shape_list:
            if (shape.selected_as_target):
                found = True

        return found

    def has_color_shape(self, color):
        found = False

        for shape in self.shape_list:
            if (shape.get_color() == color):
                found = True

        return found

    def unselect_all_shape(self):
        for shape in self.shape_list:
            shape.selected = False
            shape.selected_as_target = False

    def change_2_color(self, color):
        #pdb.set_trace()
        necessary = self.has_shape_selected()

        for shape in self.shape_list:
            if (shape.selected):
                shape.change_color(color)

        self.unselect_all_shape()

        return necessary

    def collide_2_target(self):
        if (not self.has_shape_selected() or not self.has_shape_target_selected()):
            return False

        #pdb.set_trace()
        for target in self.shape_list:
            if (not target.selected_as_target): continue
            for shape in self.shape_list:
                if (not shape.selected or shape.idx == target.idx): continue
                x0, y0, w0, h0 = shape.get_bound()
                x1, y1, w1, h1 = target.get_bound()

                #pdb.set_trace()

                if ((x1 >= x0 and x1 <= x0 + w0) or (x1 + w1 >= x0 and x1 + w1 <= x0 + w0) ):# 竖直方向碰撞
                    if (y1 > y0 + h0):# move down
                        self.move_vert_until_collision(True)
                    elif (y1 + h1 < y0):# move up
                        self.move_vert_until_collision(False)
                elif ((y1 >= y0 and y1 <= y0 + h0) or (y1 + h1 >= y0 and y1 + h1 <= y0 + h0)):# 水平方向碰撞
                    if (x1 > x0 + w0):# move right
                        self.move_hori_until_collision(True)
                    elif (x1 + w1 < x0): # move left
                        self.move_hori_until_collision(False)

        self.unselect_all_shape()

        return True

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
        #if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        #pdb.set_trace()
        for shape in self.shape_list:
            if (not shape.selected): continue

            #pdb.set_trace()
            stop = False
            delta = 0

            while True:
                for grid in shape.grid_list:
                    y = grid.y + delta + (1 if down else -1)
                    if (y > self.height - 1 or y < 0):
                        stop = True
                        break

                    shape_idx = self.get_shape_idx_from_xy(grid.x, y)
                    if (shape_idx >= 0 and shape_idx != shape.idx):
                        stop = True
                        break
                if (stop): break
                delta += (1 if down else -1)

            for grid in shape.grid_list:
                grid.move_vert(delta)

    # move hori
    def move_hori_until_collision(self, right):
        #if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return
        for shape in self.shape_list:
            #shape = self.shape_list[self.cur_sel]
            if (not shape.selected): continue
            stop = False
            delta = 0

            while True:
                for grid in shape.grid_list:
                    x = grid.x + delta + (1 if right else -1)
                    if (x > self.width - 1 or x < 0):
                        stop = True
                        break

                    shape_idx = self.get_shape_idx_from_xy(x, grid.y)
                    if (shape_idx >= 0 and shape_idx != shape.idx):
                        stop = True
                        break

                if (stop): break
                delta += (1 if right else -1)

            for grid in shape.grid_list:
                grid.move_hori(delta)

    def spread_hori(self):
        shape = self.shape_list[self.cur_sel]
        if (len(shape.grid_list) != 1): return

        grid = shape.grid_list[0]
        for i in range(self.width):
            if (i != grid.x):
                shape.grid_list.append(Grid(i, grid.y, self.width * grid.y + i, grid.color, shape))

    def spread_vert(self):
        shape = self.shape_list[self.cur_sel]
        if (len(shape.grid_list) != 1): return

        grid = shape.grid_list[0]
        for i in range(self.height):
            if (i != grid.y):
                shape.grid_list.append(Grid(grid.x, i, self.width * i + grid.x, grid.color, shape))


    def select_color(self, color):
        self.cur_sel_color = color

    def convert_2_color(self):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return
        if (self.cur_sel_color < 0 or self.cur_sel_color > self.ACTION_SEL_COLOR_9): return

        shape = self.shape_list[self.cur_sel]

        for grid in shape.grid_list:
            grid.color = self.cur_sel_color

    def bottom_align(self, from_idx, to_idx):
        if (from_idx < 0 or from_idx >= len(self.shape_list) or to_idx < 0 or to_idx >= len(self.shape_list)): return

        from_shape = self.shape_list[from_idx]
        to_shape = self.shape_list[to_idx]

        _, y0, _, h0 = to_shape.get_bound()
        _, y1, _, h1 = from_shape.get_bound()

        from_shape.move(0, (y0 + h0) - (y1 + h1) )

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

    def do_com_action(self, action):
        if (action >= ComAction.ACTION_COM_SEL_0_COLOR_0_CONVERT_COLOR.value and
            action <= ComAction.ACTION_COM_SEL_4_COLOR_9_CONVERT_COLOR.value):
            idx = action - ComAction.ACTION_COM_SEL_0_COLOR_0_CONVERT_COLOR.value
            self.select_shape(idx // 10)
            self.select_color(idx % 10)
            self.convert_2_color()
        elif (action >= ComAction.ACTION_COM_SEL_0_TOP_MOVE_UNTIL_COLISSION.value and
            action <= ComAction.ACTION_COM_SEL_4_RIGHT_MOVE_UNTIL_COLISSION.value):
            idx = action - ComAction.ACTION_COM_SEL_0_TOP_MOVE_UNTIL_COLISSION.value
            self.select_shape(idx // 4)
            self.select_direct_attension(idx % 4)
            self.move_until_collision()
        elif (action >= ComAction.ACTION_COM_SEL_0_POINT_HORI_SPREAD.value and
            action <= ComAction.ACTION_COM_SEL_4_POINT_VERT_SPEAD.value):
            idx = action - ComAction.ACTION_COM_SEL_0_POINT_HORI_SPREAD.value
            self.select_shape(idx // 4)
            if (idx % 2 == 0):
                self.spread_hori()
            else:
                self.spread_vert()
        elif (action >= ComAction.ACTION_COM_SEL_0_BOTTOM_ALIGN_TO_1.value and
            action <= ComAction.ACTION_COM_SEL_4_BOTTOM_ALIGN_TO_3.value):
            idx = action - ComAction.ACTION_COM_SEL_0_BOTTOM_ALIGN_TO_1.value
            from_idx = idx // 4
            to_idx = idx % 4
            if (to_idx >= from_idx): to_idx += 1
            self.select_shape(from_idx)
            self.bottom_align(from_idx, to_idx)

    def do_single_action(self, action):
        action += 1
        necessary = True

        if (action >= SingleAction.ACTION_SINGLE_SEL_COLOR_1.value and action <= SingleAction.ACTION_SINGLE_SEL_COLOR_9.value):
            color = action - SingleAction.ACTION_SINGLE_SEL_COLOR_1.value + 1
            necessary = self.select_shape_by_color(color)
        elif (action >= SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_1.value and action <= SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_9.value):
            color = action - SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_1.value + 1
            necessary = self.select_target_by_color(color)
        elif (action == SingleAction.ACTION_SINGLE_MOVE_TO_TARGET_UNTIL_COLISSION.value):
            necessary = self.collide_2_target()
        elif (action >= SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_1.value and action <= SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_5.value):
            idx = action - SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_1.value
            necessary = self.select_shape_by_size_rank(idx)
        elif (action >= SingleAction.ACTION_SINGLE_CHANGE_COLOR_1.value and action <= SingleAction.ACTION_SINGLE_CHANGE_COLOR_9.value):
            color = action - SingleAction.ACTION_SINGLE_CHANGE_COLOR_1.value + 1
            necessary = self.change_2_color(color)

        return necessary

        # if (action == GameEngine.ACTION_SEL_0):
        #     self.select_shape(0)
        # elif (action == GameEngine.ACTION_SEL_1):
        #     self.select_shape(1)
        # elif (action == GameEngine.ACTION_SEL_2):
        #     self.select_shape(2)
        # elif (action == GameEngine.ACTION_SEL_3):
        #     self.select_shape(3)
        # elif (action == GameEngine.ACTION_SEL_4):
        #     self.select_shape(4)
        # elif (action == GameEngine.ACTION_ATTENSION_TOP):
        #     self.select_direct_attension(self.DIRECTION_TOP)
        # elif (action == GameEngine.ACTION_ATTENSION_BOTTOM):
        #     self.select_direct_attension(self.DIRECTION_BOTTOM)
        # elif (action == GameEngine.ACTION_ATTENSION_LEFT):
        #     self.select_direct_attension(self.DIRECTION_LEFT)
        # elif (action == GameEngine.ACTION_ATTENSION_RIGHT):
        #     self.select_direct_attension(self.DIRECTION_RIGHT)
        # elif (action == GameEngine.ACTION_MOVE_UNTIL_COLISSION):
        #     self.move_until_collision()
        # elif (action == GameEngine.ACTION_SEL_COLOR_0):
        #     self.select_color(0)
        # elif (action == GameEngine.ACTION_SEL_COLOR_1):
        #     self.select_color(1)
        # elif (action == GameEngine.ACTION_SEL_COLOR_2):
        #     self.select_color(2)
        # elif (action == GameEngine.ACTION_SEL_COLOR_3):
        #     self.select_color(3)
        # elif (action == GameEngine.ACTION_SEL_COLOR_4):
        #     self.select_color(4)
        # elif (action == GameEngine.ACTION_SEL_COLOR_5):
        #     self.select_color(5)
        # elif (action == GameEngine.ACTION_SEL_COLOR_6):
        #     self.select_color(6)
        # elif (action == GameEngine.ACTION_SEL_COLOR_7):
        #     self.select_color(7)
        # elif (action == GameEngine.ACTION_SEL_COLOR_8):
        #     self.select_color(8)
        # elif (action == GameEngine.ACTION_SEL_COLOR_9):
        #     self.select_color(9)
        # elif (action == GameEngine.ACITON_CONVERT_COLOR):
        #     self.convert_2_color()

    def do_action(self, action):
        necessary = True

        if (self.action_mode == 'combo'):
            self.do_com_action(action + 1)
        else:
            necessary = self.do_single_action(action)

        done = False

        new_score= self.calc_current_score()
        if (new_score == self.finish_score):
            done = True
        progress = new_score - self.cur_score
        #print(new_score, self.cur_score, progress, self.finish_score)
        self.cur_score = new_score

        #print('current socre : %d progress : %d' % (self.cur_score, progress))

        self.features = self.shape_list_2_feature()

        if (not necessary):
            progress -= 0.1

        #print('progress %2f' % progress)

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

                if (shape.selected):
                    pygame.draw.rect(self.screen, COLOR_SELECTED, (PAD_LENGTH + grid.x * GRID_LENGTH, 
                    PAD_LENGTH + grid.y * GRID_LENGTH, GRID_LENGTH, GRID_LENGTH), 1)

                if (shape.selected_as_target):
                    pygame.draw.rect(self.screen, COLOR_SELECTED_AS_TARGET, (PAD_LENGTH + grid.x * GRID_LENGTH, 
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
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_MOVE_TO_TARGET_UNTIL_COLISSION.value - 1)
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_1.value - 1)
        elif event.key == pygame.K_1:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_1.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_1.value - 1)
        elif event.key == pygame.K_2:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_2.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_2.value - 1)
        elif event.key == pygame.K_3:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_3.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_3.value - 1)
        elif event.key == pygame.K_4:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_4.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_4.value - 1)
        elif event.key == pygame.K_5:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_5.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_SIZE_TOP_5.value - 1)
        elif event.key == pygame.K_6:
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_6.value - 1)
        elif event.key == pygame.K_7:
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_7.value - 1)
        elif event.key == pygame.K_8:
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_8.value - 1)
        elif event.key == pygame.K_9:
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_9.value - 1)
        elif event.key == pygame.K_q:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_1.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_1.value - 1)
        elif event.key == pygame.K_w:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_2.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_2.value - 1)
        elif event.key == pygame.K_e:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_3.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_3.value - 1)
        elif event.key == pygame.K_r:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_4.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_4.value - 1)
        elif event.key == pygame.K_t:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_5.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_5.value - 1)
        elif event.key == pygame.K_y:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_6.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_6.value - 1)
        elif event.key == pygame.K_u:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_7.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_7.value - 1)
        elif event.key == pygame.K_i:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_8.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_8.value - 1)
        elif event.key == pygame.K_o:
            #obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_TARGET_COLOR_9.value - 1)
            obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_CHANGE_COLOR_9.value - 1)
        # elif event.key == pygame.K_p:
        #     obs, reward, done, info = self.do_action(GameEngine.ACTION_SEL_COLOR_9)
        # elif event.key == pygame.K_TAB:
        #     obs, reward, done, info = self.do_action(GameEngine.ACITON_CONVERT_COLOR)
        # elif event.key == pygame.K_a:
        #     obs, reward, done, info = self.do_action(SingleAction.ACTION_SINGLE_SEL_COLOR_1.value - 1)
        # elif event.key == pygame.K_b:
        #     obs, reward, done, info = self.do_action(ComAction.ACTION_COM_SEL_0_BOTTOM_ALIGN_TO_1.value - 1)

        return obs, reward, done, info


if __name__ == "__main__":
    INPUT_FILE = '/Users/yao/develop/ARC/data/training/08ed6ac7.json'
    with open(INPUT_FILE,'r') as f:
        puzzle = json.load(f)
    puzzle_input = puzzle['train'][0]['input']
    puzzle_output = puzzle['train'][0]['output']

    # puzzle_input = puzzle['test'][0]['input']
    # puzzle_output = puzzle['test'][0]['output']
    
    game_engine = GameEngine(puzzle_input, puzzle_output, True, 'single')
    game_engine.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                obs, reward, done, info = game_engine.process_key(event)
                #print(reward)

        game_engine.draw_game()

    pygame.quit()