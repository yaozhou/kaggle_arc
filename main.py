import pygame
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pdb

INPUT_FILE = '/Users/yao/develop/ARC/data/training/05f2a901.json'

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
COLOR_EMPTY = COLOR_PALETTE[0]
COLOR_SELECTED = pygame.Color(0x99,0x66,0xff)

class Grid:
    def __init__(self, x, y, idx, shape_idx, color, game_engine):
        self.x = x
        self.y = y
        self.idx = idx
        self.shape_idx = shape_idx
        self.color = color
        self.game_engine = game_engine

    def move_vert(self, delta):
        self.y += delta
        self.idx += game_engine.width * delta

    def move_hori(self, delta):
        self.x += delta
        self.idx += delta

class Shape:
    def __init__(self, point_list, width, height):
        self.point_list = point_list
        self.width = width
        self.height = height
        self.shape_list = np.array([1,2])

    

class GameEngine: 
    ACTION_SEL_0 = 1
    ACTION_SEL_1 = 2
    ACTION_SEL_2 = 3
    ACTION_SEL_3 = 4
    ACTION_SEL_4 = 5
    ACTION_SEL_5 = 6
    ACTION_ATTENSION_TOP = 7
    ACTION_ATTENSION_BOTTOM = 8
    ACTION_ATTENSION_LEFT = 9
    ACTION_ATTENSION_RIGHT = 10
    ACTION_MOVE_UNTIL_COLISSION = 11

    DIRECTION_TOP = 0
    DIRECTION_BOTTOM = -1
    DIRECTION_LEFT = -2
    DIRECTION_RIGHT = -3

    def __init__(self, input):
        self.state = np.array(input).flatten()
        self.width = len(input[0])
        self.height = len(input)
        self.shape_list = np.array([])
        self.cur_sel = 0
        self.cur_attension = self.DIRECTION_TOP
        self.G = nx.Graph()


    def update_shape_list_from_state(self):
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                color = self.state[idx]
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

                self.G.add_node(idx)
                
                #print(self.width, self.height, idx, up, up_left, up_right, down, down_left, down_right, left, right)

                if (up >= 0 and color == self.state[up]):
                    self.G.add_edge(idx, up)
                if (up_left >= 0 and color == self.state[up_left]):
                    self.G.add_edge(idx, up_left)
                
                if (up_right >= 0 and color == self.state[up_right]):
                    self.G.add_edge(idx, up_right)

                if (down >= 0 and color == self.state[down]):
                    self.G.add_edge(idx, down)

                if (down_left >= 0 and color == self.state[down_left]):    
                    self.G.add_edge(idx, down_left)

                if (down_right >= 0 and color == self.state[down_right]):
                    self.G.add_edge(idx, down_right)

                if (left >= 0 and color == self.state[left]):
                    self.G.add_edge(idx, left)

                if (right >= 0 and color == self.state[right]):
                    self.G.add_edge(idx, right)

        #print(self.G.edges())
        self.shape_list = np.array(list(nx.connected_components(self.G)))
        #print(self.shape_list[0])

        for shape_idx, shape in enumerate(self.shape_list):
            grid_list = []
            for point in list(shape):
                i = point // self.width
                j = point % self.width
                #pdb.set_trace()
                grid_list.append(Grid(j, i, point, shape_idx, self.state[point], self))
            self.shape_list[shape_idx] = np.array(grid_list)

    def update_state_from_shape(self):
        self.state = np.zeros(self.width * self.height, dtype=np.int64)

        
        for shape in self.shape_list:
            for grid in shape:
                self.state[grid.idx] = grid.color


    def select_shape(self, idx):
        self.cur_sel = idx

    def select_direct_attension(self, direction):
        self.cur_attension = direction

    def get_shape_idx(self, point_idx):
        for shape in self.shape_list:
            for grid in shape:
                if (grid.idx == point_idx):
                    return grid.shape_idx
        
        return -1

    def get_shape_idx_from_xy(self, x, y):
        for shape in self.shape_list:
            for grid in shape:
                if (grid.x == x and grid.y == y):
                    return grid.shape_idx

        return -1

    def move_vert_until_collision(self, down):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        shape = self.shape_list[self.cur_sel]
        stop = False
        delta = 0

        while True:
            for grid in shape:
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

        for grid in shape:
            grid.move_vert(delta)

    def move_hori_until_collision(self, right):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        shape = self.shape_list[self.cur_sel]
        stop = False
        delta = 0

        while True:
            for grid in shape:
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

        for grid in shape:
            grid.move_hori(delta)
        

    def move_down_until_collision(self):
        if (self.cur_sel < 0 or self.cur_sel >= len(self.shape_list)): return

        shape = self.shape_list[self.cur_sel]

        stop = False
        delta = 0

        #pdb.set_trace()

        while True:
            #print(delta)
            for point in shape:
                #print(point.x, point.y)
                i = point.y + delta + 1
                j = point.x
                if (i > self.height - 1): 
                    stop = True
                    break

                next_pos_in_direction = i * self.width + j

                # i = next_pos_in_direction // self.width
                # j = next_pos_in_direction % self.width
                

                idx = self.get_shape_idx(next_pos_in_direction)
                if (idx >= 0 and idx != self.cur_sel):
                    stop = True
                    break

            if (stop):
                break
            delta += 1
            

        if (delta == 0): return

        #pdb.set_trace()
        for grid in shape:
            grid.move_down(delta)

        for grid in shape:
            print(grid.x, grid.y)


    def move_until_collision(self):
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
        elif (action == GameEngine.ACTION_SEL_5):
            self.select_shape(5)
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




def draw_input(screen, game_engine):
    row_num = game_engine.height
    col_num = game_engine.width
    #print(game_engine.state)
    #pdb.set_trace()

    for i in range(row_num):
        for j in range(col_num):
            
            color = COLOR_PALETTE[game_engine.state[col_num * i + j]]
            pygame.draw.rect(screen, color, pygame.Rect(PAD_LENGTH + j*GRID_LENGTH,
                            PAD_LENGTH + i*GRID_LENGTH, GRID_LENGTH, GRID_LENGTH))

    for i in range(row_num + 1):
        pygame.draw.line(screen, (255, 255, 255), (PAD_LENGTH + 0, PAD_LENGTH +i * GRID_LENGTH), 
                        (PAD_LENGTH +GRID_LENGTH * col_num, PAD_LENGTH +i * GRID_LENGTH), 1)
    for i in range(col_num + 1):
        pygame.draw.line(screen, (255, 255, 255), (PAD_LENGTH + i * GRID_LENGTH, PAD_LENGTH +0), 
                        (PAD_LENGTH + i * GRID_LENGTH, PAD_LENGTH + row_num * GRID_LENGTH), 1)

    font = pygame.font.Font('freesansbold.ttf', 16)
    

    for idx, shape in enumerate(game_engine.shape_list):
        for point in shape:
            text_surface = font.render(str(idx), True, (125,125,125))
            j = point.x
            i = point.y
            #print(point, i, j)
            if (idx == game_engine.cur_sel):
                pygame.draw.rect(screen, COLOR_SELECTED, (PAD_LENGTH + j*GRID_LENGTH, PAD_LENGTH + i*GRID_LENGTH, GRID_LENGTH, GRID_LENGTH), 1)
            screen.blit(text_surface, dest=(PAD_LENGTH + (j + 0.5) * GRID_LENGTH, PAD_LENGTH + (i + 0.5) * GRID_LENGTH))


    if (game_engine.cur_attension == GameEngine.DIRECTION_TOP):
        pygame.draw.circle(screen, COLOR_SELECTED, ((col_num * GRID_LENGTH + PAD_LENGTH * 2) // 2, PAD_LENGTH // 2), 15)
    elif (game_engine.cur_attension == GameEngine.DIRECTION_BOTTOM):
        pygame.draw.circle(screen, COLOR_SELECTED, ((col_num * GRID_LENGTH + PAD_LENGTH * 2) // 2, 
                            PAD_LENGTH * 2 + row_num * GRID_LENGTH - PAD_LENGTH // 2 ), 15)
    elif (game_engine.cur_attension == GameEngine.DIRECTION_LEFT):
        pygame.draw.circle(screen, COLOR_SELECTED, (PAD_LENGTH // 2, (PAD_LENGTH * 2 + row_num * GRID_LENGTH) // 2), 15)
    elif (game_engine.cur_attension == GameEngine.DIRECTION_RIGHT):
        pygame.draw.circle(screen, COLOR_SELECTED, (col_num * GRID_LENGTH + PAD_LENGTH * 2 - PAD_LENGTH // 2, 
                            (PAD_LENGTH * 2 + row_num * GRID_LENGTH) // 2), 15)


def game_init(row_num, col_num):
    pygame.init()
    screen = pygame.display.set_mode([col_num * GRID_LENGTH + PAD_LENGTH * 2, row_num * GRID_LENGTH + PAD_LENGTH * 2])
    return screen


with open(INPUT_FILE,'r') as f:
    puzzle = json.load(f)

puzzle_input = puzzle['train'][0]['input']
#print(puzzle_input)
row_num = len(puzzle_input)
col_num = len(puzzle_input[0])

screen = game_init(row_num, col_num)

game_engine = GameEngine(puzzle_input)
game_engine.update_shape_list_from_state()
#pdb.set_trace()
#print(game_engine.shape_list)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                print('pressed 0')
                game_engine.do_action(GameEngine.ACTION_SEL_0)
            elif event.key == pygame.K_1:
                game_engine.do_action(GameEngine.ACTION_SEL_1)
            elif event.key == pygame.K_2:
                game_engine.do_action(GameEngine.ACTION_SEL_2)
            elif event.key == pygame.K_3:
                game_engine.do_action(GameEngine.ACTION_SEL_3)
            elif event.key == pygame.K_4:
                game_engine.do_action(GameEngine.ACTION_SEL_4)
            elif event.key == pygame.K_5:
                game_engine.do_action(GameEngine.ACTION_SEL_5)
            elif event.key == pygame.K_UP:
                game_engine.do_action(GameEngine.ACTION_ATTENSION_TOP)
            elif event.key == pygame.K_DOWN:
                game_engine.do_action(GameEngine.ACTION_ATTENSION_BOTTOM)
            elif event.key == pygame.K_LEFT:
                game_engine.do_action(GameEngine.ACTION_ATTENSION_LEFT)
            elif event.key == pygame.K_RIGHT:
                game_engine.do_action(GameEngine.ACTION_ATTENSION_RIGHT)
            elif event.key == pygame.K_SPACE:
                game_engine.do_action(GameEngine.ACTION_MOVE_UNTIL_COLISSION)

    screen.fill(COLOR_PALETTE[0])
    game_engine.update_state_from_shape()
    draw_input(screen, game_engine)
    pygame.display.flip()

pygame.quit()