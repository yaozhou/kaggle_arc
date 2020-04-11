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

        for idx, shape in enumerate(self.shape_list):
            self.shape_list[idx] = np.array(list(shape))

    def update_state_from_shape():
        pass

    def select_shape(self, idx):
        self.cur_sel = idx

    def select_direct_attension(self, direction):
        self.cur_attension = direction

    def move_until_collision(self):
        pass

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




def draw_input(screen, input, shape_list, cur_sel, attension):
    row_num = len(input)
    col_num = len(input[0])

    for i in range(row_num):
        for j in range(col_num):
            color = COLOR_PALETTE[input[i][j]]
            pygame.draw.rect(screen, color, pygame.Rect(PAD_LENGTH + j*GRID_LENGTH, PAD_LENGTH + i*GRID_LENGTH, GRID_LENGTH, GRID_LENGTH))

    for i in range(row_num + 1):
        pygame.draw.line(screen, (255, 255, 255), (PAD_LENGTH + 0, PAD_LENGTH +i * GRID_LENGTH), 
                        (PAD_LENGTH +GRID_LENGTH * col_num, PAD_LENGTH +i * GRID_LENGTH), 1)
    for i in range(col_num + 1):
        pygame.draw.line(screen, (255, 255, 255), (PAD_LENGTH + i * GRID_LENGTH, PAD_LENGTH +0), 
                        (PAD_LENGTH + i * GRID_LENGTH, PAD_LENGTH + row_num * GRID_LENGTH), 1)

    font = pygame.font.Font('freesansbold.ttf', 16)
    

    for idx, shape in enumerate(shape_list):
        for point in shape:
            text_surface = font.render(str(idx), True, (125,125,125))
            i = point // col_num
            j = point % col_num
            #print(point, i, j)
            if (idx == cur_sel):
                pygame.draw.rect(screen, COLOR_SELECTED, (PAD_LENGTH + j*GRID_LENGTH, PAD_LENGTH + i*GRID_LENGTH, GRID_LENGTH, GRID_LENGTH), 1)
            screen.blit(text_surface, dest=(PAD_LENGTH + (j + 0.5) * GRID_LENGTH, PAD_LENGTH + (i + 0.5) * GRID_LENGTH))


    if (attension == GameEngine.DIRECTION_TOP):
        pygame.draw.circle(screen, COLOR_SELECTED, ((col_num * GRID_LENGTH + PAD_LENGTH * 2) // 2, PAD_LENGTH // 2), 15)
    elif (attension == GameEngine.DIRECTION_BOTTOM):
        pygame.draw.circle(screen, COLOR_SELECTED, ((col_num * GRID_LENGTH + PAD_LENGTH * 2) // 2, 
                            PAD_LENGTH * 2 + row_num * GRID_LENGTH - PAD_LENGTH // 2 ), 15)
    elif (attension == GameEngine.DIRECTION_LEFT):
        pygame.draw.circle(screen, COLOR_SELECTED, (PAD_LENGTH // 2, (PAD_LENGTH * 2 + row_num * GRID_LENGTH) // 2), 15)
    elif (attension == GameEngine.DIRECTION_RIGHT):
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

    screen.fill(COLOR_PALETTE[0])
    draw_input(screen, puzzle_input, game_engine.shape_list, game_engine.cur_sel, game_engine.cur_attension)
    pygame.display.flip()

pygame.quit()