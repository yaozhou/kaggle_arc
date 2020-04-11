import pygame
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

GRID_LENGTH = 100
COLOR_EMPTY = COLOR_PALETTE[0]

class Shape:
    def __init__(self, point_list, width, height):
        self.point_list = point_list
        self.width = width
        self.height = height

class GameEngine: 
    ACTION_SEL_0 = 1
    ACTION_SEL_1 = 2
    ACTION_SEL_2 = 3
    ACTION_SEL_3 = 4
    ACTION_SEL_4 = 5
    ACTION_ATTENSION_TOP = 6
    ACTION_ATTENSION_BOTTOM = 7
    ACTION_ATTENSION_LEFT = 8
    ACTION_ATTENSION_RIGHT = 9
    ACTION_MOVE_UNTIL_COLISSION = 10

    DIRECTION_TOP = 0
    DIRECTION_BOTTOM = 1
    DIRECTION_LEFT = 2
    DIRECTION_RIGHT = 3

    def __init__(self, input):
        self.state = np.array(input).flatten()
        self.width = len(input)
        self.height = len(input[0])
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

                
                print(idx)
                self.G.add_node(idx)
                
                print(up, up_left, up_right, down, down_left, down_right, left, right)

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

        print(self.G.nodes())
        nx.draw(self.G)
        plt.show()            



        self.shape_list = None

    def update_state_from_shape():
        pass

    def select_shape(idx):
        pass

    def select_direct_attension(direction):
        pass

    def move_until_collision():
        pass

    def do_action(action):
        if (action == ACTION_SEL_0):
            select_shape(0)
        elif (action == ACTION_SEL_1):
            select_shape(1)
        elif (action == ACTION_SEL_2):
            select_shape(2)
        elif (action == ACTION_SEL_3):
            select_shape(3)
        elif (action == ACTION_SEL_4):
            select_shape(4)
        elif (action == ACTION_SEL_5):
            select_shape(5)
        elif (action == ACTION_ATTENSION_TOP):
            select_direct_attension(DIRECTION_TOP)
        elif (action == ACTION_ATTENSION_BOTTOM):
            select_direct_attension(DIRECTION_BOTTOM)
        elif (action == ACTION_ATTENSION_LEFT):
            select_direct_attension(DIRECTION_LEFT)
        elif (action == ACTION_ATTENSION_RIGHT):
            select_direct_attension(DIRECTION_RIGHT)
        elif (action == ACTION_MOVE_UNTIL_COLISSION):
            move_until_colission()




def draw_input(screen, input):
    row_num = len(input)
    col_num = len(input[0])

    for i in range(row_num):
        for j in range(col_num):
            color = COLOR_PALETTE[input[i][j]]
            pygame.draw.rect(screen, color, pygame.Rect(j*GRID_LENGTH, i*GRID_LENGTH, GRID_LENGTH, GRID_LENGTH))

def game_init(row_num, col_num):
    pygame.init()
    screen = pygame.display.set_mode([row_num * GRID_LENGTH, col_num * GRID_LENGTH])
    return screen


with open("./1e0a9b12.json",'r') as f:
    puzzle = json.load(f)

puzzle_input = puzzle['train'][1]['input']
row_num = len(puzzle_input)
col_num = len(puzzle_input[0])

screen = game_init(row_num, col_num)

game_engine = GameEngine(puzzle_input)
game_engine.update_shape_list_from_state()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(COLOR_PALETTE[0])
    draw_input(screen, puzzle_input)
    pygame.display.flip()

pygame.quit()