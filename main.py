# Simple pygame program

# Import and initialize the pygame library
import pygame
import json


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




with open("./1e0a9b12.json",'r') as f:
    puzzle = json.load(f)

puzzle_input = puzzle['train'][0]['input']
row_num = len(puzzle_input)
col_num = len(puzzle_input[0])




def draw_input(screen, input):
    #print(input)
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



screen = game_init(row_num, col_num)

# Set up the drawing window
#screen = pygame.display.set_mode([500, 500])

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill(COLOR_PALETTE[0])
    draw_input(screen, puzzle_input)

    # Draw a solid blue circle in the center
    #pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

    #pygame.draw.rect(screen, pygame.Color(0x0074D9), pygame.Rect(0, 0, 100, 100), 1)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()