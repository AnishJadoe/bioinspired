import PySimpleGUI as sg
import numpy as np
import math
import random

AppFont = 'Any 16'
sg.theme('DarkGrey5')
_VARS = {'cellCount': 20, 'gridSize': 400, 'canvas': False, 'window': False,
         'playerPos': [0,0], 'cellMAP':False
         }

cellSize = _VARS['gridSize']/_VARS['cellCount']
exitPos = [_VARS['cellCount']-1, _VARS['cellCount']-1]

def make_maze(dimX, dimY):

    # Start with an empty grid
    init_map = np.zeros((dimX,dimY),dtype=int)

    # Add relevant rows and columns

    for x in range(2):
        rand_row = random.randint(1, dimX)
        rand_column = random.randint(1, dimY)
        init_map[rand_row-1:rand_row] = 1
        init_map[:,rand_column-1] = 1

        # poke holes
        for x in range(4):
            init_map[rand_row-1][random.randint(0,dimY-1)] = 0
            init_map[random.randint(0, dimX-1)][rand_column-1] = 0

    # Add blank cells for entrance, exit and around them:
    init_map[0][0] = 0
    init_map[0][1] = 0
    init_map[1][0] = 0
    init_map[dimX-1][dimY-1] = 0
    init_map[dimX-1][dimY-2] = 0
    init_map[dimX-2][dimY-1] = 0
    init_map[dimX-2][dimY-2] = 0

    return init_map


_VARS['cellMAP'] = make_maze(_VARS['cellCount'], _VARS['cellCount'])

def draw_grid():
    cells = _VARS['cellCount']
    _VARS['canvas'].TKCanvas.create_rectangle(
        1,1,_VARS['gridSize'],_VARS['gridSize'], outline='BLACK', width=1)
    for x in range(cells):
        _VARS['canvas'].TKCanvas.create_line(
            ((cellSize * x), 0), ((cellSize * x), _VARS['gridSize']),
            fill='BLACK', width=1
        )
        _VARS['canvas'].TKCanvas.create_line(
            (0 ,(cellSize * x)), (_VARS['gridSize'], (cellSize * x) ),
            fill='BLACK', width=1
        )

    return

def draw_cell(x, y,color='GREY'):
    _VARS['canvas'].TKCanvas.create_rectangle(
        x, y, x + cellSize, y + cellSize,
        outline='BLACK', fill=color, width=1
    )

    return

def place_cells():
    for row in range(_VARS['cellMAP'].shape[0]):
        for column in range(_VARS['cellMAP'].shape[1]):
            if (_VARS['cellMAP'][column][row] == 1):
                draw_cell((cellSize*row), (cellSize*column))
    return

def check_events(event):
    move = ''
    if len(event) == 1:
        if ord(event) == 119:
            move = 'Up'
        elif ord(event) == 115:
            move = 'Down'
        elif ord(event) == 97:
            move = 'Left'
        elif ord(event) == 100:
            move = 'Right'
    else:
        if event.startswith('Up'):
            move = 'Up'
        elif event.startswith('Down'):
            move = 'Down'
        elif event.startswith('Left'):
            move = 'Left'
        elif event.startswith('Right'):
            move = 'Right'

    return move

#INIT
layout = [[sg.Canvas(size=(_VARS['gridSize'], _VARS['gridSize']),
            background_color='white',
            key='canvas')],
          [sg.Exit(font=AppFont),
           sg.Text('', key='-exit-', font=AppFont, size=(15, 1)),
           sg.Button('NewMaze', font=AppFont)]]

_VARS['window'] = sg.Window('GridMaker',
                            layout, resizable=True, finalize=True,
                            return_keyboard_events=True)
_VARS['canvas'] = _VARS['window']['canvas']
draw_grid()
draw_cell(_VARS['playerPos'][0],_VARS['playerPos'][1],'TOMATO')
draw_cell(exitPos[0]*cellSize, exitPos[1]*cellSize, 'Black')
place_cells()


while True:
    event, values = _VARS['window'].read()
    if event in (None, 'Exit'):
        break

    if event == 'NewMaze':
        _VARS['playerPos'] = [0,0]
        _VARS['cellMAP'] = make_maze(_VARS['cellCount'],_VARS['cellCount'])

    xPos = int(math.ceil(_VARS['playerPos'][0]/cellSize))
    yPos = int(math.ceil(_VARS['playerPos'][1]/cellSize))

    if check_events(event) == 'Up':
        if int(_VARS['playerPos'][1] - cellSize) >= 0:
            if _VARS['cellMAP'][yPos-1][xPos] != 1:
                _VARS['playerPos'][1] = _VARS['playerPos'][1]-cellSize

    elif check_events(event) == 'Down':
        if int(_VARS['playerPos'][1] + cellSize )< _VARS['gridSize']-1:
            if _VARS['cellMAP'][yPos+1][xPos] != 1:
                _VARS['playerPos'][1] = _VARS['playerPos'][1] + cellSize

    elif check_events(event) == 'Left':
        if int(_VARS['playerPos'][0] - cellSize) >= 0:
            if _VARS['cellMAP'][yPos][xPos-1] != 1:
                _VARS['playerPos'][0] = _VARS['playerPos'][0] - cellSize

    elif check_events(event) == 'Right':
        if int(_VARS['playerPos'][0] + cellSize) < 400:
            if _VARS['cellMAP'][yPos][xPos+1] != 1:
                _VARS['playerPos'][0] = _VARS['playerPos'][0] + cellSize

    _VARS['canvas'].TKCanvas.delete('all')
    draw_grid()
    draw_cell(exitPos[0]*cellSize, exitPos[1]*cellSize, 'Black')
    draw_cell(_VARS['playerPos'][0],_VARS['playerPos'][1],'TOMATO')
    place_cells()

    xPos = int(math.ceil(_VARS['playerPos'][0]/cellSize))
    yPos = int(math.ceil(_VARS['playerPos'][1]/cellSize))

    if [xPos,yPos] == exitPos:
        _VARS['window']['-exit-'].update('Found the exit !')
    else:
        _VARS['window']['-exit-'].update('')


_VARS['window'].close()