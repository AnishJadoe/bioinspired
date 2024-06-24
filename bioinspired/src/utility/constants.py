# Map 
START_POSITION = (19, 18) # Has to be + 10
MAP_DIMS = (120,80)
CELL_SIZE = 10
MAP_SIZE = (MAP_DIMS[0]*CELL_SIZE,MAP_DIMS[1]*CELL_SIZE)
N_TOKENS = 125
# Neural Network
N_INPUTS = 10
N_HIDDEN = 5
N_OUTPUTS = 2

# COLORS
WHITE = (255,255,255,128)
DARK_GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
BLUE = (0, 0, 255)
BROWN = (139, 69, 19)
ORANGE = (255, 165, 0)
SENSOR_COLORS = [RED,DARK_GREEN,PURPLE,ORANGE,BLUE]
YELLOW = (255, 255, 0, 255)
BLACK = (0, 0, 0, 255)

# Weights
W_TOKEN = 15
W_QUICK = 300
W_CLOSE = 0.05
W_COL = 0.01

WEIGHT_RANGE=0.5
MAX_ENERGY = 500