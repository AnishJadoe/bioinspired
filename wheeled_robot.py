import math
import numpy as np
import pygame
from CTRNN import CTRNN
from scipy.sparse import csr_matrix

np.random.RandomState()


def calc_distance(coord1, coord2):
    dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    return dist


def calc_angle(coord1, coord2):
    if coord1[0] == coord2[0]:
        orient = 0
    else:
        orient = math.tan((coord1[1] - coord2[1]) / (coord1[0] - coord2[0]))

    return orient


class Walls:

    def __init__(self, size, dimensions):
        self.black = (0, 0, 0)
        self.size = size
        self.width = dimensions[1]
        self.height = dimensions[0]
        self.cellMAP = self.get_maze_map()
        self.grid = self.get_grid()
        self.obstacles = self.get_obs()

    def get_maze_map(self):

        WALL = 0
        FLOOR = 1

        np.random.seed(42)
        cellMAP = np.random.choice([1, 0], size=(int(self.height / self.size), int(self.width / self.size)),
                                   p=[0.75, 0.25])

        generations = 50
        GRIDHEIGHT = int(self.height / self.size)
        GRIDWIDTH = int(self.width / self.size)

        if generations > 1:
            for generation in range(generations):

                for row in range(GRIDHEIGHT):
                    for column in range(GRIDWIDTH):

                        subMAP = cellMAP[max(row - 1, 0):min(row + 2, GRIDHEIGHT),
                                 max(column - 1, 0):min(column + 2, GRIDWIDTH)]

                        wallcount_1away = len(np.where(subMAP.flatten() == WALL)[0])

                        subMAP = cellMAP[max(row - 2, 0):min(row + 3, GRIDHEIGHT),
                                 max(column - 2, 0):min(column + 3, GRIDWIDTH)]
                        wallcount_2away = len(np.where(subMAP.flatten() == WALL)[0])

                        # this consolidates walls
                        # for first five generations build a scaffolding of wal

                        if generation < generations - 1:
                            # if looking 1 away in all directions you see 5 or more walls
                            # consolidate this point into a wall, if that doesnt happpen
                            # and if looking 2 away in all directions you see less than
                            # 7 walls, add a wall, this consolidates and adds walls
                            if wallcount_1away >= 5 or wallcount_2away <= 7:
                                cellMAP[row][column] = WALL

                            else:
                                cellMAP[row][column] = FLOOR

                            if row == 0 or column == 0 or row == GRIDHEIGHT - self.size \
                                    or column == GRIDWIDTH - self.size:
                                cellMAP[row][column] = WALL
                        # this consolidates open space, fills in standalone walls,
                        # after generation 5 consolidate walls and increase walking space
                        # if there are more than 5 walls nearby make that point a wall,
                        # otherwise add a floor
                        else:
                            if wallcount_1away >= 5:
                                cellMAP[row][column] = WALL
                            else:
                                cellMAP[row][column] = FLOOR
        return cellMAP

    def get_grid(self):
        '''
        Draw the black rectangles based on the cellMAP obtained from the
        game of life.
        :param map:
        :return:
        '''

        grid = []

        for x in range(0, self.height, self.size):
            for y in range(0, self.width, self.size):
                rect = pygame.Rect(x, y, self.size, self.size)
                grid.append(rect)

        return grid

    def get_obs(self):

        obs = []

        for index in range(0, len(self.grid)):
            if self.cellMAP.flatten()[index] == 0:
                obs.append(self.grid[index])

        return obs

    def draw(self, map):

        for ob in self.obstacles:
            pygame.draw.rect(map, self.black, ob, 0)


class Envir:

    def __init__(self, dimensions):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)

        # map dims
        self.height = dimensions[0]
        self.width = dimensions[1]

        # windows settings
        pygame.display.set_caption('Robot World')
        self.map = pygame.display.set_mode((self.height,
                                            self.width))

        self.font = pygame.font.Font('freesansbold.ttf', 50)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimensions[0] - 600,
                                dimensions[1] - 100)
        self.trail_set = []

    def write_info(self, Vl, Vr, theta):

        txt = f"Vl = {Vl} Vr = {Vr} theta = {int(math.degrees(theta))}"
        self.text = self.font.render(txt, True, self.white, self.black)
        self.map.blit(self.text, self.textRect)

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yellow, (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]))
        if self.trail_set.__sizeof__() > 3000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)


class Robot:

    def __init__(self, startpos, robot_img, width, chromosome):

        self.m2p = 3779.52  # meters 2 pixels
        self.w = width
        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.vl = 0.01 * self.m2p
        self.vr = 0.01 * self.m2p
        self.maxspeed = 0.2 * self.m2p
        self.minspeed = -0.2 * self.m2p
        self.ls_theta = []

        self.ls_x = []
        self.ls_y = []
        self.dist_travelled = 0
        self.avg_dist = 0
        self.flip = 0

        self.sensor = [0, 0, 0, 0, 0, 0]  # 1 tactile sensor (sensor[0]) and 5 ultrasound

        self.width = 25
        self.length = 25
        self.collision = 0

        self.img = pygame.image.load(robot_img)
        self.img = pygame.transform.scale(self.img, (self.width, self.length))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x,
                                                  self.y))

        self.chromosome = chromosome

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def get_collision(self, obstacles):

        self.sensor[0] = 0

        for obstacle in range(0, len(obstacles)):
            if self.rect.colliderect(obstacles[obstacle]):
                self.collision += 1
                self.sensor[0] = 1

    def move(self, height, width, dt, event=None, auto=False):

        self.ls_x.append(self.x)
        self.ls_y.append(self.y)

        if auto:

            actions = np.dot(self.sensor, self.chromosome)

            # Instead of having 2 options, the robot now has 4
            # such that it is also able to brake

            self.vl += actions[0]
            self.vl -= actions[1]
            self.vr += actions[2]
            self.vr -= actions[3]

        else:
            if event is not None:
                if event.type == pygame.KEYDOWN:

                    if event.key == 49:  # 1 is accelerate left
                        self.vl += 0.01 * self.m2p

                    elif event.key == 51:  # 3 is decelerate left
                        self.vl -= 0.01 * self.m2p

                    elif event.key == 54:  # 8 is accelerate right
                        self.vr += 0.01 * self.m2p

                    elif event.key == 56:  # 0 is decelerate right
                        self.vr -= 0.01 * self.m2p

        if self.sensor[0]:

            if 0.5 * math.pi > abs(self.theta) >= 0:
                # print('top right')
                self.x = self.x + 0.05
                self.y = self.y + 0.05

            if 1 * math.pi > abs(self.theta) >= 0.5 * math.pi:
                # print('top left')
                self.x = self.x + 0.05
                self.y = self.y - 0.05

            if 1.5 * math.pi > abs(self.theta) >= 1 * math.pi:
                # print('bottom left')
                self.x = self.x + 0.05
                self.y = self.y + 0.05

            else:
                # print('bottom right')
                self.x = self.x - 0.05
                self.y = self.y + 0.05

        else:

            self.x += ((self.vl + self.vr) / 2) * math.cos(self.theta) * dt
            self.y -= ((self.vl + self.vr) / 2) * math.sin(self.theta) * dt

        self.theta += (self.vr - self.vl) / self.w * dt

        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.flip += 1
            self.theta = 0

        # Detects if we're within map borders

        if self.x < 0 + 0.5 * self.length:
            self.x = 0 + 0.5 * self.length

        if self.x >= height - 1.2 * self.length:
            self.x = height - 1.2 * self.length

        if self.y < 0 + 0.5 * self.width:
            self.y = 0 + 0.5 * self.width

        if self.y >= width - 1.2 * self.width:
            self.y = width - 1.2 * self.width

        # set min speed
        self.vr = max(self.vr, self.minspeed)
        self.vl = max(self.vl, self.minspeed)

        # set max speed
        self.vr = min(self.vr, self.maxspeed)
        self.vl = min(self.vl, self.maxspeed)

        self.rotated = pygame.transform.rotozoom(self.img,
                                                 math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center=(self.x,
                                                  self.y))

        return

    def get_sensor(self, obstacles, map):

        if self.sensor[0] == 0:

            self.sensor = [0, 0, 0, 0, 0, 0]

            for obstacle in obstacles:
                dist_to_wall = calc_distance((self.x, self.y), (obstacle.x, obstacle.y))

                if dist_to_wall <= 65:
                    angle_w_wall = calc_angle((self.x, self.y), (obstacle.x, obstacle.y))
                    if 0 <= angle_w_wall < (math.pi * 2 * 1 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[1] = 1

                    elif (math.pi * 2 * 1 / 5) <= angle_w_wall < (math.pi * 2 * 2 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[2] = 1

                    elif (math.pi * 2 * 2 / 5) <= angle_w_wall < (math.pi * 2 * 3 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[3] = 1

                    elif (math.pi * 2 * 3 / 5) <= angle_w_wall < (math.pi * 2 * 4 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[4] = 1

                    else:
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[5] = 1

    def get_reward(self):

        for i in range(1, len(self.ls_x)):
            # Take the line integral

            delta_x = self.ls_x[i] - self.ls_x[i - 1]
            delta_y = self.ls_y[i] - self.ls_y[i - 1]
            delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)

            self.dist_travelled += delta_r

        self.avg_dist = math.sqrt((self.init_pos[0] - self.x) ** 2 + (self.init_pos[1] - self.y) ** 2)

        reward = self.dist_travelled * 5 + self.avg_dist * 3 - self.collision - self.flip

        return reward


def run_simulation(time, ls_chromosome, n_robots=1):
    # Initialize the robot
    start = (300, 200)
    img_path = "/Users/anishjadoenathmisier/Documents/GitHub/BioInspiredIntelligence/robot.png"

    ls_robots = list()
    ls_rewards = list()
    ls_chromosome = ls_chromosome

    tot_abs_dist = list()
    tot_avg = list()
    tot_coll = list()
    tot_reward = list()
    tot_flips = list()

    for i in range(n_robots):
        ls_robots.append(Robot(start, img_path, 25, ls_chromosome[i]))

    pygame.init()

    # Dimensions of the screen
    dims = (1400, 1000)

    environment = Envir(dims)
    environment.map.fill((255, 255, 255))

    # Obtain the walls
    wall = Walls(20, dims)

    dt = 0
    lasttime = pygame.time.get_ticks()
    # Simulation loop
    while pygame.time.get_ticks() <= time:

        for robot in ls_robots:
            robot.get_collision(wall.obstacles)
            robot.get_sensor(wall.obstacles, environment.map)

        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                if event.key == 48:
                    robot.x = 200
                    robot.y = 200

            if event.type == pygame.QUIT:
                pygame.quit()

            for robot in ls_robots:
                robot.move(environment.height, environment.width, dt, event, auto=True)

        dt = (pygame.time.get_ticks() - lasttime) / 1000
        lasttime = pygame.time.get_ticks()

        for robot in ls_robots:
            robot.move(environment.height, environment.width, dt, auto=True)

        pygame.display.update()

        environment.map.fill((255, 255, 255))
        wall.draw(environment.map)
        for robot in ls_robots:
            robot.draw(environment.map)

    pygame.quit()

    for robot in ls_robots:
        ls_rewards.append(robot.get_reward())

    for robot in ls_robots:
        tot_avg.append(robot.avg_dist)
        tot_abs_dist.append(robot.dist_travelled)
        tot_coll.append(robot.collision)
        tot_reward.append(robot.get_reward())
        tot_flips.append(robot.flip)

    print(f'AVERAGE REL DISTANCE: {np.mean(tot_avg)}')
    print(f'AVERAGE TOT DISTANCE: {np.mean(tot_abs_dist)}')
    print(f'AVERAGE COLLISIONS: {np.mean(tot_coll)}')
    print(f'AVERAGE REWARD: {np.mean(tot_reward)}')
    print(f'AVERAGE FLIPS: {np.mean(tot_flips)}')

    return ls_rewards, ls_chromosome


def selection(pop, scores, k=3):
    '''
    Using a tournement style method, we obtain the best
    agent in that population.
    :param pop:
    :param scores:
    :param k:
    :return:
    '''

    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix

    return pop[selection_ix]


def genetic_algorithm(n_iter, n_robots, sim_time):
    # Construct initial population, make sure to figure out how to make this
    # flow more logically

    pop = list()
    for i in range(0, n_robots):
        pop.append(np.random.uniform(low=0, high=1, size=(6, 4)))

    print('GENERATION 0')
    init_scores, init_chromo = run_simulation(sim_time, pop, n_robots=n_robots)

    best, best_eval = 0, init_scores[0]

    for gen in range(n_iter):
        print(f'GENERATION: {gen + 1}')
        scores, chromo = run_simulation(sim_time, pop, n_robots=n_robots)

        for i in range(n_robots):
            if scores[i] > best_eval:
                best, best_eval = chromo[i], scores[i]
                print(f'Generation {gen + 1} gives a new best with score {scores[i]}')
        selected = [selection(chromo, scores) for _ in range(n_robots)]

        children = list()
        for i in range(0, n_robots, 2):
            p1, p2 = selected[i], selected[i + 1]

            for c in crossover(p1, p2):
                mutation(c)
                children.append(c)

        pop = children

    return best, best_eval


a = np.ones((6, 2))
b = np.zeros((6, 2))


def crossover(parent1, parent2):
    random_choice = np.random.sample()
    random_row = np.random.randint(low=1, high=parent1.shape[0])
    random_column = np.random.randint(low=1, high=parent1.shape[1])

    empty_child1 = np.zeros(parent1.shape)
    empty_child2 = np.zeros(parent1.shape)

    # Horizontal crossover
    if random_choice > 0.5:
        empty_child1[:random_row, :] = parent1[:random_row, :]
        empty_child1[random_row:, :] = parent2[random_row:, :]
        child1 = empty_child1

        empty_child2[:random_row, :] = parent2[:random_row, :]
        empty_child2[random_row:, :] = parent1[random_row:, :]
        child2 = empty_child2

    else:
        empty_child1[:, :random_column] = parent1[:, :random_column]
        empty_child1[:, random_column:] = parent2[:, random_column:]
        child1 = empty_child1

        empty_child2[:, :random_column] = parent2[:, :random_column]
        empty_child2[:, random_column:] = parent1[:, random_column:]
        child2 = empty_child2

    return [child1, child2]


def mutation(individual, mutation_rate=1 / 24):
    random_choice = np.random.sample()

    if random_choice > mutation_rate:

        unequal = True

        while unequal:
            # Row and column to start the mutation swap with
            random_row_s = np.random.randint(low=0, high=individual.shape[0])
            random_column_s = np.random.randint(low=0, high=individual.shape[1])

            # Row and column to be swapped
            random_row_e = np.random.randint(low=0, high=individual.shape[0])
            random_column_e = np.random.randint(low=0, high=individual.shape[1])

            if (random_row_s != random_row_e) or (random_column_s != random_column_e):
                unequal = False

        p_start = individual[random_row_s, random_column_s]
        p_end = individual[random_row_e, random_column_e]

        individual[random_row_s, random_column_s] = p_end
        individual[random_row_e, random_column_e] = p_start

    return


best, best_eval = genetic_algorithm(n_iter=15, n_robots=14, sim_time=30000)

print(best, best_eval)
