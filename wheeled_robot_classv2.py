import math
import numpy as np
import pygame
import matplotlib.pyplot as plt

# Initialize the robot
start = (300, 200)
img_path = "/Users/anishjadoenathmisier/Documents/GitHub/BioInspiredIntelligence/robot.png"

np.random.RandomState()


################ FUNCTIONS ################

def get_init_pop(n_robots):
    population = list()
    for i in range(0, n_robots):
        population.append(np.random.randint(low=-255, high=255, size=(10, 2)))
    return population


def calc_distance(coord1, coord2):
    dist = math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    return dist


def calc_angle(coord1, coord2):
    if coord1[0] == coord2[0]:
        orient = 0
    else:
        orient = math.tan((coord1[1] - coord2[1]) / (coord1[0] - coord2[0]))

    return orient


def run_simulation(time, pop, n_robots):
    pygame.init()

    global start
    global img_path

    clock = pygame.time.Clock()
    ls_robots = list()
    scores = list()

    tot_abs_dist = list()
    tot_avg = list()
    tot_coll = list()
    tot_reward = list()
    tot_flips = list()

    dims = (1200, 800)

    environment = Envir(dims)
    environment.map.fill((255, 255, 255))

    # Obtain the walls
    wall = Walls(20, dims)


    draw = Draw(dims)

    for i in range(n_robots):
        ls_robots.append(Robot(start, img_path, 15, pop[i]))

    dt = 0
    lasttime = pygame.time.get_ticks()
    # Simulation loop
    while pygame.time.get_ticks() <= time:
        clock.tick(120)
        
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

       

        wall.get_rewards()
        environment.map.fill((255, 255, 255))
        wall.draw(environment.map, ls_robots)

        for robot in ls_robots:
            robot.get_sensor(wall.obstacles, environment.map)
            if sum(robot.sensor[1:]) > 0:
                robot.get_collision(wall.obstacles)
            robot.move(environment.height, environment.width, dt, auto=True)

            robot.draw(environment.map)

        draw.write_info(gen=GA.gen, time=pygame.time.get_ticks() / 1000, map=environment.map)

        pygame.display.update()
       
    pygame.quit()

    for robot in ls_robots:
        scores.append(robot.get_reward(time))

    for robot in ls_robots:
        tot_avg.append(robot.avg_dist)
        tot_abs_dist.append(robot.dist_travelled)
        tot_coll.append(robot.collision)
        tot_reward.append(robot.get_reward(time))
        tot_flips.append(robot.flip)

    GA.reward_gen.append(np.mean(tot_reward))
    GA.dist_gen.append(np.mean(tot_abs_dist))
    GA.rel_dist_gen.append(np.mean(tot_avg))
    GA.coll_gen.append(np.mean(tot_coll))

    return scores


################ ClASSES ################

class Walls:  # change name to world

    def __init__(self, size, dimensions):
        self.black = (0, 0, 0)
        self.yellow = (255, 255, 0)

        self.size = size
        self.width = dimensions[1]
        self.height = dimensions[0]
        self.amount_rewards = 20

        self.cellMAP = self.get_maze_map()
        self.grid = self.get_grid()
        self.obstacles = self.get_obs()
        self.rewards = list()

    def get_maze_map(self):

        WALL = 0
        FLOOR = 1

        np.random.seed(42)
        cellMAP = np.random.choice([1, 0], size=(int(self.height / self.size), int(self.width / self.size)),
                                   p=[0.80, 0.20])

        generations = 2
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

    def get_rewards(self):

        ls_coors = []

        for index in range(0, len(self.grid)):
            if self.cellMAP.flatten()[index] == 1:
                ls_coors.append(self.grid[index])

        for i in range(0, self.amount_rewards):
            reward_idx = np.random.randint(low=0, high=len(ls_coors))
            if len(self.rewards) < 20:
                self.rewards.append(ls_coors[reward_idx])

    def draw(self, map, robots):

        for ob in self.obstacles:
            pygame.draw.rect(map, self.black, ob, 0)

        for reward in self.rewards:
            pygame.draw.rect(map, self.yellow, reward, 0)

            for robot in robots:
                if reward.colliderect(robot.rect):
                    self.rewards.pop(self.rewards.index(reward))
                    break


class Envir:

    def __init__(self, dimensions):

        # map dims
        self.height = dimensions[0]
        self.width = dimensions[1]

        # windows settings
        pygame.display.set_caption('Robot World')
        self.map = pygame.display.set_mode((self.height,
                                            self.width))

    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map, self.yellow, (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]))
        if self.trail_set.__sizeof__() > 3000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)


class Draw:

    def __init__(self, dimensions):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)

        self.font = pygame.font.Font('freesansbold.ttf', 50)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimensions[0] - 600,
                                dimensions[1] - 100)
        self.trail_set = []

    def write_info(self, map, gen, time):
        txt = f"Generation: {gen}, Time: {time}"
        self.text = self.font.render(txt, True, self.white, self.black)
        map.blit(self.text, self.textRect)


class Robot:

    def __init__(self, startpos, robot_img, width, chromosome):

        self.m2p = 3779.52  # meters 2 pixels
        self.w = width * 10
        self.init_pos = startpos
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.vl = 1
        self.vr = 1
        self.maxspeed = 0.01 * self.m2p
        self.minspeed = -0.01 * self.m2p

        self.ls_tot_speed = list()

        self.ls_theta = []
        self.ls_x = []
        self.ls_y = []
        self.omega = 0

        self.dist_travelled = 0
        self.avg_dist = 0
        self.flip = 0
        self.stuck = 0

        self.sensor = [0, 0, 0, 0, 0, 0]  # 1 tactile sensor (sensor[0]) and 5 ultrasound

        self.width = width
        self.length = width
        self.collision = 0
        self.reward = 0

        self.img = pygame.image.load(robot_img)
        self.img = pygame.transform.scale(self.img, (self.width, self.length))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center=(self.x,
                                                  self.y))

        self.chromosome = chromosome

    def draw(self, world):
        world.blit(self.rotated, self.rect)

    def get_collision(self, obstacles):

        self.sensor[0] = 0

        for obstacle in obstacles:
            if self.rect.colliderect(obstacle):
                self.collision += 1
                self.sensor[0] = 1

    def get_reward(self, rewards):

        for reward in rewards:
            if self.rect.colliderect(reward):
                self.reward += 1
                rewards.pop(reward)

    def move(self, height, width, dt, event=None, auto=False):

        self.ls_x.append(self.x)
        self.ls_y.append(self.y)
        self.ls_theta.append(self.theta)

        if self.x == self.ls_x[-1] and self.y == self.ls_y[-1]:
            self.stuck += 1

        self.ls_tot_speed.append(self.vl + self.vr)

        if auto:

            states = np.zeros(shape=(1, 10), dtype=object)
            states[:, 0:-4] = self.sensor
            # Give the robot its own state as input
            states[:, -4] = self.theta
            states[:, -3] = self.omega
            states[:, -2] = self.vl
            states[:, -1] = self.vl

            actions = np.dot(states, self.chromosome)

            # Instead of having 2 options, the robot now has 4
            # such that it is also able to brake

            self.vl += actions[:, 0] * 0.015
            self.vr += actions[:, 1] * 0.015

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

                # check to see if the rotational velocity of the robot is not exceding 
        # the maximum rotational velocity 

        self.omega = (self.vr - self.vl) / self.w

        if self.omega >= 0.02 * math.pi:
            self.flip += 1
            self.omega = 0.02 * math.pi

        self.theta += self.omega * dt

        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.theta = 0

        if self.sensor[0]:

            self.x += (-0.1 * (self.vr + self.vl) * 0.5 * math.cos(self.theta) * dt)[0]
            self.y -= (-0.1 * (self.vr + self.vl) * 0.5 * math.sin(self.theta) * dt)[0]

        else:

            self.x += (((self.vl + self.vr) / 2) * math.cos(self.theta) * dt)[0]
            self.y -= (((self.vl + self.vr) / 2) * math.sin(self.theta) * dt)[0]




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

                if dist_to_wall <= 75:
                    angle_w_wall = calc_angle((self.x, self.y), (obstacle.x, obstacle.y))

                    if 0 <= angle_w_wall < (math.pi * 2 * 1 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[1] = 1 * 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 1 / 5) <= angle_w_wall < (math.pi * 2 * 2 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[2] = 1 * 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 2 / 5) <= angle_w_wall < (math.pi * 2 * 3 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[3] = 1 * 1 / max(1, dist_to_wall)

                    elif (math.pi * 2 * 3 / 5) <= angle_w_wall < (math.pi * 2 * 4 / 5):
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[4] = 1 * 1 / max(1, dist_to_wall)

                    else:
                        pygame.draw.line(map, (255, 0, 0), (self.x, self.y), (obstacle.x, obstacle.y))
                        self.sensor[5] = 1 * 1 / max(1, dist_to_wall)

    def get_reward(self, time):

        for i in range(1, len(self.ls_x)):
            # Take the line integral

            delta_x = float(self.ls_x[i] - self.ls_x[i - 1])
            delta_y = float(self.ls_y[i] - self.ls_y[i - 1])
            delta_r = np.sqrt(delta_x ** 2 + delta_y ** 2)
            self.dist_travelled += delta_r


        self.avg_dist = math.sqrt((self.init_pos[0] - self.x) ** 2 + (self.init_pos[1] - self.y) ** 2)
        avg_vel = self.avg_dist / time

        score = self.dist_travelled * 1 + self.avg_dist * 20  \
                - self.collision * 10  \
                + self.reward * 50 + avg_vel * 10

        fitness = score / (
                self.dist_travelled + self.avg_dist  + self.collision + avg_vel + self.reward)

        return float(fitness)


class GeneticAlgorithm:

    def __init__(self, n_robots, n_iter, cross_rate, mut_rate):

        self.n_robots = n_robots
        self.n_iter = n_iter
        self.scores = list()
        self.best_eval = 0
        self.pop = get_init_pop(self.n_robots)
        self.best = self.pop[0]

        self.gen = 0

        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.clone = False

        self.ls_robots = list()
        self.ls_rewards = list()
        self.tot_abs_dist = list()
        self.tot_avg = list()
        self.tot_coll = list()
        self.tot_reward = list()
        self.tot_flips = list()

        self.reward_gen = list()
        self.coll_gen = list()
        self.dist_gen = list()
        self.rel_dist_gen = list()
        self.flips_gen = list()

    def selection(self):  # k=5
        '''
        Using a tournement style method, we obtain the best
        agent in that population.
        :param pop:
        :param scores:
        :param k:
        :return:
        '''

        index = min(-1, -math.floor(self.n_robots * 0.20))  # take best 10%
        best_index = np.argsort(np.array(self.scores))[index:]
        best_pop = [self.pop[i] for i in best_index]
        other_index = np.argsort(np.array(self.scores))[:index]

        other_pop = [self.pop[i] for i in other_index]

        # selection_ix = np.random.randint(len(self.pop))
        # for ix in np.random.randint(0, len(self.pop), k - 1):
        #     if self.ls_rewards[ix] < self.ls_rewards[selection_ix]:
        #         selection_ix = ix

        return best_pop, other_pop

    def mutation(self, individual):
        random_choice = np.random.sample()

        if random_choice > self.mut_rate:

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

    def crossover(self, parent1, parent2):

        random_choice = np.random.sample()
        random_row = np.random.randint(low=1, high=parent1.shape[0])
        random_column = np.random.randint(low=1, high=parent1.shape[1])

        empty_child1 = np.zeros(parent1.shape)
        empty_child2 = np.zeros(parent1.shape)
        self.clone = True

        if self.cross_rate > random_choice:
            self.clone = False

            # Horizontal crossover
            if random_choice > 0.5:
                empty_child1[:random_row, :] = parent1[:random_row, :]
                empty_child1[random_row:, :] = parent2[random_row:, :]
                child1 = empty_child1

                empty_child2[:random_row, :] = parent2[:random_row, :]
                empty_child2[random_row:, :] = parent1[random_row:, :]
                child2 = empty_child2
            # Vertical crossover
            else:
                empty_child1[:, :random_column] = parent1[:, :random_column]
                empty_child1[:, random_column:] = parent2[:, random_column:]
                child1 = empty_child1

                empty_child2[:, :random_column] = parent2[:, :random_column]
                empty_child2[:, random_column:] = parent1[:, random_column:]
                child2 = empty_child2
        else:
            child1, child2 = parent1, parent2

        return [child1, child2]

    def main(self, sim_time):

        # Construct initial population, make sure to figure out how to make this
        # flow more logically

        print('GENERATION 0')

        self.scores = run_simulation(sim_time, self.pop, self.n_robots)
        self.best_eval = self.scores[0]

        for gen in range(self.n_iter):
            self.gen += 1
            print(f'GENERATION: {gen + 1}')
            self.scores = run_simulation(sim_time, self.pop, self.n_robots)

            for i in range(self.n_robots):
                if self.scores[i] > self.best_eval:
                    self.best, self.best_eval = self.pop[i], self.scores[i]
                    print(f'Generation {gen + 1} gives a new best with score {self.scores[i]}')
            # selected = [self.selection() for _ in range(self.n_robots)]
            ls_p1, ls_p2 = self.selection()

            children = list()
            mating = True
            while mating:

                random_parent1 = np.random.randint(0, len(ls_p1))
                parent1 = ls_p1[random_parent1]
                flip = np.random.uniform(0, 1)

                if flip > 0.4:
                    random_parent2 = np.random.randint(0, len(ls_p1))
                    parent2 = ls_p1[random_parent2]
                else:
                    random_parent2 = np.random.randint(0, len(ls_p2))
                    parent2 = ls_p2[random_parent2]

                for c in self.crossover(parent1, parent2):

                    if not self.clone:
                        self.mutation(c)
                        children.append(c)
                    else:
                        children.append(c)

                if len(children) >= self.n_robots:
                    mating = False

            # for i in range(0, self.n_robots, 2):
            #     p1, p2 = selected[i], selected[i + 1]
            #
            #     for c in self.crossover(p1, p2):
            #         if not self.clone:
            #             self.mutation(c)
            #             children.append(c)
            #         else:
            #             children.append(c)
            self.pop = children

        return self.best, self.best_eval

    def get_results(self, result):

        if result == 'fitness':
            result = self.reward_gen

        if result == 'tot_dist':
            result = self.dist_gen

        if result == 'rel_dist':
            result = self.rel_dist_gen

        if result == 'flip':
            result = self.flips_gen

        if result == 'coll':
            result = self.coll_gen

        return result

    def best_gen(self):
        return self.best


epochs = 25
GA = GeneticAlgorithm(n_robots=2 * 15, n_iter=epochs, cross_rate=0.8,
                      mut_rate=1 / 10)  # n_robots should be an equal number for crossover to work

best, best_eval = GA.main(60000)

x = range(0, epochs + 1)
plt.figure()
plt.title('FITNESS')
plt.grid()
plt.plot(x, GA.get_results('fitness'))

plt.figure()
plt.title('TOT DIST')
plt.grid()
plt.plot(x, GA.get_results('tot_dist'))

plt.figure()
plt.title('TOT COLL')
plt.grid()
plt.plot(x, GA.get_results('coll'))

plt.figure()
plt.title('REL DIST')
plt.grid()
plt.plot(x, GA.get_results('rel_dist'))

plt.show()
print(best, best_eval)
