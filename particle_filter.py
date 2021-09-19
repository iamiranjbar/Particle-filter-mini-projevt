import math
import pygame, sys
import numpy as np
from pygame.locals import *
from random import randrange, uniform

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
LINE_COLOR = (255,0,0)
BACKGROUND_COLOR = (0, 0, 0)
CIRCLE_COLOR = (0, 0, 255)
SAMPLES_COUNT = 1000
STAY_RATE = 0.9
BEST_SELECT_RATE = 0.3


def generate_random_pos(count):
    random_poses = []
    for _ in range(count):
        random_x = randrange(SCREEN_WIDTH)
        random_y = randrange(SCREEN_HEIGHT)
        random_pos = (random_x, random_y)
        random_poses.append(random_pos)
    return random_poses

def calculate_distance(first_point, second_point):
    first_x, first_y = first_point
    second_x, second_y = second_point
    return math.sqrt((second_x - first_x)**2 + (second_y - first_y)**2)

def calculate_angle(destination, source):
    source_x, source_y = source
    destination_x, destination_y = destination

    dx = destination_x - source_x
    dy = destination_y - source_y
    return math.atan2(dy, dx)

def estimate_location(best_poses):
    xs = []
    ys = []
    for pos in best_poses:
        x, y = pos
        xs.append(x)
        ys.append(y)
    location_x = np.mean(xs)
    location_y = np.mean(ys)
    print("Location: x = {} y = {}".format(location_x, location_y))

def generate_next_iteration_poses_best_select(iteration_poses, sensor_pos):
    distances = []
    for pos in iteration_poses:
        distances.append(calculate_distance(pos, sensor_pos))
    distances = np.array(distances)
    sorted_indexs = (-distances).argsort()
    next_iter_poses = []
    for index in sorted_indexs[::-1]:
        next_iter_poses.append(iteration_poses[index])
    stay_count = math.floor(STAY_RATE * SAMPLES_COUNT)
    next_iter_poses = next_iter_poses[:stay_count]
    estimate_location(next_iter_poses)
    new_count = SAMPLES_COUNT - stay_count
    new_random_poses = generate_random_pos(new_count)
    next_iter_poses.extend(new_random_poses)
    return next_iter_poses

def generate_next_iteration_poses_rouleetee_wheel(iteration_poses, sensor_pos):
    fitness = []
    for pos in iteration_poses:
        # fitness.append(1000 - calculate_distance(pos, sensor_pos))
        fitness.append(1/calculate_distance(pos, sensor_pos))
    probs = [fit/sum(fitness) for fit in fitness]
    next_iter_poses = []
    stay_count = math.ceil(STAY_RATE * SAMPLES_COUNT)
    next_iter_poses_indexes = np.random.choice(len(iteration_poses), p=probs, size=stay_count)
    next_iter_poses = np.array(iteration_poses)[next_iter_poses_indexes].tolist()
    new_count = SAMPLES_COUNT - stay_count
    new_random_poses = generate_random_pos(new_count)
    next_iter_poses.extend(new_random_poses)
    return next_iter_poses

def generate_next_iteration_poses_hybrid(iteration_poses, sensor_pos):
    distances = []
    for pos in iteration_poses:
        distances.append(calculate_distance(pos, sensor_pos))
    distances = np.array(distances)
    sorted_indexs = (-distances).argsort()
    next_iter_poses = []
    for index in sorted_indexs[::-1]:
        next_iter_poses.append(iteration_poses[index])
    select_best_count = math.floor(BEST_SELECT_RATE * SAMPLES_COUNT)
    next_iter_poses = next_iter_poses[:select_best_count]
    fitness = []
    for pos in iteration_poses:
        # fitness.append(1000 - calculate_distance(pos, sensor_pos))
        fitness.append(1/calculate_distance(pos, sensor_pos))
    probs = [fit/sum(fitness) for fit in fitness]
    stay_count = math.ceil(STAY_RATE * SAMPLES_COUNT) - select_best_count
    next_iter_poses_indexes = np.random.choice(len(iteration_poses), p=probs, size=stay_count)
    next_iter_poses.extend(np.array(iteration_poses)[next_iter_poses_indexes].tolist())
    new_count = SAMPLES_COUNT - stay_count
    new_random_poses = generate_random_pos(new_count)
    next_iter_poses.extend(new_random_poses)
    return next_iter_poses

def move_all_particles(poses, radius, angle):
    for pos_index, pos in enumerate(poses):
        pos_x, pos_y = pos
        pos_x += (radius * math.cos(angle))
        pos_y += (radius * math.sin(angle))
        poses[pos_index] = (pos_x, pos_y)
    return poses

def run():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    screen.fill(BACKGROUND_COLOR)
    pygame.display.set_caption('Particle Filter')
    pygame.display.flip()
    random_poses = generate_random_pos(SAMPLES_COUNT)
    for pos in random_poses:
        pygame.draw.circle(screen, (255, 255, 255), pos, 4)
    previous_mouse_x = 0
    previous_mouse_y = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if mouse_x == previous_mouse_x and mouse_y == previous_mouse_y:
            continue
        
        mouse_pos = (mouse_x, mouse_y)
        previous_mouse_pos = (previous_mouse_x, previous_mouse_y)
        motion_radius = calculate_distance(mouse_pos, previous_mouse_pos)
        motion_angle = calculate_angle(mouse_pos, previous_mouse_pos)
        motion_noisy_radius = motion_radius + np.random.normal(loc=0.1, scale=5)
        random_poses = move_all_particles(random_poses, motion_noisy_radius, motion_angle)

        sensor_x = mouse_x * uniform(0.6, 1.4) + 0.2
        sensor_y = mouse_y * uniform(0.6, 1.4)
        sensor_pos = (sensor_x, sensor_y)

        pygame.draw.line(screen, LINE_COLOR, (0, SCREEN_HEIGHT), (mouse_x, mouse_y))
        pygame.draw.line(screen, LINE_COLOR, (SCREEN_WIDTH, SCREEN_HEIGHT), (mouse_x, mouse_y))
        pygame.draw.circle(screen, CIRCLE_COLOR, (10, SCREEN_HEIGHT-10), 10)
        pygame.draw.circle(screen, CIRCLE_COLOR, (SCREEN_WIDTH-10, SCREEN_HEIGHT-10), 10)
        pygame.display.update()
        random_poses = generate_next_iteration_poses_best_select(random_poses, sensor_pos)
        # random_poses = generate_next_iteration_poses_rouleetee_wheel(random_poses, sensor_pos)
        # random_poses = generate_next_iteration_poses_hybrid(random_poses, sensor_pos)
        screen.fill(BACKGROUND_COLOR)
        for pos in random_poses:
            pygame.draw.circle(screen, (255, 255, 255), pos, 4)
        pygame.time.delay(10)

        previous_mouse_x = mouse_x
        previous_mouse_y = mouse_y

run()
