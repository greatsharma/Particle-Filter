import os
import cv2
import time
import math
import shutil
from collections import deque
import random
from itertools import product

import matplotlib.pyplot as plt


MIN_XRANGE = 1
MIN_YRANGE = 1
MAX_XRANGE = 50
MAX_YRANGE = 50
TRUE_CLOSEST_POINT = [25, 25]


def do_measurement(particles):
    for idx in range(len(particles)):
        particles[idx][2] = math.sqrt((particles[idx][0]-TRUE_CLOSEST_POINT[0])**2 + (particles[idx][1]-TRUE_CLOSEST_POINT[1])**2) + random.randint(0,2)
    return particles


def update_weights(particles):
    weights = [1 / (p[2] + 1e-5) for p in particles]
    weights_sum = sum(weights)
    return [w/weights_sum for w in weights]


def resample(particles, num_particles, randomness=0):
    particles = random.choices(particles, weights, k=num_particles)
    return [[p[0]+random.randint(-randomness,randomness), p[1]+random.randint(-randomness,randomness), p[2]] for p in particles]


def plot_particles(particles):
    x_coords = [p[0] + random.random() * random.choices([-1, 1], [1, 1])[0] for p in particles]
    y_coords = [p[1] + random.random() * random.choices([-1, 1], [1, 1])[0] for p in particles]
    plt.scatter(x_coords, y_coords, s=5, alpha=0.5)
    plt.scatter(TRUE_CLOSEST_POINT[0], TRUE_CLOSEST_POINT[1], s=5, c="r")
    # x_coords = [p[0] for p in particles]
    # y_coords = [p[1] for p in particles]
    # plt.scatter(x_coords, y_coords, s=2, c='black')
    ax = plt.gca()
    ax.set_xlim([MIN_XRANGE-5, MAX_XRANGE+5])
    ax.set_ylim([MIN_YRANGE-5, MAX_YRANGE+5])


if __name__ == "__main__":
    dir_path = "temp_images/"
    os.mkdir(dir_path)

    num_particles = 100

    x = random.choices(range(MIN_XRANGE, MAX_XRANGE), [1] * (MAX_XRANGE - MIN_XRANGE), k=int(math.sqrt(num_particles)))
    y = random.choices(range(MIN_YRANGE, MAX_YRANGE), [1] * (MAX_YRANGE - MIN_YRANGE), k=int(math.sqrt(num_particles)))

    distances = [100 for _ in range(num_particles)]
    avgdist_list = deque()

    weights = [1/num_particles for _ in range(num_particles)]

    particles = list(product(x, y))
    particles = [[p[0], p[1], d] for p,d in zip(particles, distances)]

    fig = plt.figure(figsize=(6, 6))

    for i in range(100):
        if len(avgdist_list) < 6:
            avgdist_list.append(sum([p[2] for p in particles])/ num_particles)
        else:
            avgdist_list.rotate(-1)
            avgdist_list[-1] = sum([p[2] for p in particles])/ num_particles
        
        global_avgcoord = (
            round(sum([p[0] for p in particles]) / len(particles), 2),
            round(sum([p[1] for p in particles]) / len(particles), 2)
        )

        print(f"iteration {i}: avg distance from goal is {round(avgdist_list[-1], 4)}: global coord is {global_avgcoord}")

        plot_particles(particles)
        plt.annotate(f"iteration {i}", (-2, 52), color="r")
        plt.annotate(f"{num_particles} Particles", (-2, 49), color="r")
        plt.annotate(f"avg distance from goal is {round(avgdist_list[-1], 4)}", (20, 52), color="g")
        plt.annotate(f"global coord is {global_avgcoord}", (20, 49), color="g")

        if avgdist_list[-1] < 2:
            plt.title("Converged")
            print("Converged !")
            break
        
        plt.savefig(f"{dir_path}particlefilter_{i}.jpg")
        plt.savefig(f"particle_filter.jpg")
        plt.close()

        if len(avgdist_list) < 8:
            particles = resample(particles, num_particles=100, randomness=1)
        elif avgdist_list[-1] < sum(avgdist_list)/len(avgdist_list):
            num_particles = max(50, num_particles - int(num_particles * 0.1))
            particles = resample(particles, num_particles=num_particles)
        else:
            num_particles = min(200, num_particles + int(num_particles * 0.2))
            particles = resample(particles, num_particles=num_particles, randomness=2)

        particles = do_measurement(particles)

        weights = update_weights(particles)

        time.sleep(0.5)

    plt.savefig(f"{dir_path}particlefilter_{i}.jpg")
    plt.savefig(f"particle_filter.jpg")
    plt.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter("particle_filter.mp4", fourcc, 1, (640,480))

    for im in sorted(os.listdir(dir_path)):
        img = cv2.imread(dir_path + im)
        videowriter.write(img)

    videowriter.release()
    shutil.rmtree(dir_path)