import os
import cv2
import time
import math
import shutil
import random
from collections import deque
from itertools import product

import matplotlib.pyplot as plt


MIN_XRANGE = 1
MIN_YRANGE = 1
MAX_XRANGE = 50
MAX_YRANGE = 50
GOAL_POINT = [25, 25]


def do_measurement(particles, measurement_noise=0):
    for idx in range(len(particles)):
        particles[idx][2] = abs(math.sqrt((particles[idx][0]-GOAL_POINT[0])**2 + (particles[idx][1]-GOAL_POINT[1])**2) + random.gauss(0.0, measurement_noise))
    return particles


def update_weights(particles):
    weights = [1 / (p[2] + 1e-5) for p in particles]
    weights_sum = sum(weights)
    return [w/weights_sum for w in weights]


def resample(particles, weights, num_particles, sampling_variance=0):
    particles_indices = random.choices(range(len(particles)), weights, k=num_particles)

    sampled_particles = []

    for idx in particles_indices:
        x = particles[idx][0] + random.gauss(0.0, sampling_variance)
        y = particles[idx][1] + random.gauss(0.0, sampling_variance)
        d = abs(math.sqrt((x-GOAL_POINT[0])**2 + (y-GOAL_POINT[1])**2) + random.gauss(0.0, sampling_variance))
        sampled_particles.append([x, y, d])

    return sampled_particles


def plot_particles(particles, iter_num, avg_dist_from_goal, global_avgcoord):
    x_coords = [p[0] for p in particles]
    y_coords = [p[1] for p in particles]
    plt.scatter(x_coords, y_coords, s=3, alpha=0.25)
    plt.scatter(sum(x_coords)/len(particles), sum(y_coords)/len(particles), s=6, color="black")

    plt.scatter(GOAL_POINT[0], GOAL_POINT[1], s=6, c="r")

    plt.annotate(f"iteration {iter_num}", (-2, 52), color="r")
    plt.annotate(f"{len(particles)} Particles", (-2, 49), color="r")
    plt.annotate(f"avg distance from goal is {avg_dist_from_goal}", (20, 52), color="g")
    plt.annotate(f"global coord is {global_avgcoord}", (20, 49), color="g")

    ax = plt.gca()
    ax.set_xlim([MIN_XRANGE-5, MAX_XRANGE+5])
    ax.set_ylim([MIN_YRANGE-5, MAX_YRANGE+5])

    plt.savefig(f"{dir_path}{iter_}.jpg")
    plt.savefig(f"particle_filter.jpg")
    plt.close()


if __name__ == "__main__":
    dir_path = ".temp_images/"
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    num_iter = 50
    num_particles = 250

    particles = [[random.randint(MIN_XRANGE, MAX_XRANGE), random.randint(MIN_YRANGE, MAX_YRANGE), 100] for _ in range(num_particles)]
    weights = [1/num_particles for _ in range(num_particles)]

    avgdist_list = deque()

    for iter_ in range(num_iter):
        if len(avgdist_list) < 6:
            avgdist_list.append(sum([p[2] for p in particles])/ num_particles)
        else:
            avgdist_list.rotate(-1)
            avgdist_list[-1] = sum([p[2] for p in particles])/ num_particles
        
        global_avgcoord = (
            round(sum([p[0] for p in particles]) / len(particles), 2),
            round(sum([p[1] for p in particles]) / len(particles), 2)
        )

        plot_particles(particles, iter_, round(avgdist_list[-1], 4), global_avgcoord)

        if avgdist_list[-1] < 1.2:
            print("Converged !")
            break
        
        if len(avgdist_list) < 8:
            particles = resample(particles, weights, num_particles=num_particles, sampling_variance=0.5)
        elif avgdist_list[-1] < sum(avgdist_list)/len(avgdist_list):
            num_particles = max(50, num_particles - int(num_particles * 0.1))
            particles = resample(particles, weights, num_particles=num_particles, sampling_variance=0)
        else:
            num_particles = min(200, num_particles + int(num_particles * 0.2))
            particles = resample(particles, weights, num_particles=num_particles, sampling_variance=1.5)

        particles = do_measurement(particles, measurement_noise=1)

        weights = update_weights(particles)

        time.sleep(0.5)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter("particle_filter.mp4", fourcc, 1, (640,480))

    for im in sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0])):
        img = cv2.imread(dir_path + im)
        videowriter.write(img)

    videowriter.release()
    shutil.rmtree(dir_path)