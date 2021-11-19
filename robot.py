import time
import random
from math import *
import matplotlib.pyplot as plt


class Robot():
    def __init__(self, map: list, landmarks: list = None):
        self.map = map
        self.x_range = map[0]
        self.y_range = map[1]
        self.x_pos = random.uniform(0., self.x_range)
        self.y_pos = random.uniform(0., self.y_range)
        self.forward_noise = 0.0
        self.sense_noise = 0.0

        if landmarks:
            if any(len(lm) != 2 for lm in landmarks):
                raise ValueError(
                    'all landmarks should of length 2 containing [x_pos, y_pos]')
            else:
                self.landmarks = landmarks
        
    def set_position(self, pos: list):
        if pos[0] < 0. or pos[0] > self.x_range:
            raise ValueError('x position out of range')
        if pos[1] < 0. or pos[1] > self.y_range:
            raise ValueError('y position out of range')

        self.x_pos = pos[0]
        self.y_pos = pos[1]

    def set_noise(self, noise: list):
        if any(n < 0. for n in noise):
            raise ValueError('noise cannot be negative')

        self.forward_noise = noise[0]
        self.sense_noise = noise[1]

    def sense(self):
        sensor_z = []
        for lm in self.landmarks:
            dist = sqrt((self.x_pos-lm[0])**2 + (self.y_pos-lm[1])**2)
            dist += random.gauss(0.0, self.sense_noise)
            sensor_z.append(dist)

        return sensor_z

    def move(self, x_move, y_move):
        if x_move < 0.0 or y_move < 0.0:
            raise ValueError('moves cannot be negative')

        if x_move:
            self.x_pos += x_move + random.gauss(0., self.forward_noise)
            self.x_pos %= self.x_range

        if y_move:
            self.y_pos += y_move + random.gauss(0., self.forward_noise)
            self.y_pos %= self.y_range

    def measurement_likelihood(self, sensor_z: list):
        if len(sensor_z) != len(self.landmarks):
            raise ValueError('incomplete sensor measurements')

        likelihood = 1.
        for z, lm in zip(sensor_z, self.landmarks):
            dist = sqrt((self.x_pos-lm[0])**2 + (self.y_pos-lm[1])**2)
            likelihood *= exp(-((dist - z)**2)/(self.sense_noise**2) /
                              2.) / sqrt(2.0 * pi * (self.sense_noise ** 2))

        return likelihood

    def __getitem__(self, index):
        if index == 0:
            return self.x_pos
        elif index == 1:
            return self.y_pos
        else:
            raise ValueError('invalid index')

    def __setitem__(self, index, val):
        if index == 0:
            if val < 0. or val > self.x_range:
                raise ValueError('x position out of range')
            self.x_pos = val
        elif index == 1:
            if val < 0. or val > self.y_range:
                raise ValueError('y position out of range')
            self.y_pos = val
        else:
            raise ValueError('invalid index')

    def __eq__(self, robot):
        if not isinstance(robot, Robot):
            return False

        return self.x_pos == robot.x_pos and self.y_pos == robot.y_pos

    def __repr__(self):
        return '[x_pos: {}  y_pos: {}]'.format(self.x_pos, self.y_pos)


def plot_model(map, landmarks, robot, particles):
    plt.scatter([lm[0] for lm in landmarks], [lm[1] for lm in landmarks], s=60, color="black")

    xcoords = [p.x_pos for p in particles]
    ycoords = [p.y_pos for p in particles]
    plt.scatter(xcoords, ycoords, s=5, alpha=0.25, color="blue")

    plt.scatter(robot.x_pos, robot.y_pos, s=5, alpha=1, color="red")

    ax = plt.gca()
    ax.set_xlim([-10, map[0]+30])
    ax.set_ylim([-10, map[1]+30])


def loss(robot, particles, map):
    sum = 0.
    for p in particles:
        dx = ((p.x_pos - robot.x_pos + map[0]/2.) % map[0]) - map[0]/2.
        dy = ((p.y_pos - robot.y_pos + map[1]/2.) % map[1]) - map[1]/2.
        err = sqrt(dx*dx + dy*dy)
        sum += err

    return round(sum / len(particles), 4)


def resample(particles, weights, map, landmarks, num_particles, randomness=0):
    particles_indices = random.choices(range(len(particles)), weights, k=num_particles)

    sampled_particles = []

    for idx in particles_indices:
        p = Robot(map, landmarks)
        p.set_noise([0.0, 2.0])
        p.x_pos = particles[idx].x_pos + random.gauss(0.0, randomness)
        p.x_pos %= p.x_range
        p.y_pos = particles[idx].y_pos  + random.gauss(0.0, randomness)
        p.y_pos %= p.y_range
        sampled_particles.append(p)

    del particles

    return sampled_particles


def apply_particle_filter(map, landmarks, n_particles, n_iter):
    robot = Robot(map, landmarks)
    robot.set_noise([0.0, 1.5])

    robot.x_pos = 40
    robot.y_pos = 50
 
    # initalize particles
    particles = []
    for i in range(n_particles):
        p = Robot(map, landmarks)
        p.set_noise([0.0, 2.0])
        particles.append(p)

    # run particle filter
    for iter in range(n_iter):
        print(f"iteration {iter}")

        sensor_z = robot.sense()

        plot_model(map, landmarks, robot, particles)
        
        plt.annotate(f"Iteration {iter}", (10, 120), color="r")
        plt.annotate(f"Total particles {n_particles}", (10, 113), color="r")
        plt.annotate(f"Loss {loss(robot, particles, map)}", (70, 120), color="r")

        plt.annotate(f"x:{round(robot.x_pos)}", (robot.x_pos+6, robot.y_pos+8), color="g")
        plt.annotate(f"y:{round(robot.y_pos)}", (robot.x_pos+6, robot.y_pos+2), color="g")
        plt.annotate(f"z:{[round(z) for z in sensor_z]}",(robot.x_pos+6, robot.y_pos-4),color="g")

        for idx, lm in enumerate(landmarks):
           plt.annotate(f"Z{idx+1}", (lm[0], lm[1]-8), color="black")

        plt.savefig(f"robot.jpg")
        plt.close()

        weights = []
        for p in particles:
            weights.append(p.measurement_likelihood(sensor_z))

        w_norm = sum(weights)
        weights = [w/(w_norm + 1e-6) for w in weights]

        # resampling
        if loss(robot, particles, map) < 2:
            n_particles = max(50, n_particles - int(n_particles * 0.1))
        else:
            n_particles = min(200, n_particles + int(n_particles * 0.2))

        particles = resample(particles, weights, map, landmarks, num_particles=n_particles, randomness=1)

        x_move = random.randint(1, 4)
        y_move = random.randint(1, 4)
        robot.move(x_move, y_move)

        for p in particles:
            p.move(x_move, y_move)

        time.sleep(1)

    return robot, particles


if __name__ == '__main__':
    map = [100, 100]
    landmarks = [[20., 20.], [80., 80], [20., 80.], [80., 20.]]
    robot, particles = apply_particle_filter(map, landmarks, 100, n_iter=50)