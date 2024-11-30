import numpy as np
import os

import matplotlib.pyplot as plt


class Box:
    def __init__(self, l, b, x, y):
        self.l = l
        self.b = b
        self.x = x
        self.y = y

    def isinside(self, xp, yp, pad):
        xbound = (xp > self.x - 0.5 * self.l - pad) and (
            xp < self.x + 0.5 * self.l + pad
        )
        ybound = (yp > self.y - 0.5 * self.b - pad) and (
            yp < self.y + 0.5 * self.b + pad
        )
        inside = xbound and ybound
        return inside


class Maze:
    def __init__(self, xrange=(-0.9, 0.9), yrange=(-0.9, 0.9)):
        self.obstacles = []
        self.xrange = xrange
        self.yrange = yrange

        self.areas = np.array([obs.area for obs in self.obstacles])

    def sample(self, pad):
        while True:
            x = np.random.uniform(*self.xrange)
            y = np.random.uniform(*self.yrange)

            inside = False
            for obs in self.obstacles:
                if obs.isinside(x, y, pad=pad):
                    inside = True
                    break
            if not inside:
                return x, y

class MazeA(Maze):
    def __init__(self, xrange=(-1.2, 1.2), yrange=(-1.2, 1.2)):
        super().__init__(xrange=xrange, yrange=yrange)

        self.obstacles = [
            Box(0.7, 0.1, 0.6, 0.3),
            Box(0.1, 0.7, 0.3, 0.6),
            Box(0.7, 0.1, -0.6, -0.3),
            Box(0.1, 0.7, -0.3, -0.6),
            Box(0.7, 0.1, 0.6, -0.3),
            Box(0.1, 0.7, 0.3, -0.6),
            Box(0.7, 0.1, -0.6, 0.3),
            Box(0.1, 0.7, -0.3, 0.6),
        ]

class MazeB(Maze):
    def __init__(self, xrange=(-0.9, 0.9), yrange=(-0.9, 0.9)):
        super().__init__(xrange=xrange, yrange=yrange)

        self.obstacles = [
            Box(0.1, 0.8, -0.4, 0.0),
            Box(0.3, 0.1, -0.25, 0.4),
            Box(0.3, 0.1, -0.25, -0.4),
            Box(0.1, 0.8, 0.4, 0.0),
            Box(0.3, 0.1, 0.25, 0.4),
            Box(0.3, 0.1, 0.25, -0.4),
        ]


class MazeC(MazeB):
    def __init__(self, xrange=(-1.2, 1.2), yrange=(-1.2, 1.2)):
        super().__init__(xrange=xrange, yrange=yrange)

        self.obstacles.extend(
            [
                Box(1.4, 0.1, 0.0, 0.7),
                Box(0.1, 0.5, -0.7, 0.45),
                Box(0.1, 0.5, 0.7, 0.45),
                Box(1.4, 0.1, 0.0, -0.7),
                Box(0.1, 0.5, -0.7, -0.45),
                Box(0.1, 0.5, 0.7, -0.45),
            ]
        )


def generate(maze, nsample=100, pad=0.05):
    states = np.zeros((nsample, 4))

    for i in range(nsample):
        x, y = maze.sample(pad)
        states[i, 0] = x
        states[i, 1] = y

    return states

def generate_robot_obj_pos(maze, nsample=100, pad=0.1, min_dist=0.1, max_dist=0.5):
    """
    Generate robot and object positions with distance constraint
    """
    states = np.zeros((nsample, 4))

    for i in range(nsample):
        x_obj, y_obj = maze.sample(pad)
        states[i, 0] = x_obj
        states[i, 1] = y_obj

        # Generate object position based on robot position
        while True:
            x_robot, y_robot = maze.sample(pad)
            dist = np.linalg.norm([x_robot - x_obj, x_robot - y_obj])
            if dist > min_dist and dist < max_dist:
                break
        states[i, 2] = x_robot
        states[i, 3] = y_robot

    return states


def plot(states, pngfile):
    plt.plot(states[:, 0], states[:, 1], "b.")
    plt.savefig(pngfile)

def plot_robot_obj(states, pngfile):
    plt.plot(states[:, 0], states[:, 1], "b.")
    plt.plot(states[:, 2], states[:, 3], "r.")
    plt.savefig(pngfile)


if __name__ == "__main__":
    # maze = MazeC()
    # name = "maze_c"
    #
    # npfile = "tasks/maze/assets/reset_states/{}.npy".format(name)
    # pngfile = "tasks/maze/assets/reset_states/{}.png".format(name)
    # os.makedirs(os.path.dirname(npfile), exist_ok=True)
    # state = generate(maze, nsample=10000)
    # state = state.astype(dtype=np.float32)
    #
    # with open(npfile, "wb") as f:
    #     np.save(f, state)
    #
    # plot(state, pngfile)

    maze = MazeB()
    name = "maze_b"

    npfile = "tasks/maze/assets/reset_states/{}.npy".format(name)
    pngfile = "tasks/maze/assets/reset_states/{}.png".format(name)
    os.makedirs(os.path.dirname(npfile), exist_ok=True)
    state = generate_robot_obj_pos(maze, nsample=1000, min_dist=0.05, max_dist=0.1)
    state = state.astype(dtype=np.float32)

    with open(npfile, "wb") as f:
        np.save(f, state)

    plot_robot_obj(state, pngfile)
