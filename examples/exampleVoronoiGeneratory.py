import os
import numpy as np
from scipy.spatial import voronoi_plot_2d, Voronoi
import matplotlib
import matplotlib.pyplot as plt


class VoronoiGenerator:
    def __init__(self, num_of_nodes, min_range=0, max_range=10):
        self.points = np.random.uniform(low=min_range, high=max_range, size=(num_of_nodes, 2))
        self.voronoi = Voronoi(self.points)


if __name__ == "__main__":
    voronoi_gen = VoronoiGenerator(200)

    plt.scatter(voronoi_gen.points[:,0], voronoi_gen.points[:,1])
    # plt.show()

    voronoi_plot_2d(voronoi_gen.voronoi)
    # plt.show()

    print(voronoi_gen.voronoi.vertices)