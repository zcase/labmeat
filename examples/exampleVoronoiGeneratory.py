import os
import numpy as np
from scipy.spatial import voronoi_plot_2d, Voronoi
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.lines as mlines

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

class VoronoiGenerator:
    def __init__(self, num_of_nodes, dim=2, min_range=0, max_range=10):
        self.ponits = None
        # TODO(zcase): Bound this within an area and all lines touching a boundry become a vertice

        if dim == 2:
            self.points = np.random.uniform(low=min_range, high=max_range, size=(num_of_nodes, dim))
        if dim >= 3:
            dim_val = 3
            self.points = np.random.uniform(low=min_range, high=max_range, size=(num_of_nodes, dim_val))

        # print(self.points)

        self.voronoi = Voronoi(self.points)


if __name__ == "__main__":
    voronoi_gen = VoronoiGenerator(5)

    print('Vertices: ')
    print(voronoi_gen.voronoi.vertices)

    print('\n\nRidge Vertices: ')
    print(voronoi_gen.voronoi.ridge_vertices)

    print('\n\nRidge Points: ')
    print(voronoi_gen.voronoi.ridge_points)

    ptp_bound = voronoi_gen.voronoi.points.ptp(axis=0)
    line_segments = []
    center = voronoi_gen.voronoi.points.mean(axis=0)
    for pointidx, simplex in zip(voronoi_gen.voronoi.ridge_points, voronoi_gen.voronoi.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = voronoi_gen.voronoi.points[pointidx[1]] - voronoi_gen.voronoi.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi_gen.voronoi.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi_gen.voronoi.vertices[i] + direction * ptp_bound.max()

            line_segments.append([(voronoi_gen.voronoi.vertices[i, 0], voronoi_gen.voronoi.vertices[i, 1]),
                                  (far_point[0], far_point[1])])

    print('line segments: ', len(line_segments))
    print(line_segments)

    plt.scatter(voronoi_gen.points[:,0], voronoi_gen.points[:,1])
    # plt.show()

    voronoi_plot_2d(voronoi_gen.voronoi)
    plt.show()

    fig = plt.figure()

    # # Mark the Voronoi vertices.
    # plt.plot(voronoi_gen.voronoi.vertices[:,0], voronoi_gen.voronoi.vertices[:, 1], 'ko', ms=8)

    # for vpair in voronoi_gen.voronoi.ridge_vertices:
    #     if vpair[0] >= 0 and vpair[1] >= 0:
    #         v0 = voronoi_gen.voronoi.vertices[vpair[0]]
    #         v1 = voronoi_gen.voronoi.vertices[vpair[1]]

    #         print('\n\n Edges: ')
    #         print(v0, v1)
    #         # Draw a line from v0 to v1.
    #         plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)

    # plt.show()