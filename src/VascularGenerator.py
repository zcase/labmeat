import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from scipy.spatial import Delaunay
from skimage.draw import line
from bresenham import bresenham

class VascularGenerator:
    def __init__(self, min_range=1, max_range=10, dim=2, num_of_nodes=20):
        # Create Wall start and end points
        self.left_wall_startend = np.array([[min_range, min_range], [min_range, max_range]])
        self.right_wall_startend = np.array([[max_range, min_range], [max_range, max_range]])

        # Create Left and Right Walls
        self.left_wall, self.right_wall = self.create_walls(min_range, max_range, num_of_nodes)

        # Generate random points
        self.pts = self.generate_pts(min_range, max_range, num_of_nodes, dim,
                                     self.left_wall, self.right_wall,
                                     self.left_wall_startend, self.right_wall_startend)

        # Create the Planar graph using the Delaunay Method
        self.tri = Delaunay(self.pts)

        # Create Edges
        self.edges = self.generate_edges(self.tri, self.pts)

        # Convert Edges to image
        self.img = self.convert_to_img2(self.edges, max_range)


    def generate_edges(self, tri, pts):
        edges = []
        for _, s in enumerate(tri.simplices):
            tri_pts = pts[s]

            for i, pt in enumerate(tri_pts):
                if i == 0:
                    continue
                elif pt[0] == tri_pts[-1][0] and pt[1] == tri_pts[-1][1]:
                    edges.append((pt, tri_pts[0]))

                edges.append((tri_pts[i-1], pt))

        return edges


    def convert_to_img2(self, edges, max_range):
        img = np.zeros((max_range+1, max_range+1), dtype=float)

        for edge in edges:
            pt1, pt2 = edge
            pts_on_line = np.array(list(bresenham(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))))

            img[pts_on_line[:,0], pts_on_line[:,1]] = 1
            img[int(pt1[0]), int(pt1[1])] = 2
            img[int(pt2[0]), int(pt2[1])] = 2

        img = np.pad(np.rot90(img), (0,0), 'linear_ramp', end_values=(5, -4))

        return img

    # ====================== #
    # ==== Create Walls ==== #
    # ====================== #
    def create_walls(self, min, max, num_of_nodes):
        # Number of Wall nodes = 1% of number of nodes
        high_node_val = int(num_of_nodes / 5)
        num_left_nodes = 1
        num_right_nodes = 1
        if high_node_val != 0:
            num_left_nodes = np.random.randint(1, high=high_node_val)
            num_right_nodes = np.random.randint(1, high=high_node_val)

        left_wall = np.empty([num_left_nodes, 2], dtype=float)
        right_wall = np.empty([num_right_nodes, 2], dtype=float)

        # Create Nodes for Left Wall (In)
        for i in range(num_left_nodes):
            node = np.array([[min, np.random.randint(min + 1, max - 1)]], dtype=float)
            if left_wall != np.array([]) and np.all(left_wall[:, 0] == node):
                i -= 1
            else:
                left_wall[i] = node[0]

        # Create Nodes for Right Wall (Out)
        for i in range(num_right_nodes):
            node = np.array([[max, np.random.randint(min + 1, max - 1)]], dtype=float)
            if right_wall != np.array([]) and np.all(right_wall[:, 0] == node):
                i -= 1
            else:
                right_wall[i] = node[0]

        return left_wall, right_wall

    def calculate_distance(self, point, neighbor):
        '''
        Euclidean distance
        '''
        dist = np.linalg.norm(point - neighbor)
        return dist

    def generate_pts(self, min_range, max_range, num_pts, dim, lt_wall, rht_wall, lt_startend, rht_startend):
        pts = np.random.uniform(low=min_range, high=max_range, size=(num_pts, dim))

        pts = np.append(pts, lt_wall, axis=0)
        pts = np.append(pts, rht_wall, axis=0)
        pts = np.append(pts, lt_startend, axis=0)
        pts = np.append(pts, rht_startend, axis=0)

        return pts

    def count_nodes(self, *argv):
        count = 0
        for arg in argv:
            count += len(arg)
        return count


    def print_images(self):
        for j, s in enumerate(self.tri.simplices):
            p = self.pts[s].mean(axis=0)
            plt.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        plt.triplot(self.pts[:,0], self.pts[:,1], self.tri.simplices)
        plt.plot(self.pts[:,0], self.pts[:,1], 'o')
        plt.savefig('Vasc_Graph.png')

        plt.imsave('Vasc2D_img.png', self.img, cmap='Greys')
