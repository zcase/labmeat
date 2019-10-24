import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from scipy.spatial import Delaunay

class VascularGenerator:
    def __init__(self, min_range=1, max_range=10, dim=2, num_of_nodes=20):
        # if num_of_nodes < 10:
        #     num_of_nodes = 10

        self.left_wall_startend = np.array([[min_range, min_range], [min_range, max_range]])
        self.right_wall_startend = np.array([[max_range, min_range], [max_range, max_range]])

        self.left_wall, self.right_wall = self.create_walls(min_range, max_range, num_of_nodes)


        self.left_wall_nodes = self.count_nodes(self.left_wall, self.left_wall_startend)
        self.right_wall_nodes = self.count_nodes(self.right_wall, self.right_wall_startend)

        # print("left: ", self.left_wall_nodes)
        # print("right: ", self.right_wall_nodes)

        self.pts = self.generate_pts(min_range, max_range, num_of_nodes, dim,
                                     self.left_wall, self.right_wall,
                                     self.left_wall_startend, self.right_wall_startend)

        self.tri = Delaunay(self.pts)
        self.edges = self.generate_edges(self.tri, self.pts)

    def generate_edges(self, tri, pts):
        edges = []
        for j, s in enumerate(tri.simplices):
            p = pts[s].mean(axis=0)
            tri_pts = pts[s]
            for i, pt in enumerate(tri_pts):
                if i == 0:
                    continue
                else:
                    edges.append((tri_pts[i-1], pt))

        #     plt.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        # print('len: ', len(pts))
        # plt.triplot(pts[:,0], pts[:,1], tri.simplices)
        # plt.plot(pts[:,0], pts[:,1], 'o')
        # plt.savefig('test.png')

        # print(edges)
        return edges


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
        # print('pts    : ', pts)
        # print('lt_wall: ', lt_wall)
        # print('rt_wall: ', rht_wall)
        # print('lt_star: ', lt_startend)
        # print('rt_star: ', rht_startend)
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


if __name__ == "__main__":
    vas_structure = VascularGenerator(num_of_nodes=3)


