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

        # self.edges = self.generate_edges(self.pts)


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


    # def generate_edges(self, np_pts):
    #     edges = []
    #     # for point in np_pts:
    #     #     print(point)

    #     # print('\npts: ')
    #     # print(np_pts)
    #     # print('\nsorted pts: ')
    #     # np_pts.sort(axis=0)
    #     np_pts.view('i8,i8').sort(order=['f0'], axis=0)
    #     # print(np_pts, type(np_pts))


    #     # print(np_pts[0])
    #     # print(np_pts[-1])

    #     min_ind = np.array([i for i,v in enumerate(np_pts[:,0]) if v == np_pts[0][0]])
    #     max_ind = np.array([i for i,v in enumerate(np_pts[:,0]) if v == np_pts[-1][0]])

    #     # print('min: ')
    #     # print(np_pts[min_ind])
    #     # print('max: ')
    #     # print(np_pts[max_ind])

    #     # Create Left Vert Line
    #     for i, point in enumerate(np_pts[min_ind]):
    #         if i == 0:
    #             continue
    #         else:
    #             edges.append((np_pts[min_ind][i-1], point))

    #     # Create Right Vert Line
    #     for i, point in enumerate(np_pts[max_ind]):
    #         if i == 0:
    #             continue
    #         else:
    #             edges.append((np_pts[max_ind][i-1], point))


    #     for point in np_pts:
    #         # print(point)
    #         # indexs = np.where(np.all(np_pts[:, 0] != point) and np_pts > point[0])
    #         # print('    Point: ', point, point[0], point[1])
    #         # print('    Points: ', np_pts[:,0])
    #         ind = np.array([i for i,v in enumerate(np_pts[:,0]) if v > point[0]])
    #         if ind != []:
    #             ind = ind[:2]
    #             # https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array
    #             # print('ind: ', ind)
    #             num_of_edges = np.random.randint(low=1, high=len(ind))
    #             idx = np.random.randint(len(np_pts[ind]), size=num_of_edges)
    #             p = np_pts[ind]
    #             # print('    ind         : ', ind)
    #             # print('    pts[ind]    : \n', np_pts[ind], '\n END pts[ind]')
    #             # print('    num_of_edges: ', num_of_edges)
    #             # print('      p[idx]    :', p[idx], )

    #             for chosen_pt in p[idx]:
    #                 edges.append((point, chosen_pt))
    #             num_of_edges = np.random.randint(low=0, high=6)
    #             print(num_of_edges)
    #             # if num_of_edges >= 3:
    #             #     edges.append((point, np_pts[ind]))

    #     # print(edges)

    #     return edges

    def is_planar(self, np_edges):
        pass
    # There should be a package called planar that can check this

def find_neighbors(pindex, triang):
    neighbors = list()
    for simplex in triang.vertices:
        if pindex in simplex:
            neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
            '''
            this is a one liner for if a simplex contains the point we`re interested in,
            extend the neighbors list by appending all the *other* point indices in the simplex
            '''
    #now we just have to strip out all the dulicate indices and return the neighbors list:
    return list(set(neighbors))
if __name__ == "__main__":
    vas_structure = VascularGenerator(num_of_nodes=3)

    # print('\n\n\nEDGES:')
    # for edge in vas_structure.edges:
    #     print(edge)

    # for pt_1, pt_2 in vas_structure.edges:
    #     # print('p1: ', pt_1, '  ', )
    #     # print('p2: ', pt_2)
    #     combined = np.concatenate((np.array([pt_1]), np.array([pt_2])))
    #     # print('combined: ', combined)
    #     # print('\n\n')
    #     plt.plot(combined[:,0], combined[:,1], 'ro-')

    # plt.savefig('test.png')




    tri = Delaunay(vas_structure.pts)
    # print(tri.simplices)
    print('len: ', len(vas_structure.pts))
    for pt in vas_structure.pts:
        print('Pt: ', pt)
    print('\n')
    #     # print('    ', pt[tri.simplices])

    edges = []
    for j, s in enumerate(tri.simplices):
        p = vas_structure.pts[s].mean(axis=0)
        print(s)
        print('    Pts : ', vas_structure.pts[s])
        tri_pts = vas_structure.pts[s]
        for i, pt in enumerate(tri_pts):
            if i == 0:
                continue
            else:
                edges.append((tri_pts[i-1], pt))

        print('    Mean:', p)
        plt.text(p[0], p[1], '#%d' % j, ha='center') # label triangles
    print('len: ', len(vas_structure.pts))
    plt.triplot(vas_structure.pts[:,0], vas_structure.pts[:,1], tri.simplices)
    plt.plot(vas_structure.pts[:,0], vas_structure.pts[:,1], 'o')
    plt.savefig('test.png')

    print(edges)

