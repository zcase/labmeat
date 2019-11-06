import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 


from scipy.spatial import Delaunay
from skimage.draw import line
from bresenham import bresenham


class VascularGenerator:
    def __init__(self, min_range=1, max_range=10, dim=2, num_of_nodes=20):
        self.graph = defaultdict(list)
        # Create Wall start and end points
        self.min_range = min_range
        self.max_range = max_range
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


    def add_edge_to_graph(self, np_pt1, np_pt2):
        # Compares the x values
        if np_pt1[0] < np_pt2[0]:
            if not self.arreq_in_list(np_pt2, self.graph[tuple(np_pt1)]):
                self.graph[tuple(np_pt1)].append(np_pt2)
        elif np_pt1[0] == np_pt2[0] and np_pt1[1] > np_pt2[1]:              # Added this to enforce bottom-right node as exit
            if not self.arreq_in_list(np_pt2, self.graph[tuple(np_pt1)]):
                self.graph[tuple(np_pt1)].append(np_pt2)
        else:
            if not self.arreq_in_list(np_pt1, self.graph[tuple(np_pt2)]):
                self.graph[tuple(np_pt2)].append(np_pt1)

    def arreq_in_list(self, myarr, list_arrays):
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

    def arreq_in_list_wIdx(self, myarr, list_arrays):
        return next((idx for idx, elem in enumerate(list_arrays) if np.array_equal(elem, myarr)), -1)

    def create_pt_edge_dict(self, pts, edges):
        for pt1, pt2 in edges:
            # print('Pt1: ', pt1, '  Pt2: ', pt2)
            self.add_edge_to_graph(pt1, pt2)

    def depth_first_search(self):
        # https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
        num_nodes = len(self.pts)
        visted_nodes = [False] * num_nodes

        for i in range(num_nodes):
            if not visted_nodes[i]:
                node = self.pts[i]
                self.dfs_helper(i, node, visted_nodes)

    def dfs_helper(self, node_idx, node, visted_nodes):
        visted_nodes[node_idx] = True
        print(node)

        for i, node in enumerate(self.graph[tuple(node)]):
            if not visted_nodes[node_idx]:
                self.dfs_helper(i, node, visted_nodes)

    def generate_edges(self, tri, pts):
        edges = []
        for _, s in enumerate(tri.simplices):
            tri_pts = pts[s]

            for i, pt in enumerate(tri_pts):
                if i == 0:
                    continue
                elif pt[0] == tri_pts[-1][0] and pt[1] == tri_pts[-1][1]:
                    self.add_edge_to_graph(pt, tri_pts[0])

                self.add_edge_to_graph(tri_pts[i-1], pt)

        topl_to_topR = (np.array([self.min_range, self.max_range]), np.array([self.max_range, self.max_range]))
        btml_to_btmR = (np.array([self.min_range, self.min_range]), np.array([self.max_range, self.min_range]))

        for pt_pair in [topl_to_topR, btml_to_btmR]:
            pt1, pt2 = pt_pair
            if self.arreq_in_list(pt2, self.graph[tuple(pt1)]):
                idx = self.arreq_in_list_wIdx(pt2, self.graph[tuple(pt1)])
                del self.graph[tuple(pt1)][idx]
            elif self.arreq_in_list(pt1, self.graph[tuple(pt2)]):
                idx = self.arreq_in_list_wIdx(pt1, self.graph[tuple(pt2)])
                del self.graph[tuple(pt1)][idx]

        edges = list()
        for tuple_pt, np_pt_list in self.graph.items():
            for np_pt in np_pt_list:
                edges.append((np.asarray(tuple_pt), np_pt))

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
    
    def add_flows_to_img(self, flow_dict):
        for pt_key, flow_val in flow_dict:
            pt1, pt2 = pt_key
            pt1 = np.asarray(pt1)
            pt2 = np.asarray(pt2)
            pts_on_line = np.array(list(bresenham(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))))
            self.img[pts_on_line[:,0], pts_on_line[:,1]] = flow_val

        return self.img

    # ====================== #
    # ==== Create Walls ==== #
    # ====================== #
    def create_walls(self, min, max, num_of_nodes):
        # Number of Wall nodes = 1% of number of nodes
        high_node_val = int(num_of_nodes / 5)
        num_left_nodes = 1
        num_right_nodes = 1
        if high_node_val > 1:
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

        pts = np.array(sorted(pts, key=lambda k: (-k[0], k[1]),  reverse=True))

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

    def pretty(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))
