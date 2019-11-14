# import numpy as np
import autograd.numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict 
from scipy.spatial import Delaunay
from skimage.draw import line
from bresenham import bresenham
import os


class VascularGenerator:
    def __init__(self, min_range=1, max_range=10, dim=2, num_of_nodes=20):
        self.graph = defaultdict(list)
        self.update_count = 0
        # Create Wall start and end points
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.left_wall_startend = np.array([[self.min_range, self.min_range], [self.min_range, self.max_range]])
        self.right_wall_startend = np.array([[self.max_range, self.min_range], [self.max_range, self.max_range]])

        # Create Left and Right Walls
        self.left_wall, self.right_wall = self.create_walls(self.min_range, self.max_range, num_of_nodes)

        # Generate random points
        self.pts, self.moveable_pts = self.generate_pts(self.min_range, self.max_range, num_of_nodes, dim,
                                                        self.left_wall, self.right_wall,
                                                        self.left_wall_startend, self.right_wall_startend)

        # Create the Planar graph using the Delaunay Method
        self.tri = Delaunay(self.pts)

        # Create Edges
        self.edges = self.generate_edges(self.tri, self.pts)

        # Convert Edges to image
        self.img = self.convert_to_img2(self.edges, max_range)


    def add_edge_to_graph(self, np_pt1, np_pt2):
        # print('Line 42: ', np_pt1, type(np_pt1))
        # print('Add edge: line 40: ')
        # print('  ', np_pt1, list(np_pt1))
        if type(np_pt1) != type(np.array((2,4))):
            np_pt1 = list(np_pt1._value)
            np_pt2 = list(np_pt2._value)
        else:
            np_pt1 = list(np_pt1)
            np_pt2 = list(np_pt2)
        # print('Adding To Graph: ', np_pt1, np_pt2)
        if tuple(np_pt1) not in self.graph.keys():
            self.graph[tuple(np_pt1)]
        if tuple(np_pt2) not in self.graph.keys():
            self.graph[tuple(np_pt2)]
        # Compares the x values

        if np_pt1[0] < np_pt2[0]:
            # if not self.arreq_in_list(np_pt2, self.graph[tuple(np_pt1)]):
            if not np_pt2 in self.graph[tuple(np_pt1)]:
                self.graph[tuple(np_pt1)].append(np_pt2)
        elif np_pt1[0] == np_pt2[0] and np_pt1[1] > np_pt2[1]:              # Added this to enforce bottom-right node as exit
            # if not self.arreq_in_list(np_pt2, self.graph[tuple(np_pt1)]):
            if not np_pt2 in self.graph[tuple(np_pt1)]:
                self.graph[tuple(np_pt1)].append(np_pt2)
        else:
            # if not self.arreq_in_list(np_pt1, self.graph[tuple(np_pt2)]):
            if not np_pt1 in self.graph[tuple(np_pt2)]:
                self.graph[tuple(np_pt2)].append(np_pt1)


    def arreq_in_list(self, myarr, list_arrays):
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

    def arreq_in_list_wIdx(self, myarr, list_arrays):
        return next((idx for idx, elem in enumerate(list_arrays) if np.array_equal(elem, myarr)), -1)

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
        # print(node)

        for i, node in enumerate(self.graph[tuple(node)]):
            if not visted_nodes[node_idx]:
                self.dfs_helper(i, node, visted_nodes)

    def generate_edges(self, tri, pts):
        edges = []
        for _, s in enumerate(tri.simplices):
            tri_pts = pts[s]
            # print('Line 98: ', tri_pts, list(tri_pts))

            for i, pt in enumerate(tri_pts):
                if i == 0:
                    continue
                elif pt[0] == tri_pts[-1][0] and pt[1] == tri_pts[-1][1]:
                    # print('Line 104: ', tri_pts[0], list(tri_pts[0]))
                    self.add_edge_to_graph(pt, tri_pts[0])

                self.add_edge_to_graph(tri_pts[i-1], pt)

        topl_to_topR = (np.array([self.min_range, self.max_range]), np.array([self.max_range, self.max_range]))
        btml_to_btmR = (np.array([self.min_range, self.min_range]), np.array([self.max_range, self.min_range]))

        for pt_pair in [topl_to_topR, btml_to_btmR]:
            pt1, pt2 = pt_pair
            pt1 = [i for i in pt1]
            pt2 = [i for i in pt2]

            if pt2 in self.graph[tuple(pt1)]:
            # if self.arreq_in_list(pt2, self.graph[tuple(pt1)]):
                idx = self.arreq_in_list_wIdx(pt2, self.graph[tuple(pt1)])
                del self.graph[tuple(pt1)][idx]
            elif pt1 in self.graph[tuple(pt2)]:
            # elif self.arreq_in_list(pt1, self.graph[tuple(pt2)]):
                idx = self.arreq_in_list_wIdx(pt1, self.graph[tuple(pt2)])
                del self.graph[tuple(pt1)][idx]

        edges = []
        for tuple_pt, np_pt_list in self.graph.items():
            for np_pt in np_pt_list:
                # edges.append((np.asarray(tuple_pt), list(np_pt)))
                edges.append((list(tuple_pt), list(np_pt)))

        return edges

    def convert_to_img2(self, edges, max_range):
        img = []
        for _ in range(int(max_range + 1.0)):
            img.append([0.0 for i in range(int(max_range + 1.0))])

        for edge in edges:
            pt1, pt2 = edge
            # print(pt1, type(pt1))
            # print(pt1, np.array(pt1))
            # print('line 142: ', [i for i in list(pt1)], pt1[0][0])
            pt1 = [int(i) for i in list(pt1)]
            pt2 = [int(i) for i in list(pt2)]
            pts_on_line = list(bresenham(pt1[0], pt1[1], pt2[0], pt2[1]))
            for x, y in pts_on_line:
                if (x == pt1[0] and y == pt1[1]) or (x == pt2[0] and y == pt2[1]):
                    img[x][y] = 2
                else:
                    img[x][y] = 1

        return np.array(img)
    
    def add_flows_to_img(self, flow_dict):
        img = []
        for _ in range(int(self.max_range + 1)):
            img.append([0 for i in range(int(self.max_range + 1))])

        for pt_key, flow_val in flow_dict.items():
            pt1, pt2 = pt_key
            pt1 = [int(i) for i in list(pt1)]
            pt2 = [int(i) for i in list(pt2)]
            pts_on_line = list(bresenham(pt1[0], pt1[1], pt2[0], pt2[1]))
            for x, y in pts_on_line:
                img[x][y] = flow_val

        self.img = np.array(img)

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

        left_wall = []
        right_wall = []

        # Create Nodes for Left Wall (In)
        for i in range(num_left_nodes):
            node = [[min, np.random.randint(min + 1, max - 1)]]
            if node[0] in left_wall:
                i -= 1
            else:
                left_wall.append(node[0])

        # Create Nodes for Right Wall (Out)
        for i in range(num_right_nodes):
            node = [[max, np.random.randint(min + 1, max - 1)]]
            if node[0] in right_wall:
                i -= 1
            else:
                right_wall.append(node[0])

        return np.array(left_wall), np.array(right_wall)

    def calculate_distance(self, point, neighbor):
        '''
        Euclidean distance
        '''
        dist = np.linalg.norm(point - neighbor)
        return dist

    def flatten_mvable_pts(self):

        # flattened_pts = sorted(self.moveable_pts, key=lambda k: (-k[0], k[1]),  reverse=True)
        # print('VasGen Line 216: ', flattened_pts)
        # flattened_pts = np.hstack(flattened_pts)
        # flattened_pts = ()
        # for np_pt in list(sorted(self.moveable_pts, key=lambda k: (-k[0], k[1]),  reverse=True)):
        #     flattened_pts += tuple(np_pt)

        # return flattened_pts
        return self.moveable_pts


    def update_moveable_pts(self, new_mvable_pts):
        tuple_of_pts = ()
        prev = None
        cur = None
        for i in range(2, len(new_mvable_pts)+2, 2):
            cur  = new_mvable_pts[i-2:i]
            if i > 2:
                tuple_of_pts += (prev, cur)
            prev = cur

        # print('tup of pts: ', tuple_of_pts)
        # print('np.vstack: ', np.vstack((tuple_of_pts)))
        # print('test  :\n', np.vstack({tuple(row) for row in tuple_of_pts}))
        # val = np.vstack((tuple_of_pts))
        print('VasGen Line 238: ', type(new_mvable_pts), new_mvable_pts, np.vstack(new_mvable_pts))
        print('VasGen Line 239: ', type(tuple_of_pts), tuple_of_pts)
        # self.moveable_pts = np.vstack((tuple_of_pts))
        self.moveable_pts = np.vstack(new_mvable_pts)
        # new_mvable_pts = np.unique(val, axis=0)
        # new_mvable_pts = val
        # pts = new_mvable_pts
        pts = self.moveable_pts
        pts = np.append(pts, self.left_wall, axis=0)
        pts = np.append(pts, self.right_wall, axis=0)
        pts = np.append(pts, self.left_wall_startend, axis=0)
        pts = np.append(pts, self.right_wall_startend, axis=0)
        self.pts = np.array(sorted(pts, key=lambda k: (-k[0], k[1]),  reverse=True))

        # Update all points off of image
        # self.pts = pts

        self.graph = defaultdict(list)
        # self.moveable_pts = new_mvable_pts
        # print('pts: \n', self.pts)
        # print(type(self.pts), str(type(self.pts)), type(np.array((2, 4))))
        if type(self.pts) != type(np.array((2, 4))):
            # print(self.pts._value)
            self.tri = Delaunay(self.pts._value)
        else:
            self.tri = Delaunay(self.pts)

        # self.tri = Delaunay(self.pts)


        self.edges = self.generate_edges(self.tri, self.pts)
        self.img = None
        self.img = self.convert_to_img2(self.edges, self.max_range)
        self.update_count += 1
        # self.print_images(graph_name='Vasc_Graph_' + str(self.update_count) + '.png', img_name='Vasc2D_img_' +str(self.update_count)+ '.png')


    def generate_pts(self, min_range, max_range, num_pts, dim, lt_wall, rht_wall, lt_startend, rht_startend):
        pts = np.random.uniform(low=min_range, high=max_range, size=(num_pts, dim))
        pts = [list(pt) for pt in pts]
        moveable_pts = pts
        # print('VasGen Line 280 ', moveable_pts)
        # print('VasGen Line 281 ', pts)
        # print('VasGen Line 282 ', lt_wall)
        # print('VasGen Line 283 ', rht_wall)
        # print('VasGen Line 284 ', lt_startend)
        # print('VasGen Line 285 ', rht_startend)

        # for pt in 

        # for pt in list(lt_wall)+list(rht_wall)+list(lt_startend)+list(rht_startend):
        #     pts.append(list(pt))

        # print('VasGen Line 290 ', pts)
        # print('VasGen Line 290 ', np.array(pts))
        # pts.append([list(pt) for pt in lt_wall])
        # pts.append([list(pt) for pt in rht_wall])
        # pts.append([list(pt) for pt in lt_startend])
        # pts.append([list(pt) for pt in rht_startend])

        pts = np.append(pts, lt_wall, axis=0)
        pts = np.append(pts, rht_wall, axis=0)
        pts = np.append(pts, lt_startend, axis=0)
        pts = np.append(pts, rht_startend, axis=0)
        # print(pts)
        # print(np.array(pts))

        pts = np.array(sorted(np.array(pts), key=lambda k: (-k[0], k[1]),  reverse=True))

        # print('VasGen Line 306: \n', pts)

        # os.sys.exit()

        return pts, moveable_pts

    def count_nodes(self, *argv):
        count = 0
        for arg in argv:
            count += len(arg)
        return count


    def print_images(self, graph_name='Vasc_Graph.png', img_name='Vasc2D_img.png'):
        fig = plt.figure()
        for j, s in enumerate(self.tri.simplices):
            p = self.pts[s].mean(axis=0)
            plt.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        plt.triplot(self.pts[:,0], self.pts[:,1], self.tri.simplices)
        plt.plot(self.pts[:,0], self.pts[:,1], 'o')
        # plt.savefig(graph_name)
        fig.savefig(graph_name)
        plt.close(fig)

        # https://stackoverflow.com/questions/38191855/zero-pad-numpy-array
        img = np.pad(self.img, ((2, 3), (2, 3)), 'constant')
        plt.imsave(img_name, np.rot90(img), cmap='jet')

    def pretty(self, d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))
