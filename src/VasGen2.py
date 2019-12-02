from autograd.builtins import isinstance, tuple, list
import autograd.numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from bresenham import bresenham
import os
import random


class VasGen2:
    def __init__(self, min_range=1, max_range=10, x=10, side_nodes=True, dim=2, num_of_nodes=20):
        self.graph = dict()
        self.update_count = 0
        self.has_side_nodes = side_nodes

        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.left_wall_startend = [[self.min_range, self.min_range], [self.min_range, self.max_range]]
        self.right_wall_startend = [[self.max_range, self.min_range], [self.max_range, self.max_range]]

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
        self.img = self.convert_to_img(self.edges, max_range)
        self.diffused_img = None

        self.pts_on_vessels = []

    # ====================== #
    # ==== Create Walls ==== #
    # ====================== #
    def create_walls(self, min, max, num_of_nodes):
        # Number of Wall nodes = 1% of number of nodes
        high_node_val = int(num_of_nodes / 5)
        num_left_nodes = 1
        num_right_nodes = 1
        if high_node_val > 1:
            num_left_nodes = random.randint(1, high_node_val)
            num_right_nodes = random.randint(1, high_node_val)

        left_wall = []
        right_wall = []

        # Create Nodes for Left Wall (In)
        for i in range(num_left_nodes):
            node = [[min, float(random.randint(min + 1, max - 1))]]
            if node[0] in left_wall:
                i -= 1
            else:
                left_wall.append(node[0])

        # Create Nodes for Right Wall (Out)
        for i in range(num_right_nodes):
            node = [[max, float(random.randint(min + 1, max - 1))]]
            if node[0] in right_wall:
                i -= 1
            else:
                right_wall.append(node[0])

        return left_wall, right_wall

    # ====================== #
    # ==== Generate Pts ==== #
    # ====================== #
    def generate_pts(self, min_range, max_range, num_pts, dim, lt_wall, rht_wall, lt_startend, rht_startend):
        pts = []
        for _ in range(num_pts):
            pts.append([random.uniform(min_range, max_range) for _ in range(2)])

        moveable_pts = pts

        if self.has_side_nodes:
            pts = pts + lt_wall + rht_wall + lt_startend + rht_startend
        else:
            pts = pts + lt_startend + rht_startend
        pts = sorted(pts ,key=lambda x: (-x[0],x[1]), reverse=True)

        return pts, moveable_pts

    # ======================== #
    # ==== Generate Edges ==== #
    # ======================== #
    def generate_edges(self, tri, pts):
        edges = []
        for _, s in enumerate(tri.simplices):
            tri_pts = [ pts[i] for i in list(s)]

            for i, pt in enumerate(tri_pts):
                if i == 0:
                    continue
                elif pt == tri_pts[-1]: # Wrap around
                    self.add_edge_to_graph(pt, tri_pts[0])
                
                self.add_edge_to_graph(tri_pts[i-1], pt)

        topl_to_topR = ([self.min_range, self.max_range], [self.max_range, self.max_range])
        btml_to_btmR = ([self.min_range, self.min_range], [self.max_range, self.min_range])

        for pt_pair in [topl_to_topR, btml_to_btmR]:
            pt1, pt2 = pt_pair

            if pt2 in self.graph[tuple(pt1)]:
                idx = self.graph[tuple(pt1)].index(pt2)
                del self.graph[tuple(pt1)][idx]
            elif pt1 in self.graph[tuple(pt2)]:
                idx = self.graph[tuple(pt2)].index(pt1)
                del self.graph[tuple(pt1)][idx]

        for tuple_pt, lst_pt_list in self.graph.items():
            for lst_pt in lst_pt_list:
                # edges.append((np.asarray(tuple_pt), list(np_pt)))
                edges.append((list(tuple_pt), lst_pt))
        

        return edges

    # =========================== #
    # ==== Add Edge To Graph ==== #
    # =========================== #
    def add_edge_to_graph(self, lst_pt1, lst_pt2):
        if tuple(lst_pt1) not in self.graph.keys():
            self.graph[tuple(lst_pt1)] = []
        if tuple(lst_pt2) not in self.graph.keys():
            self.graph[tuple(lst_pt2)] = []

        if lst_pt1[0] < lst_pt2[0]:
            if not lst_pt2 in self.graph[tuple(lst_pt1)]:
                self.graph[tuple(lst_pt1)].append(lst_pt2)
        elif lst_pt1[0] == lst_pt2[0] and lst_pt1[1] > lst_pt2[1]:              # Added this to enforce bottom-right node as exit
            if not lst_pt2 in self.graph[tuple(lst_pt1)]:
                self.graph[tuple(lst_pt1)].append(lst_pt2)
        else:
            if not lst_pt1 in self.graph[tuple(lst_pt2)]:
                self.graph[tuple(lst_pt2)].append(lst_pt1)

    # ========================== #
    # ==== Convert To Image ==== #
    # ========================== #
    def convert_to_img(self, edges, max_range):
        img = []
        for _ in range(int(max_range + 1.0)):
            img.append([0.0 for i in range(int(max_range + 1.0))])

        for edge in edges:
            pt1, pt2 = edge
            pt1 = [int(i) for i in pt1]
            pt2 = [int(i) for i in pt2]
            # TODO(zcase): Account for it points go <min or > max for both x and y
            # print(pt1)
            # print(pt2)
            pts_on_line = list(bresenham(pt1[0], pt1[1], pt2[0], pt2[1]))
            print(pts_on_line)
            for x, y in pts_on_line:
                if (x == pt1[0] and y == pt1[1]) or (x == pt2[0] and y == pt2[1]):
                    img[x][y] = 2
                else:
                    img[x][y] = 1

        return img

    # ============================ #
    # ==== Add Flows To Image ==== #
    # ============================ #
    def add_flows_to_img(self, flow_dict):
        # print('In add_flows_to_img')
        img = []
        self.pts_on_vessels = []
        for _ in range(int(self.max_range + 1)):
            img.append([0.0 for i in range(int(self.max_range + 1))])

        for pt_key, flow_val in flow_dict.items():
            pt1, pt2 = pt_key
            pt1 = [int(i) for i in list(pt1)]
            pt2 = [int(i) for i in list(pt2)]
            pts_on_line = list(bresenham(pt1[0], pt1[1], pt2[0], pt2[1]))
            self.pts_on_vessels += pts_on_line
            for x, y in pts_on_line:
                img[x][y] = flow_val

        self.img = img

        return self.img

    def flatten_mvable_pts(self):
        flat_list = []
        for lst_pt in self.moveable_pts:
            for xORy_val in lst_pt:
                flat_list.append(xORy_val)

        return flat_list


    def checkIfDuplicates(self, listOfElems):
        ''' Check if given list contains any duplicates '''
        setOfElems = set()
        dup_idx_lst = []
        for idx, elem in enumerate([(round(pt[0], 6), round(pt[1], 6)) for pt in listOfElems]):
            if elem in setOfElems:
                dup_idx_lst.append(idx)
            else:
                setOfElems.add(elem)

        return dup_idx_lst

    def remove_dup_pts(self):
        dup_idx_lst = self.checkIfDuplicates(self.moveable_pts)
        while(dup_idx_lst):
            for dup_idx in dup_idx_lst:
                pt_val_lst = self.moveable_pts[dup_idx]
                noise = [random.uniform(-1, 1) for _ in range(2)]
                pt_val_lst = pt_val_lst
                pt_val_lst = [sum(x) for x in zip(pt_val_lst, noise)]
                x = pt_val_lst[0]
                y = pt_val_lst[1]
                if x < self.min_range+1:
                    x = float(self.min_range+1)
                elif x > self.max_range-1:
                    x = float(self.max_range-1)
                
                if y < self.min_range+1:
                    y = float(self.min_range+1)
                elif y > self.max_range-1:
                    y = float(self.max_range-1)
                pt_val_lst = [x, y]
                self.moveable_pts[dup_idx] = pt_val_lst
                dup_idx_lst = self.checkIfDuplicates(self.moveable_pts)

    def update_moveable_pts(self, new_mvable_pts):
        # print('In update_moveable_pts')
        # print('VasGen2 236: ', new_mvable_pts)
        # print(type(new_mvable_pts))
        self.graph = dict()
        pts_lst = []
        prev = None
        cur = None
        # print('\n IN UPDATE PTS')
        # print('new_mv_pts: ', new_mvable_pts)
        # for i in range(2, len(list(new_mvable_pts)._value)+2, 2):
        for i in range(2, len(list(new_mvable_pts))+2, 2):
            # cur  = list(new_mvable_pts)._value[i-2:i]
            cur  = list(new_mvable_pts)[i-2:i]
            x = float(cur[0])
            y = float(cur[1])
            if x < self.min_range+1:
                x = float(self.min_range+1)
            elif x > self.max_range-1:
                x = float(self.max_range-1)
            
            if y < self.min_range+1:
                y = float(self.min_range+1)
            elif y > self.max_range-1:
                y = float(self.max_range-1)

            cur = [x, y]
            # print('First Cur: ', cur)

            # if i > 2:
            #     prev = [float(i) for i in prev]
            #     cur = [float(i) for i in cur]
            #     print('prev: ', prev)
            #     print('cur : ', cur)
            #     pts_lst = pts_lst + [prev, cur]
            # prev = cur
            pts_lst = pts_lst + [cur]

        # print('\nhere: ')
        # print(pts_lst, type(pts_lst), len(pts_lst))
        # os.sys.exit()
        # print('hERE: ', list(new_mvable_pts)._value)
        self.moveable_pts = pts_lst
        self.remove_dup_pts()

        # print('HERE 2:         ', self.moveable_pts)

        pts = []
        if self.has_side_nodes:
            pts = self.moveable_pts + self.left_wall + self.right_wall + self.left_wall_startend + self.right_wall_startend
        else:
            pts = self.moveable_pts + self.left_wall_startend + self.right_wall_startend
        # print('VasGen2: ', np.array(pts), type(pts))
        self.pts = sorted(pts, key=lambda x: (-x[0],x[1]), reverse=True)

        self.tri = Delaunay(self.pts)
        self.edges = self.generate_edges(self.tri, self.pts)

        self.img = None
        self.img = self.convert_to_img(self.edges, self.max_range)

        self.update_count = self.update_count + 1
        # self.print_images(graph_name='Vasc_Graph_' + str(self.update_count) + '.png', img_name='Vasc2D_img_' +str(self.update_count)+ '.png')

    def print_images(self, graph_name='Vasc_Graph.png', img_name='Vasc2D_img.png'):
        fig = plt.figure()
        for j, s in enumerate(self.tri.simplices):
            p = np.array(self.pts)[s].mean(axis=0)
            plt.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        plt.triplot(np.array(self.pts)[:,0], np.array(self.pts)[:,1], self.tri.simplices)
        plt.plot(np.array(self.pts)[:,0], np.array(self.pts)[:,1], 'o')
        fig.savefig(graph_name)
        plt.close(fig)

        # https://stackoverflow.com/questions/38191855/zero-pad-numpy-array
        img = np.pad(np.array(self.img), ((2, 3), (2, 3)), 'constant')
        plt.imsave(img_name, np.rot90(img), cmap='jet')

    def update_hillclimb_pts(self, new_mvable_pts):
        self.graph = dict()
        pts_lst = []
        prev = None
        cur = None

        for i in range(0, len(list(new_mvable_pts))):
           
            cur  = list(new_mvable_pts)[i]
            print('updating points', cur)
            x = float(cur[0])
            y = float(cur[1])
            if x < self.min_range+1:
                x = float(self.min_range+1)
            elif x > self.max_range-1:
                x = float(self.max_range-1)
            
            if y < self.min_range+1:
                y = float(self.min_range+1)
            elif y > self.max_range-1:
                y = float(self.max_range-1)

            cur = [x, y]
            pts_lst = pts_lst + [cur]

        self.moveable_pts = pts_lst
        self.remove_dup_pts()

        pts = []
        if self.has_side_nodes:
            pts = self.moveable_pts + self.left_wall + self.right_wall + self.left_wall_startend + self.right_wall_startend
        else:
            pts = self.moveable_pts + self.left_wall_startend + self.right_wall_startend
        
        self.pts = sorted(pts, key=lambda x: (-x[0],x[1]), reverse=True)

        self.tri = Delaunay(self.pts)
        self.edges = self.generate_edges(self.tri, self.pts)

        self.img = None
        self.img = self.convert_to_img(self.edges, self.max_range)

        self.update_count = self.update_count + 1