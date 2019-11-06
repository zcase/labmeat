from VascularGenerator import VascularGenerator
from equations import *

import numpy as np

if __name__ == "__main__":

    # TODO: Make this (num_of_nodes) a command line argument
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=3)
    vas_structure.print_images()
    print('\n Depth First Search')
    vas_structure.depth_first_search()

    # print('\nNum Nodes: ', len(vas_structure.pts))

    # X is ascending, Y is decending
    for key in sorted(vas_structure.graph, key=lambda element: (-element[0], element[1]),  reverse=True):
        print('Key: ', key, '  : ', vas_structure.graph[key])

    computeFlow(vas_structure)