from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse

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

    flowDict = computeFlow(vas_structure)

    print('\n\n\n')
    for key in sorted(flowDict, key=lambda element: (element[0], element[1]),  reverse=True):
        print('Key:', key, '   :  ', flowDict[key])

    vas_structure.add_flows_to_img(flowDict)
    vas_structure.print_images()

    lab_meat_diffuse(vas_structure.img, 100, 0.5, 10)