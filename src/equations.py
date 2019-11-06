# Equations found at: https://www.frontiersin.org/articles/10.3389/fbioe.2017.00027/full

# For each edge i, j,
#     Q(i, j) = [pi*(r^4)*deltaPI(i, j)] / 8*n*l

# For each node, i with σ(i) neighbors,
#     ∑k∈σ(i)  Q(i,k) = 0

# For each lacunae cycle ϕ(k),
#     ∑(i,j)∈ϕ(k)  ΔP(i,j) = 0


# The rate of nutrient supply along the vessel (i, j) is defined as:
#     partial_N(i,j)/partial_t = p_sub_n * (2*PI*r) * [Q(i,j) / (k_out + Q(i,j))] * [k_l / (k_l + N)]

# The availability of nutrient will activate the producer cells that will begin to consume N and produce X. The controlling equation is given as:
#     partial_X / partial_t = u_p * (N / (N + k_p)) * (k_i / (X + k_i)) * M_p + D_p*(v^2)*X

# The rate of product uptake along the vessel (i, j) is defined as:
#     partial_X(i,j)/partial_t = -P_p * (2*PI*r) * [Q(i,j) / (k_in + Q(i,j))] * [X / (k_p + X)]

# Nutrient will be consumed by the producer cells in direct correspondence to the production of X, but at a different reaction rate μn:
#     partial_N / partial_t = -u_n * (N / (N + k_p)) * (k_i / (X + k_i)) * M_p + D_p*(v^2)*N

from VascularGenerator import VascularGenerator

import math
import numpy as np
from collections import defaultdict 


def computeFlow(vas_structure):
    RADIUS = 1          # ? Not sure what we will want this to be
    VISCOSITY = 0.00089 # Viscosity of water at 25C

    numNodes = len(vas_structure.pts)
    numEdges = len(vas_structure.edges)

    matrixSize = numNodes + 2*numEdges
    flowMatrix = np.zeros((matrixSize, matrixSize))

    nodeOrderLookup = []    # References for consistency of columns
    edgeOrderLookup = []

    # Put all edges into ordered lists for reference
    for i, key in enumerate(sorted(vas_structure.graph, key=lambda element: (-element[0], element[1]),  reverse=True)):
        nodeOrderLookup.append(key)
        for connection in vas_structure.graph[key]:
            edgeOrderLookup.append((key, (connection.tolist()[0], connection.tolist()[1])))


     # Initial section of matrix (2 rows)
    flowMatrix[0][0] = 1
    flowMatrix[1][numNodes-1] = 1

    # Would be 2, but we skip row in section 4 for first node, because no inward flow
    currentRow = 1

    # Section 4 of matrix (Row for each node, besides first and last)
    for i, edge in enumerate(edgeOrderLookup):
        if (nodeOrderLookup.index(edge[0])) != 0:
            flowMatrix[currentRow+nodeOrderLookup.index(edge[0])][numNodes+i] = 1
        if (nodeOrderLookup.index(edge[1])) != numNodes-1:
            flowMatrix[currentRow+nodeOrderLookup.index(edge[1])][numNodes+i] = -1

    # Similar to above, subtracting 1 because section 4's last node is also skipped, because no outward flow
    currentRow += numNodes-1

    # Section 3 of matrix (Row for each edge)
    for i, edge in enumerate(edgeOrderLookup):
        length = getLength(edge)
        flowMatrix[currentRow][numNodes+i] = 1
        flowMatrix[currentRow][numNodes+numEdges+i] = -((math.pi*RADIUS**4)/(8*VISCOSITY*length))
        currentRow += 1

    #
    # Seems equation (5) may be unnecessary, matrix is already square and solvable without it
    # Equation (5) still seems to be valid/represented in the values being output
    #

    # Definition section of matrix (Row for each edge)
    for i, edge in enumerate(edgeOrderLookup):
        flowMatrix[currentRow][nodeOrderLookup.index(edge[0])] = 1
        flowMatrix[currentRow][nodeOrderLookup.index(edge[1])] = -1
        flowMatrix[currentRow][numNodes+numEdges+i] = -1
        currentRow += 1


    # All values in matrix 'B' are zero, except for first row for pressure at first node ('initial' section)
    answerMatrix = np.zeros((matrixSize, 1))
    answerMatrix[0] = 1000

    print('\nNodes:\n')
    for node in nodeOrderLookup:
        print(node)

    print('\nEdges:\n')
    for edge in edgeOrderLookup:
        print(edge)

    variables = np.linalg.solve(flowMatrix, answerMatrix)

    print('\nSolved Variables:\n')
    for value in variables:
        print(value)
    
def getLength(edge):
    pt1 = edge[0]
    pt2 = edge[1]

    return (((pt2[0]-pt1[0])**2)+((pt2[1]-pt1[1])**2))**0.5
