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
# import numpy as np
import autograd.numpy as np
from collections import defaultdict 
from autograd.builtins import isinstance, tuple


def computeFlow(vas_structure):
    # print('In computeFlow')
    RADIUS = 1.0          # ? Not sure what we will want this to be
    VISCOSITY = 0.00089 # Viscosity of water at 25C

    numNodes = len(vas_structure.pts)
    numEdges = len(vas_structure.edges)

    matrixSize = numNodes + 2*numEdges
    flowMatrix = [[0 for x in range(matrixSize)] for y in range(matrixSize)]
    

    nodeOrderLookup = []    # References for consistency of columns
    edgeOrderLookup = []

    # Put all edges into ordered lists for reference
    for i, key in enumerate(sorted(vas_structure.graph, key=lambda element: (-element[0], element[1]),  reverse=True)):
        nodeOrderLookup.append(key)
        for connection in vas_structure.graph[key]:
            # print('# ==================== #')
            # print('# ==== CONNECTION ==== #\n')
            # print('Equations: Line 53:, ', connection, type(connection))
            # print(type(connection))
            # if type(connection) != type(np.array((2, 4))):
            #     # print(connection[0]._value)
            #     # print(connection._value)
            #     # print(tuple(connection._value))
            #     # print(key)
            #     t, tt = key
            #     # print(t, tt)
            #     tx = t._value
            #     ty = tt._value
            #     # print(tx, ty)

            #     edgeOrderLookup.append(((tx, ty), (connection[0]._value, connection[1]._value)))
            # else:
                # edgeOrderLookup.append((key, (connection.tolist()[0], connection.tolist()[1])))
            edgeOrderLookup.append((key, (connection[0], connection[1])))
            
            # edgeOrderLookup.append((key, (connection.tolist()[0], connection.tolist()[1])))


     # Initial section of matrix (2 rows)
    flowMatrix[0][0] = 1
    flowMatrix[1][numNodes-1] = 1

    # Would be 2, but we skip row in section 4 for first node, because no inward flow
    currentRow = 1

    # Section 4 of matrix (Row for each node, besides first and last)
    # print(flowMatrix)
    # print(flowMatrix.shape)
    for i, edge in enumerate(edgeOrderLookup):
        # print('\n\nEDGE: ', edge)
        # print('EDGE[0] : ', edge[0])
        # print('EDGE[1] : ', edge[1])
        # print('NumNodes: ', numNodes)
        # print('NumEdges: ', numEdges)
        # print('currentRow+nodeOrderLookup.index(edge[1]): ', currentRow+nodeOrderLookup.index(edge[1]), '\n\n')
        if (nodeOrderLookup.index(edge[0])) != 0:
            flowMatrix[currentRow+nodeOrderLookup.index(edge[0])][numNodes+i] = 1
        if (nodeOrderLookup.index(edge[1])) != numNodes-1:
            flowMatrix[currentRow+nodeOrderLookup.index(edge[1])][numNodes+i] = -1

    # Similar to above, subtracting 1 because section 4's last node is also skipped, because no outward flow
    currentRow = currentRow + numNodes-1

    # Section 3 of matrix (Row for each edge)
    for i, edge in enumerate(edgeOrderLookup):
        length = getLength(edge)
        flowMatrix[currentRow][numNodes+i] = 1
        flowMatrix[currentRow][numNodes+numEdges+i] = -((math.pi*RADIUS**4)/(8*VISCOSITY*length))
        currentRow = currentRow + 1

    #
    # Seems equation (5) may be unnecessary, matrix is already square and solvable without it
    # Equation (5) still seems to be valid/represented in the values being output
    #

    # Definition section of matrix (Row for each edge)
    for i, edge in enumerate(edgeOrderLookup):
        flowMatrix[currentRow][nodeOrderLookup.index(edge[0])] = 1
        flowMatrix[currentRow][nodeOrderLookup.index(edge[1])] = -1
        flowMatrix[currentRow][numNodes+numEdges+i] = -1
        currentRow = currentRow + 1


    # All values in matrix 'B' are zero, except for first row for pressure at first node ('initial' section)
    answerMatrix = [0]*matrixSize
    answerMatrix[0] = 1000
    # answerMatrix[0] = 1
    
    # answerMatrix[0] = 1

    # print('\nNodes:\n')
    # for node in nodeOrderLookup:
    #     print(node)

    # print('\nEdges:\n')
    # for edge in edgeOrderLookup:
    #     print(edge)

    
    flowMatrix = np.array(flowMatrix)
    # https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests
    # flowMatrixdfDirty = flowMatrix+0.00001*np.random.rand(matrixSize, matrixSize)
    answerMatrix = np.array(answerMatrix)

    variables = np.linalg.solve(flowMatrix, answerMatrix)
    # variables = np.linalg.solve(flowMatrixdfDirty, answerMatrix)

    # print('\nSolved Variables:\n')
    # printSolvedVariables(nodeOrderLookup, edgeOrderLookup, variables)


    flowDict = defaultdict(int)
    
    for i, edge in enumerate(edgeOrderLookup):
        flowDict[edge] = abs(variables[i+numNodes])

    return flowDict
    
def getLength(edge):
    pt1 = edge[0]
    pt2 = edge[1]

    return (((pt2[0]-pt1[0])**2)+((pt2[1]-pt1[1])**2))**0.5

def printSolvedVariables(nodeLookup, edgeLookup, variables):
    index = 0
    for i, node in enumerate(nodeLookup):
        print('P'+str(i+1)+':', variables[index], ' Node:', node)
        index = index +  1
    
    for i, edge in enumerate(edgeLookup):
        print('Q'+str(i+1)+':', variables[index], ' Edge:', edge)
        index = index +  1

    for i, edge in enumerate(edgeLookup):
        print('𝝙P'+str(i+1)+':', variables[index], ' Edge:', edge)
        index = index +  1
