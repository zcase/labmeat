import autograd.numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import autograd.scipy.signal as sig
import math
# seed the pseudorandom number generator
from random import seed
from random import random
from random import randint
from timeit import default_timer as timer
import matplotlib as mpl
import os
import time

VarCount = 2

# Basic non-linear rate functions
def hillPlus(A, k, n):
    return np.power(A, n) / (np.power(A, n) + np.power(k, n))
    
def hillMinus(R, k, n):
    return (1 - hillPlus(R, k, n))
    
def hillMinusOne(R, k):
    return hillMinus(R, k, 1)
      
def hillPlusOne(A, k):
    return hillPlus(A, k, 1)

## regulatory mechanisms
def regulatePlusMinus(P_plus, k_plus, P_minus, k_minus):
    # standard regulatory system, one activating and one inhibiting
    # equations 6, 7, 9
    return hillPlusOne(P_plus, k_plus) * hillMinusOne(P_minus, k_minus)
    
def regulatePlusPlus(P_plus0, k_plus0, P_plus1, k_plus1):
    # standard regulatroy system, two activating
    # equations 8
    return hillPlusOne(P_plus0, k_plus0) * hillPlusOne(P_plus1, k_plus1)

#################################################################################
################## DYNAMIC MODEL BASED ON PAPER #################################
#################################################################################
## Update for one cell
def oneCell_ODE(oneCell, params, cellId, OnePdeDelta, vas_structure):
    # returns a Delta for each of the protiens liste, d in oneCell [P0, P1] 
    # accumulate the product as the fitness
    # first unpack the parameters
    (sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx) = params
    (Nu, Xm) = oneCell #Nutrient first, product second
    (dNu, dXm) = OnePdeDelta
    if  vas_structure.pts_on_vessels[cellId]: #vessel cell
        # Vessels secreate nuetrients and takeup waste
        # rate multiplier due to the walls of the vessel
        vesselRate = 1.0 #math.pi * 2 * radius[cellId]
        # Equation 6, Nutrient production from the vessel
        deltaNutrient = +1*sigmaNu * vesselRate * regulatePlusMinus(vas_structure.Q[cellId], k_out, Nu, k_i)
        # Equation 8, Waste Uptake into the vessel
        #deltaProduct = -1*sigmaXm * vesselRate * regulatePlusPlus(Q[cellId], k_in, Xm, k_p)
        # eliminate the positive feedback loop, simple rate bound uptake
        # print('dXm', dXm)
        # print('q', vas_structure.Q[cellId])
        deltaProduct = max(-1*sigmaXm * (vesselRate * hillPlusOne(vas_structure.Q[cellId], k_in)), -1* dXm)
        #deltaProduct = -1*sigmaXm * (vesselRate * hillPlusOne(Q[cellId], k_in))
    else: # Meat cells secreate waste and takeup nutrients
        # each grid location has a cell
        # Equation 9, pull nutrient from the domain
        deltaNutrient = -1*muNu * regulatePlusMinus(Nu, k_p, Xm, k_i)
        # Equation 7 add product 
        #deltaProduct = +1*muXm *regulatePlusMinus(Nu, k_p, Xm, k_i)
        deltaProduct = +1*muXm *hillPlusOne(Nu, k_p) * hillMinus(Xm, k_i, 1)
        #print("Xn = %.10f regulate = %.10f   %.10f" % (Xm, hillPlusOne(Nu, k_p) , hillMinusOne(Xm, k_i))) #regulatePlusMinus(Nu, k_p, Xm, k_i)))
    return np.array([deltaNutrient, deltaProduct])
    
def step_ODE(values, params, pdeDelta, vas_structure, shape_of_img):
    # Computes the changes to the protein values for all the cells
    # based on each cells ODE
    # ode_Delta = np.zeros(np.array(vas_structure.img).shape+(2,))
    # # for cellId in range(0, HowManyCells):
    # for ix,iy in np.ndindex(np.array(vas_structure.img).shape):
    #     # update by GRN, one cell is [P0, P1]
    #     cellId = (ix, iy)
    #     # values = [vas_structure.nutrient_values[cellId], vas_structure.product_values[cellId]]
    #     # print('pdeDelta: ', pdeDelta, pdeDelta.shape)
    #     # ode_Delta.append(oneCell_ODE(values[cellId,:], params, cellId, pdeDelta[cellId,:], vas_structure))
    #     ode_Delta[cellId] = oneCell_ODE(values[cellId], params, cellId, pdeDelta[cellId], vas_structure)

    # shape = np.array(vas_structure.img).shape
    # print(shape)
    r, c = shape_of_img
    ode_Delta = []
    for _ in range(r):
        ode_Delta.append([0 for _ in range(c)])

    for ix,iy in np.ndindex(shape_of_img):
        cellId = (ix, iy)
        ode_Delta[ix][iy] = oneCell_ODE(values[cellId], params, cellId, pdeDelta[cellId], vas_structure)
    return np.array(ode_Delta, dtype=np.float64)
    
def step_PDE(values, params, vas_structure, shape_of_img, nonLinear = False, movablePts = []):
    # Update the values based on diffusion of the proteins to nearby cells
    (sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx) = params
    diffusion = np.array([Dn, Dx]) #get the diffusion parameters
    # pde_Delta = np.zeros((HowManyCells, VarCount))
    pde_Delta = np.zeros(shape_of_img + (VarCount,), dtype=np.float64)
    values = values.T # by protein rather than cell
    newDiff = []
    for i in range(0, VarCount): # for each protein
        D = diffusion[i] #get the diffusion parameter
        if nonLinear: # precompute the adjustments needed for the moveable points
            # start = time.time()
            adjustmentPDE = D * nonLinearAdjustment(movablePts, shape_of_img)
            # end = time.time()
            # print('Nonlinear Adjust = ', end-start, 'seconds')
            #print(adjustmentPDE)
        #simple diffusion is just a convolution
        # convolveLinear = np.array([1*D,-2*D,1*D]) 
        convolveLinear = D * np.array([[1, 1, 1],
                                       [1, -8, 1],
                                       [1, 1, 1]])
        # convolveLinear = np.array([[(1-np.sqrt(2))*D, 1*D,(1-np.sqrt(2))*D],
        #                      [1*D,(-8+(4*np.sqrt(2)))*D,1*D],
        #                      [(1-np.sqrt(2))*D, 1*D, (1-np.sqrt(2))*D]])

        oldValues =  values[i]
        # accumulate the changes due to diffusion 
        for rep in range(0, 50):
            #linear diffusion
            oldValues =  oldValues + sig.convolve(oldValues, convolveLinear)[1:-1, 1:-1] #take off first and last
            # non-linear diffusion, add the adjustment
            if nonLinear: #only if moving the vessels
               oldValues = oldValues + np.multiply(oldValues, adjustmentPDE)
        # the total update returned is the difference between the original values and the values after diffusion
        newDiff.append(oldValues - values[i])
        
    newDiff = np.array(newDiff)
    pde_Delta = pde_Delta + newDiff.T #switch diff by cell order not protein order
    return newDiff.T 

def addSources(params, shape):
    # No sources in system
    sourceRates = np.zeros(shape, dtype=np.float64)
    return np.array([sourceRates, sourceRates]).T #flip by cell
    
############################################################################
### Non Linear PDE 
def nonLinearAdjustment(movablePts, shape):
    # adds an adjustment to the material transfer to take into account
    # the actural position of the cell point in space
    # adjustment is constant for each simulation, because the points do
    # not move so compute once
    r, c = shape
    allAdjustment = np.zeros(shape, dtype=np.float64)
    # for x in movablePts: #only single numbers in x one D
    for i in range(len(movablePts)):
        pt = movablePts[i]
        try:
            pointX = int(pt[0]._value)
            pointY = int(pt[1]._value)
        except AttributeError as e:
            pointX = int(pt[0])
            pointY = int(pt[1])
        int_x = 0
        int_y = 0

        if type(pt[0]) != type(np.array((1,1))) and type(pt[0]) != type(1) and \
            (type(movablePts[i][0]) != float and type(movablePts[i][0]) != np.float64):
            try:
                int_x = int(np.array(movablePts[i][0]._value)) 
                int_y = int(np.array(movablePts[i][1]._value))
            except AttributeError as e:
                int_x = int(np.array(movablePts[i][0])) 
                int_y = int(np.array(movablePts[i][1]))
        else:
            int_x = int(np.array(movablePts[i][0]))
            int_y = int(np.array(movablePts[i][1]))

        # # thisAdj= []
        # thisAdj= []
        # for _ in range(r):
        #     thisAdj.append([0 for _ in range(c)])

        np_pt = np.array([pt[0], pt[1]])
        inc = 0.5
        dist_0 = distToConc(np.linalg.norm(np_pt - np.array([pointX-1 + inc, pointY-1 + inc])))
        dist_1 = distToConc(np.linalg.norm(np_pt - np.array([pointX + inc, pointY-1 + inc])))
        dist_2 = distToConc(np.linalg.norm(np_pt - np.array([pointX+1 + inc, pointY-1 + inc])))

        dist_3 = distToConc(np.linalg.norm(np_pt - np.array([pointX-1 + inc, pointY + inc])))
        dist_5 = distToConc(np.linalg.norm(np_pt - np.array([pointX+1 + inc, pointY + inc])))

        dist_6 = distToConc(np.linalg.norm(np_pt - np.array([pointX-1 + inc, pointY+1 + inc])))
        dist_7 = distToConc(np.linalg.norm(np_pt - np.array([pointX + inc, pointY+1 + inc])))
        dist_8 = distToConc(np.linalg.norm(np_pt - np.array([pointX+1 + inc, pointY+1 + inc])))

        totalAdj = np.sum([d for d in [dist_0, dist_1, dist_2, dist_3, dist_5, dist_6, dist_7, dist_8]]) * -1
        convolution = [[dist_0, dist_1, dist_2], 
                       [dist_3, totalAdj, dist_5], 
                       [dist_6, dist_7, dist_8]]

        deltaDomain2 = get_submatrix_add(allAdjustment, (pointX, pointY), convolution)
        allAdjustment = allAdjustment + np.array(deltaDomain2)

        # totalAdj = 0 # accumulate the changes around the center point
        # # for xI in range(0, HowManyCells): #TODO: 2D adjustment
        # for xI in range(0, r): #TODO: 2D adjustment
        #     for yI in range(0, c): #TODO: 2D adjustment
        #         if ((pointX == xI - 1 and pointX > 0) and (pointY == yI - 1 and pointY > 0) or           #just before the movable point
        #             (pointX == xI - 1 and pointX < r) and (pointY == yI + 1 and pointY < c) or
        #             (pointX == xI + 1 and pointX < r) and (pointY == yI - 1 and pointY > 0) or
        #             (pointX == xI and pointX < r) and (pointY == yI - 1 and pointY > 0) or
        #             (pointX == xI and pointX < r) and (pointY == yI + 1 and pointY < c) or
        #             (pointX == xI-1 and pointX > 0) and (pointY == yI and pointY > 0) or
        #             (pointX == xI-1 and pointX < r) and (pointY == yI and pointY < c) or
        #             (pointX == xI+1 and pointX > 0) and (pointY == yI and pointY > 0) or
        #             (pointX == xI+1 and pointX < r) and (pointY == yI and pointY < c) or
        #             (pointX == xI + 1 and pointX > 0) and (pointY == yI + 1 and pointY < c)):
        #         # if ((pointI == xI - 1 and pointI > 0) or           #just before the movable point
        #         #     (pointI == xI + 1 and pointI < HowManyCells)): # center of right to x
        #             # deltaConc = distToConc(abs(pt[0] - (xI+0.5)))
        #             # thisAdj.append(deltaConc) #center of left to x)
        #             # totalAdj = totalAdj + deltaConc #accun
        #             np_pt = np.array([pt[0], pt[1]])
        #             dist = np.linalg.norm(np_pt - np.array([xI + 0.5, yI + 0.5]))
        #             deltaConc = distToConc(dist)
        #             thisAdj[xI][yI] = deltaConc
        #             totalAdj = totalAdj + deltaConc
        #     # Otherwise no adjustment   
        #     # else:
        #     #     thisAdj.append(0) 
        # #print(thisAdj)
        #  # do a second pass to place the -1* total in the middle 
        # newAdj = []
        # for _ in range(r):
        #     newAdj.append([0 for _ in range(c)])
        # # for xI in range(0, HowManyCells): #TODO 2D adjustment
        # for xI in range(0, r): #TODO 2D adjustment
        #     for yI in range(0, c): #TODO 2D adjustment
        #         # if pointX == xI: #the grid location of this movable point
        #         #     newAdj.append(-1*totalAdj) #center of left to x)
        #         # else:
        #         #     newAdj.append(thisAdj[xI]) #no change
        #         if pointX == xI and pointY == yI:
        #             newAdj[xI][yI] = -1 * totalAdj
        #         else:
        #             newAdj[xI][yI] = thisAdj[xI][yI]
        # #accumulate this movable point into the total adjustment 
        # allAdjustment = allAdjustment + np.array(newAdj)
    return allAdjustment


def get_submatrix_add(lst_matrix, center_pt_tuple, convolution):
    np_matrix = np.array(lst_matrix, dtype=np.float64)
    r, c = np_matrix.shape
    lt_padding = 0 + (center_pt_tuple[0] - 1)
    rt_padding = (c-1) - (center_pt_tuple[0] + 1)
    top_padding = 0 + (center_pt_tuple[1] - 1)
    btm_padding = (r-1) - (center_pt_tuple[1] + 1)
    row_start = 0
    row_end = np.array(convolution).shape[0]
    col_start = 0
    col_end = np.array(convolution).shape[1]

    if lt_padding < 0:
        lt_padding = 0
        row_start = row_start + 1
    if rt_padding < 0:
        rt_padding = 0
        row_end = row_end - 1
    if top_padding < 0:
        top_padding = 0
        col_start = col_start + 1
    if btm_padding < 0:
        btm_padding = 0
        col_end = col_end - 1

    padded_convo = np.pad(np.array(convolution)[row_start:row_end, col_start:col_end], ((top_padding, btm_padding), (lt_padding, rt_padding)), mode='constant', constant_values=(0.0, 0.0))

    try:
        new_matrix = np_matrix + padded_convo
    except Exception as e:
        print(e)
        print('left pad: ', lt_padding)
        print('right pad: ', rt_padding)
        print('top pad: ', top_padding)
        print('btm pad: ', btm_padding)
        print('r start:', row_start)
        print('r end:', row_end)
        print('c start:', col_start)
        print('c end:', col_end)
        print(np.array(convolution)[row_start:row_end, col_start:col_end])
        print(np.array(convolution)[row_start:row_end, col_start:col_end].shape)

    return new_matrix



def distToConc(distance):
    # maps the distance between two points (in thise case one dimention)
    # positive closer, zero if 1, negative if further away
    return 1 - distance
    
#################################################
### MAIN SOLVER, SIMULATOR OF THE DYNAMIC SYSTEM    
#################################################

def solveModelODEPDE(vas_structure, times, params = (), nonLinear = False, movablePts = []):
    #Simple solver that just uses the ODE PDE functions as update rules
    # Unrolls the dynamics for the number of sample times
    # Returns the dynamics which is a list of shape [HowManyCells,VarCount] 
    # Returns the fitness for each iteration, the list of updates on the product 
    # at the vessel cells
    (params,) = params
    trajectories = [] 
    fitnessList = []
    odeDeltaList = []
    pdeDeltaList = []
    detlat = times[1]-times[0]
    shape_of_img = np.array(vas_structure.product_values).shape
    values = np.zeros(shape_of_img + (2,), dtype=np.float64) 
    vessel_points = vas_structure.pts_on_vessels
    vesselCellIds = [(ix, iy) for ix,iy in np.ndindex(shape_of_img) if vessel_points[(ix,iy)] > 0.0]
    pdeDelta = np.zeros(shape_of_img + (VarCount,), dtype=np.float64)
    # then do the ODE and PDE
    for t in range(0,len(times)):
        # t_start = time.time()
        trajectories.append(values)

        # ode_start = time.time()
        odeDelta = step_ODE(values, params, pdeDelta, vas_structure, shape_of_img) #ODE updates
        # ode_end = time.time()
        # print('    step ODE: ', ode_end-ode_start, 'seconds')

        # values_start = time.time()
        values = values + odeDelta*detlat # add in ODE updates
        # values_end = time.time()
        # print('    Values: ', values_end-values_start, 'seconds')

        # pde_start = time.time()
        pdeDelta = step_PDE(values, params, vas_structure, shape_of_img, nonLinear = nonLinear, movablePts=movablePts) #PDE updates
        # pde_end = time.time()
        # print('    step PDE: ', pde_end-pde_start, 'seconds')

        # values_start2 = time.time()
        values = values + pdeDelta*detlat # add in PDE updates
        # values_end2 = time.time()
        # print('    Values2: ', values_end2-values_start2, 'seconds')

        # values_start3 = time.time()
        values = values + addSources(params, shape_of_img) # add sources (0 for meat domain)
        # values_end3 = time.time()
        # print('    Values3: ', values_end3-values_start3, 'seconds')

        # update the fitness
        # get the values of the second protein (the waste)
        # sum up the ODE updates along the vessels. 
        # total product removed from the environment
        deltaList = []
        for ix,iy,iz in np.ndindex(odeDelta.shape):
            if iz == 1: #product 
                deltaList.append(odeDelta[(ix,iy,iz)])
        # deltaList = odeDelta[vesselCellIds][:,1][0]
        fitnessList.append(-1*np.sum(deltaList))
        # remember
        odeDeltaList.append(odeDelta)
        pdeDeltaList.append(pdeDelta)

        # t_end = time.time()
        # print(f'TIME iter {t}: ', t_end-t_start, 'seconds')
    # print(values[(1,1)])
    for ix,iy,iz in np.ndindex(values.shape):
        if iz == 0:
            try:
                vas_structure.nutrient_values[ix][iy] = values[(ix,iy,iz)]._value
            except AttributeError as e:
                vas_structure.nutrient_values[ix][iy] = values[(ix,iy,iz)]
        elif iz == 1:
            try:
                vas_structure.product_values[ix][iy] = values[(ix,iy,iz)]._value
            except AttributeError as e:
                vas_structure.product_values[ix][iy] = values[(ix,iy,iz)]
   
    return (np.array(trajectories), fitnessList, np.array(odeDeltaList), np.array(pdeDeltaList))
    
def getDynamics(vas_structure, params, nonLinear = False, movablePts = [], runParameters = None):
    #returns a trajectory of this ODE_PDE
    if runParameters == None:
        (max_T, count) = getSampleParameters()
    else:
        (max_T, count) = runParameters
    # make the samples
    times = np.linspace(0., max_T, count)
    # run the ODE_PDE solver, returns an array [[list of values of state variables @ t0], [list of values of state variables @ t1] .....]
    return solveModelODEPDE(vas_structure, times, params=tuple((params,)), nonLinear = nonLinear, movablePts=movablePts)

# #         
# ############################################
# #### FITNESS FUNCTION IS THE SUM OF PRODUCT
# ############################################
# def fitness(trueDeltas, vessel_points):
#     # given the dynamics as the history of updates for the product
#     # select the cells that are vascular cells and sum the changes
#     # total product generated
#     product = 0
#     # for i in range(0, HowManyCells):
#     for ix,iy in np.ndindex(img.shape):
#         if vessel_points[(ix,iy)] > 0.0:
#             product = product + sum(trueDeltas[:,i,1]) #TODO 2d select of trueDeltas
#     return product


############################################
#### CONTROL PARAMETERS AND INITIAL VALUES 
############################################
# def makeRunName(batchSize , minTimeSample):
#     #make the file names from this run record the variable settings for this experimental run
#     return runNAME

# def getInitialValues():
#     # just zero
#     return np.array([np.zeros(HowManyCells),
#                       np.zeros(HowManyCells)]).T
                      
def getSampleParameters():
    # how many seconds to run the ODE/PDE for one trajectory, and how many samples
    # return (1000, 1000) 
    return (300, 300)
    #return (2,2)


ParameterNames = ['sigmaNu', 'sigmaXm','muNu', 'muXm', 'k_out', 'k_in', 'k_p', 'k_i', 'k_l', 'Dn', 'Dx']
def getTrueParameters():
    # These values are loosely based on the bioreactor
    (sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx) = (  0.052, 0.002, #sigma Nu, sigma Xm 0.002
                                                                            0.06, 0.06, #production rates Nu Xm
                                                                            0.3, 0.3, # k Out, k In
                                                                            7.3, 1e-10, 0.3, #k p, k i, k l saturation
                                                                            0.0045, 0.0055)  #diffusion
    return np.array([sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx])