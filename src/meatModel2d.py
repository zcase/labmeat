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
    
def step_ODE(values, params, pdeDelta, vas_structure):
    # Computes the changes to the protein values for all the cells
    # based on each cells ODE
    ode_Delta = np.zeros(np.array(vas_structure.img).shape+(2,))
    # for cellId in range(0, HowManyCells):
    for ix,iy in np.ndindex(np.array(vas_structure.img).shape):
        # update by GRN, one cell is [P0, P1]
        cellId = (ix, iy)
        # values = [vas_structure.nutrient_values[cellId], vas_structure.product_values[cellId]]
        # print('pdeDelta: ', pdeDelta, pdeDelta.shape)
        # ode_Delta.append(oneCell_ODE(values[cellId,:], params, cellId, pdeDelta[cellId,:], vas_structure))
        ode_Delta[cellId] = oneCell_ODE(values[cellId], params, cellId, pdeDelta[cellId], vas_structure)
    return np.array(ode_Delta)
    
def step_PDE(values, params, vas_structure, nonLinear = False, movablePts = []):
    # Update the values based on diffusion of the proteins to nearby cells
    (sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx) = params
    diffusion = np.array([Dn, Dx]) #get the diffusion parameters
    # pde_Delta = np.zeros((HowManyCells, VarCount))
    pde_Delta = np.zeros(np.array(vas_structure.img).shape + (VarCount,))
    values = values.T # by protein rather than cell
    newDiff = []
    for i in range(0, VarCount): # for each protein
        D = diffusion[i] #get the diffusion parameter
        if nonLinear: # precompute the adjustments needed for the moveable points
            adjustmentPDE = D * nonLinearAdjustment(movablePts)
            #print(adjustmentPDE)
        #simple diffusion is just a convolution
        # convolveLinear = np.array([1*D,-2*D,1*D]) 
        convolveLinear = np.array([[(1-np.sqrt(2))*D, 1*D,(1-np.sqrt(2))*D],
                             [1*D,(-8+(4*np.sqrt(2)))*D,1*D],
                             [(1-np.sqrt(2))*D, 1*D, (1-np.sqrt(2))*D]])

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
    sourceRates = np.zeros(shape)
    return np.array([sourceRates, sourceRates]).T #flip by cell
    
############################################################################
### Non Linear PDE 
def nonLinearAdjustment(movablePts, shape):
    # adds an adjustment to the material transfer to take into account
    # the actural position of the cell point in space
    # adjustment is constant for each simulation, because the points do
    # not move so compute once
    allAdjustment = np.zeros(shape)
    for x in movablePts: #only single numbers in x one D
        pointI = int(x)
        thisAdj= []
        totalAdj =0 # accumulate the changes around the center point
        for xI in range(0, HowManyCells): #TODO: 2D adjustment
            if ((pointI == xI - 1 and pointI > 0) or           #just before the movable point
                (pointI == xI + 1 and pointI < HowManyCells)): # center of right to x
                deltaConc = distToConc(abs(x - (xI+0.5)))
                thisAdj.append(deltaConc) #center of left to x)
                totalAdj = totalAdj + deltaConc #accun
            # Otherwise no adjustment   
            else:
                thisAdj.append(0) 
        #print(thisAdj)
         # do a second pass to place the -1* total in the middle 
        newAdj = []
        for xI in range(0, HowManyCells): #TODO 2D adjustment
            if pointI == xI: #the grid location of this movable point
                newAdj.append(-1*totalAdj) #center of left to x)
            else:
                newAdj.append(thisAdj[xI]) #no change
        #accumulate this movable point into the total adjustment 
        allAdjustment = allAdjustment + np.array(newAdj)
    return allAdjustment
        
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
    # then do the ODE and PDE
    # print(vas_structure.nutrient_values, vas_structure.product_values)
    # values = np.array([vas_structure.nutrient_values, vas_structure.product_values])
    print('values shape', np.array(vas_structure.product_values).shape)
    values = np.array(np.zeros(np.array(vas_structure.product_values).shape + (2,)))
    # vesselCellIds = np.array([i for i in range(0,HowManyCells) if VesselCells[i]])
    vessel_points = vas_structure.pts_on_vessels
    vesselCellIds = [(ix, iy) for ix,iy in np.ndindex(np.array(vas_structure.img).shape) if vessel_points[(ix,iy)] > 0.0]
    pdeDelta = np.zeros(np.array(vas_structure.img).shape + (VarCount,))
    for t in range(0,len(times)):
        # deltas are always shape [HowManyCells,VarCount]
        trajectories.append(values)
        odeDelta = step_ODE(values, params, pdeDelta, vas_structure) #ODE updates
        values = values + odeDelta*detlat # add in ODE updates
        pdeDelta = step_PDE(values, params, vas_structure, nonLinear = nonLinear, movablePts = movablePts) #PDE updates
        values = values + pdeDelta*detlat # add in PDE updates
        values = values + addSources(params, np.array(vas_structure.img).shape) # add sources (0 for meat domain)
        #values = np.clip(values, 0, 100)
        # update the fitness
        # get the values of the second protein (the waste)
        # sum up the ODE updates along the vessels. 
        # total product removed from the environment
        # print(odeDelta)
        # print('ode', odeDelta.shape)
        # print('ode1', odeDelta[vesselCellIds][1].shape)
        # print('ids', np.array(vesselCellIds).shape)
        # print(vesselCellIds)
        deltaList = []
        for ix,iy,iz in np.ndindex(odeDelta.shape):
            if iz == 0:
                deltaList.append(odeDelta[(ix,iy,iz)])
        # deltaList = odeDelta[vesselCellIds][:,1][0]
        fitnessList.append(-1*sum(deltaList))
        # remember
        odeDeltaList.append(odeDelta)
        pdeDeltaList.append(pdeDelta)
    print(values[(1,1)])
    #(dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = getDynamics(
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
    return solveModelODEPDE(vas_structure, times, params=tuple((params,)), nonLinear = nonLinear, movablePts = movablePts)

#         
############################################
#### FITNESS FUNCTION IS THE SUM OF PRODUCT
############################################
def fitness(trueDeltas, vessel_points):
    # given the dynamics as the history of updates for the product
    # select the cells that are vascular cells and sum the changes
    # total product generated
    product = 0
    # for i in range(0, HowManyCells):
    for ix,iy in np.ndindex(img.shape):
        if vessel_points[(ix,iy)] > 0.0:
            product = product + sum(trueDeltas[:,i,1]) #TODO 2d select of trueDeltas
    return product


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