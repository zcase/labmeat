import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
# from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
import autograd.scipy.signal as sig

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


# P0 is Nu and P1 is Xp

mpl.rcParams["font.size"] = 12
mpl.rcParams['font.family'] = 'monospace'

PATH = './'
ResultsPATH = 'Python Code/Results/'
ResultsPATH = 'meatModel/Results/'

# Only two cell tyeps, a vessel cell then True, else a meat cell
# Need to upgrade to a 2D domain with each grid location (x, y) 
# having either a vascular cell, a meat cell or empty space

# learning parameters
batchSize = 5 #3 how many times to learn over one time slice
minTimeSample = 5 # 5 and 10 works great, 1 not so great. how much time to include in the time slice sample
downSampleAnimate = 1 # times batchsize is the sampling rate for 
StepSize = 0.005
TotalIterations = 350 #How many times to do a gradient update

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
def oneCell_ODE(oneCell, params, cellId, OnePdeDelta):
    # returns a Delta for each of the protiens liste, d in oneCell [P0, P1] 
    # accumulate the product as the fitness
    # first unpack the parameters
    (sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx) = params
    (Nu, Xm) = oneCell #Nutrient first, product second
    (dNu, dXm) = OnePdeDelta
    if  VesselCells[cellId]: #vessel cell
        # Vessels secreate nuetrients and takeup waste
        # rate multiplier due to the walls of the vessel
        vesselRate = 1.0 #math.pi * 2 * radius[cellId]
        # Equation 6, Nutrient production from the vessel
        qval = Q[cellId]
        # deltaNutrient = +1*sigmaNu * vesselRate * regulatePlusMinus(Q[cellId], k_out, Nu, k_i)
        deltaNutrient = +1*sigmaNu * vesselRate * regulatePlusMinus(qval, k_out, Nu, k_i)
        # Equation 8, Waste Uptake into the vessel
        #deltaProduct = -1*sigmaXm * vesselRate * regulatePlusPlus(Q[cellId], k_in, Xm, k_p)
        # eliminate the possitive feedback loop, simple rate bound uptake
        # deltaProduct = max(-1*sigmaXm * (vesselRate * hillPlusOne(Q[cellId], k_in)), -1* dXm)
        deltaProduct = max(-1*sigmaXm * (vesselRate * hillPlusOne(qval, k_in)), -1* dXm)
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
    
def step_ODE(values, params, pdeDelta):
    # Computes the changes to the protein values for all the cells
    # based on each cells ODE
    ode_Delta = []
    for cellId in range(0, HowManyCells):
        # update by GRN, one cell is [P0, P1]
        val = values[cellId,:]
        delt = pdeDelta[cellId,:]
        # ode_Delta.append(oneCell_ODE(values[cellId,:], params, cellId, pdeDelta[cellId,:]))
        ode_Delta.append(oneCell_ODE(val, params, cellId, delt))
    return np.array(ode_Delta)
    
def step_PDE(values, params, nonLinear = False, movablePts = []):
    # Update the values based on diffusion of the proteins to nearby cells
    (sigmaNu, sigmaXm, muNu, muXm, k_out, k_in, k_p, k_i, k_l, Dn, Dx) = params
    diffusion = np.array([Dn, Dx]) #get the diffusion parameters
    pde_Delta = np.zeros((HowManyCells, VarCount))
    values = values.T # by protein rather than cell
    newDiff = []
    for i in range(0, VarCount): # for each protein
        D = diffusion[i] #get the diffusion parameter
        if nonLinear: # precompute the adjustments needed for the moveable points
            adjustmentPDE = D * nonLinearAdjustment(movablePts)
            #print(adjustmentPDE)
        #simple diffusion is just a convolution
        convolveLinear = np.array([1*D,-2*D,1*D]) 
        oneDif = np.zeros(HowManyCells)
        oldValues =  values[i]
        # accumulate the changes due to diffusion 
        for rep in range(0, 50):
            #linear diffusion
            oldValues =  oldValues + sig.convolve(oldValues, convolveLinear)[1:-1] #take off first and last
            # non-linear diffusion, add the adjustment
            if nonLinear: #only if moving the vessels
               oldValues = oldValues + np.multiply(oldValues, adjustmentPDE)
        # the total update returned is the difference between the original values and the values after diffusion
        newDiff.append(oldValues - values[i])
        
    newDiff = np.array(newDiff)
    pde_Delta = pde_Delta + newDiff.T #switch diff by cell order not protein order
    return newDiff.T 
    
def addSources(params):
    # No sources in system
    sourceRates = np.zeros(HowManyCells)
    return np.array([sourceRates, sourceRates]).T #flip by cell
    
############################################################################
### Non Linear PDE 
def nonLinearAdjustment(movablePts):
    # adds an adjustment to the material transfer to take into account
    # the actural position of the cell point in space
    # adjustment is constant for each simulation, because the points do
    # not move so compute once
    allAdjustment = np.zeros(HowManyCells)
    for x in movablePts: #only single numbers in x one D
        pointI = None
        if type(x) == np.float64 or type(x) == float:
            pointI = int(x)
        else:
            pointI = int(x._value)
        thisAdj= []
        totalAdj =0 # accumulate the changes around the center point
        for xI in range(0, HowManyCells):
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
        for xI in range(0, HowManyCells):
            if pointI == xI: #the grid location of this movable point
                newAdj.append(-1*totalAdj) #center of left to x)
            else:
                newAdj.append(thisAdj[xI]) #no change
        #accumulate this movable point into the total adjustment 
        allAdjustment = allAdjustment + np.array(newAdj)
        # print(np.array(allAdjustment))
        # print('\n\n')
    return allAdjustment
        
def distToConc(distance):
    # maps the distance between two points (in thise case one dimention)
    # positive closer, zero if 1, negative if further away
    return 1 - distance
    
    
    

#################################################
### MAIN SOLVER, SIMULATOR OF THE DYNAMIC SYSTEM    
#################################################

def solveModelODEPDE(values, times, params = (), nonLinear = False, movablePts = []):
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
    t1 = times[1]
    t0 = times[0]
    # detlat = times[1]-times[0]
    detlat = t1 - t0
    # then do the ODE and PDE
    vesselCellIds = np.array([i for i in range(0,HowManyCells) if VesselCells[i]])
    pdeDelta = np.zeros((HowManyCells, VarCount))
    for t in range(0,len(times)):
        # deltas are always shape [HowManyCells,VarCount]
        trajectories.append(values)
        odeDelta = step_ODE(values, params, pdeDelta) #ODE updates
        values = values + odeDelta*detlat # add in ODE updates
        pdeDelta = step_PDE(values, params, nonLinear = nonLinear, movablePts = movablePts) #PDE updates
        values = values + pdeDelta*detlat # add in PDE updates
        values = values + addSources(params) # add sources (0 for meat domain)
        #values = np.clip(values, 0, 100)
        # update the fitness
        # get the values of the second protein (the waste)
        # sum up the ODE updates along the vessels. 
        # total product removed from the environment
        odeDeltaVal = odeDelta[:,1][vesselCellIds]
        # fitnessList.append(-1*sum(odeDelta[:,1][vesselCellIds]))
        fitnessList.append(-1*sum(odeDeltaVal))
        # remember
        odeDeltaList.append(odeDelta)
        pdeDeltaList.append(pdeDelta)
    #(dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = getDynamics(
    return (np.array(trajectories), fitnessList, np.array(odeDeltaList), np.array(pdeDeltaList))
    
def getDynamics(values, params, nonLinear = False, movablePts = [], runParameters = None):
    #returns a trajectory of this ODE_PDE
    if runParameters == None:
        (max_T, count) = getSampleParameters()
    else:
        (max_T, count) = runParameters
    # make the samples
    times = np.linspace(0., max_T, count)
    # run the ODE_PDE solver, returns an array [[list of values of state variables @ t0], [list of values of state variables @ t1] .....]
    return solveModelODEPDE(values, times, params=tuple((params,)), nonLinear = nonLinear, movablePts = movablePts)
    
    
   

#         
############################################
#### FITNESS FUNCTION IS THE SUM OF PRODUCT
############################################
def fitness(trueDeltas):
    # given the dynamics as the history of updates for the product
    # select the cells that are vascular cells and sum the changes
    # total product generated
    product = 0
    for i in range(0, HowManyCells):
        if VesselCells[i]:
            product = product + sum(trueDeltas[:,i,1])
    return product

############################################
#### CONTROL PARAMETERS AND INITIAL VALUES 
############################################
def makeRunName(batchSize , minTimeSample):
    #make the file names from this run record the variable settings for this experimental run
    return runNAME

def getInitialValues():
    # just zero
    return np.array([np.zeros(HowManyCells),
                      np.zeros(HowManyCells)]).T
                      
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
   
# sigmaNu = 1.0 # flow rate of Nu out of the vessel walls
# sigmaXm = 1.0 # flow rate of Xm in to the vessel walls
# muNu = 1.0 # the consumption rate of nutrient by the meat cells
# muXm = 1.0 # the production rate of the
# k_out  = 1.0 #saturation of flow out of the vessel
# k_in  = 1.0 #saturation of the flow into the vessel
# k_p = 1.0 #saturation of Xm with Xm inhibition at teh vessel
# k_i = 1.0 #saturation from inhibition of Nu by Xm
# k_l = 1.0 #saturation of Nu with Nu inhibition
# Dn = 0.001 # diffusion of nutrient
# Dx = 0.001 # diffusion of prod

# P0 is Nu and P1 is Xp


 
#CLOSE in middleVesselSpacing = 80//2
# VesselSpacing = HowManyCells//2
# VesselCells = [False for i in range(0,HowManyCells)]
# for i in range(0, HowManyCells, VesselSpacing):
#     VesselCells[i] = True
# VesselCells[HowManyCells//2+2] = True # one next to the center in the middle
# VesselCells[HowManyCells//2+4] = True # one next to the center in the middle
# VesselCells[HowManyCells-1] = True #at the high side
# # create the movable points at the centers of each grid point.
# # No op
# movablePts = []
# for i in range(0,HowManyCells):
#     if VesselCells[i]:
#         if i == HowManyCells//2+2:
#             movablePts.append(i + 0.9)
#         else:
#             movablePts.append(i + 0.5)
#         
# InitialValues = getInitialValues()
# testDynamicsPlot(InitialValues, "centered-0", nonLinear = False, movablePts = movablePts)
# testDynamicsPlot(InitialValues, "centered-non", nonLinear = True, movablePts = movablePts)
# # # Spread out in middle
# VesselSpacing = HowManyCells//4+1
# VesselCells = [False for i in range(0,HowManyCells)]
# for i in range(0, HowManyCells, VesselSpacing):
#     VesselCells[i] = True
# VesselCells[HowManyCells-1] = True #at the high side
# # 

def createVesselsCenter(spacing):
    center = HowManyCells//2
    VesselIds = [center+spacing, center, center - spacing]
    VesselCells = createVesselCells(VesselIds)
    movablePts = createMovablePts(VesselIds)
    return (VesselIds, movablePts, VesselCells)
    
def createVesselsRegular(spacing):
    VesselIds =  list(range(0, HowManyCells, spacing))
    VesselCells = createVesselCells(VesselIds)
    movablePts = createMovablePts(VesselIds)
    return (VesselIds, movablePts, VesselCells)
    
def createVesselsRandom(howMany):
    VesselIds = [randint(0, HowManyCells) for _ in range(0, howMany)]
    VesselCells = createVesselCells(VesselIds)
    movablePts = createMovablePts(VesselIds)
    return (VesselIds, movablePts, VesselCells)
    
def createMovablePts(vesselIds):
    return [(i + 0.5) for i in vesselIds]
    
def randomStep(vesselIds, movablePts):
    # Pick one vessel and move it
    changeID = randint(0, len(vesselIds))
    newPts = []
    print(changeID)
    for i in range(0, len(movablePts)):
        pt = movablePts[i]
        if changeID == i:
            print(pt)
            newPts.append(max(pt + ((random()-0.5)* 1.0), 0.01))
        else:
            newPts.append(pt)
    newVessels = [int(pt) for pt in newPts]
    return (np.array(newVessels), np.array(newPts), createVesselCells(newVessels))
    
def createVesselCells(vesselIds):
    return np.array([i in vesselIds for i in range(0, HowManyCells)])
    
def saveImageOne(iteration):
    #print pathOut + fileName
    fileName = str(iteration).rjust(3,'0')
    onePath = PATH + ResultsPATH + makeRunName(batchSize , minTimeSample) + '/'
    if not os.path.exists(onePath):
        os.makedirs(onePath)
    print(onePath + fileName + '.png')
    fig.savefig(onePath + fileName + '.png',  dpi=100)
    
if __name__ == '__main__':
    runNAME = 'Factory_4' #change this for different eFxperiments
    start = timer()
    # Run the optimizer multiple times over different "slices" of the observed dynamics
    allMovablePts = []
    allFitness = []
    allIterations = []
    # domain size
    VarCount = 2
    HowManyCells = 100
    MaxIterations = 400
    ### Dynamic parameters computed from the current vessel configuration
    radius = np.ones(HowManyCells) # the radius of vessel i
    Q = np.ones(HowManyCells)*1 # the flow rate of vessel i
    initialConditions = getInitialValues()

    def fitnessValue(movablePoints, iteration):
        # only the moveable points are passed in since they will be differentiated
        # need to run the dynamics in the function! Not outside the function then pass it in
        (dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = getDynamics(initialConditions, 
                                                                            getTrueParameters(), 
                                                                            nonLinear = True, 
                                                                            movablePts = movablePoints,
                                                                            runParameters = (max_t, count))
        #only use the last 30% for the fitness, once the system has reached a stable state
        last30 = fitnessList[int(count*.30):-1]
        fitness = np.sum(last30)
        fitPrint = fitness
        if type(fitPrint) != np.float64:
            fitPrint = fitPrint._value
        print("%s, fitness = %.10E " %    (runNAME, fitPrint))
        return fitness


       
    fig = plt.figure(figsize=(18, 3), facecolor='white')
    # use LaTeX fonts in the plot
    #plt.rc('font', family='serif')
    ax_spaceTime1 = fig.add_subplot(1, 4,1, frameon=True)
    ax_spaceTime2 = fig.add_subplot(1, 4,2, frameon=True)
    ax_phase = fig.add_subplot(1, 4, 3, frameon=True)
    ax_fitness = fig.add_subplot(1, 4, 4, frameon=True)
    def callback(iteration):
        # all the dynamic values are dynamically bound
        # DRAW the values for dynamics at specific times over X
        # ax_phase.plot(odeDeltaList[count-1,:,1],'-', color = 'g', label = 'ODE')
        # ax_phase.plot(pdeDeltaList[count-1,:,1],'-', color = 'b', label = 'PDE')
        # allDeltaList = odeDeltaList + pdeDeltaList
        # ax_phase.plot(allDeltaList[count-1,:,1],'-', color = 'r')
        ax_phase.cla()
        ax_phase.plot(dynamicsTrue[count-1,:,1],'-', color = 'b')
        ax_phase.set_title("Production")
        ax_phase.set_xlabel('Position')
        ax_phase.set_ylabel('Concentration')
        ax_phase.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax_fitness.cla()
        ax_fitness.set_title("Production")
        ax_fitness.set_xlabel('Learning Iterations')
        ax_fitness.set_ylabel('Productivity')        

        ax_fitness.plot(allIterations,allFitness, color = 'k')
        ax_fitness.set_xlim(0, MaxIterations)
        ax_fitness.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

        vesselPts = np.array(allMovablePts).T
        print('Callback: (453) ', vesselPts)
        print('Callback: (454) ', allIterations)
        for i in range(0, len(vesselPts)):
            print('    Callback: (456) ', vesselPts[i,:])
            ax_spaceTime2.plot(allIterations, vesselPts[i,:],"-")
        ax_spaceTime2.set_xlim(0, MaxIterations)
        ax_spaceTime2.set_ylim(0, HowManyCells)        # plot the vessel pts as a function of run iteration
        ax_spaceTime2.set_title("Vessel Positions")
        ax_spaceTime2.set_xlabel('Learning Iterations')
        ax_spaceTime2.set_ylabel('Position')
        #dx_dy = HowManyCells*1.0/len(dynamicsTrue[:,0,0])
        # ax_spaceTime1.set_title("Nutrient")
        # #print(dynamicsTrue[start:-1,:,0].shape())
        # ax_spaceTime1.imshow(dynamicsTrue[:,:,0], interpolation='none', cmap='jet', aspect = dx_dy)#, vmin=0, vmax=1)
        # ax_spaceTime2.set_title("Product")
        # ax_spaceTime2.imshow(dynamicsTrue[:,:,1], interpolation='none', cmap='jet', aspect = dx_dy)#, vmin=0, vmax=1)
        # show fitness:
        # (max_T, count) = getSampleParameters()
        # times = np.linspace(0., max_T, count)
        # ax_fitness.plot(times, np.cumsum(fitnessList))
        print("%s, fitness = %.10E " %    (runNAME, np.cumsum(fitnessList[int(count*.30):-1])[-1]))
        plt.draw()
        saveImageOne(iteration)
        plt.pause(0.001)
 
        
    ## RUN A SIMULATION to start the optimizer on a random distribution of vessels
    (VesselIds, movablePts, VesselCells) = createVesselsRandom(4)
    
    # run it once, to get stable state dynamics
    # (dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = getDynamics(getInitialValues(), 
    #                     getTrueParameters(), 
    #                     nonLinear = True, 
    #                     movablePts = movablePts,
    #                     runParameters = getSampleParameters())
    # # Now identified stable dynamics, slice out the last bit of all the dynamics
    # (max_t, count) = getSampleParameters()
    # stablePercent = 0.8
    # startIndex = int(count * stablePercent)
    # # slice out the last bit of dynamics
    # (dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = (dynamicsTrue[startIndex: -1, :, :],
    #                                                             fitnessList[startIndex: -1], 
    #                                                             odeDeltaList[startIndex: -1, :, :],
    #                                                             pdeDeltaList[startIndex: -1, :, :])
    # # start the optimizing iterations for the last 20% of the dynamics
    # (max_t, count) = (max_t*(1-stablePercent), int(count*(1-stablePercent)))
    # movablePts = tuple(movablePts)
    movablePts = [(val) for val in movablePts]
    print('Movable Pts: ', movablePts)
    bestFitness = 0
    bestVesselPts = movablePts
    (max_t, count) = getSampleParameters()
    gradFitness = grad(fitnessValue)
    for iteration in range(0, MaxIterations):
        grad_pts = gradFitness(movablePts, iteration)
        print(grad_pts)
        movablePts = tuple(np.array(movablePts) + np.array(grad_pts) * StepSize)

        # movablePts = np.array(movablePts) + np.array(grad_pts)
#        # initial conditions are at the beginning of the dynamics just calcluated
        initialConditions = getInitialValues()
#         # make a random mutation
#         (VesselIds, movablePts, VesselCells) = randomStep(VesselIds, movablePts)
#         #print(movablePts)
#         # recompute the dynamics
        (dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) =     getDynamics(initialConditions, 
                        getTrueParameters(), 
                        nonLinear = True, 
                        movablePts = movablePts,
                        runParameters = getSampleParameters())
        #only the last 30% of the dynamics
        newFitness = np.cumsum(fitnessList[int(count*.30):-1])[-1]
#         if newFitness > bestFitness:
#             bestFitness = newFitness
#             bestVesselPts = movablePts
#             print("********Selected %s \n     Fitness = %.10E \n" % (str(bestVesselPts), bestFitness))
        # allMovablePts.append(bestVesselPts)
        allMovablePts.append(movablePts)
        allIterations.append(iteration)
        # allFitness.append(bestFitness)
        allFitness.append(newFitness)
        callback(iteration) #display the current solution
            

    

# #testDynamicsPlot(InitialValues, "nonLinear-m", nonLinear = True, movablePts = movablePts)
# #last 80 %, initialConditions set to that starting time
