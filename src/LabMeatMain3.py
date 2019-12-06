import os
import imageio
import math
import matplotlib

if os.sys.platform == "linux" or os.sys.platform == "linux2":
    matplotlib.use('TKAgg')
elif os.sys.platform == "darwin":
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

import random
import time

from natsort import natsorted, ns
from timeit import default_timer as timer

# import jax.numpy as np
# from jax import grad, jit, vmap
# from jax.builtins import tuple

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
import autograd.scipy.signal as sig

from VasGen2 import VasGen2
from equations import *
from meatModel2d import getDynamics, getTrueParameters, getSampleParameters

def create_remove_imgs():
    path_to_diffuse_pngs = 'LabMeatMain3/diffusePngs/'
    sim_img_folder = 'LabMeatMain3/imgs/'
    sim_graph_folder = 'LabMeatMain3/graphs/'
    sim_fig_folder = 'LabMeatMain3/figs/'
    if not os.path.exists(path_to_diffuse_pngs):
        os.makedirs(path_to_diffuse_pngs)

    if not os.path.exists(sim_img_folder):
        os.makedirs(sim_img_folder)

    if not os.path.exists(sim_graph_folder):
        os.makedirs(sim_graph_folder)

    if not os.path.exists(sim_fig_folder):
        os.makedirs(sim_fig_folder)

    for img_file in os.listdir(path_to_diffuse_pngs):
        os.remove(path_to_diffuse_pngs + img_file)

    if os.path.exists(sim_img_folder):
        for img_file in os.listdir(sim_img_folder):
            os.remove(sim_img_folder + img_file)

    if os.path.exists(sim_graph_folder):
        for img_file in os.listdir(sim_graph_folder):
            os.remove(sim_graph_folder + img_file)

    if os.path.exists(sim_fig_folder):
        for img_file in os.listdir(sim_fig_folder):
            os.remove(sim_fig_folder + img_file)

def saveImageOne(iteration):
    fig.savefig('LabMeatMain3/figs/' + str(iteration) + '.png', size=[1600,400])

if __name__ == "__main__":
    print("Autograd LabMeat")
    numNodes = 2
    stepSize = 0.35 # 0.008
    path_to_diffuse_pngs = 'LabMeatMain3/diffusePngs/'
    sim_img_folder = 'LabMeatMain3/imgs/'
    sim_graph_folder = 'LabMeatMain3/graphs/'
    
    create_remove_imgs()

    vas_structure = VasGen2(max_range=20, num_of_nodes=numNodes, side_nodes=False)
    vas_structure.print_images(graph_name='LabMeatMain3/LabMeatMain3_startGraph.png', img_name='LabMeatMain3/LabMeatMain3_startImg.png')
    flowDict = computeFlow(vas_structure)
    vas_structure.add_flows_to_img(flowDict)
    img = np.array(vas_structure.img)
    vas_structure.Q = img

    mvable_pts = vas_structure.moveable_pts

    all_loss = []
    time_lst = []
    fitnessList = []

    def fitnessValue(movablePoints, iteration):
        global fitnessList
        # only the moveable points are passed in since they will be differentiated
        # need to run the dynamics in the function! Not outside the function then pass it in
        (dynamicsTrue, fitnessList, odeDeltaList, pdeDeltaList) = getDynamics(vas_structure, 
                                                                            getTrueParameters(), 
                                                                            nonLinear=True, 
                                                                            movablePts=movablePoints,
                                                                            runParameters = getSampleParameters())
        #only use the last 30% for the fitness, once the system has reached a stable state
        (max_t, count) = getSampleParameters()
        loss = np.cumsum(fitnessList[int(count*.30):-1])[-1]
        return loss

    # Set up figures
    fig = plt.figure(figsize=(16, 6), facecolor='white')
    fig.suptitle('Automatic Differentiation', fontsize=16)
    ax_loss         = fig.add_subplot(231, frameon=True)
    ax_cpu          = fig.add_subplot(232, frameon=True)
    ax_node_graph   = fig.add_subplot(233, frameon=True)
    ax_img          = fig.add_subplot(234, frameon=True)
    ax_product      = fig.add_subplot(235, frameon=True)
    ax_nutrient     = fig.add_subplot(236, frameon=True)
    plt.show(block=False)

    def callback(mvable_pts, iter, nowLoss, time_duration):
        # ==== LOSS as a function of TIME ==== #
        ax_loss.cla()
        ax_loss.set_title('Optimization Gain')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Fitness')
        nowLoss = nowLoss
        all_loss.append(nowLoss)
        iteration = np.arange(0, len(all_loss), 1)

        ax_loss.plot(iteration, all_loss, '-', linestyle = 'solid', label='Gain')
        ax_loss.set_xlim(iteration.min(), iteration.max())
        ax_loss.legend(loc = "upper left")

        # ==== CPU Time ==== #
        ax_cpu.cla()
        ax_cpu.set_title('CPU Time Per Iteration')
        ax_cpu.set_xlabel('Iteration')
        ax_cpu.set_ylabel('CPU Time (s)')
        time_lst.append(time_duration)
        ax_cpu.plot(iteration, time_lst, '-', linestyle = 'solid', label='CPU Time')
        ax_cpu.legend(loc = "upper left")

        # ==== Plots the Node Graph ==== #
        ax_node_graph.cla()
        ax_node_graph.set_title('Node Graph')
        for j, s in enumerate(vas_structure.tri.simplices):
            p = np.array(vas_structure.pts)[s].mean(axis=0)
            ax_node_graph.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        ax_node_graph.triplot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], vas_structure.tri.simplices)
        ax_node_graph.plot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], 'o')

        # ==== Plot Flow Img ==== #
        ax_img.cla()
        ax_img.set_title('Flow Image')
        ax_img.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_img.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_img.imshow(np.rot90(np.array(vas_structure.img)[1:,1:]), cmap='jet')

        # ==== Plot Product Image ==== #
        ax_product.cla()
        ax_product.set_title('Product Image')
        ax_product.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_product.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_product.imshow(np.rot90(np.array(vas_structure.product_values)), cmap='jet')
        
        # # ==== Plot Nutrient Image ==== #
        ax_nutrient.cla()
        ax_nutrient.set_title('Nutrient Image')
        ax_nutrient.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax_nutrient.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_nutrient.imshow(np.rot90(np.array(vas_structure.nutrient_values)), cmap='jet')

        plt.subplots_adjust( wspace = 0.5, hspace = 0.5 )
        plt.draw()
        saveImageOne(iter)
        plt.pause(0.001)
        return 3

    gradFitness = grad(fitnessValue)
    m = np.zeros(np.array(mvable_pts).shape)
    v = np.zeros(np.array(mvable_pts).shape)
    b1=0.9
    b2=0.999
    eps=10**-8
    (max_t, count) = getSampleParameters()
    for i in range(200):
        start = time.time()
        grad_pts = gradFitness(mvable_pts, i)

        m = (1 - b1) * np.array(grad_pts)      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (np.array(grad_pts)**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))

        mvable_pts = tuple(np.array(mvable_pts) + np.array(grad_pts))
        mvable_pts = mvable_pts + stepSize * mhat / (np.sqrt(vhat) + eps)
        new_pts = []
        for val in mvable_pts:
            x = float(val[0])
            y = float(val[1])
            if x < 0:
                x = 0.0
            elif x >= vas_structure.max_range:
                x = float(vas_structure.max_range - 1)

            if y < 0:
                y = 0.0
            elif y >= vas_structure.max_range:
                y = float(vas_structure.max_range - 1)
            
            new_pts += [[x, y]]

        flat_list = []
        for lst_pt in new_pts:
            for xORy_val in lst_pt:
                flat_list.append(xORy_val)
        mvable_pts = flat_list
        print('Updated mvable_pts:\n', mvable_pts)
        vas_structure.update_moveable_pts(mvable_pts)
        newFitness = np.cumsum(fitnessList[int(count*.30):-1])[-1]

        flowDict = computeFlow(vas_structure)
        vas_structure.add_flows_to_img(flowDict)
        mvable_pts = vas_structure.moveable_pts

        end = time.time()
        elapsedSec = (end - start)
        callback(mvable_pts, i, newFitness._value, elapsedSec)
