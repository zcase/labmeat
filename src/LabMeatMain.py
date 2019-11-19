from __future__ import print_function
from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse
import os
import math
import matplotlib

if os.sys.platform == "linux" or os.sys.platform == "linux2":
    matplotlib.use('TKAgg')
elif os.sys.platform == "darwin":
    matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

import imageio
from natsort import natsorted, ns
from sklearn.preprocessing import minmax_scale

from optimizers import adamVas as AdamTwo
from autograd.builtins import isinstance, tuple, list

from VasGen2 import VasGen2
import random


# import numpy as np

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
# from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
import autograd.scipy.signal as sig
from timeit import default_timer as timer
from autograd.tracer import trace, Node

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def jax_sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def gaussian(x):
    return np.exp(-x**2)

def random_walk():
    path_to_diffuse_pngs = 'diffusePngs/'
    sim_img_folder = 'simulation_imgs/imgs/'
    sim_graph_folder = 'simulation_imgs/graphs/'

    vas_structure = VascularGenerator(max_range=100, num_of_nodes=2)
    vas_structure.print_images(graph_name='AutoGrad_startGraph.png', img_name='AutoGrad_startImg.png')
    mvable_pts = tuple(vas_structure.flatten_mvable_pts())
    print(mvable_pts)

    test_movement = np.array([-30, -20, -10, 0, 10, 20, 30])

    loss_lst = []
    time_lst = []
    for i in range(51):
        vas_structure.img = simulate(mvable_pts, i, vas_structure)
        vas_structure.print_images(graph_name=sim_graph_folder+'test_graph'+str(i)+'.png', img_name=sim_img_folder+'test_img'+str(i)+'.png')
        # print(vas_structure.img)
        loss = loss_health(vas_structure.img)
        loss_lst.append(loss)
        time_lst.append(i)
        print(i, 'LOSS:', loss)

        index = np.random.choice(vas_structure.moveable_pts.shape[0], 1, replace=False)
        inc_index_x = np.random.choice(test_movement, 1, replace=False)
        inc_index_y = np.random.choice(test_movement, 1, replace=False)

        test_x = vas_structure.moveable_pts[index][0][0] + inc_index_x[0]
        test_y = vas_structure.moveable_pts[index][0][1] + inc_index_y[0]

        if test_x < 0:
            test_x = 0
        if test_x > 100:
            test_x = 100

        if test_y < 0:
            test_y = 0
        if test_y > 100:
            test_y = 100

        vas_structure.moveable_pts[index] = [test_x, test_y]
        mvable_pts = tuple(vas_structure.flatten_mvable_pts())


    # colors = ("red", "green", "blue")
    fig = plt.figure()
    plt.scatter(time_lst, loss_lst, alpha=0.5)
    plt.title('Loss Over Time')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Loss', fontsize=16)
    fig.savefig('LossGraph.png')

    images = []
    for file_name in natsorted(os.listdir(sim_img_folder), key=lambda y: y.lower()):
        if file_name.endswith('.png'):
            file_path = os.path.join(sim_img_folder, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('Vasc_Random_Walk.gif', images, fps=50)
    os.sys.exit()



def get_submatrix_add(lst_matrix, center_pt_tuple, convolution, submatrix_size=2):
    w, h = len(lst_matrix), len(lst_matrix[0])
    orig_row, orig_col = center_pt_tuple
    row, col = orig_row, orig_col
    row_start = -1
    row_end = -1
    col_start = -1
    col_end = -1

    if row >= 0 and row < h:
        row_start = row - submatrix_size + 1 if (row - submatrix_size + 1) > 0 else 0
        row_end = row + submatrix_size if (row + submatrix_size) <= h else h
    elif row == h:
        row -= 1
        row_start = row - submatrix_size + 1 if (row - submatrix_size + 1) > 0 else 0
        row_end = row + submatrix_size if (row + submatrix_size) <= h else h
    elif row < 0 or row > h:
        print('Bad row dimension of ', row)

    if col >= 0 and col < w:
        col_start = col - submatrix_size + 1 if (col - submatrix_size + 1) > 0 else 0
        col_end = col + submatrix_size if (col + submatrix_size) <= w else w
    elif col == w:
        col -= 1
        col_start = col - submatrix_size + 1 if (col - submatrix_size + 1) > 0 else 0
        col_end = col + submatrix_size if (col + submatrix_size) <= w else w
    elif col < 0 or col > w:
        print('Bad col dimension of ', col)

    conv_row_start = 0
    conv_row_end = 3
    conv_col_start = 0
    conv_col_end = 3
    new_convolution = None

    if (orig_row - submatrix_size + 1) < 0:
        conv_row_start = 1
    if (orig_row + submatrix_size ) >= h:
        conv_row_end = 1

    if (orig_col - submatrix_size + 1) < 0:
        conv_col_start = 1
    if (orig_col + submatrix_size) >= w:
        conv_col_end = 1

    new_convolution = convolution[conv_row_start:conv_row_end][conv_col_start:conv_col_end]

    new_matrix = []
    for row_idx, lst_row in enumerate(lst_matrix):
        count = 0
        if row_idx >= row_start and row_idx < row_end and count < conv_col_end:
            val2 = new_convolution[conv_col_start + count]
            updated_lst_vals = [sum(x) for x in zip(lst_row[col_start:col_end], val2)]

            new_row=[]
            row_count = 0
            for cur_col_idx, val in enumerate(lst_row):
                if cur_col_idx >= col_start and cur_col_idx < col_end:
                    new_row.append(updated_lst_vals[0+row_count])
                    row_count = row_count + 1
                else:
                    new_row.append(val)

            count = count + 1
            lst_row = new_row

        new_matrix.append(lst_row)
    return new_matrix


def getSampleParameters():
    # how many iterations to run the ODE for one trajectory, and the time step
    #     (max_T, samples per trajectory) = getSampleParameters()
    return (5, 25) #3 15 works

def loss_health(img, iter):
    # 2D array of neutrient values
    # sum of sigmoid values (high N is low low, low N is high loss)
    total_loss = 0.0
    for ix,iy in np.ndindex(img.shape):
        loss = gaussian(img[ix,iy])
        total_loss = total_loss +  (loss *-1)

    print('LabMeatMain Line 171 LOSS:                           ', total_loss, iter)
    return total_loss

def simulate(mvble_pts, t, vasc_structure):
    # print('Int SIMULATE')
    # Updtae Vascular Structure Movable Points
    vasc_structure.update_moveable_pts(mvble_pts)

    # Solve for flow
    flowDict = computeFlow(vasc_structure)

    # Add flows to image
    vasc_structure.add_flows_to_img(flowDict)

    # vasc_structure.print_images(graph_name=sim_graph_folder + 'sim_graph_'+str(t)+'.png',
    #                             img_name=sim_img_folder + 'sim_img_'+str(t)+'.png')

    diffused_img = diffusion(vasc_structure.moveable_pts, vasc_structure.img)

    return diffused_img

def diffusion(mvble_pts, img):
    # D is the defusion constant
    D = .225
    B = D / 10

    #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
    for _ in range(0, 20): # how many times you run a diffusion update
        convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        deltaDiffusion = sig.convolve(np.array(img), convolve)[1:-1, 1:-1] #take off first and last
        deltaDiffusion = deltaDiffusion + np.array(img)

        # the update to the img from one step of diffusion
        img = np.array(np.array(img) + np.array(deltaDiffusion) + np.array(nonlinearDiffusion(mvble_pts, img)))
        img = img - (B*img)
    #     print(type(img))
    #     img_pic = np.pad(img, ((2, 3), (2, 3)), 'constant')
    #     plt.imsave('diffusePngs/TestDiffuse_'+str(i)+'.png', np.rot90(np.array(list(img_pic))), cmap='jet')

    # path_to_img_dir = 'diffusePngs/'
    # images = []
    # for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
    #     if file_name.endswith('.png'):
    #         file_path = os.path.join(path_to_img_dir, file_name)
    #         images.append(imageio.imread(file_path))
    # imageio.mimsave('VascDiffuse.gif', images, fps=5)

    # newimg = np.array(minmax_scale(np.array(img._value)))
    np_img = np.array(img)
    mn, mx = np_img.min(), np_img.max()
    np_img = (np_img - mn) / (mx - mn)
    return np_img

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

# None linear diffusion (compute each convoution for each location)
def nonlinearDiffusion(mvble_pts, img):
    #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
    #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
    h, w = np.array(img).shape
    deltaDomain2 = []
    for _ in range(w):
        deltaDomain2.append([0.0 for _ in range(h)])

    for i in range(len(mvble_pts)):
        pt = mvble_pts[i]
        x = pt[0]
        y = pt[1]
        int_x = 0
        int_y = 0


        if type(x) != type(np.array((1,1))) and type(x) != type(1):
            int_x = int(np.array(mvble_pts[i][0]._value))
            int_y = int(np.array(mvble_pts[i][1]._value))
        else:
            int_x = int(np.array(mvble_pts[i][0]))
            int_y = int(np.array(mvble_pts[i][1]))
        # int_x = int(np.array(mvble_pts[i][0]._value))
        # int_y = int(np.array(mvble_pts[i][1]._value))
        np_pt = np.array([x, y])

        dist_0 = np.linalg.norm(np_pt - np.array([int_x-1, int_y-1]))
        dist_1 = np.linalg.norm(np_pt - np.array([int_x, int_y-1]))
        dist_2 = np.linalg.norm(np_pt - np.array([int_x+1, int_y-1]))

        dist_3 = np.linalg.norm(np_pt - np.array([int_x-1, int_y]))
        # dist_4 = np.linalg.norm(np_pt - np.array([int_x, int_y]))
        dist_5 = np.linalg.norm(np_pt - np.array([int_x+1, int_y]))

        dist_6 = np.linalg.norm(np_pt - np.array([int_x-1, int_y+1]))
        dist_7 = np.linalg.norm(np_pt - np.array([int_x, int_y+1]))
        dist_8 = np.linalg.norm(np_pt - np.array([int_x+1, int_y+1]))

        X = -sigmoid(dist_0 - 1) - sigmoid(dist_1 - 1) - sigmoid(dist_2 - 1) - sigmoid(dist_3 - 1) - sigmoid(dist_5 - 1) - sigmoid(dist_6 - 1) - sigmoid(dist_7 - 1) - sigmoid(dist_8 - 1)

        convolution = [[sigmoid(dist_0 - 1), sigmoid(dist_1 - 1), sigmoid(dist_2 - 1)], [sigmoid(dist_3 - 1), X, sigmoid(dist_5 - 1)], [sigmoid(dist_6 - 1), sigmoid(dist_7 - 1), sigmoid(dist_8 - 1)]]
        deltaDomain2 = get_submatrix_add(deltaDomain2, (int_x, int_y), convolution)

    return deltaDomain2

def create_remove_imgs():
    path_to_diffuse_pngs = 'diffusePngs/'
    sim_img_folder = 'simulation_imgs/imgs/'
    sim_graph_folder = 'simulation_imgs/graphs/'
    if not os.path.exists(path_to_diffuse_pngs):
        os.makedirs(path_to_diffuse_pngs)

    if not os.path.exists(sim_img_folder):
        os.makedirs(sim_img_folder)

    if not os.path.exists(sim_graph_folder):
        os.makedirs(sim_graph_folder)

    for img_file in os.listdir(path_to_diffuse_pngs):
        os.remove(path_to_diffuse_pngs + img_file)

    if os.path.exists(sim_img_folder):
        for img_file in os.listdir(sim_img_folder):
            os.remove(sim_img_folder + img_file)

    if os.path.exists(sim_graph_folder):
        for img_file in os.listdir(sim_graph_folder):
            os.remove(sim_graph_folder + img_file)

def saveImageOne(iteration):
    #print pathOut + fileName
    fileName = RunNAME + str(iteration).rjust(3,'0')
    fig.savefig(PATH + ResultsPATH + fileName + '.png', size=[1600,400])#, size=[1000,1000]) #, size=[700,700] IF 1000, renders each quadrant separately



if __name__ == "__main__":
    start = timer()
    total_iterations = 100
    all_params = []
    all_loss = []

    create_remove_imgs()

    (max_T, count) = getSampleParameters()
    t = np.linspace(0., max_T, count)

    print('Creating Vas')
    # vas_structure = VascularGenerator(max_range=100, num_of_nodes=2)
    # vas_structure = VasGen2(max_range=100, num_of_nodes=2)
    vas_structure = VasGen2(max_range=20, num_of_nodes=2)
    print('CreatED Vas')
    vas_structure.print_images(graph_name='AutoGrad_startGraph.png', img_name='AutoGrad_startImg.png')

    # Solve for flow
    flowDict = computeFlow(vas_structure)
    vas_structure.add_flows_to_img(flowDict)
    img = vas_structure.img
    diffused_img = img


    def fitness(fit_mvable_pts, iter):
        diffused_img = diffusion(fit_mvable_pts, img)
        return loss_health(diffused_img, iter)


    # Setup display figures
    fig = plt.figure(figsize=(16, 4), facecolor='white')
    # use LaTeX fonts in the plot
    #plt.rc('font', family='serif')
    ax_loss     = fig.add_subplot(141, frameon=True)
    ax_node_graph    = fig.add_subplot(142, frameon=True)
    ax_img = fig.add_subplot(143, frameon=True)
    ax_diffused_img = fig.add_subplot(144, frameon=True)
    plt.show(block=False)
    count = 1
    # Plot Data through callback
    def callback(mvable_pts, iter, g):
        # print('callback', mvable_pts, iter)
        ####################################
        # LOSS as a function of TIME
        ax_loss.cla()
        ax_loss.set_title('Train Loss')
        ax_loss.set_xlabel('t')
        ax_loss.set_ylabel('loss')
        #colors =['b', 'b', 'g', 'g', 'r', 'r']
        nowLoss = fitness(mvable_pts, iter)
        flowDict = computeFlow(vas_structure)
        vas_structure.add_flows_to_img(flowDict)

        all_loss.append(nowLoss)
        time = np.arange(0, len(all_loss), 1)
        
        ax_loss.plot(time, all_loss, '-', linestyle = 'solid', label='loss') #, color = colors[i]
        ax_loss.set_xlim(time.min(), time.max())
        ax_loss.legend(loc = "upper left")

        ## Plots the Node Graph
        ax_node_graph.cla()
        ax_node_graph.set_title('Node Graph')
        for j, s in enumerate(vas_structure.tri.simplices):
            p = np.array(vas_structure.pts)[s].mean(axis=0)
            ax_node_graph.text(p[0], p[1], 'Cell #%d' % j, ha='center') # label triangles
        ax_node_graph.triplot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], vas_structure.tri.simplices)
        ax_node_graph.plot(np.array(vas_structure.pts)[:,0], np.array(vas_structure.pts)[:,1], 'o')

        # ==== Plot Img Version ==== #
        pltimg = np.pad(np.array(vas_structure.img), ((2, 3), (2, 3)), 'constant')
        # plt.imsave(img_name, np.rot90(pltimg), cmap='jet')
        ax_img.set_title('Flow Image')
        ax_img.imshow(np.rot90(pltimg))

        # ==== Plot Diffused Img Version ==== #
        diffused_img_plt1 = diffusion(mvable_pts, vas_structure.img)
        diffused_img_plt1 = np.pad(np.array(diffused_img_plt1), ((2, 3), (2, 3)), 'constant')
        ax_diffused_img.set_title('Diffused Image')
        ax_diffused_img.imshow(np.rot90(diffused_img_plt1))

        plt.draw()
        # saveImageOne(iter)

        plt.pause(0.001)
        return 3

    pts = np.array(vas_structure.pts)
    print('Starting AutoGrad\n')
    print('Original MvPts: ', vas_structure.moveable_pts)
    # optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=1, num_iters=total_iterations, callback=callback)
    optimized_mvble_pts = AdamTwo(grad(fitness), vas_structure.moveable_pts, vas_structure=vas_structure, step_size=0.1, num_iters=total_iterations, callback=callback)
    print('Finished AutoGrad\n')

    print('    Optimized Pts:')
    print('      ', optimized_mvble_pts)
    print('\n')

    # print('Optimized PTs')
    # print(vas_structure.pts)
    # vas_structure.update_moveable_pts(optimized_mvble_pts)
    print('Setting up optimized points')
    print(np.array(vas_structure.pts))
    # print()
    # print(pts)
    # vas_structure.print_images(graph_name='Optimized_AutoGrad_graph.png', img_name='Optimized_AutoGrad_img.png')
    end = timer()
    print('Time per iteration: ', str((end-start) / total_iterations))

    def img_path_generator(path_to_img_dir):
        for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
            if file_name.endswith('.png'):
                file_path = os.path.join(path_to_img_dir, file_name)
                yield imageio.imread(file_path)

    sim_img_folder = 'simulation_imgs/imgs/'
    sim_graph_folder = 'simulation_imgs/graphs/'

    imageio.mimsave('AutoDiff_Img.gif', img_path_generator(sim_img_folder), fps=50)
    imageio.mimsave('AutoDiff_Graph.gif', img_path_generator(sim_graph_folder), fps=50)
