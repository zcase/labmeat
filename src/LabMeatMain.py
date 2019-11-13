from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse
import os
import math
import matplotlib.pyplot as plt

import imageio
from natsort import natsorted, ns
from sklearn.preprocessing import minmax_scale

from optimizers import adam as AdamTwo


# import numpy as np

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
import autograd.scipy.signal as sig
from timeit import default_timer as timer

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gaussian(x):
    return math.exp(-x**2)

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



def get_submatrix_add(np_matrix, center_pt_tuple, convolution, submatrix_size=2):
    h, w = np_matrix.shape
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

    new_convolution = np.array(convolution[conv_row_start:conv_row_end, conv_col_start:conv_col_end])
    # np_matrix[row_start:row_end, col_start:col_end] += new_convolution
    new_matrix = []
    for row_idx, row in enumerate(np_matrix):
        count = 0
        if row_idx >= row_start and row_idx < row_end and count < conv_col_end:
            val2 = list(new_convolution[conv_col_start + count])
            updated_lst_vals = [sum(x) for x in zip(row[col_start:col_end], val2)]
            row[col_start:col_end] = updated_lst_vals # list assignment not np.array assignment
            count += 1

        new_matrix.append(row)
    
    new_matrix = np.array(new_matrix)

    return new_matrix


def getSampleParameters():
    # how many iterations to run the ODE for one trajectory, and the time step
    #     (max_T, samples per trajectory) = getSampleParameters()
    return (5, 25) #3 15 works

def loss_health(img):
    # 2D array of neutrient values
    # sum of sigmoid values (high N is low low, low N is high loss)
    newimg = np.array(minmax_scale(np.array(img)))
    total_loss = 0
    for ix,iy in np.ndindex(newimg.shape):
        loss = gaussian(newimg[ix,iy])
        total_loss += loss

    return total_loss

def simulate(mvble_pts, t, vasc_structure):
    # Updtae Vascular Structure Movable Points
    vasc_structure.update_moveable_pts(mvble_pts)

    sim_img_folder = 'simulation_imgs/imgs/'
    sim_graph_folder = 'simulation_imgs/graphs/'
    if not os.path.exists(sim_img_folder):
        os.makedirs(sim_img_folder)
    if not os.path.exists(sim_graph_folder):
        os.makedirs(sim_graph_folder)

    # Solve for flow
    flowDict = computeFlow(vasc_structure)

    # Add flows to image
    vasc_structure.add_flows_to_img(flowDict)

    # vasc_structure.print_images(graph_name=sim_graph_folder + 'sim_graph_'+str(t)+'.png',
    #                             img_name=sim_img_folder + 'sim_img_'+str(t)+'.png')

    # run the diffusion
    # diffused_img = lab_meat_diffuse(vas_structure.img, 100, 1, 50)
    diffused_img = diffusion(mvble_pts, vasc_structure.img)

    return diffused_img

def diffusion(mvble_pts, img):
    # D is the defusion constant
    D = .225

    #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
    for i in range(0, 20): # how many times you run a diffusion update
        convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        deltaDiffusion = sig.convolve(img, convolve)[1:-1, 1:-1] #take off first and last
        # if i > 0:
        deltaDiffusion += np.array(img)

        # the update to the img from one step of diffusion
        img = np.array(img + deltaDiffusion + nonlinearDiffusion(mvble_pts, img))
        img_pic = np.pad(img, ((2, 3), (2, 3)), 'constant')
        plt.imsave('diffusePngs/TestDiffuse_'+str(i)+'.png', np.rot90(img_pic), cmap='jet')

    path_to_img_dir = 'diffusePngs/'
    images = []
    for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
        if file_name.endswith('.png'):
            file_path = os.path.join(path_to_img_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('VascDiffuse.gif', images, fps=5)

    return img


# None linear diffusion (compute each convoution for each location)
def nonlinearDiffusion(mvble_pts, img):
    #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
    #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
    h, w = img.shape
    deltaDomain = np.zeros((h, w))
    for i in range(1, len(mvble_pts), 2):
        x = mvble_pts[i-1]
        y = mvble_pts[i]
        # print(type(x), type(1))
        if type(x) != np.float64 and type(x) != type(1):
            x = x._value
            y = y._value
        int_x = int(x)
        int_y = int(y)
        
        np_pt = np.array([x, y])
        # int_np_pt = np.array([int_x, int_y])
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

        convolution = np.array([[sigmoid(dist_0 - 1), sigmoid(dist_1 - 1), sigmoid(dist_2 - 1)], [sigmoid(dist_3 - 1), X, sigmoid(dist_5 - 1)], [sigmoid(dist_6 - 1), sigmoid(dist_7 - 1), sigmoid(dist_8 - 1)]])
        deltaDomain = get_submatrix_add(deltaDomain, (int_x, int_y), convolution)

    return deltaDomain

def create_remove_imgs():
    path_to_diffuse_pngs = 'diffusePngs/'
    sim_img_folder = 'simulation_imgs/imgs/'
    sim_graph_folder = 'simulation_imgs/graphs/'
    if not os.path.exists(path_to_diffuse_pngs):
        os.makedirs(path_to_diffuse_pngs)

    for img_file in os.listdir(path_to_diffuse_pngs):
        os.remove(path_to_diffuse_pngs + img_file)

    if os.path.exists(sim_img_folder):
        for img_file in os.listdir(sim_img_folder):
            os.remove(sim_img_folder + img_file)

    if os.path.exists(sim_graph_folder):
        for img_file in os.listdir(sim_graph_folder):
            os.remove(sim_graph_folder + img_file)

if __name__ == "__main__":
    start = timer()
    total_iterations = 10
    all_params = []
    all_loss = []

    create_remove_imgs()

    (max_T, count) = getSampleParameters()
    t = np.linspace(0., max_T, count)


    # test = np.zeros((11, 11))
    # test[5][5] = 255
    # t = np.array(test)
    # # diffused_test = diffusion((5,5), test)
    # for t in te:
    #     print(t)
    # print(te)
    # res = np.array([np.linalg.norm(th) for th in te])
    # print(te.shape[0])
    # test = np.arange(100.0).reshape(10,10)
    new_convolution = np.ones((3,3)) * 100
    # row_start = 4
    # row_end = 7


    # col_start = 4
    # col_end = 7

    # print(new_convolution[1, 1:3])

    # new_matrix = []
    # for row_idx, row in enumerate(test):
    #     row = list(row)
    #     if row_idx >= row_start and row_idx < row_end:
    #         val2 = list(new_convolution[(row_idx-len(new_convolution)) - 1])
    #         updated_lst_vals = [sum(x) for x in zip(row[col_start:col_end], val2)]
    #         row[col_start:col_end] = updated_lst_vals

    #     new_matrix.append(row)
    
    # new_matrix = np.array(new_matrix)

    # deltaDomain = np.arange(100.0).reshape(10,10)

    # print(deltaDomain)
    # deltaDomain = get_submatrix_add(deltaDomain, (0,0), new_convolution)
    # print('\n')
    # print(deltaDomain)

    # os.sys.exit()


    # TODO: Make this (num_of_nodes) a command line argument
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=2)
    vas_structure.print_images(graph_name='AutoGrad_startGraph.png', img_name='AutoGrad_startImg.png')
    mvable_pts = tuple(vas_structure.flatten_mvable_pts())
    print(mvable_pts)


    def fitness(mvable_pts, iter):
        diffused_sim_img = simulate(mvable_pts, iter, vas_structure)
        return loss_health(diffused_sim_img)


    # Setup display figures

    # Plot Data through callback
    def callback(mvable_pts, iter, g):
        print(iter)
        print(mvable_pts)
        print(g)
        return 3

    # callback(mvable_pts, 0, 0)
    grad_fitness = grad(fitness)
    grad_fitness(mvable_pts, 1)

    print('Starting AutoGrad\n')
    optimized_mvble_pts = AdamTwo(grad(fitness), mvable_pts, step_size=0.005, num_iters=total_iterations, callback=callback)
    print('Finished AutoGrad\n')

    print('    Optimized Pts:')
    print('      ', optimized_mvble_pts)
    print('\n')
    vas_structure.update_moveable_pts(optimized_mvble_pts)
    vas_structure.print_images(graph_name='AutoGrad_graph.png', img_name='AutoGrad_img.png')
    end = timer()
    print('Time per iteration: ', str((end-start) / total_iterations))
