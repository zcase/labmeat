from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse
import os
import math
import matplotlib.pyplot as plt

import imageio
from natsort import natsorted, ns
from sklearn.preprocessing import minmax_scale


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

    new_convolution = convolution[conv_row_start:conv_row_end, conv_col_start:conv_col_end]
    np_matrix[row_start:row_end, col_start:col_end] += new_convolution

    return np_matrix


def adamMy(grad, lossFn, x, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    m = np.full((len(x),), 0.001)
    loss0 = 0
    changeStepSize = 0.01
    for i in range(0, num_iters):
        loss1 = lossFn(x,i)
        lossChange = abs(loss0-loss1)
        if lossChange < 0.01:
            step_size = 0.009
        # if lossChange < 0.008:
        #     step_size = 0.015
        g = grad(x,i)
        #g = np.minimum(g, m)
        print("in ADAM Dloss = %.4f, step size = %.4f" % (lossChange, step_size))
        #print(loss)
        #print(x)
        if callback: 
            callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x

def getSampleParameters():
    # how many iterations to run the ODE for one trajectory, and the time step
    #     (max_T, samples per trajectory) = getSampleParameters()
    return (5, 25) #3 15 works

def loss_health(img):
    # 2D array of neutrient values
    # sum of sigmoid values (high N is low low, low N is high loss)
    minmax_scale(img, copy=False)
    total_loss = 0
    for ix,iy in np.ndindex(img.shape):
        loss = gaussian(img[ix,iy])
        print(loss, ix, iy, img[ix,iy])
        total_loss += loss

    return total_loss

def simulate(mvble_pts, t, vasc_structure):
    # Updtae Vascular Structure Movable Points
    vasc_structure.update_moveable_pts(mvble_pts)

    # Solve for flow
    flowDict = computeFlow(vasc_structure)

    # Add flows to image
    vasc_structure.add_flows_to_img(flowDict)

    # run the diffusion
    # diffused_img = lab_meat_diffuse(vas_structure.img, 100, 1, 50)
    diffused_img = diffusion(mvble_pts, vasc_structure.img)

    return diffused_img

def diffusion(mvble_pts, img):
    # D is the defusion constant
    D = 0.1

    #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
    for i in range(0, 200): # how many times you run a diffusion update
        convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        deltaDiffusion = sig.convolve(img, convolve)[1:-1, 1:-1] #take off first and last
        if i > 0:
            deltaDiffusion += img

        # the update to the img from one step of diffusion
        img = img + deltaDiffusion + nonlinearDiffusion(mvble_pts, img)
        img_pic = np.pad(img, ((2, 3), (2, 3)), 'constant')
        plt.imsave('diffusePngs/TestDiffuse_'+str(i)+'.png', np.rot90(img_pic), cmap='jet')

    path_to_img_dir = 'diffusePngs/'
    images = []
    for file_name in natsorted(os.listdir(path_to_img_dir), key=lambda y: y.lower()):
        if file_name.endswith('.png'):
            file_path = os.path.join(path_to_img_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('VascDiffuse.gif', images, fps=50)

    return img


# None linear diffusion (compute each convoution for each location)
def nonlinearDiffusion(mvble_pts, img):
    #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
    #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
    w, h = img.shape
    deltaDomain = np.zeros((w, h))
    for i in range(1, len(mvble_pts), 2):
        x = mvble_pts[i-1]
        y = mvble_pts[i]
        int_x = int(x)
        int_y = int(y)
        
        np_pt = np.array([x, y])
        # int_np_pt = np.array([int_x, int_y])
        dist_0 = np.linalg.norm(np_pt - np.array([int_x-1, int_y-1]))
        dist_1 = np.linalg.norm(np_pt - np.array([int_x, int_y-1]))
        dist_2 = np.linalg.norm(np_pt - np.array([int_x+1, int_y-1]))

        dist_3 = np.linalg.norm(np_pt - np.array([int_x-1, int_y]))
        dist_4 = np.linalg.norm(np_pt - np.array([int_x, int_y]))
        dist_5 = np.linalg.norm(np_pt - np.array([int_x+1, int_y]))

        dist_6 = np.linalg.norm(np_pt - np.array([int_x-1, int_y+1]))
        dist_7 = np.linalg.norm(np_pt - np.array([int_x, int_y+1]))
        dist_8 = np.linalg.norm(np_pt - np.array([int_x+1, int_y+1]))

        X = -sigmoid(dist_0 - 1) - sigmoid(dist_1 - 1) - sigmoid(dist_2 - 1) - sigmoid(dist_3 - 1) - sigmoid(dist_5 - 1) - sigmoid(dist_6 - 1) - sigmoid(dist_7 - 1) - sigmoid(dist_8 - 1)

        convolution = np.array([[sigmoid(dist_0 - 1), sigmoid(dist_1 - 1), sigmoid(dist_2 - 1)], [sigmoid(dist_3 - 1), X, sigmoid(dist_5 - 1)], [sigmoid(dist_6 - 1), sigmoid(dist_7 - 1), sigmoid(dist_8 - 1)]])
        deltaDomain = get_submatrix_add(deltaDomain, (int_x, int_y), convolution)

    return deltaDomain

if __name__ == "__main__":
    start = timer()
    total_iterations = 800
    all_params = []
    all_loss = []

    path_to_diffuse_pngs = 'diffusePngs/'
    if not os.path.exists(path_to_diffuse_pngs):
        os.makedirs(path_to_diffuse_pngs)

    for img_file in os.listdir(path_to_diffuse_pngs):
        os.remove(path_to_diffuse_pngs + img_file)

    (max_T, count) = getSampleParameters()
    t = np.linspace(0., max_T, count)

    # testdiffuse = np.zeros((11, 11))
    # testdiffuse[5][5] = 255
    # plt.imsave('TEST_diff_start.png', np.rot90(testdiffuse), cmap='Greys')
    # diff_img = diffusion((5,5), testdiffuse)
    # plt.imsave('TEST_diff_DIFFUSED.png', np.rot90(diff_img), cmap='Greys')

    # print('\n Diffused: \n', diff_img)


    # TODO: Make this (num_of_nodes) a command line argument
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=2)
    vas_structure.print_images()
    mvable_pts = tuple(vas_structure.flatten_mvable_pts())

    vas_structure.img = simulate(mvable_pts, t, vas_structure)
    vas_structure.print_images(img_name='TEST_simulate.png')
    print('DONE')
    print(vas_structure.img)
    print("LOSS:", loss_health(vas_structure.img))
    os.sys.exit()

    def fitness(params, iter):
        return loss_health(simulate(params, t, vas_structure))


    # Setup display figures

    # Plot Data through callback
    def callback(params, iter, g):
        pass



    optimized_mvble_pts = adamMy(grad(fitness), fitness, mvable_pts, step_size = 0.005,
                            num_iters=total_iterations, callback=callback)

    print('Optimized Pts')
    print(optimized_mvble_pts)
    end = timer()
    print('Time per iteration: ', str((end-start) / total_iterations))
