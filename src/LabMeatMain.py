from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse
import os
import math

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

def get_submatrix(np_matrix, center_pt_tuple, submatrix_size=2):
    h, w = np_matrix.shape
    row, col = center_pt_tuple
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

    return np_matrix[row_start:row_end, col_start:col_end]


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
    return

def simulate(mvble_pts, t, vasc_structure):
    # Updtae Vascular Structure Movable Points
    vasc_structure.update_moveable_pts(mvble_pts)

    # Solve for flow
    flowDict = computeFlow(vasc_structure)

    # Add flows to image
    vasc_structure.add_flows_to_img(flowDict)

    # run the diffusion
    diffused_img = diffusion(mvble_pts, vasc_structure.img)

    return diffused_img

def diffusion(mvble_pts, img):
    # D is the defusion constant
    D = 0.1
    #https://programtalk.com/python-examples/autograd.scipy.signal.convolve/
    for _ in range(0, 100): # how many times you run a diffusion update
        convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
        deltaDiffusion = sig.convolve(vas_structure.img, convolve)[1:-1, 1:-1] #take off first and last

        # the update to the img from one step of diffusion
        img = img + deltaDiffusion + nonlinearDiffusion(mvble_pts, img)
    return img


# None linear diffusion (compute each convoution for each location)
def nonlinearDiffusion(mvble_pts, img):
    #http://greg-ashton.physics.monash.edu/applying-python-functions-in-moving-windows.html
    #https://stackoverflow.com/questions/12816293/vectorize-this-convolution-type-loop-more-efficiently-in-numpy
    w, h = img.shape
    deltaDomain = np.ones((w, h))
    for i in range(1, len(mvable_pts), 2):
        x = mvable_pts[i-1]
        y = mvable_pts[i]
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

        deltaDomain += sig.convolve(deltaDomain, convolution)[1:-1, 1:-1] #take off first and last
    return deltaDomain

if __name__ == "__main__":
    start = timer()
    total_iterations = 800
    all_params = []
    all_loss = []

    (max_T, count) = getSampleParameters()
    t = np.linspace(0., max_T, count)

        # TODO: Make this (num_of_nodes) a command line argument
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=2)
    vas_structure.print_images()
    mvable_pts = tuple(vas_structure.flatten_mvable_pts())

    # vas_structure.img = simulate(mvable_pts, t, vas_structure)
    # vas_structure.print_images(img_name='TEST_simulate.png')
    # print('DONE')

    flowDict = computeFlow(vas_structure)
    vas_structure.add_flows_to_img(flowDict)
    vas_structure.print_images(img_name='TEST_0.png')

    D = 0.1
    convolve = np.array([[1*D, 1*D, 1*D],[1*D,-8*D,1*D], [1*D, 1*D, 1*D]])
    deltaDiffusion = sig.convolve(vas_structure.img, convolve)[1:-1, 1:-1] #take off first and last
    print(convolve)
    print(deltaDiffusion)
    print(deltaDiffusion.shape)

    print('\n')
    print(mvable_pts)
    print('\n')

    w, h = vas_structure.img.shape
    deltaDomain_NonLinear = np.zeros((w, h))
    # d_0   d_1   d_2
    # d_3   d_4   d_5
    # d_6   d_7   d_8
    for i in range(1, len(mvable_pts), 2):
        x = mvable_pts[i-1]
        y = mvable_pts[i]
        int_x = int(x)
        int_y = int(y)
        
        np_pt = np.array([x, y])
        int_np_pt = np.array([int_x, int_y])
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

        deltaDomain_NonLinear += sig.convolve(deltaDomain_NonLinear, convolution)[1:-1, 1:-1] #take off first and last

        print("convoluation:\n", convolution)
        print('# =============== #')
        print('DeltaDomain :\n', deltaDomain_NonLinear)
        print('# =============== #')


        dist_4 = np.linalg.norm(np_pt-int_np_pt)
        print(np_pt, int_np_pt, dist_4)
    


    vas_structure.img += deltaDiffusion + deltaDomain_NonLinear
    vas_structure.print_images(img_name='TEST.png')
    print(vas_structure.img)
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


    # # Update points
    # for i in range(0, len(flattened_pts), 2):
    #     if flattened_pts[i] + 10 < vas_structure.max_range:
    #         flattened_pts[i] = flattened_pts[i] + 10
    #     else:
    #         flattened_pts[i] += (vas_structure.max_range - 1) - flattened_pts[i]

    # vas_structure.update_moveable_pts(flattened_pts)
    # vas_structure.print_images(graph_name='Vasc_Graph_' + str(vas_structure.update_count) + '.png', img_name='Vasc2D_img_' +str(vas_structure.update_count)+ '.png')

    # print('\n Depth First Search')
    # vas_structure.depth_first_search()

    # flowDict = computeFlow(vas_structure)

    # print('\n\n\n')
    # for key in sorted(flowDict, key=lambda element: (element[0], element[1]),  reverse=True):
    #     print('Key:', key, '   :  ', flowDict[key])

    # vas_structure.add_flows_to_img(flowDict)
    # vas_structure.print_images()

    # lab_meat_diffuse(vas_structure.img, 100, 0.5, 10)


    # # # Generate data from true dynamics.
    # # true_y0 = np.array([2., 0.]).T
    # # t = np.linspace(0., max_T, N)
    # # true_A = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    # # true_y = odeint(func, true_y0, t, args=(true_A,))

    # def train_loss(params, iter):
    #     return 1
    #     # pred = ode_pred(params, true_y0, t)
    #     # return L1_loss(pred, true_y)

    # # Set up figure
    # fig = plt.figure(figsize=(12, 4), facecolor='white')
    # ax_traj     = fig.add_subplot(131, frameon=False)
    # ax_phase    = fig.add_subplot(132, frameon=False)
    # ax_vecfield = fig.add_subplot(133, frameon=False)
    # plt.show(block=False)

    # # Plots data and learned dynamics.
    # def callback(params, iter, g):
    #     pass

    #     # # pred = ode_pred(params, true_y0, t)

    #     # print("Iteration {:d} train loss {:.6f}".format(
    #     #       iter, L1_loss(pred, true_y)))

    #     # ax_traj.cla()
    #     # ax_traj.set_title('Trajectories')
    #     # ax_traj.set_xlabel('t')
    #     # ax_traj.set_ylabel('x,y')
    #     # ax_traj.plot(t, true_y[:, 0], '-', t, true_y[:, 1], 'g-')
    #     # ax_traj.plot(t, pred[:, 0], '--', t, pred[:, 1], 'b--')
    #     # ax_traj.set_xlim(t.min(), t.max())
    #     # ax_traj.set_ylim(-2, 2)
    #     # ax_traj.xaxis.set_ticklabels([])
    #     # ax_traj.yaxis.set_ticklabels([])
    #     # ax_traj.legend()

    #     # ax_phase.cla()
    #     # ax_phase.set_title('Phase Portrait')
    #     # ax_phase.set_xlabel('x')
    #     # ax_phase.set_ylabel('y')
    #     # ax_phase.plot(true_y[:, 0], true_y[:, 1], 'g-')
    #     # ax_phase.plot(pred[:, 0], pred[:, 1], 'b--')
    #     # ax_phase.set_xlim(-2, 2)
    #     # ax_phase.set_ylim(-2, 2)
    #     # ax_phase.xaxis.set_ticklabels([])
    #     # ax_phase.yaxis.set_ticklabels([])

    #     # ax_vecfield.cla()
    #     # ax_vecfield.set_title('Learned Vector Field')
    #     # ax_vecfield.set_xlabel('x')
    #     # ax_vecfield.set_ylabel('y')
    #     # ax_vecfield.xaxis.set_ticklabels([])
    #     # ax_vecfield.yaxis.set_ticklabels([])

    #     # # vector field plot
    #     # y, x = npo.mgrid[-2:2:21j, -2:2:21j]
    #     # dydt = nn_predict(np.stack([x, y], -1).reshape(21 * 21, 2), 0,
    #     #     params).reshape(-1, 2)
    #     # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    #     # dydt = (dydt / mag)
    #     # dydt = dydt.reshape(21, 21, 2)

    #     # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    #     # ax_vecfield.set_xlim(-2, 2)
    #     # ax_vecfield.set_ylim(-2, 2)

    #     # fig.tight_layout()
    #     # plt.draw()
    #     # plt.pause(0.001)


    # # Train neural net dynamics to match data.
    # # init_params = init_nn_params(0.1, layer_sizes=[D, 150, D])
    # # optimized_params = adam(grad(train_loss), init_params,
    # #                         num_iters=1000, callback=callback)
