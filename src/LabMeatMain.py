from VascularGenerator import VascularGenerator
from equations import *
from diffu2D_u0 import lab_meat_diffuse

# import numpy as np

import autograd.numpy as np
from autograd import grad
from autograd.scipy.integrate import odeint
from autograd.builtins import tuple
from autograd.misc.optimizers import adam
import autograd.numpy.random as npr


if __name__ == "__main__":

    # TODO: Make this (num_of_nodes) a command line argument
    vas_structure = VascularGenerator(max_range=100, num_of_nodes=3)
    vas_structure.print_images()
    flattened_pts = vas_structure.flatten_mvable_pts()

    print('Num Pts  : ', len(vas_structure.pts))
    print('Num edges: ', len(vas_structure.edges))

    # Update points
    for i in range(0, len(flattened_pts), 2):
        if flattened_pts[i] + 10 < vas_structure.max_range:
            flattened_pts[i] = flattened_pts[i] + 10
        else:
            flattened_pts[i] += (vas_structure.max_range - 1) - flattened_pts[i]

    print('Flattened_pts: ', flattened_pts)

    print('Num edges: ', len(vas_structure.edges))
    vas_structure.update_moveable_pts(flattened_pts)
    print('Num edges: ', len(vas_structure.edges))
    vas_structure.print_images(graph_name='Vasc_Graph_' + str(vas_structure.update_count) + '.png', img_name='Vasc2D_img_' +str(vas_structure.update_count)+ '.png')

    print('Num Pts  : ', len(vas_structure.pts))
    print('Num edges: ', len(vas_structure.edges))

    

    
    print('\n Depth First Search')
    vas_structure.depth_first_search()

    flowDict = computeFlow(vas_structure)

    print('\n\n\n')
    for key in sorted(flowDict, key=lambda element: (element[0], element[1]),  reverse=True):
        print('Key:', key, '   :  ', flowDict[key])

    vas_structure.add_flows_to_img(flowDict)
    # vas_structure.print_images()

    lab_meat_diffuse(vas_structure.img, 100, 0.5, 10)


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
