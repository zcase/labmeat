"""Some standard gradient-based stochastic optimizers.
These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.
These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""
# https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py
from __future__ import absolute_import
from builtins import range
import os
from VasGen2 import VasGen2
from equations import *

import autograd.numpy as np
from autograd.misc import flatten
from autograd.wrap_util import wraps

def unflatten_optimizer(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(grad, x0, vas_structure, callback=None, *args, **kwargs):
        print('Optimzers: 21 = ', x0)
        print('VasStruct: ', vas_structure)
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g))
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, vas_structure, _callback, *args, **kwargs))

    return _optimize

def unflatten_optimizer2(optimize):
    """Takes an optimizer that operates on flat 1D numpy arrays and returns a
    wrapped version that handles trees of nested containers (lists/tuples/dicts)
    with arrays/scalars at the leaves."""
    @wraps(optimize)
    def _optimize(grad, x0, imgsize, callback=None, *args, **kwargs):
        print('Optimzers: 21 = ', x0)
        _x0, unflatten = flatten(x0)
        _grad = lambda x, i: flatten(grad(unflatten(x), i))[0]
        if callback:
            _callback = lambda x, i, g: callback(unflatten(x), i, unflatten(g))
        else:
            _callback = None
        return unflatten(optimize(_grad, _x0, imgsize, _callback, *args, **kwargs))

    return _optimize

@unflatten_optimizer
def sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size * velocity
    return x

@unflatten_optimizer
def rmsprop(grad, x, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return x

@unflatten_optimizer2
def adamTest(grad, x, imgsize, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    # print('X = ', x, type(x))

    for i in range(0, num_iters):
        # print('HELLO')
        g = grad(x,i)
        # print('    g    =', g)
        # print('    i    =', i)

        if callback: 
            callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        # print('       m = ', m)
        # print('       v = ', v)
        # print('       x = ', x)
        # print('    mhat = ', mhat)
        # print('    vhat = ', vhat)
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
        print('THIS IS X: ', x)

        new_pts = []
        for i in range(2, len(list(x))+2, 2):
            cur  = list(x)[i-2:i]
            xi = float(cur[0])
            yi = float(cur[1])
            if xi < 0:
                xi = float(0.0)
            elif xi >= imgsize:
                xi = float(imgsize -1)
            
            if yi < 0.0:
                yi = float(0.0)
            elif yi >= imgsize:
                yi = float(imgsize)

            cur = [xi, yi]
            # print('First Cur: ', cur)

            # if i > 2:
            #     prev = [float(i) for i in prev]
            #     cur = [float(i) for i in cur]
            #     print('prev: ', prev)
            #     print('cur : ', cur)
            #     pts_lst = pts_lst + [prev, cur]
            # prev = cur
            new_pts = new_pts + [cur]


        # new_pts = []
        # for val in x:
        #     print(x)
        #     x = float(val[0])
        #     y = float(val[1])
        #     if x < 0:
        #         x = 0.0
        #     elif x >= imgsize:
        #         x = float(imgsize -1)

        #     if y < 0:
        #         y = 0.0
        #     elif y >= imgsize:
        #         y = float(imgsize -1)
            
        #     new_pts += [[x, y]]
        print(new_pts)
        x = np.array(new_pts)

        # os.sys.exit()
        print('Updated: point: ', x)
        # vesselImage
        # vesselSecretion
        # img, vesselImage, vesselPts = updateVesselImg(vesselImage, vesselSecretion, x, img)



        # os.sys.exit()
        # print('    x= ', x)
        # vas_structure.update_moveable_pts(x)
        # x = np.array(vas_structure.flatten_mvable_pts())
        # print('After update: ', x)

        # Solve for flow
        # flowDict = computeFlow(vas_structure)

        # Add flows to image
        # vas_structure.add_flows_to_img(flowDict)
        # img = vas_structure.img
    return x

def updateVesselImg(vesselImage, vesselSecretion, mvable_pts, img):
    vesselPts = [(int(x), int(y)) for (x, y) in mvable_pts]
    for (xi, yi) in vesselPts:
        vesselImage[xi, yi] += vesselSecretion
    img = vesselImage
    return img, vesselImage, vesselPts


@unflatten_optimizer
def adamVas(grad, x, vas_structure, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    # print('X = ', x, type(x))

    for i in range(0, num_iters):
        # print('HELLO')
        g = grad(x,i)
        # print('    g    =', g)
        # print('    i    =', i)

        if callback: 
            callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        # print('       m = ', m)
        # print('       v = ', v)
        # print('       x = ', x)
        # print('    mhat = ', mhat)
        # print('    vhat = ', vhat)
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
        # print('THIS IS X: ', x)


        # os.sys.exit()
        # print('    x= ', x)
        vas_structure.update_moveable_pts(x)
        x = np.array(vas_structure.flatten_mvable_pts())
        print('After update: ', x)

        # Solve for flow
        flowDict = computeFlow(vas_structure)

        # Add flows to image
        vas_structure.add_flows_to_img(flowDict)
        img = vas_structure.img
        # print(vas_structure.diffused_img)
        # path_to_diffuse_pngs = 'diffusePngs/'
        # sim_img_folder = 'simulation_imgs/imgs/'
        # sim_graph_folder = 'simulation_imgs/graphs/'
        # vas_structure.print_images(graph_name=sim_graph_folder+'test_graph'+str(i)+'.png', img_name=sim_img_folder+'test_img'+str(i)+'.png')
    return x