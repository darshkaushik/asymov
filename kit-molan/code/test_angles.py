#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 achandrasekaran <arjun.chandrasekaran@tuebingen.mpg.de>
#
# Distributed under terms of the MIT license.

import os, sys, pdb
import numpy as np
import torch

import unittest

from scipy.spatial.transform import Rotation as R

sys.path.append('/ps/project/conditional_action_gen/language2motion/packages/Complextext2animation/src/')
from common import quaternion as Q
from common import transforms3dbatch as T

"""
Script to test angles conversions
"""

class TestEuler2Quat():

    def in_rads(self, e=np.array([[0., 0., 0.]])):
        print('In Radians')
        print('-'*20)
        e1, e2 = np.copy(e), np.copy(e)
        print('Original angle (xyz) = ', e1)
        q_T = T.euler2quatbatch(e1, axes='sxyz')
        print('Transforms3d angle (sxyz) = ', q_T)

        print('Original angle (xyz) = ', e2)
        a = R.from_euler('xyz', e2, degrees=False)
        q_sc = R.as_quat(a)
        print('Scipy angle (xyzs) = ', q_sc)
        pdb.set_trace()
        # self.assertAlmostEqual(q_T, q_sc)

    def in_degs(self, e=np.array([[0., 0., 0.]])):
        print('In degrees')
        print('-'*20)
        e1, e2 = np.copy(e), np.copy(e)
        print('Original angle = ', e1)
        q_T = T.euler2quatbatch(e1, axes='sxyz')
        print('Transforms3d angle = ', q_T)

        print('Original angle = ', e2)
        a = R.from_euler('xyz', e2, degrees=True)
        q_sc = R.as_quat(a)
        print('Scipy angle = ', q_sc)
        # self.assertAlmostEqual(q_T, q_sc)

# obj = TestEuler2Quat()

# obj.in_rads()
# obj.in_degs()
# obj.in_rads(np.array([[1., 2., 3.]]))
# obj.in_degs(np.array([[1., 2., 3.]]))
# obj.in_rads(np.array([[3.14, 3.14, 3.14/2.]]))

def q2e(q, degrees=False):
    '''Assume input quaternion is in sxyz format'''
    xyzs = q[..., [1, 2, 3, 0]]
    r = R.from_quat(xyzs)
    e = r.as_euler('xyz', degrees)
    return e
    