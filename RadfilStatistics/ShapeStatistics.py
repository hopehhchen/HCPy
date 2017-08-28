import time
import numbers
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

##
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
#
from HCPy import *

class ShapeStatistics(object):
    '''
    Statistics object that takes in a radfil object and calculates properties
    related to the shape of the spline.

    Parameters
    ----------
    radfil: a RadFil object with `xbeforespline` and `ybeforespline`.  That is,
            after `build_profile` is run.

    samp_int: integer (default = 1)
              Similar to `samp_int` in RadFil, it is used to determine the knots
              that are used to calculate the statistics.  Since no cutting is
              performed, `samp_int` can be set to a smaller value without
              delaying the performance much.

    Attributes
    ----------
    distance: the distance along the spline. [pix]

    PA: the position angle of the tangent vector. [deg]

    PArate: the rate at which PA changes along the spline. [deg/pix]

    mcurvature: the Menger 3-point curvature.  [1/pix]
    '''

    def __init__(self, radfil, samp_int = 1):
        # Check if the radfil object is properly made.
        if hasattr(radfil, 'xbeforespline') and hasattr(radfil, 'ybeforespline'):
            self.radfil = radfil
            self.xbeforespline = radfil.xbeforespline
            self.ybeforespline = radfil.ybeforespline
        else:
            raise ValueError('Please run build_profile before input.')

        # Check for imgscale, which users can use for unit conversion.
        # Check for units at the same time.
        if hasattr(radfil, 'imgscale'):
            self.imgscale = radfil.imgscale
        else:
            raise ValueError('There is no imgscale in the radfil obj.')

        # Check for `samp_int`
        if isinstance(samp_int, numbers.Number):
            self.samp_int = samp_int
        else:
            raise ValueError('samp_int needs to be a number.')

        # Full spline
        tckp, up, = splprep([self.xbeforespline, self.ybeforespline],
                            k = 3,
                            nest = -1)
        self.points = np.array(splev(up, tckp)).T
        self.fprime = np.array(splev(up, tckp, der = 1)).T

        ## Quantities below are for points[1:-1].

        # Distance
        ## distance to the next point
        distance1 = np.diagonal(distance.cdist(self.points, self.points),
                                offset = 1)
        self.distance = np.array([np.sum(distance1[:i])\
                                  for i in range(1, len(distance1))])
        self.distance = self.distance[::samp_int]

        # PA
        PA = PositionAngle(self.fprime)
        ## Making sure that the output is continuous
        PAcont = [PA[0]]
        sign = False
        for i in range(1, len(PA)):
            ## record distcontinuous changes by sign
            if abs(PA[i]-PA[i-1]) >= 175.:
                sign = ~sign

                if PA[i-1] >= 90.:
                    offset = 180.
                else:
                    offset = -180.

            ## store based on sign and the direction
            if sign:
                PAcont.append(PA[i]+offset)
            else:
                PAcont.append(PA[i])
        self.PA = np.array(PAcont)[1:-1:samp_int]
        ## PA changing rate
        self.PArate = np.diff(PAcont)/distance1
        self.PArate = self.PArate[:-1:samp_int]

        # Menger Curvature
        mcurvature = MengerCurvature(self.points)
        self.mcurvature = mcurvature[::samp_int]
