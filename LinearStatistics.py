import time
import numbers
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

##
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
import scipy.stats as stats
#
from HCPy import *

class LinearStatistics(object):
    '''
    The statistics object that calculates statistical properties along the spline.

    Parameters
    ------
    radfil: a RadFil object with `xbeforespline` and `ybeforespline`.  That is,
            after `build_profile` is run.

    samp_int: integer (default = 3)
              Similar to `samp_int` in RadFil, it is used to determine the knots
              that are used to calculate the statistics.  Since no cutting is
              performed, the only limit of `samp_int` is the resolution of the
              original observations.

    Attributes:
    ------
    distance: distance along the spline [pix]

    masks: list of masks that define the regions used to calculate the statistics.
    [a list of numpy.ndarray with boolean values]

    values: list of (sets of) values extracted from the image using the masks.
    [a list of 1D numpy.ndarray with float numbers]
    '''

    def __init__(self, radfil, samp_int = 3):
        # Check for the image stored in the radfil object.
        if hasattr(radfil, 'image'):
            self.image = radfil.image
        else:
            raise ValueError('There is no image input in the radfil obejct.')

        # Check if the radfil object is properly made.
        if hasattr(radfil, 'xbeforespline') and hasattr(radfil, 'ybeforespline'):
            self.radfil = radfil
            self.xbeforespline = radfil.xbeforespline
            self.ybeforespline = radfil.ybeforespline
        else:
            raise ValueError('Please run build_profile before input.')

        # Check for cutdist and use it as the edge out to which the stats are
        # derived.  Check for units at the same time.
        if hasattr(radfil, 'cutdist') and hasattr(radfil, 'imgscale'):
            self.imgscale = radfil.imgscale
            self.cutdist = radfil.cutdist
        else:
            raise ValueError('There is no cutdist nor imgscale in the radfil obj.')

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
        ## we may not need the tangent line.
        ##self.fprime = np.array(splev(up, tckp, der = 1)).T

        # Distance
        ## distance to the next point
        distance1 = np.diagonal(distance.cdist(self.points, self.points),
                                offset = 1)
        self.distance = np.array([np.sum(distance1[:i])\
                                  for i in range(1, len(distance1))])
        self.distance = self.distance[::samp_int]


        # Masks
        ### This is set up to be similar to the mask when cut=False in Radfil,
        ### except that the distance to the spline (instead of the spine) is
        ### used to keep the results consistent with the distance measurements
        ### above.
        xmesh, ymesh = np.meshgrid(np.arange(self.image.shape[1]),
                                   np.arange(self.image.shape[0]))
        dcube = np.array([np.hypot(xmesh-self.points[i, 0],
                                   ymesh-self.points[i, 1])
                          for i in range(len(self.points))])
        ## the distance grid
        dgrid = np.min(dcube,
                       axis = 0)
        dgrid = dgrid*self.imgscale.value ## put it in the proper units
        ### By the design of RadFil, the units of imgscale and cutdist should
        ### always be the same.
        ## the index grid
        igrid = np.argmin(dcube,
                          axis = 0).astype(float)
        mask_all = (igrid != 0.)&\
                   (igrid != igrid.max())&\
                   (dgrid < self.cutdist.value)
        igrid[~mask_all] = np.nan
        ## list of masks
        mask_list = [((igrid//samp_int) == ii)
                     for ii in list(set((igrid//samp_int)[np.isfinite(igrid)]))]
        self.masks = mask_list


        # Sets of values based on the orignal image
        self.values = [self.image[mask]
                       for mask in self.masks]

    def statistics(self, statimage = None, statistic = ('moment', 0)):
        '''
        Parameters
        ------
        statimage: numpy.ndarray with a shape the same as the original image.
            The image used to claculate the statistics from.  The default is to
            use the original image.

        statistic: a length-2 object, with the 1st being a string, the 2nd a number.
            The statistics to calculate.  The 1st item should be a string of either
            'moment' or 'percentile'.
                - When statistic[0] = 'moment', statistic[1] would be used to
                determine which moment to calculate.  The calculation is done
                with scipy.stats.describe, which applies a DOF correction.  I.e.
                n = nobs - 1.
                - When statistic[0] = 'percentile', statistic[1] would be used
                to determine the percentile to calculate.  Note that ('percentile',
                50.) would give you the median.

        Attributes:
        ------
        statimage: the image that the statistics is calculated from.
        [a 2D numpy.ndarray with the same shape as the original image]

        statvalues: list of (sets of) values extracted from statimage using the masks.
        [a list of 1D numpy.ndarray with float numbers]

        stats: a 1D numpy.ndarray of the chosen statistics. [with float numbers]
        '''
        # Check for the input image
        if isinstance(statimage, np.ndarray) and (statimage.shape == self.image.shape):
            self.statimage = statimage
            self.statvalues = [self.statimage[mask]
                               for mask in self.masks]
        elif statimage is None:
            self.statimage = self.image
            self.statvalues = self.values
        else:
            raise ValueError('The input has to be a numpy.ndarray object with the same shape as the orignal image.')

        # Check for the statistics to be calculated.
        if (len(statistic) == 2) and isinstance(statistic[0], str) and\
        (statistic[0].lower() in ['moment', 'percentile']) and\
        isinstance(statistic[1], numbers.Number):
            # Calculate the moments; except the mean, others are central.
            ### The nan is always omitted.
            ### Note that scipy.stats applies the correction in DOF. i.e. n = nobs - 1.
            ### Note that only finite numbers are taken into consideration.
            if (statistic[0].lower() == 'moment'):
                ## number of objects
                if statistic[1] == 0.:
                    self.stats = np.array([stats.describe(values[np.isfinite(values)]).nobs
                                           for values in self.statvalues])
                ## mean
                elif statistic[1] == 1.:
                    self.stats = np.array([stats.describe(values[np.isfinite(values)]).mean
                                           for values in self.statvalues])
                ## variance
                elif statistic[1] == 2.:
                    self.stats = np.array([stats.describe(values[np.isfinite(values)]).variance
                                           for values in self.statvalues])
                ## skewness
                elif statistic[1] == 3.:
                    self.stats = np.array([stats.describe(values[np.isfinite(values)]).skewness
                                           for values in self.statvalues])
                ## kurtosis
                elif statistic[1] == 4.:
                    self.stats = np.array([stats.describe(values[np.isfinite(values)]).kurtosis
                                           for values in self.statvalues])
                ## others
                else:
                    raise ValueError('When calculating moments, the number needs to be 0, 1, 2, 3, or 4.')
            # Calculate the percentiles.
            elif (statistic[0].lower() == 'percentile'):
                self.stats = np.array([np.percentile(values[np.isfinite(values)], statistic[1])
                                       for values in self.statvalues])
            # errors
            else:
                raise ValueError('Check the input `statistic` and the documentation.')

        return self
