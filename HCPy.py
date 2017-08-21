import numpy as np

def PA(vector):
    vx, vy = vector[0], vector[1]
    PA_E = np.degrees(np.arctan2(vx, vy))

    if (PA_E >= 90.) and (PA_E <= 180.):
        PA = PA_E - 90.
    elif (PA_E >= -90.) and (PA_E < 90.):
        PA = PA_E + 90.
    elif (PA_E >= -180.) and (PA_E < -90.):
        PA = PA_E + 270.
    else:
        PA = np.nan

    return PA

def PointsDist(point1, point2):

    return np.hypot(point1[0]-point2[0], point1[1]-point2[1])
