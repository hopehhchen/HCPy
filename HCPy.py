import numpy as np

def PositionAngleBase(vector):
    # Calculate the PA of a vector, East of North
    vx, vy = vector[0], vector[1]
    PA_E = np.degrees(np.arctan2(vy, vx))

    if (PA_E >= 90.) and (PA_E <= 180.):
        PA = PA_E - 90.
    elif (PA_E >= -90.) and (PA_E < 90.):
        PA = PA_E + 90.
    elif (PA_E >= -180.) and (PA_E < -90.):
        PA = PA_E + 270.
    else:
        PA = np.nan

    return PA

def PositionAngle(vectors):

    if np.size(vectors) == 2:
        return PositionAngleBase(vectors)
    elif np.size(vectors) > 2 and np.shape(vectors)[1] == 2:
        return np.apply_along_axis(PositionAngleBase, 1, vectors)
    else:
        raise ValueError('The input has to be an (N, 2) array-like object.')



def PointsDist(point1, point2):
    # Calculate the Euclidean distance between two points

    return np.hypot(point1[0]-point2[0], point1[1]-point2[1])
