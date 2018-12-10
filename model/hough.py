""" 
    Hough transformation in the projection space
"""
import matplotlib.pyplot as plt
import numpy as np

def hough(projection, points, discretization = 100):
    """
        Computes hough transforms of the points
    
        Arguments:
            projection {Projection} -- Object obtained from CAA
            points -- Datapoints to project and anlyze 
    """
    # Coordinates in the projection space
    projected = projection.projectPoints(points)

    # Coordinate in the Hough space
    r = np.sqrt(projected[:,0]**2 + projected[:,1]**2)
    theta = np.arctan2(projected[:,1], projected[:,0])

    return np.column_stack([r, theta])

def displayHough(projection, points):
    """
        Computes hough transforms of the points and displays it
    
        Arguments:
            projection {Projection} -- Object obtained from CAA
            points -- Datapoints to project and anlyze 
    """
    hough_coordinate = hough(projection, points)

    plt.figure("Hough Projections Points")
    plt.scatter(hough_coordinate[:,1], hough_coordinate[:,0])
    plt.ylabel("Radius")
    plt.xlabel("Angle")
    plt.show()