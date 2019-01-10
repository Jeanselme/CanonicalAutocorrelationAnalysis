""" 
    Hough transformation in the projection space
"""
import matplotlib.pyplot as plt
import numpy as np

def hough(projection, points, discretizationRadius = 1000, discretizationAngle = 180):
    """
        Computes hough transforms of the points
    
        Arguments:
            projection {Projection} -- Object obtained from CAA
            points -- Datapoints to project and analyze 

            discretizationRadius {Int} 
            discretizationAngle {Int}
        
    """
    # Coordinates in the projection space
    projected = projection.projectPoints(points)

    # Polar coordinate of all points
    r = np.sqrt(projected[:,0]**2 + projected[:,1]**2)
    theta = np.arctan2(projected[:,1], projected[:,0])

    # Hough space
    houghSpace = np.zeros((discretizationRadius, discretizationAngle))
    radiusBins = np.linspace(- np.max(r), np.max(r), discretizationRadius + 1)
    angleBins = np.linspace(0, np.pi, discretizationAngle)
    for i, phi in enumerate(angleBins):
        # For the given theta compute the radius of the line cutting the points
        radPhi = np.cos(phi - theta) * r
        hist, _ = np.histogram(radPhi, bins=radiusBins)
        houghSpace[:, i] += hist

    return np.flip(houghSpace, 0), angleBins * 180 / np.pi, radiusBins

def displayHough(projection, points):
    """
        Computes hough transforms of the points and displays it
        Call this multiple times to display different serieses in different colors
    
        Arguments:
            projection {Projection} -- Object obtained from CAA
            points -- Datapoints to project and anlyze
            plt -- Plot to display points. 
            color -- Color name for plotting 'points' 
    """
    houghSpace, xaxis, yaxis = hough(projection, points)

    plt.figure()
    plt.imshow(houghSpace, extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], aspect='auto', cmap='gray')
    plt.ylabel("Radius")
    plt.xlabel("Angle(in deg)")
    plt.show()
