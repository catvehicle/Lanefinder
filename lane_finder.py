#!/usr/bin/env python
from __future__ import division
from __future__ import absolute_import
from sklearn.cluster import KMeans
import sys
from sklearn import metrics
from scipy.spatial.distance import cdist
import cv2
import time
import math
import numpy as np
from numpy import linalg as LA
import os
from itertools import imap

try:
    from ros_classes import *
except:
    print "ROS import error; can only run visualization - no publishing paths for you"

rng = np.random.RandomState(42)


class lane_finder():
    '''
    A class to find lane points given an image that has been inverse perspective mapped and scrubbed of most features
    other than the lanes.
    '''

    def __init__(self, image, scale_long=1, scale_lat=1, longOffset=1.3, latOffset=.2, step_size=25, seek_angle=.17,
                 seek_max=10, base_size=.2):
        #### Hyperparameters ####

        # the distance a window slides before recentering on lane; too small can lead to overfitting
        # too large can lead to too much generalization and lower curve capturing
        self.step_size = step_size
        # the percentage of the image to consider for placing initial points (starts at bottom and expands upward)
        self.base_size = base_size
        # meters per pixel along the horizontal image axis
        self.scale_long = scale_long
        # meters per pixel along the vertical image axis
        self.scale_lat = scale_lat
        # offset in meters that the camera is from the center of object (car) - horizontal image axis
        self.longOffset = longOffset
        #  offset in meters that the camera is from the center of object (car) - vertical image axis
        self.latOffset = latOffset
        # the size in pixels of the square windows used to search
        self.windowSize = self.step_size
        # the angle to swivel seek forward behavior along
        self.seek_angle = seek_angle
        # the number of times to seek forward on a seek forward attempt
        self.seek_max = seek_max
        ###########################

        self.image = self.binarizeImage(image)
        # used for visualization
        self.vis = image
        self.height, self.width = self.image.shape
        self.initialPoints = self.initialPoints(1)
        self.lanes = []
        # create an anonymous function to map points to a new space based on the heading of the car
        self.rotate = lambda points, theta: [int(points[0] * math.cos(theta) + points[1] * math.sin(theta)),
                                             int(points[1] * math.cos(theta) - points[0] * math.sin(theta))]
        for point in self.initialPoints:
            self.lanes.append(self.findLanePoints(point))
        self.midPoints = self.extractMidPoints(250)

    def initialPoints(self, clusters):
        '''
        Finds the expected starting points of lanes in the image recursively. When called with one cluster, it assumes
        there is one lane. If the calculated start of this lane does not land on a white pixel in the image itself, the
        function calls itself, increasing the cluster count (and thus, expected lane count)
        :param clusters: integer number of lanes to find starting points for
        :return: the centers of the lanes in pixel coordinates as tuples
        '''

        # Crop the search space
        bottom = (self.height - int(self.base_size * self.height))
        base = self.image[bottom:self.height, 0:self.width]

        # Find white pixels
        whitePixels = np.argwhere(base == 255)

        # Attempt to run kmeans (the kmeans parameters were not chosen with any sort of hard/soft optimization)
        try:
            kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=3, max_iter=150).fit(whitePixels)
        except:
            # If kmeans fails increase the search space unless it is the whole image, then it fails
            if self.base_size > 1:
                return None
            else:
                self.base_size = self.base_size * 1.5
            return self.initialPoints(clusters)
        # conver centers to integer values so can be used as pixel coords
        centers = [list(imap(int, center)) for center in kmeans.cluster_centers_]
        # Lamda function to remap the y coordiates of the clusters into the image space
        increaseY = lambda points: [points[0] + int((1 - self.base_size) * self.height), points[1]]
        # map the centers in terms of the image space
        modifiedCenters = [increaseY(center) for center in centers]

        # check to see if any centers not in white pixel territory
        for center in modifiedCenters:
            if center not in whitePixels:
                return self.initialPoints(clusters + 1)
        # return a list of tuples for centers
        return modifiedCenters

    def findLanePoints(self, initialPoint):
        '''
        Find the points and confidence values of each point for a lane
        :param initialPoint: the starting point of the lane in image coordinates
        :return: a tuple of points and confidence values for a lane
        '''
        # empty lists to store the information of lane
        points = []
        lines = []  # currently unused for anything
        polygons = []  # currently unused for anything
        confidenceValues = [1]

        # oldCenter is the slide starting location
        oldCenter = initialPoint
        # store the oldCenter as a lane point
        points.append(oldCenter)
        # the initial change vector is a just straight upwards in the image
        changeVector = [-self.step_size, 0]
        # store the initial change vector
        changeVectors = [changeVector]
        # counter to handle over-iteration
        counter = 0

        # Each iteration adds a lane point
        while (True):
            counter += 1
            # emergency escape case
            if counter > 50:
                break
            # slide along the changeVector and mark our location
            tempCenter = np.add(oldCenter, changeVector)
            # create a search window in this space
            top, bottom, left, right = makeBox(tempCenter, self.windowSize)
            # if outside the image space break!
            if top < 0 or bottom > self.height - 1 or left < 0 or right > self.width - 1:
                break
            # index into pic to create the search window
            block = self.image[top:bottom, left: right]
            # find white pixels
            whitePixels = np.argwhere(block == 255)
            avgPixel = np.mean(whitePixels, axis=0)

            try:
                # Reposition ourselves based on the average location of white pixels
                newCenter = [top + int(avgPixel[0]), left + int(avgPixel[1])]
            except:
                # There are no white pixels

                # First perform seek forward behavior
                avgPixel, whitePixels, newCenter = self.seekForward(changeVector, oldCenter)

                # If first seek forward fails, perform two more shifted to the left and right by (no
                # concrete reason to use default .17)
                if avgPixel is None:
                    avgPixel, whitePixels, newCenter = self.seekForward(self.rotate(changeVector, self.seek_angle),
                                                                        oldCenter)
                    if avgPixel is None:
                        avgPixel, whitePixels, newCenter = self.seekForward(self.rotate(changeVector, -self.seek_angle),
                                                                            oldCenter)
                        if avgPixel is None:
                            break

            # add the confidence value correpsonding to point
            # this is represented by the number of white pixels used to get the point
            # less white pixels is lower confidence (direct inverse proportional)
            confidenceValues.append(len(whitePixels))
            # the unweighted new change vector
            rawChangeVector = np.subtract(newCenter, oldCenter)

            # Normalize the change vector's magnitude against the step size for consistent stepping distances
            magnitude = LA.norm(rawChangeVector)
            scaleFactor = self.step_size / magnitude
            changeVector = np.multiply(scaleFactor, rawChangeVector).astype(int)

            # Set the newCenter to the scaled Unweighted vector
            newCenter = np.add(oldCenter, changeVector)
            oldCenter = newCenter
            points.append(newCenter)

            # Weight the change vector based on the previous change vector
            # The two are weighted using their correpsonding confidence values and added
            totalConfidence = confidenceValues[-1] + confidenceValues[-2]
            oldVectorWeighted = np.multiply(changeVectors[-1], confidenceValues[-2] / totalConfidence)
            currentVectorWeighted = np.multiply(changeVector, confidenceValues[-1] / totalConfidence)
            changeVector = np.add(oldVectorWeighted, currentVectorWeighted)
            magnitude = LA.norm(changeVector)
            scaleFactor = self.step_size / magnitude
            changeVector = np.multiply(scaleFactor, changeVector).astype(int)

            # Store this weighted change Vector
            changeVectors.append(changeVector)

        return points, lines, polygons, confidenceValues

    def seekForward(self, changeVector, oldCenter):
        '''
        Continues searching from the last known lane point along the last known change vector until a new lane is found
        or the maximum iteration count is reached
        :param changeVector: the last known change vector
        :param oldCenter: the last known lane point
        :return: the average pixel loc inside the search window; the number of white pixels in the search window, the
                 pixel coordinates of the new lane point
        '''
        counter = 0
        # Each iteration is a fresh search attempt farther out from the last known lane point
        while (1):
            counter += 1
            # Break out if max iteartion exceeded
            if counter > self.seek_max:
                return None, None, None
            topLeft, bottomLeft, bottomRight, topRight, = getRect(changeVector, oldCenter, self.windowSize)
            if topLeft[0] < 0 or bottomLeft[0] > self.height - 1 or bottomLeft[1] < 0 or bottomRight[
                1] > self.width - 1:
                return None, None, None
            block = self.image[topLeft[0]:bottomLeft[0], bottomLeft[1]: bottomRight[1]]
            whitePixels = np.argwhere(block == 255)
            avgPixel = np.mean(whitePixels, axis=0)
            try:
                newCenter = [topLeft[0] + int(avgPixel[0]), topLeft[1] + int(avgPixel[1])]
                return avgPixel, whitePixels, newCenter
            except Exception, e:
                oldCenter = np.add(oldCenter, changeVector)
                cv2.circle(self.vis, tuple(oldCenter[::-1]), 5, (0, 0, 255), -1)

    '''def expandSearch(self, changeVector, oldCenter, polygons):
        topLeft, bottomLeft, bottomRight, topRight, = getRect(changeVector, oldCenter, self.windowSize)
        pts = np.array([topLeft[::-1], bottomLeft[::-1], bottomRight[::-1], topRight[::-1]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        polygons.append(pts)
        # index into pic to create the search window
        block = self.image[topLeft[0]:bottomLeft[0], bottomLeft[1]: bottomRight[1]]
        whitePixels = np.argwhere(block == 255)
        avgPixel = np.mean(whitePixels, axis=0)
        try:
            newCenter = [topLeft[0] + int(avgPixel[0]), topLeft[1] + int(avgPixel[1])]
            return avgPixel, whitePixels, newCenter
        except:
            return None, None, None'''

    def pathGen(self):
        '''
        Converts the midpoints of the lanes in the image to a path object for use with ROS
        :return: path object from ros_classes
        '''
        midPoints = self.getMidpoints()
        # Adjust the midpoints' coordinates in terms of the bottom center of the image (location of camera)
        ys = np.subtract(np.full(len(midPoints), self.height), midPoints[:, 0])
        xs = np.add(midPoints[:, 1], np.full(len(midPoints), - self.width / 2))
        # scale the pixel values to meters and subtract offsets
        ys_scaled = np.multiply(ys, self.scale_long)
        ys_scaled = np.add(ys_scaled, self.longOffset)
        xs_scaled = np.multiply(xs, self.scale_lat)
        xs_scaled = np.subtract(xs_scaled, self.latOffset)
        # if there are enough points in path return the path object with xs, ys, and all 0s for phi
        # phi is no longer calculated by this program
        if len(self.lanes) > 1 and len(ys_scaled) > 3:
            myPath = path(xs_scaled[:-3], ys_scaled[:-3], np.zeros(len(ys_scaled - 3)))
            return myPath
        else:
            return None

    def extractMidPoints(self, adjust):
        '''
        Extracts the midpoints of the lanes in the image using the assumed left and right lanes of the road
        :param adjust: the amount of pixels to shift lane points horizontally if there are no corresponding points in
                       the other lane
        :return: a list of tuples that are the pixel coordinates of the lanes
        '''
        imageMid = int(self.width / 2)
        points = np.array(self.initialPoints).flatten()[1::2]

        #### All of this simply gets the left and right lanes

        # left lane is the closest lane in the left half of the image to the center of the image
        # right lane is the closest lane in the right half of the image to the center of the image
        leftPoints = points[np.argwhere(points < imageMid)]
        rightPoints = points[np.argwhere(points > imageMid)]
        midPoints = []
        try:
            leftLane = self.lanes[(np.argwhere(points == max(leftPoints)[0]))[0][0]]
        except:
            leftLane = ([], [], [], [])
        try:
            rightLane = self.lanes[(np.argwhere(points == min(rightPoints)[0]))[0][0]]
        except:
            rightLane = ([], [], [], [])
        #######################################################

        if len(leftLane[0]) > len(rightLane[0]):
            longerLane = leftLane
            shorterLane = rightLane
        else:
            longerLane = rightLane
            shorterLane = leftLane
            adjust = -adjust

        # As long as there are still points in each lane, calculate the midpoint of lane by averaging corresponding
        # points
        for i in xrange(0, len(shorterLane[0])):
            sumPoint = np.add(longerLane[0][i], shorterLane[0][i])
            avgPoint = np.divide(sumPoint, 2)
            midPoints.append(avgPoint)
        # Once you run out of corresponding points, simply horizontally offset the remaining points
        for i in xrange(len(shorterLane[0]), len(longerLane[0])):
            midPoints.append(np.add(longerLane[0][i], [0, adjust]))

        midPoints = [list(imap(int, midPoint)) for midPoint in midPoints]
        return midPoints

    def visualize(self):
        for points, lines, polygons, confidence in self.lanes:
            for point in points:
                pass
                cv2.circle(self.vis, tuple(point[::-1]), 10, (0, 255, 0), -1)
            for line in lines:
                cv2.line(self.vis, line[0], line[1], (255, 255, 0), 5)
            for polygon in polygons:
                cv2.polylines(self.vis, [polygon], True, (0, 0, 255))

        for midPoint in self.midPoints:
            cv2.circle(self.vis, tuple(midPoint[::-1]), 10, (255, 0, 255), -1)

        return self.vis

    def binarizeImage(self, image):
        '''
        Binarize an image
        :param image: cv2 image
        :return: the binarized version of the image
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
        return binary

    def getMidpoints(self):
        '''
        Lane midpoint accessor
        :return: list of tuples representing the midpoints of the lanes in pixels coordinates
        '''
        return np.array(self.midPoints)


def makeBox(center, size):
    '''
    Create a square box of given size around a point
    :param center: The center of the box, tuple of coordinates
    :param size: the length and width in pixels
    :return: the four points of the box (each is a coordinate tuple)
    '''
    top = int(center[0] - size / 2);
    bottom = int(center[0] + size / 2);
    left = int(center[1] - size / 2);
    right = int(center[1] + size / 2);
    return top, bottom, left, right


def getRect(vector, center, size):
    angle = np.arctan2(vector[0], vector[1]) * 180 / math.pi
    if angle < 0:
        angle = angle + 360
    if 45 <= angle and angle <= 135:
        topLeft = np.add(center, [0, -size])
        bottomLeft = np.add(center, [size, -size])
        bottomRight = np.add(center, [size, size])
        topRight = np.add(center, [0, size])
    elif 135 <= angle and angle <= 225:
        topLeft = np.add(center, [-size, -size])
        bottomLeft = np.add(center, [size, -size])
        bottomRight = np.add(center, [size, 0])
        topRight = np.add(center, [-size, 0])
    elif 225 <= angle and angle <= 315:
        topLeft = np.add(center, [-size, -size])
        bottomLeft = np.add(center, [0, -size])
        bottomRight = np.add(center, [0, size])
        topRight = np.add(center, [-size, size])
    elif 315 <= angle or angle <= 45:
        topLeft = np.add(center, [-size, 0])
        bottomLeft = np.add(center, [size, 0])
        bottomRight = np.add(center, [size, size])
        topRight = np.add(center, [-size, size])

    return topLeft, bottomLeft, bottomRight, topRight


if __name__ == u'__main__':
    # make sure you're feeding in an image path as an argument
    pathToImage = sys.argv[1]
    image = cv2.imread(pathToImage)
    lf = lane_finder(image, step_size=50, base_size=.2)
    img = lf.visualize()
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
