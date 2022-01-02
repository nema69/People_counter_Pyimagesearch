import numpy as np


class TrackableObject:  # class = objekt

    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]  # list with history of location
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False
        self.x_direction = 0
        self.y_direction = 0

    def direction_vertical(self):

        # c[1] y component of location history
        y = [c[1] for c in self.centroids]
        # negative = up , positive = down
        direction_value_y = self.centroids[0][1] - np.mean(y)
        self.y_direction = direction_value_y

        return direction_value_y

    def direction_horizontal(self):

        # c[1] y component of location history
        x = [c[0] for c in self.centroids]
        # negative = right , positive = left
        direction_value_x = self.centroids[0][0] - np.mean(x)
        self.x_direction = direction_value_x

        return direction_value_x
