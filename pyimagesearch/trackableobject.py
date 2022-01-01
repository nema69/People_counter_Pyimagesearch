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

    def direction_horizontal(self):

        # c[1] y component of location history
        y = [c[1] for c in self.centroids]
        # negative = up , positive = down
        direction_value = self.centroids[1] - np.mean(y)
        return direction_value
