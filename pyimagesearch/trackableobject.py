class TrackableObject:  # class = objekt
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]  # list with history of location
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False
