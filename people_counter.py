# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from write_csv import create_csv_file
from write_csv import write_new_value
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def update_trackers_cv2(tracker_list, current_frame):
    rects_ = []
    for tracker_ in tracker_list:

        rect_cv2 = tracker_.update(current_frame)
        left, top, right, bottom = rect_cv2[1]
        # dlib_rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))
        # add the bounding box coordinates to the rectangles list
        rects_.append((left, top, right, bottom))
    return rects_


def initialize_cv2_tracker(input_frame, input_bb):
    tracker = cv2.legacy.TrackerMOSSE_create()
    tracker.init(input_frame, input_bb)
    return tracker


def initialize_dlib_tracker(input_frame, input_bb):
    tracker = dlib.correlation_tracker()
    (startX, startY, endX, endY) = input_bb
    rect = dlib.rectangle(startX, startY, endX, endY)
    tracker.start_track(input_frame, rect)
    return tracker


def preprocess_frame_detection(input_frame):
    nh, nw = 224, 224
    h, w, _ = input_frame.shape
    if h < w:
        off = (w - h) / 2
        input_frame = input_frame[:, off:off + h]
    else:
        off = (h - w) / 2
        input_frame = input_frame[off:off + h, :]
    frame_resized = imutils.resize(input_frame, [nh, nw])
    output_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    return output_frame


def preprocess_frame(input_frame, desired_width):
    resized_frame = imutils.resize(input_frame, width=desired_width)

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    sharpened_frame = cv2.filter2D(resized_frame, -1, kernel)
    return cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)


def update_trackers_dlib(tracker_list, current_frame):
    rects_ = []
    for tracker_ in tracker_list:

        tracker_.update(current_frame)
        pos = tracker_.get_position()

        # unpack the position object
        startX_ = int(pos.left())
        startY_ = int(pos.top())
        endX_ = int(pos.right())
        endY_ = int(pos.bottom())

        # add the bounding box coordinates to the rectangles list
        rects_.append((startX_, startY_, endX_, endY_))
    return rects_


class PeopleCounter:
    def __init__(self, prototxt, model, *args, **kwargs):
        self.prototxt = prototxt
        self.model = model
        self.input = kwargs.get('input', None)
        self.output = kwargs.get('output', None)
        self.confidence = kwargs.get('confidence', 0.4)
        self.skip_frames = kwargs.get('skip_frames', 30)

        # Person dict for better Direction detection
        self.person_dict = dict()

        self.frame_counts = 110
        self.frame_counts_up = 60

        # Setup csv file
        self.current_file = create_csv_file()
        print('Printing to: %s' % self.current_file)
        self.total = 0

        # initialize the list of class labels MobileNet SSD was trained to
        # detect
        self.CLASSES = ["background", "person"]

        # ["background", "aeroplane", "bicycle", "bird", "boat",
        #           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        #           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        #           "sofa", "train", "tvmonitor"]

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        # if a video path was not supplied, grab a reference to the webcam
        if self.input is None:
            print("[INFO] starting video stream...")
            self.vs = VideoStream(src=0).start()
            time.sleep(2.0)
        # otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            self.vs = cv2.VideoCapture(self.input)

        # initialize the video writer (we'll instantiate later if need be)
        self.writer = None

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(maxDisappeared=40)  # , maxDistance=50
        self.trackers = []
        self.trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0

        # start the frames per second throughput estimator
        self.fps = FPS().start()

    def run_detection_on_frame(self, input_frame):
        # grab the frame dimensions and convert the frame to a blob
        input_frame = (input_frame)
        (h, w) = input_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            input_frame, 0.007843, (w, h), (127, 127, 127))
        # pass the blob through the network and obtain the detections
        # and predictions
        self.net.setInput(blob)

        return self.net.forward()

    def build_list_of_bounding_boxes(self, input_detections):

        out_list_bounding_boxes = []
        out_list_labels = []
        for i in np.arange(0, input_detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = input_detections[0, 0, i, 2]
            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > self.confidence:
                # extract the index of the class label from the
                # detections list
                idx = int(self.detections[0, 0, i, 1])
                label = self.CLASSES[idx]
                # if the class label is not a person, ignore it
                if self.CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                (h, w) = self.frame.shape[:2]
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)
                out_list_bounding_boxes.append(bb)
                out_list_labels.append(label)

        return out_list_bounding_boxes, out_list_labels

    def draw_counting_lines(self, input_frame, orientation, distance, colour):
        (h, w) = self.frame.shape[:2]
        half_dist = distance//2
        if orientation:

            cv2.line(input_frame, (0, h // 2 + half_dist),
                     (w, h // 2 + half_dist), colour, 2)
            cv2.line(input_frame, (0, h // 2 - half_dist),
                     (w, h // 2 - half_dist), colour, 2)
        else:
            cv2.line(input_frame, (w // 2 + half_dist, 0),
                     (w // 2 + half_dist, h), colour, 2)
            cv2.line(input_frame, (w // 2 - half_dist, 0),
                     (w // 2 - half_dist, h), colour, 2)

    def main_loop(self):
        # loop over frames from the video stream
        while True:
            start_time_frame = time.time()

            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            nframe = self.vs.read()
            self.frame = nframe[1] if self.input is not None else nframe

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if self.input is not None and self.frame is None:
                break

            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib

            # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            # frame = preprocess_frame(frame, 500)
            self.frame = imutils.resize(self.frame, height=500)

            # if the frame dimensions are empty, set them
            if self.W is None or self.H is None:
                (self.H, self.W) = self.frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if self.output is not None and self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter(self.output, fourcc, 30,
                                              (self.W, self.H), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []
            print("Height: {}".format(self.H))
            print("Width: {}".format(self.W))
            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if self.totalFrames % self.skip_frames == 0:
                start_time_detection = time.time()
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                self.trackers = []

                self.detections = self.run_detection_on_frame(self.frame)

                list_of_bounding_boxes, list_of_labels = self.build_list_of_bounding_boxes(
                    self.detections)

                for bounding_box in list_of_bounding_boxes:

                    # (startX, startY, endX, endY) = bounding_box
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = initialize_cv2_tracker(
                        self.frame, bounding_box)
                    # tracker = initialize_dlib_tracker(frame, bounding_box)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)

                    # otherwise, we should utilize our object *trackers* rather than
                    # object *detectors* to obtain a higher frame processing throughput
                end_time_detection = time.time()
                print('updating detection: {}'.format(
                    end_time_detection - start_time_detection))

            else:
                # loop over the trackers
                start_time_trackers = time.time()
                rects = update_trackers_cv2(self.trackers, self.frame)
                # rects = update_trackers_dlib(trackers, frame)
                end_time_trackers = time.time()
                print('updating trackers: {}'.format(
                    end_time_trackers - start_time_trackers))
            # draw 2 horizontal lines in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            # moving 'up' or 'down'
            # cv2.line(frame, (0, H // 5), (W, H // 5), (0, 255, 255), 2)
            # cv2.line(frame, (0, H // 3 * 2), (W, H // 3 * 2), (0, 255, 255), 2)
            self.draw_counting_lines(self.frame, 0, 20, (255, 255, 255))

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids

            objects = self.ct.update(rects)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = self.trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not to.counted:
                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        if direction < 0 and centroid[1] < self.H // 3 and self.person_dict[str(objectID)] > self.frame_counts_up:
                            self.totalUp += 1
                            to.counted = True

                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction > 0 and centroid[1] > self.H // 3 * 2:
                            if self.person_dict[str(objectID)] > self.frame_counts:
                                self.totalDown += 1
                                to.counted = True

                # store the trackable object in our dictionary
                self.trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                if str(objectID) in self.person_dict:
                    self.person_dict[str(objectID)] += 1
                else:
                    self.person_dict[str(objectID)] = 0
                text = "ID {}".format(objectID)
                cv2.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(
                    self.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Up", self.totalUp),
                ("Down", self.totalDown),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(self.frame, text, (10, self.H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if self.writer is not None:
                self.writer.write(self.frame)

            # show the output frame
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # Write new info to csv file
            # Determine if total has gone up
            if self.total < self.totalUp + self.totalDown:

                if self.totalUp > last_totalUp:  # Determine Direction
                    direction = 'up'
                else:
                    direction = 'down'

                self.total = self.totalUp + self.totalDown  # Calculate total
                # Write new values to file
                write_new_value(self.current_file, direction, self.total)

            last_totalUp = self.totalUp
            last_totalDown = self.totalDown

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            self.totalFrames += 1
            self.fps.update()
            end_time_frame = time.time()
            print('frame_time: {}'.format(end_time_frame - start_time_frame))
        # stop the timer and display FPS information
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if self.writer is not None:
            self.writer.release()

        # if we are not using a video file, stop the camera video stream
        if self.input is None:
            self.vs.stop()

        # otherwise, release the video file pointer
        else:
            self.vs.release()

        # close any open windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pc = PeopleCounter("mobilenet_ssd/MBSSD_PED_deploy.prototxt",
                       "mobilenet_ssd/MBSSD_PED.caffemodel", input="videos/MOT20-02.webm")
    pc.main_loop()
