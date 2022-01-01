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
from functions import two_point_box_2_width_height_box
from functions import width_height_box_2_two_point_box


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

    # output_frame = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)

    return sharpened_frame


def run_detection_on_frame(input_frame):
    # grab the frame dimensions and convert the frame to a blob
    input_frame = (input_frame)
    (h, w) = input_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(input_frame, 0.007843, (w, h), (127, 127, 127))
    # pass the blob through the network and obtain the detections
    # and predictions
    net.setInput(blob)
    output_detections = net.forward()

    return output_detections


def build_list_of_bounding_boxes(input_detections):

    out_list_bounding_boxes = []
    out_list_labels = []
    for i in np.arange(0, input_detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = input_detections[0, 0, i, 2]
        # filter out weak detections by requiring a minimum
        # confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            (h, w) = frame.shape[:2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            bb = (startX, startY, endX, endY)
            out_list_bounding_boxes.append(bb)
            out_list_labels.append(label)

    return out_list_bounding_boxes, out_list_labels


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


def update_trackers_cv2(tracker_list, current_frame):

    rects_ = []

    for tracker_ in tracker_list:

        # get width height rect from tracker
        success, rect_cv2 = tracker_.update(current_frame)
        # transform into 2 point rect
        two_point_rect = width_height_box_2_two_point_box(rect_cv2)
        # add the bounding box coordinates to the rectangles list
        rects_.append(two_point_rect)

    return rects_


def initialize_cv2_tracker(input_frame, input_bb):
    # Transform BB from left,bot,right,top to x,y,w,h
    new_bb = two_point_box_2_width_height_box(input_bb)

    # Create new tracker
    tracker_ = cv2.legacy.TrackerMOSSE_create()

    # Init new tracker
    tracker_.init(input_frame, new_bb)

    return tracker_


def initialize_dlib_tracker(input_frame, input_bb):
     tracker = dlib.correlation_tracker()
     (startX, startY, endX, endY) = input_bb
     rect = dlib.rectangle(startX, startY, endX, endY)
     tracker.start_track(input_frame, rect)
     return tracker


def draw_counting_lines(input_frame, orientation, distance, colour):
    (h, w) = frame.shape[:2]
    half_dist = distance//2
    if not orientation:

        cv2.line(input_frame, (0, h // 2 + half_dist), (w, h // 2 + half_dist), colour, 2)
        cv2.line(input_frame, (0, h // 2 - half_dist), (w, h // 2 - half_dist), colour, 2)

    else:
        cv2.line(input_frame, (w // 2 + half_dist, 0), (w // 2 + half_dist, h), colour, 2)
        cv2.line(input_frame, (w // 2 - half_dist, 0), (w // 2 - half_dist, h), colour, 2)


def draw_box_from_bb(in_frame, bb):
    if bb != (0.0,0.0,0.0,0.0):
        startpoint = int(bb[0]), int(bb[1])
        endpoint = int(bb[2]), int(bb[3])
        out_frame = cv2.rectangle(in_frame, startpoint, endpoint, (0, 0, 255), 2)
        return out_frame
    else:
        return in_frame


def determine_direction(centroid, orientation):

    # 0 = Vertical
    if orientation == 0:
        # c[1] y component of location history
        y = [c[1] for c in to.centroids]
        # negative = up , positive = down
        direction_value = centroid[1] - np.mean(y)
        return direction_value
    else:
        # c[1] y component of location history
        y = [c[0] for c in to.centroids]
        # negative = up , positive = down
        direction_value = centroid[0] - np.mean(y)
        return direction_value


def check_if_count(centroid, direction, orientation, totalUp, totalDown):

    if orientation == 0:
        if direction < 0 and centroid[1] < H // 3 and person_dict[str(objectID)] > frame_counts_up:
            totalUp += 1
            to.counted = True

        # if the direction is positive (indicating the object
        # is moving down) AND the centroid is below the
        # center line, count the object
        elif direction > 0 and centroid[1] > H // 3 * 2:
            if person_dict[str(objectID)] > frame_counts:
                totalDown += 1
                to.counted = True
        return totalUp, totalDown

    else:
        if direction < 0 and centroid[0] < W // 3 and person_dict[str(objectID)] > frame_counts_up:
            totalUp += 1
            to.counted = True

        # if the direction is positive (indicating the object
        # is moving down) AND the centroid is below the
        # center line, count the object
        elif direction > 0 and centroid[0] > W // 3 * 2:
            if person_dict[str(objectID)] > frame_counts:
                totalDown += 1
                to.counted = True
        return totalUp, totalDown


def count(objects, orientation,current_count):

    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

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
            # y = [c[1] for c in to.centroids]
            # direction = centroid[1] - np.mean(y)
            # 0 = Vertical
            if orientation == 0:
                # c[1] y component of location history
                y = [c[1] for c in to.centroids]
                # negative = up , positive = down
                direction = centroid[1] - np.mean(y)

            else:
                # c[1] y component of location history
                y = [c[0] for c in to.centroids]
                # negative = up , positive = down
                direction = centroid[0] - np.mean(y)

        to.centroids.append(centroid)

        # check to see if the object has been counted or not
        if not to.counted:
            if orientation == 0:
                if direction < 0 and centroid[1] < H // 3 and person_dict[str(objectID)] > frame_counts_up:
                    current_count[0] += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 3 * 2:
                    if person_dict[str(objectID)] > frame_counts:
                        current_count[1] += 1
                        to.counted = True

            else:
                if direction < 0 and centroid[0] < W // 3 and person_dict[str(objectID)] > frame_counts_up:
                    current_count[0] += 1
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[0] > W // 3 * 2:
                    if person_dict[str(objectID)] > frame_counts:
                        current_count[1] += 1
                        to.counted = True

    trackableObjects[objectID] = to
    return current_count



    # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

# Person dict for better Direction detection
person_dict = dict()
frame_counts = 110
frame_counts_up = 60
Vertical = 0
Horizontal = 1
Counting_direction = ["Vertical", "Horizontal"]
orientation = Horizontal
TotalUp = 0
TotalDown = 0

# Setup csv file
current_file = create_csv_file()
print('Printing to: %s' % current_file)
total = 0
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
          "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
          "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
          "sofa", "train", "tvmonitor"]




#["background", "person"]
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=20)  # , maxDistance=50
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    start_time_frame = time.time()

    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame_detection = preprocess_frame(frame, 500)
    frame = imutils.resize(frame, height=600)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []
    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args["skip_frames"] == 0:
        start_time_detection = time.time()
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        detections = run_detection_on_frame(frame_detection)

        list_of_bounding_boxes, list_of_labels = build_list_of_bounding_boxes(detections)

        for bounding_box in list_of_bounding_boxes:

            # (startX, startY, endX, endY) = bounding_box
            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = initialize_cv2_tracker(frame, bounding_box)
            # tracker = initialize_dlib_tracker(frame, bounding_box)

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            trackers.append(tracker)
            frame = draw_box_from_bb(frame, bounding_box)
            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
        end_time_detection = time.time()
        print('updating detection: {}'.format(end_time_detection - start_time_detection))

    else:
        # loop over the trackers
        start_time_trackers = time.time()
        rects = update_trackers_cv2(trackers, frame)
        # rects = update_trackers_dlib(trackers, frame)

        for rec in rects:
            frame = draw_box_from_bb(frame, rec)

        end_time_trackers = time.time()
        print('updating trackers: {}'.format(end_time_trackers - start_time_trackers))


    # draw 2 horizontal lines in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    # cv2.line(frame, (0, H // 5), (W, H // 5), (0, 255, 255), 2)
    # cv2.line(frame, (0, H // 3 * 2), (W, H // 3 * 2), (0, 255, 255), 2)
    draw_counting_lines(frame, orientation, 100, (255, 255, 255))

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids

    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

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
            # y = [c[1] for c in to.centroids]
            # direction = centroid[1] - np.mean(y)
            direction = centroid.direction_horizontal
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                # if direction < 0 and centroid[1] < H // 3 and person_dict[str(objectID)] > frame_counts_up:
                #     totalUp += 1
                #     to.counted = True
                #
                # # if the direction is positive (indicating the object
                # # is moving down) AND the centroid is below the
                # # center line, count the object
                # elif direction > 0 and centroid[1] > H // 3 * 2:
                #     if person_dict[str(objectID)] > frame_counts:
                #         totalDown += 1
                #         to.counted = True
                totalUp, totalDown = check_if_count(centroid, direction, orientation, TotalUp, TotalDown)
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        if str(objectID) in person_dict:
            person_dict[str(objectID)] += 1
        else:
            person_dict[str(objectID)] = 0
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Write new info to csv file
    # Determine if total has gone up
    if total < totalUp + totalDown:

        if totalUp > last_totalUp:  # Determine Direction
            direction = 'up'
        else:
            direction = 'down'

        total = totalUp + totalDown  # Calculate total
        write_new_value(current_file, direction, total)  # Write new values to file

    last_totalUp = totalUp
    last_totalDown = totalDown

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()
    end_time_frame = time.time()
    print('frame_time: {}'.format(end_time_frame - start_time_frame))
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
