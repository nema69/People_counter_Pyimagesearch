# import the necessary packages
from imutils.video import FPS
import multiprocessing
from multiprocessing import Pool
import numpy as np
import argparse
import imutils
import dlib
import cv2
from os import getpid
import concurrent.futures


def start_tracker(box, label, rgb, inputQueue, outputQueue):
    # construct a dlib rectangle object from the bounding box
    # coordinates and then start the correlation tracker
    t = dlib.correlation_tracker()
    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
    t.start_track(rgb, rect)
    tracker_confidence = 9
    # loop indefinitely -- this function will be called as a daemon
    # process so we don't need to worry about joining it
    while True:
        # attempt to grab the next frame from the input queue
        rgb = inputQueue.get()
        # if there was an entry in our queue, process it
        if rgb is not None:
            # update the tracker and grab the position of the tracked
            # object
            tracker_confidence = t.update(rgb)
            print("ProcessID:{}".format(getpid()))
            print("Confidence: {}".format(tracker_confidence))
            pos = t.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            # add the label + bounding box coordinates to the output
            # queue
            outputQueue.put((label, (startX, startY, endX, endY), tracker_confidence))

        if tracker_confidence < 8:
            return


def preprocess_frame(input_frame, desired_width):

    resized_frame = imutils.resize(input_frame, width=desired_width)

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    sharpened_frame = cv2.filter2D(resized_frame, -1, kernel)

    output_frame = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2RGB)

    return output_frame


def run_detection_on_frame(input_frame):
    # grab the frame dimensions and convert the frame to a blob
    (h, w) = input_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(input_frame, 0.007843, (w, h), 127.5)
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

# initialize our lists of queues -- both input queue and output queue
# for *every* object that we will be tracking
inputQueues = []
outputQueues = []

skip_frames = args["skip_frames"]
frames_since_detection = 0

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "person"]
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["input"])
writer = None
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:

    # grab the next frame from the video file
    (grabbed, frame) = vs.read()
    # check to see if we have reached the end of the video file
    if frame is None:
        break
    # resize the frame for faster processing and then convert the
    # frame from BGR to RGB ordering (dlib needs RGB ordering)
    frame = preprocess_frame(frame, 500)
    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # if our list of queues is empty then we know we have yet to
    # create our first object tracker
    if frames_since_detection % skip_frames == 0:
        print('if')
        detections = run_detection_on_frame(frame)
        print('running detection...')
        frames_since_detection = 1

        list_bounding_boxes, list_labels = build_list_of_bounding_boxes(detections)

        for i in range(len(list_bounding_boxes)):

            # grab the corresponding class label for the detection
            # and draw the bounding box
            (startX, startY, endX, endY) = list_bounding_boxes[i]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, list_labels[i], (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        #cv2.imshow("Frame", frame)

    # otherwise, we've already performed detection so let's track
    # multiple objects
    else:
        print('else')
        # loop over each of our input ques and add the input RGB
        # frame to it, enabling us to update each of the respective
        # object trackers running in separate processes
        for iq in inputQueues:
            iq.put(rgb)
        # loop over each of the output queues
        for oq in outputQueues:
            # grab the updated bounding box coordinates for the
            # object -- the .get method is a blocking operation so
            # this will pause our execution until the respective
            # process finishes the tracking update
            (label, (startX, startY, endX, endY), tracker_confidence) = oq.get()

            #if tracker_confidence < 8:

            # draw the bounding box from the correlation object
            # tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.putText(frame, str(tracker_confidence), (startX, startY - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()
        frames_since_detection = frames_since_detection + 1
        #print(frames_since_detection)
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()
# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()