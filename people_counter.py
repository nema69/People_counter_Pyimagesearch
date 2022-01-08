# import the necessary packages
from os import stat
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from write_csv import create_csv_file
from write_csv import write_new_value
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import dlib
import cv2
from functions import two_point_box_2_width_height_box
from functions import width_height_box_2_two_point_box
import concurrent.futures

import queue


class PeopleCounter:
    def __init__(self, prototxt, model, **kwargs) -> None:
        self.prototxt = prototxt
        self.model = model
        self.input = kwargs.get('input', None)
        self.output = kwargs.get('output', None)
        self.confidence = kwargs.get('confidence', 0.4)
        self.skip_frames = kwargs.get('skip_frames', 30)

    @staticmethod
    def preprocess_frame(input_frame, roi, longest_side):
        cropped_frame = input_frame[int(roi[1]):int(
            roi[1]+roi[3]-1), int(roi[0]):int(roi[0]+roi[2]-1)]

        h, w, _ = cropped_frame.shape
        if w > h and w > longest_side:
            dw = longest_side
            dh = int(h/w*longest_side)
        elif h > longest_side:
            dh = longest_side
            dw = int(w/h * longest_side)
        else:
            dh = int(h)
            dw = int(w)
        dim = (dw, dh)

        resized_frame = cv2.resize(
            cropped_frame, dim, interpolation=cv2.INTER_LINEAR)

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        sharpened_frame = cv2.filter2D(resized_frame, -1, kernel)

        return resized_frame, sharpened_frame

    @staticmethod
    def update_trackers_dlib(tracker_list, input_frame):
        rects_ = []
        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        for tracker_ in tracker_list:

            tracker_.update(rgb_frame)
            pos = tracker_.get_position()

            # unpack the position object
            startX_ = int(pos.left())
            startY_ = int(pos.top())
            endX_ = int(pos.right())
            endY_ = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects_.append((startX_, startY_, endX_, endY_))
        return rects_

    @staticmethod
    def update_trackers_cv2(tracker_list, current_frame):
        rects_ = []
        success_list_ = []

        for tracker_ in tracker_list:

            # get width height rect from tracker
            success_, rect_cv2 = tracker_.update(current_frame)
            # transform into 2 point rect
            two_point_rect = width_height_box_2_two_point_box(rect_cv2)
            # add the bounding box coordinates to the rectangles list
            rects_.append(two_point_rect)
            success_list_.append(success_)
        return rects_, success_list_

    @staticmethod
    def initialize_cv2_tracker(input_frame, input_bb):
        # Transform BB from left,bot,right,top to x,y,w,h
        new_bb = two_point_box_2_width_height_box(input_bb)

        # Create new tracker
        tracker_ = cv2.TrackerKCF_create()

        # Init new tracker
        tracker_.init(input_frame, new_bb)

        return tracker_

    @staticmethod
    def initialize_dlib_tracker(input_frame, input_bb):
        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        tracker = dlib.correlation_tracker()
        (startX, startY, endX, endY) = input_bb
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb_frame, rect)
        return tracker

    def run_detection_on_frame(self, input_frame):
        # grab the frame dimensions and convert the frame to a blob
        (h, w) = input_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            input_frame, 0.007843, (w, h), (127, 127, 127))
        # pass the blob through the network and obtain the detections
        # and predictions
        self.net.setInput(blob)
        output_detections = self.net.forward()

        return output_detections

    def build_list_of_bounding_boxes(self, input_detections):
        out_list_bounding_boxes = []
        out_list_labels = []
        for detection_no in np.arange(0, input_detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = input_detections[0, 0, detection_no, 2]
            # filter out weak detections by requiring a minimum
            # confidence
            if confidence > self.confidence:
                # extract the index of the class label from the
                # detections list
                idx = int(self.detections[0, 0, detection_no, 1])
                label = self.CLASSES[idx]
                # if the class label is not a person, ignore it
                if self.CLASSES[idx] != "person":
                    continue

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                (h, w) = self.frame.shape[:2]
                box = self.detections[0, 0, detection_no, 3:7] * \
                    np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bb = (startX, startY, endX, endY)
                out_list_bounding_boxes.append(bb)
                out_list_labels.append(label)

        return out_list_bounding_boxes, out_list_labels

    def draw_counting_lines(self, input_frame, orientation, distance, colour):
        (h, w) = self.frame.shape[:2]
        half_dist = distance//2
        if not orientation:

            cv2.line(input_frame, (0, h // 2 + half_dist),
                     (w, h // 2 + half_dist), colour, 2)
            cv2.line(input_frame, (0, h // 2 - half_dist),
                     (w, h // 2 - half_dist), colour, 2)

        else:
            cv2.line(input_frame, (w // 2 + half_dist, 0),
                     (w // 2 + half_dist, h), colour, 2)
            cv2.line(input_frame, (w // 2 - half_dist, 0),
                     (w // 2 - half_dist, h), colour, 2)

    @staticmethod
    def draw_box_from_bb(in_frame, bb):
        if bb != (0.0, 0.0, 0.0, 0.0):
            startpoint = int(bb[0]), int(bb[1])
            endpoint = int(bb[2]), int(bb[3])
            out_frame = cv2.rectangle(
                in_frame, startpoint, endpoint, (0, 0, 255), 2)
            return out_frame
        else:
            return in_frame

    def determine_direction(self, centroid, orientation):
        # 0 = Vertical
        if orientation == 0:
            # c[1] y component of location history
            y = [c[1] for c in self.to.centroids]
            # negative = up , positive = down
            direction_value = centroid[1] - np.mean(y)
            return direction_value
        else:
            # c[1] y component of location history
            y = [c[0] for c in self.to.centroids]
            # negative = up , positive = down
            direction_value = centroid[0] - np.mean(y)
            return direction_value

    def check_if_count(self, trackable_object, orientation, totalUp, totalDown, offset):
        # orientation = 1: objects are moving along the x axis
        if orientation:
            # calculate horizontal direction
            trackable_object.direction_horizontal()
            # direction < 0: objects are moving right (x getting more pos), centroid x position ([0][0]) is right of W//3 * 2
            if trackable_object.x_direction < 0 and trackable_object.centroids[-1][0] > self.W // 2 + offset and self.person_dict[str(trackable_object.objectID)] > self.frame_counts_up:
                totalUp += 1
                trackable_object.counted = True

            # direction > 0: objects are moving left (x getting more neg), centroid x position ([0][0]) is left of W//3
            elif trackable_object.x_direction > 0 and trackable_object.centroids[-1][0] < self.W // 2 - offset and self.person_dict[str(trackable_object.objectID)] > self.frame_counts_up:
                totalDown += 1
                self.to.counted = True
            return totalUp, totalDown

        else:
            if self.direction < 0 and self.centroid[0] < self.W // 2 - offset and self.person_dict[str(self.objectID)] > self.frame_counts_up:
                totalUp += 1
                self.to.counted = True

            # if the direction is positive (indicating the object
            # is moving down) AND the centroid is below the
            # center line, count the object
            elif self.direction > 0 and self.centroid[0] > self.W // 2 + offset:
                if self.person_dict[str(self.objectID)] > self.frame_counts:
                    totalDown += 1
                    self.to.counted = True
            return totalUp, totalDown

    def capture_frame(self, src, queue_, framerate):
        period = 1/framerate
        self.cap = cv2.VideoCapture(src)
        frame_no = 0
        while True:
            start = time.time()
            _, frame_ = self.cap.read()
            frame_no = frame_no+1
            queue_.put((frame_no, frame_))
            if time.time()-start < period:
                time.sleep(period-(time.time()-start))
            if frame_ is None:
                continue

    @staticmethod
    def get_roi(input_frame, scale):
        h, w, _ = input_frame.shape
        nh = int(h * scale)
        nw = int(w * scale)
        resized_frame = cv2.resize(input_frame, (nw, nh))
        roi_scaled = cv2.selectROI('ROI', resized_frame, False)
        cv2.destroyWindow('ROI')
        roi = tuple(x/scale for x in roi_scaled)
        return roi

    def main_loop(self):
        # Setup for Counting and Direction
        offset_dist = 0
        self.person_dict = dict()
        self.frame_counts = 10
        self.frame_counts_up = 10
        self.Vertical = 0
        self.Horizontal = 1
        self.Counting_direction = ["Vertical", "Horizontal"]
        self.orientation = self.Horizontal
        direction_dict = {1: 'Up', 2: 'Down', 3: 'Right', 4: 'Left'}

        # Setup csv file
        current_file = create_csv_file()
        print('Printing to: %s' % current_file)
        total = 0

        # initialize the list of class labels MobileNet SSD was trained to
        # detect
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        #["background", "person"]

        # load our serialized model from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # if a video path was not supplied, grab a reference to the webcam
        if self.input is None:
            print("[INFO] starting video stream...")
            #self.vs = VideoStream(src=0).start()
            self.vs = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 !\
        nvvidconv ! video/x-raw, format=BGRx, width=640, height=360 ! videoconvert ! video/x-raw, format=BGR  ! appsink")
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
        # each of our trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(
            maxDisappeared=self.skip_frames*2, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0

        # start the frames per second throughput estimator
        self.fps = FPS().start()
        # Get one frame for determining ROI
        _, roiframe = self.vs.read()
        roi_coord = self.get_roi(roiframe, 0.5)

        # Build queue for frames, normal queue for video read( preloads frames) and lifoqueue for "live" video"
        q = queue.Queue(maxsize=10)
        max_frame = 0

        # init argument tuple for capture frame thread
        params = (self.input, q, 30)

        # start thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # pack arguments with lamda
            f1 = executor.submit(lambda p: self.capture_frame(*p), params)

            # loop over frames from the video stream
            while True:
                start_time_frame = time.time()

                # grab the next frame and handle if we are reading from either
                # VideoCapture or VideoStream
                #frame = vs.read()
                #frame = frame[1] if args.get("input", False) else frame

                # get new frame from queue
                frame_num, new_frame = q.get(timeout=1)

                # if frame is none exit the loop
                if new_frame is None:
                    break

                # print(f'current_frame:{frame_num} max frame:{max_frame}')
                # check if new_frame isnt an earlier frame to prevent rubberbanding
                if frame_num >= max_frame and new_frame is not None:
                    max_frame = frame_num
                    self.frame = new_frame
                    backup_frame = self.frame
                # if it is an older frame reuse the most recent frame
                else:
                    self.frame = backup_frame

                # if we are viewing a video and we did not grab a frame then we
                # have reached the end of the video
                # if args["input"] is not None and frame is None:
                #    break

                # crop the frame to the roi, resize the frame to have a maximum size
                # and sharpen it for detection
                self.frame, frame_detection = self.preprocess_frame(
                    self.frame, roi_coord, 400)

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
                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if self.totalFrames % self.skip_frames == 0:
                    start_time_detection = time.time()
                    # set the status and initialize our new set of object trackers
                    status = "Detecting"
                    self.trackers = []

                    # run detections on frame and return all detected objects
                    self.detections = self.run_detection_on_frame(
                        frame_detection)

                    # get relevant bounding boxes from detections
                    list_of_bounding_boxes, list_of_labels = self.build_list_of_bounding_boxes(
                        self.detections)

                    for bounding_box in list_of_bounding_boxes:

                        # start a tracker for each bounding box
                        tracker = self.initialize_cv2_tracker(
                            self.frame, bounding_box)
                        #tracker = initialize_dlib_tracker(frame, bounding_box)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        self.trackers.append(tracker)

                        # draw box around detections
                        self.frame = self.draw_box_from_bb(
                            self.frame, bounding_box)

                    end_time_detection = time.time()
                    print('updating detection: {}'.format(
                        end_time_detection - start_time_detection))

                # otherwise, we should utilize our object *trackers* rather than
                # object *detectors* to obtain a higher frame processing throughput
                else:
                    start_time_trackers = time.time()

                    # update trackers
                    rects, success_list = self.update_trackers_cv2(
                        self.trackers, self.frame)
                    #rects = update_trackers_dlib(trackers, frame)

                    # draw boxes around successfull tracks
                    for count, rec in enumerate(rects):
                        success = success_list[count]
                        if success:
                            self.frame = self.draw_box_from_bb(self.frame, rec)
                        else:
                            print('Tracking Error')
                    end_time_trackers = time.time()
                    print('updating trackers: {}'.format(
                        end_time_trackers - start_time_trackers))

                # draw the lines on the other side of which we will count
                self.draw_counting_lines(self.frame, self.orientation,
                                         offset_dist, (255, 255, 255))

                # use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids
                start_time_update_ct = time.time()
                objects = self.ct.update(rects)

                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    # check to see if a trackable object exists for the current
                    # object ID
                    self.to = self.trackableObjects.get(objectID, None)

                    # if there is no existing trackable object, create one
                    if self.to is None:
                        self.to = TrackableObject(objectID, centroid)

                    # otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                    else:
                        # the difference between the y-coordinate of the *current*
                        # centroid and the mean of *previous* centroids will tell
                        # us in which direction the object is moving (negative for
                        # 'up' and positive for 'down')
                        if self.orientation:
                            direction = self.to.direction_horizontal()
                        else:
                            direction = self.to.direction_vertical()

                        self.to.centroids.append(centroid)

                        # check to see if the object has been counted or not
                        if not self.to.counted:
                            # check if we can count the object
                            self.totalUp, self.totalDown = self.check_if_count(
                                self.to, self.orientation, self.totalUp, self.totalDown, offset_dist)

                    # store the trackable object in our dictionary
                    self.trackableObjects[objectID] = self.to

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
                    (direction_dict[1+(2*self.orientation)], self.totalUp),
                    (direction_dict[2+(2*self.orientation)], self.totalDown),
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
                end_time_update_ct = time.time()
                print('updating ct and counting: {}'.format(
                    end_time_update_ct - start_time_update_ct))

                # Write new info to csv file
                # Determine if total has gone up
                if total < self.totalUp + self.totalDown:

                    if self.totalUp > last_totalUp:  # Determine Direction
                        self.direction = direction_dict[1+(2*self.orientation)]
                    else:
                        self.direction = direction_dict[2+(2*self.orientation)]

                    total = self.totalUp + self.totalDown  # Calculate total
                    # Write new values to file
                    write_new_value(current_file, direction, total)

                last_totalUp = self.totalUp
                last_totalDown = self.totalDown

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    executor.shutdown(wait=False)
                    break

                # increment the total number of frames processed thus far and
                # then update the FPS counter
                self.totalFrames += 1
                self.fps.update()
                end_time_frame = time.time()
                print('frame_time: {}'.format(
                    end_time_frame - start_time_frame))

            # stop the timer and display FPS information
            self.fps.stop()
            print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
            print("Total Frames:{}".format(self.totalFrames))

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
    pc = PeopleCounter(prototxt="mobilenet_ssd/MobileNetSSD_deploy_py.prototxt",
                       model="mobilenet_ssd/MobileNetSSD_deploy_py.caffemodel", input="videos/test_video.mp4", skip_frames=10)

    pc.main_loop()
