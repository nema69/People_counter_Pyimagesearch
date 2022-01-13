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
from functions import bb_intersection_over_union
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
        self.roi = kwargs.get('roi', True)
        self.queue = kwargs.get('queue', True)
        self.frame_counts_up = kwargs.get('frame_counts_up', 8)
        # 0: Vertical   1: Horizontal
        self.orientation = kwargs.get('orientation', 1)
        self.offset_dist = kwargs.get('offset_dist', 2)
        self.border_dist = kwargs.get('border_dist', 50)
        self.webserver = kwargs.get('webserver', False)

    def preprocess_frame(self, longest_side):
        if self.roi:
            self.frame = self.frame[int(self.roi_coord[1]):int(
                self.roi_coord[1]+self.roi_coord[3]-1), int(self.roi_coord[0]):int(self.roi_coord[0]+self.roi_coord[2]-1)]

        h, w, _ = self.frame.shape
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

        self.frame = cv2.resize(
            self.frame, dim, interpolation=cv2.INTER_LINEAR)

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

        sharpened_frame = cv2.filter2D(self.frame, -1, kernel)

        return sharpened_frame

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

    def update_trackers_cv2(self):
        rects_ = []
        success_list_ = []

        for tracker_ in self.trackers:

            # get width height rect from tracker
            success_, rect_cv2 = tracker_.update(self.frame)
            # transform into 2 point rect
            two_point_rect = width_height_box_2_two_point_box(rect_cv2)
            # add the bounding box coordinates to the rectangles list
            rects_.append(two_point_rect)
            success_list_.append(success_)
        return rects_, success_list_

    def initialize_cv2_tracker(self, input_bb):
        # Transform BB from left,bot,right,top to x,y,w,h
        new_bb = two_point_box_2_width_height_box(input_bb)

        # Create new tracker
        tracker_ = cv2.TrackerKCF_create()

        # Init new tracker
        tracker_.init(self.frame, new_bb)

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

    def build_list_of_bounding_boxes(self):
        out_list_bounding_boxes = []
        out_list_labels = []
        for detection_no in np.arange(0, self.detections.shape[2]):
            # extract the confidence (i.e., probability) associated
            # with the prediction
            confidence = self.detections[0, 0, detection_no, 2]
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

    def draw_counting_lines(self, colour):
        (h, w) = self.frame.shape[:2]
        half_dist = self.offset_dist//2
        if not self.orientation:

            cv2.line(self.frame, (0, h // 2 + half_dist),
                     (w, h // 2 + half_dist), colour, 2)
            cv2.line(self.frame, (0, h // 2 - half_dist),
                     (w, h // 2 - half_dist), colour, 2)

        else:
            cv2.line(self.frame, (w // 2 + half_dist, 0),
                     (w // 2 + half_dist, h), colour, 2)
            cv2.line(self.frame, (w // 2 - half_dist, 0),
                     (w // 2 - half_dist, h), colour, 2)

    def draw_box_from_bb(self, bb):
        if bb != (0.0, 0.0, 0.0, 0.0):
            startpoint = int(bb[0]), int(bb[1])
            endpoint = int(bb[2]), int(bb[3])
            out_frame = cv2.rectangle(
                self.frame, startpoint, endpoint, (0, 0, 255), 2)

    def determine_direction(self, to, centroid, orientation):
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

    def check_if_count(self, to):
        # orientation = 1: objects are moving along the x axis
        if self.orientation:
            # calculate horizontal direction
            to.direction_horizontal()
            # direction < 0: objects are moving right (x getting more pos), centroid x position ([0][0]) is right of W//3 * 2
            if to.x_direction < 0 and to.centroids[-1][0] > self.W // 2 + self.offset_dist and self.person_dict[str(to.objectID)] > self.frame_counts_up:
                self.totalUp += 1
                to.counted = True

            # direction > 0: objects are moving left (x getting more neg), centroid x position ([0][0]) is left of W//3
            elif to.x_direction > 0 and to.centroids[-1][0] < self.W // 2 - self.offset_dist and self.person_dict[str(to.objectID)] > self.frame_counts_up:
                self.totalDown += 1
                to.counted = True

        else:
            if self.direction < 0 and self.centroid[0] < self.W // 2 - self.offset_dist and self.person_dict[str(self.objectID)] > self.frame_counts_up:
                self.totalUp += 1
                to.counted = True

            # if the direction is positive (indicating the object
            # is moving down) AND the centroid is below the
            # center line, count the object
            elif self.direction > 0 and self.centroid[0] > self.W // 2 + self.offset_dist:
                if self.person_dict[str(self.objectID)] > self.frame_counts_up:
                    self.totalDown += 1
                    to.counted = True

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

    def get_roi(self, command_queue, output_queue):
        scale = 0.25 if self.webserver else 0.5
        h, w, _ = self.frame.shape
        nh = int(h * scale)
        nw = int(w * scale)
        resized_frame = cv2.resize(self.frame, (nw, nh))
        if self.webserver:
            output_queue.put(resized_frame)
            while(True):
                try:
                    print("trying")
                    roi_scaled = command_queue.get(timeout=2)
                    if type(roi_scaled) is not tuple:
                        continue
                    print("done")
                    break
                except:
                    print("Waiting for user input...")
        else:
            roi_scaled = cv2.selectROI('ROI', resized_frame, False)
            cv2.destroyWindow('ROI')
        roi = tuple(x/scale for x in roi_scaled)

        print(roi)
        return roi

    def add_tracked_via_iou(self, tracker_bbs, detection_bbs, iou_threshold):
        if len(tracker_bbs) == 0:
            return detection_bbs
        if len(detection_bbs) == 0:
            return tracker_bbs
        new_bb_list = []
        for detection_bb in detection_bbs:
            for tracker_bb in tracker_bbs:
                if bb_intersection_over_union(tracker_bb, detection_bb) < 1 - iou_threshold:
                    new_bb_list.append(tracker_bb)
        if new_bb_list:
            detection_bbs.extend(new_bb_list)
            return detection_bbs
        return detection_bbs

    def init_videoparameters(self, command_queue, output_queue):
        # Get one frame to setup videoparameters
        _, self.frame = self.vs.read()

        # --- ROI
        if self.roi:
            self.roi_coord = self.get_roi(command_queue, output_queue)

        # --- If the frame dimensions are empty, set them
        self.preprocess_frame(400)
        (self.H, self.W) = self.frame.shape[:2]

        # --- If we are supposed to be writing a video to disk, initialize
        # the writer
        if self.output is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(self.output, fourcc, 30,
                                          (self.W, self.H), True)

    def check_tracker_valid(self, rec, h, w, border_dist, orientation_):
        if rec[0] < border_dist or rec[2] > w - border_dist and orientation_:
            return False
        if rec[1] < border_dist or rec[3] > h - border_dist and not orientation_:
            return False
        return True

    def main_loop(self, people_counter_command_queue, people_counter_output_queue):
        # --- Setup for Counting and Direction
        self.person_dict = dict()   # Store known persons

        direction_dict = {1: 'Up', 2: 'Down', 3: 'Right', 4: 'Left'}

        # --- Setup CSV file
        current_file = create_csv_file()
        print('Printing to: %s' % current_file)
        total = 0

        # --- Initialize the list of class labels MobileNet SSD was trained to detect
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        # ---

        # --- Load our serialized model from disk
        print("[INFO] Loading model...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # --- If no video path was supplied, grab a reference to the webcam
        if self.input is None:
            print("[INFO] starting video stream...")
            # self.vs = VideoStream(src=0).start()
            self.vs = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 !\
        nvvidconv ! video/x-raw, format=BGRx, width=640, height=360 ! videoconvert ! video/x-raw, format=BGR  ! appsink")
            time.sleep(2.0)

        else:   # Otherwise, grab a reference to a video file
            print("[INFO] opening video file...")
            self.vs = cv2.VideoCapture(self.input)

        # --- Instantiate our centroid tracker
        self.ct = CentroidTracker(
            maxDisappeared=self.skip_frames*2, maxDistance=50)
        self.trackers = []              # List to store each tracker
        self.trackableObjects = {}      # Map eache unique objectID to a trackable object

        # --- Counters
        self.totalFrames = 0    # Number of processed frames
        self.totalDown = 0      # Counter for objects that moved up
        self.totalUp = 0        # # Counter for objects that moved down

        self.H = None
        self.W = None
        self.roi_coord = None
        self.rects_for_iou = None

        self.draw_output = False if self.webserver else True

        # --- Start the frames per second throughput estimator
        self.fps = FPS().start()

        # --- Build queue for frames, normal queue for video read( preloads frames) and lifoqueue for "live" video"
        if self.queue:
            q = queue.Queue(maxsize=10)
            max_frame = 0

            # --- Init argument tuple for capture frame thread
            params = (self.input, q, 30)

        self.init_videoparameters(
            people_counter_command_queue, people_counter_output_queue)

        # --- Start thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Pack arguments with lamda
            if self.queue:
                f1 = executor.submit(lambda p: self.capture_frame(*p), params)

            # Loop over frames from the video stream
            while True:
                start_time_frame = time.time()
                web_output = False
                if self.webserver:
                    try:
                        if (people_counter_command_queue.get(timeout=0.0001) == "frame"):
                            web_output = True
                            self.draw_output = True
                    except:
                        self.draw_output = False

                # Grab next frame
                if self.queue:  # Get new frame from queue
                    frame_num, new_frame = q.get(timeout=1)

                    # If frame is none exit the loop
                    if new_frame is None:
                        break

                    # Check if new_frame isnt an earlier frame to prevent rubberbanding
                    if frame_num >= max_frame and new_frame is not None:
                        max_frame = frame_num
                        self.frame = new_frame
                        backup_frame = self.frame
                    # If it is an older frame reuse the most recent frame
                    else:
                        self.frame = backup_frame

                else:   # Get next frame frome VideoCapture
                    _, self.frame = self.vs.read()
                    # Exit when no frame left
                    if self.frame is None:
                        print("[WARNING] No frame left. Leaving the loop...")
                        break

                # Crop the frame to the roi, resize the frame to have a maximum size
                # and sharpen it for detection
                frame_detection = self.preprocess_frame(400)

                # Initialize the current status along with our list of bounding
                # box rectangles returned by either (1) our object detector or
                # (2) the correlation trackers
                status = "Waiting"
                rects = []
                # Check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if self.totalFrames % self.skip_frames == 0:
                    start_time_detection = time.time()
                    # Set the status and initialize our new set of object trackers
                    status = "Detecting"
                    self.trackers = []

                    # Run detections on frame and return all detected objects
                    self.detections = self.run_detection_on_frame(
                        frame_detection)

                    # Get relevant bounding boxes from detections
                    list_of_bounding_boxes, _ = self.build_list_of_bounding_boxes()

                    if self.rects_for_iou is not None:
                        list_of_bounding_boxes = self.add_tracked_via_iou(
                            self.rects_for_iou, list_of_bounding_boxes, 0.9)

                    for bounding_box in list_of_bounding_boxes:
                        # Start a tracker for each bounding box
                        tracker = self.initialize_cv2_tracker(bounding_box)
                        #tracker = initialize_dlib_tracker(frame, bounding_box)

                        # Add the tracker to our list of trackers so we can
                        # utilize it during skiped frames
                        self.trackers.append(tracker)

                        # Draw box around detections
                        if self.draw_output:
                            self.draw_box_from_bb(bounding_box)

                    end_time_detection = time.time()
                    print('Updating detection: {}'.format(
                        end_time_detection - start_time_detection))

                # Otherwise, we should utilize our object *trackers* rather than
                # object *detectors* to obtain a higher frame processing throughput
                else:
                    start_time_trackers = time.time()

                    # Update trackers
                    rects, success_list = self.update_trackers_cv2()
                    #rects = update_trackers_dlib(trackers, frame)

                    # Draw boxes around successfull tracks
                    if self.draw_output:
                        for count, rec in enumerate(rects):
                            success = success_list[count]

                            if success:
                                self.draw_box_from_bb(rec)
                            else:
                                print('Tracking Error')
                    end_time_trackers = time.time()
                    print('Updating trackers: {}'.format(
                        end_time_trackers - start_time_trackers))

                    if (self.totalFrames + 1) % self.skip_frames == 0:
                        # rects_for_iou = rects

                        self.rects_for_iou = [rec for rec in rects if
                                              success_list and self.check_tracker_valid(rec, self.H, self.W, self.border_dist, self.orientation)]

                # Draw the lines on the other side of which we will count
                if self.draw_output:
                    self.draw_counting_lines((255, 255, 255))

                # Use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids
                start_time_update_ct = time.time()
                objects = self.ct.update(rects)

                # Loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    # Check to see if a trackable object exists for the current
                    # object ID
                    to = self.trackableObjects.get(objectID, None)

                    # If there is no existing trackable object, create one
                    if to is None:
                        to = TrackableObject(objectID, centroid)

                    # Otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                    else:
                        # The difference between the y-coordinate of the *current*
                        # centroid and the mean of *previous* centroids will tell
                        # us in which direction the object is moving (negative for
                        # 'up' and positive for 'down')
                        if self.orientation:
                            direction = to.direction_horizontal()
                        else:
                            direction = to.direction_vertical()

                        to.centroids.append(centroid)

                        # Check to see if the object has been counted or not
                        if not to.counted:
                            # Check if we can count the object
                            self.check_if_count(to)

                    # Store the trackable object in our dictionary
                    self.trackableObjects[objectID] = to

                    # Draw both the ID of the object and the centroid of the
                    # object on the output frame
                    if str(objectID) in self.person_dict:
                        self.person_dict[str(objectID)] += 1
                    else:
                        self.person_dict[str(objectID)] = 0
                    if self.draw_output:
                        text = "ID {}".format(objectID)
                        cv2.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(
                            self.frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                # Construct a tuple of information we will be displaying on the
                # frame
                if self.draw_output:
                    info = [
                        (direction_dict[1+(2*self.orientation)], self.totalUp),
                        (direction_dict[2+(2*self.orientation)],
                         self.totalDown),
                        ("Status", status),
                    ]
                    # Loop over the info tuples and draw them on our frame
                    for (i, (k, v)) in enumerate(info):
                        text = "{}: {}".format(k, v)
                        cv2.putText(self.frame, text, (10, self.H - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Pass output to webserver
                    if web_output:
                        print("Sending output to webserver")
                        people_counter_output_queue.put(self.frame)
                    # Show the output frame
                    else:
                        cv2.imshow("People Counter", self.frame)

                # Check to see if we should write the frame to disk
                if self.output is not None:
                    self.writer.write(self.frame)

                key = cv2.waitKey(1) & 0xFF
                end_time_update_ct = time.time()
                print('Updating ct and counting: {}'.format(
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

                # If the `q` key was pressed, break from the loop
                if key == ord("q"):
                    executor.shutdown(wait=False)
                    break

                # Increment the total number of frames processed thus far and
                # then update the FPS counter
                self.totalFrames += 1
                self.fps.update()
                end_time_frame = time.time()
                print('frame_time: {}'.format(
                    end_time_frame - start_time_frame))

            # Stop the timer and display FPS information
            self.fps.stop()
            print("[INFO] Elapsed time: {:.2f}".format(self.fps.elapsed()))
            print("[INFO] Approx. FPS: {:.2f}".format(self.fps.fps()))
            print("Total Frames:{}".format(self.totalFrames))

            # Check to see if we need to release the video writer pointer
            if self.output is not None:
                self.writer.release()

            # If we are not using a video file, stop the camera video stream
            if self.input is None:
                self.vs.stop()

            # Otherwise, release the video file pointer
            else:
                self.vs.release()

            # Close any open windows
            cv2.destroyAllWindows()


if __name__ == "__main__":
    pc = PeopleCounter(prototxt="mobilenet_ssd/MobileNetSSD_deploy_py.prototxt",
                       model="mobilenet_ssd/MobileNetSSD_deploy_py.caffemodel", input="videos/test_video.mp4", skip_frames=10, queue=False, roi=True)

    pc.main_loop(None, None)
