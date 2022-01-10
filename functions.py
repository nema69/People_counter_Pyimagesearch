def two_point_box_2_width_height_box(two_point_box):

    # Transform BB from left,bot,right,top to x,y,w,h
    left = two_point_box[0]
    bot = two_point_box[1]
    right = two_point_box[2]
    top = two_point_box[3]
    width_height_box = (left, bot, right - left, top - bot)
    return width_height_box


def width_height_box_2_two_point_box(width_height_box):

    # Transform BB from left,bot,right,top to x,y,w,h
    left = width_height_box[0]
    bot = width_height_box[1]
    w = width_height_box[2]
    h = width_height_box[3]
    two_point_box = (left, bot, left + w, bot + h)
    return two_point_box


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
