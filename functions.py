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
