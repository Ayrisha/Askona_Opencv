import cv2
import numpy
import background_detection as b
import body_posture as p
import colour
import yolov8_detection as yolov

# The way to the full-face photo
path_image_fullface = 'image/fullface.jpg'
# Real background size
background_size = 2700


def find_real_size(pixel_size, background_pixel_size):
    """
    find_real_size determines the actual size of the subject relative to the known background

        :argument
            pixel_size: number of pixels in the item
            background_pixel_size: number of pixels in the background

        :return
            item_size: item size
    """
    item_size = pixel_size * background_size / background_pixel_size
    return item_size


def detect_image_fullface(image):
    """
    detect_image_fullface processes a full-face photo

        :argument
            image: photo

        :returns
            human: processed photo
            human growth
            head size
            pose: 1, if the person's posture is correct; 0, if incorrect
            shldr_inclination: angle of deviation in the shoulders

    """
    human_pixel_size = 0
    pose = 0

    dominate_color_hsv = b.main_color_hsv(path_image_fullface)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_background = b.find_mask_background(dominate_color_hsv, image_hsv)
    background = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_background)

    background_pixel_size, upper, lower = b.find_background_pixel_size(background)

    human = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8)
    human[:, :] = colour.COLOR_LIGHT_BLUE

    h, w, lm, lmPose = p.find_pose(image)

    # Left shoulder
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    # Right shoulder
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    # Left ear
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

    # Right ear
    r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
    r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)

    human_head_size = l_ear_x - r_ear_x
    shldr_inclination = p.find_angle(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
    shldr_inclination = abs(90 - shldr_inclination)

    if shldr_inclination < 1:
        pose = 1

        cv2.line(human, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), colour.COLOR_GREEN, 4)

    else:
        pose = 0

        cv2.line(human, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), colour.COLOR_RED, 4)

    # Drawing lines
    cv2.circle(human, (l_ear_x, l_ear_y), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (r_ear_x, r_ear_y), 7, colour.COLOR_GREEN, -1)
    cv2.line(human, (l_ear_x, l_ear_y), (r_ear_x, r_ear_y), colour.COLOR_GREEN, 4)
    cv2.circle(human, (l_shldr_x, l_shldr_y), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (r_shldr_x, r_shldr_y), 7, colour.COLOR_GREEN, -1)

    boxes, classes, segmentations = yolov.yolov8_detect(image)

    # Drawing the outline of a person and the main points
    for box, class_id, seg in zip(boxes, classes, segmentations):
        (x, y, x2, y2) = box
        if class_id == 0:
            human_pixel_size = y2 - y
            cv2.polylines(human, [seg], True, colour.COLOR_BLUE, 4)
            x = x + (int(x2) - int(x)) / 2
            cv2.circle(human, (int(x), int(y)), 7, colour.COLOR_GREEN, -1)
            cv2.circle(human, (int(x), int(y2)), 7, colour.COLOR_GREEN, -1)
            cv2.line(human, (int(x), int(y)), (int(x), int(y2)), colour.COLOR_GREEN, 4)

    # cv2.line(human, (0, upper), (w, upper), COLOR_RED, 4)
    # cv2.line(human, (0, lower), (w, lower), COLOR_RED, 4)

    return human, round(find_real_size(human_pixel_size, background_pixel_size)), \
           round(find_real_size(human_head_size, background_pixel_size)), pose, round(shldr_inclination)
