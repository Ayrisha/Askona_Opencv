import cv2

import body_posture as b
import colour
import numpy
import yolov8_detection as yolov


def detect_image_profile(image):
    """
    detect_image_profile processes a profile photo

        :argument
            image: foto

        :returns
            human: processed photo
            pose: 1, if the person's posture is correct; 0, if incorrect
            neck_inclination: angle of inclination in the lower back
            torso_inclination: angle of inclination in the back

    """
    human = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8)
    human[:, :] = colour.COLOR_LIGHT_BLUE

    h, w, lm, lmPose = b.find_pose(image)

    # Left shoulder
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    # Right shoulder
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    # Left ear
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

    # Left hip
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    # Calculation of deflection angles
    neck_inclination = b.find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = b.find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Drawing points
    cv2.circle(human, (l_shldr_x, l_shldr_y), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (l_ear_x, l_ear_y), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (l_shldr_x, l_shldr_y - 100), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (r_shldr_x, r_shldr_y), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (l_hip_x, l_hip_y), 7, colour.COLOR_GREEN, -1)
    cv2.circle(human, (l_hip_x, l_hip_y - 100), 7, colour.COLOR_GREEN, -1)

    # Determining the correctness of the pose
    if neck_inclination < 20 and torso_inclination < 10:
        pose = 1

        cv2.line(human, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), colour.COLOR_GREEN, 4)
        cv2.line(human, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), colour.COLOR_GREEN, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), colour.COLOR_GREEN, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), colour.COLOR_GREEN, 4)

    else:
        pose = 0

        cv2.line(human, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), colour.COLOR_RED, 4)
        cv2.line(human, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), colour.COLOR_RED, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), colour.COLOR_RED, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), colour.COLOR_RED, 4)

    # Drawing the outline of a person
    boxes, classes, segmentations = yolov.yolov8_detect(image)
    for box, class_id, seg in zip(boxes, classes, segmentations):
        if class_id == 0:
            cv2.polylines(human, [seg], True, colour.COLOR_BLUE, 4)

    return human, pose, round(neck_inclination), round(torso_inclination)
