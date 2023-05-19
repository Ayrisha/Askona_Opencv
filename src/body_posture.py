import cv2
import mediapipe as mp
import math as m


def find_pose(image):
    """
    find_pose definition of the pose of the person in the photo

        :argument
            image: photo

        :returns
            h: height of the image
            w: width of the image
            lm: normalized x and y coordinates of landmarks
            lmPose: one pose reference in the detected pose
    """
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    keypoints = pose.process(image)
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return h, w, lm, lmPose


def find_angle(x1, y1, x2, y2):
    """
    find_angle finding the angle of inclination between points

        :argument
            x1: x coordinate of the first point
            y1: y coordinate of the first point
            x2: x coordinate of the second point
            y2: y coordinate of the second point


        :returns
            degree: deviation angle
    """
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta

    return degree
