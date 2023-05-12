import cv2
import mediapipe as mp
import math as m


def find_pose(image):
    """
    find_pose определение позы человека на фото

        Параметры:
            image: фото

        Возвращаемое значение:
            h: высота снимка
            w: ширина снимка
            lm: нормализованные координаты x и y ориентиров
            lmPose: один ориентир позы в обнаруженной позе
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
    find_angle нахождение угла наклона между точками

        Параметры:
            x1: координата x первой точки
            y1: координата y первой точки
            x2: координата x второй точки
            y2: координата y второй точки


        Возвращаемое значение:
            degree: угол отклонения
    """
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta

    return degree
