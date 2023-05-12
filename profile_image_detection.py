from body_posture_detection import *
from colour import *
import numpy
from yolov8_detection import yolov8_detect


def detect_image_profile(image):
    """
    detect_image_profile обрабатывает фотографию в профиль

        Параметры:
            image: фото

        Возвращаемое значение:
            human: обработанная фотография
            pose: 1, если поза человека правильная; 2, если неправильная

    """
    human = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8)
    human[:, :] = COLOR_LIGHT_BLUE

    h, w, lm, lmPose = find_pose(image)

    # Левое плечо
    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    # Правое плечо
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    # Левый ухо
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

    # Левое бедро
    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    # Расчет углов отклонения
    neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Отрисовка точек
    cv2.circle(human, (l_shldr_x, l_shldr_y), 7, COLOR_GREEN, -1)
    cv2.circle(human, (l_ear_x, l_ear_y), 7, COLOR_GREEN, -1)
    cv2.circle(human, (l_shldr_x, l_shldr_y - 100), 7, COLOR_GREEN, -1)
    cv2.circle(human, (r_shldr_x, r_shldr_y), 7, COLOR_GREEN, -1)
    cv2.circle(human, (l_hip_x, l_hip_y), 7, COLOR_GREEN, -1)
    cv2.circle(human, (l_hip_x, l_hip_y - 100), 7, COLOR_GREEN, -1)

    # Определение правильности позы
    if neck_inclination < 20 and torso_inclination < 10:
        pose = 1

        cv2.line(human, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), COLOR_GREEN, 4)
        cv2.line(human, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), COLOR_GREEN, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), COLOR_GREEN, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), COLOR_GREEN, 4)

    else:
        pose = 0

        cv2.line(human, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), COLOR_RED, 4)
        cv2.line(human, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), COLOR_RED, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), COLOR_RED, 4)
        cv2.line(human, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), COLOR_RED, 4)

    # Отрисовка контура человека
    boxes, classes, segmentations = yolov8_detect(image)
    for box, class_id, seg in zip(boxes, classes, segmentations):
        if class_id == 0:
            cv2.polylines(human, [seg], True, COLOR_BLUE, 4)

    return human, pose
