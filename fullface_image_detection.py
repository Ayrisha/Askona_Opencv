from background_detection import *
from body_posture_detection import *
from colour import *
from yolov8_detection import *

# Путь к фото в анфас
path_image_fullface = 'image/fullface.jpg'
# Реальный размер фона
background_size = 2500


def find_real_size(pixel_size, background_pixel_size):
    """
    find_real_size определяет реальный размера предмета относительно известного фона

        Параметры:
            pixel_size: кол-во пикселей в предмете
            background_pixel_size: кол-во пикселей в фоне

        Возвращаемое значение:
            размер предмета
    """
    return pixel_size * background_size / background_pixel_size


def detect_image_fullface(image):
    """
    detect_image_fullface обрабатывает фотографию в анфас

        Параметры:
            image: фото

        Возвращаемое значение:
            human: обработанная фотография
            рост человека
            размер головы
            pose: 1, если осанка человека правильная; 2, если неправильная
            shldr_inclination: угол отклонения в плечах

    """
    human_pixel_size = 0
    pose = 0

    dominate_color_hsv = main_color_hsv(path_image_fullface)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_background = find_mask_background(dominate_color_hsv, image_hsv)
    background = cv2.bitwise_and(image_hsv, image_hsv, mask=mask_background)

    background_pixel_size, upper, lower = find_background_pixel_size(background)

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

    # Правое ухо
    r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w)
    r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h)

    human_head_size = l_ear_x - r_ear_x
    shldr_inclination = find_angle(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
    shldr_inclination = abs(90 - shldr_inclination)

    if shldr_inclination < 1:
        pose = 1

        cv2.line(human, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), COLOR_GREEN, 4)

    else:
        pose = 0

        cv2.line(human, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), COLOR_RED, 4)

    # Отрисовка линий
    cv2.circle(human, (l_ear_x, l_ear_y), 7, COLOR_GREEN, -1)
    cv2.circle(human, (r_ear_x, r_ear_y), 7, COLOR_GREEN, -1)
    cv2.line(human, (l_ear_x, l_ear_y), (r_ear_x, r_ear_y), COLOR_GREEN, 4)
    cv2.circle(human, (l_shldr_x, l_shldr_y), 7, COLOR_GREEN, -1)
    cv2.circle(human, (r_shldr_x, r_shldr_y), 7, COLOR_GREEN, -1)

    boxes, classes, segmentations = yolov8_detect(image)

    # Отрисовка контура человека и основных точек
    for box, class_id, seg in zip(boxes, classes, segmentations):
        (x, y, x2, y2) = box
        if class_id == 0:
            human_pixel_size = y2 - y
            cv2.polylines(human, [seg], True, COLOR_BLUE, 4)
            x = x + (int(x2) - int(x)) / 2
            cv2.circle(human, (int(x), int(y)), 7, COLOR_GREEN, -1)
            cv2.circle(human, (int(x), int(y2)), 7, COLOR_GREEN, -1)
            cv2.line(human, (int(x), int(y)), (int(x), int(y2)), COLOR_GREEN, 4)

    return human, round(find_real_size(human_pixel_size, background_pixel_size)), \
           round(find_real_size(human_head_size, background_pixel_size)), pose, shldr_inclination
