import cv2
from fullface_image_detection import detect_image_fullface
from profile_image_detection import detect_image_profile

# Путь к фото в анфас
path_image_fullface = 'image/fullface.jpg'
# Путь к фото в профиль
path_image_profile = 'image/profile.jpeg'
# Путь к результату после обработки фото в анфас
path_result_fullface = 'image/result_fullface.jpg'
# Путь к результату после обработки фото в профиль
path_result_profile = 'image/result_profile.jpg'


if __name__ == '__main__':
    # Чтение входных фото
    image_fullface = cv2.imread(path_image_fullface)
    image_profile = cv2.imread(path_image_profile)

    # Обработка фотограйия
    result_fullface, human_height, head_size = detect_image_fullface(image_fullface)
    result_profile, pose_mark = detect_image_profile(image_profile)

    # Отображение результатов
    if pose_mark == 0:
        print("Bad pose")
    else:
        print("Good pose")

    print(human_height)
    print(head_size)

    # Записование результатов
    cv2.imwrite(path_result_fullface, result_fullface)
    cv2.imwrite(path_result_profile, result_profile)
