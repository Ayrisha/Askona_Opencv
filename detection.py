import cv2
from fullface_image_detection import detect_image_fullface
from profile_image_detection import detect_image_profile

# Путь к фото в анфас
path_image_fullface = 'image/fullface.jpg'
# Путь к фото в профиль
path_image_profile = 'image/profile2.jpg'
# Путь к результату после обработки фото в анфас
path_result_fullface = 'image/result_fullface.jpg'
# Путь к результату после обработки фото в профиль
path_result_profile = 'image/result_profile.jpg'


if __name__ == '__main__':
    try:
        # Чтение входных фото
        image_fullface = cv2.imread(path_image_fullface)
        image_profile = cv2.imread(path_image_profile)

        # Обработка фотограйия
        result_fullface, human_height, head_width, pose_fullface, shldr_inclination = \
            detect_image_fullface(image_fullface)
        result_profile, pose_profile, neck_inclination, torso_inclination = detect_image_profile(image_profile)

        # Отображение результатов
        if pose_profile == 0:
            print("Bad pose profile")
        else:
            print("Good pose profile")

        if pose_fullface == 0:
            print("Bad pose fullface")
        else:
            print("Good pose fullface")

        print("Height: ", human_height)
        print("Head: ", head_width)
        print("Shoulder inclination: ", shldr_inclination)
        print("Neck inclination: ", neck_inclination)
        print("Torso inclination: ", torso_inclination)

        down_width = 20
        down_height = 10
        down_points = (down_width, down_height)
        cv2.resize(result_fullface, down_points, interpolation=cv2.INTER_LINEAR)
        cv2.resize(result_profile, down_points, interpolation=cv2.INTER_LINEAR)

        # Записование результатов
        cv2.imwrite(path_result_fullface, result_fullface)
        cv2.imwrite(path_result_profile, result_profile)
    except FileNotFoundError:
        raise FileNotFoundError
