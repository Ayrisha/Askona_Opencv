import cv2
from src import fullface_detection as f
import profile_detection as p


# Path to full-face photo
path_image_fullface = 'image/fullface.jpg'
# Path to profile photo
path_image_profile = 'image/profile.jpeg'
# Path to the result after processing the full-face photo
path_result_fullface = 'image/result_fullface.jpg'
# Path to the result after processing the photo in the profile
path_result_profile = 'image/result_profile.jpg'


if __name__ == '__main__':
    """
    The detection package helps to process a photo of a person in profile and full-face and get human parameters.

    Author: Kuchinskaya Arina(https://github.com/Ayrisha)
    """
    try:
        # Reading input photos
        image_fullface = cv2.imread(path_image_fullface)
        image_profile = cv2.imread(path_image_profile)

        # Photo processing
        result_fullface, human_height, head_width, pose_fullface, shldr_inclination = \
            f.detect_image_fullface(image_fullface)
        result_profile, pose_profile, neck_inclination, torso_inclination = p.detect_image_profile(image_profile)

        # Displaying results
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

        # Recording results
        cv2.imwrite(path_result_fullface, result_fullface)
        cv2.imwrite(path_result_profile, result_profile)
    except FileNotFoundError:
        raise FileNotFoundError
