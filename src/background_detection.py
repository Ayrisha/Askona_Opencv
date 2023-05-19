import colorsys
import numpy
import cv2
from colorthief import ColorThief

# Color sensitivity for finding the dominant color
sensitivity = 15


def main_color_hsv(filename):
    """
    main_color_hsv determines the dominant color in the photo

        :argument
            filename: path to the photo

        :returns
            dominate_color_hsv: HSV code of the dominant color
    """
    color_thief = ColorThief(filename)
    color = color_thief.get_color(quality=1)
    color_hsv = colorsys.rgb_to_hsv(color[0] / 255, color[1] / 255, color[2] / 255)
    dominate_color_hsv = int(color_hsv[0] * 179), int(color_hsv[1] * 255), int(color_hsv[2] * 255)
    return dominate_color_hsv


def find_mask_background(dominate_color_hsv, image_hsv):
    """
    find_mask_background finds the background mask

        :argument
            dominate_color_hsv: dominant color in HSV
            image_hsv: photo in HSV

        :returns
            photo with background mask
    """
    m_color = []
    p_color = []
    j = 0

    for color_hsv in dominate_color_hsv:
        if color_hsv > sensitivity:
            m_color.append(color_hsv - sensitivity)
        else:
            m_color.append(color_hsv)

        if j == 0:
            if color_hsv + sensitivity < 180:
                p_color.append(color_hsv + sensitivity)
            else:
                p_color.append(color_hsv)
        else:
            if color_hsv + sensitivity < 256:
                p_color.append(color_hsv + sensitivity)
            else:
                p_color.append(color_hsv)
        j = j + 1

    lower_color = numpy.array(
        (m_color[0], m_color[1], m_color[2]))
    upper_color = numpy.array(
        (p_color[0], p_color[1], p_color[2]))
    return cv2.inRange(image_hsv, lower_color, upper_color)


def find_background_pixel_size(background):
    """
    find_background_pixel_size finds the number of pixels in the background

        :argument
            background: background mask

        :returns
            distinction: number of background pixels
            upper: the highest point of the background
            lower: the lowest point of the background
    """
    upper = background.shape[0]
    lower = 0
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    ret, background_thresh = cv2.threshold(background_gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(background_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for value in contours:
        x, y, w, h = cv2.boundingRect(value)
        if y > lower:
            lower = y
        if y < upper:
            upper = y

    distinction = lower - upper

    return distinction, upper, lower
