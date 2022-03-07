import os.path
from typing import Tuple, Optional

import numpy as np
import cv2

"""
    План дальнейшей работы:
    1) Распознавать границы бумаги и многоугольника - DONE
    2) Распознавать границы объектов - IN PROCESS - пробую разные решения
    3) Алгоритм возможности расположения объектов внутри многоугольника - TO IMPLEMENT
        3.1) Простой вариант - параллельный перенос
        3.2) Более сложный вариант - алгоритм позволяющий каждым объектом манипулировать по-отдельности, вращать объекты
"""


def find_paper_and_polygon_contours(folder: str, file_name: str, path_to_save: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    image_path = os.path.join(folder, file_name)
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = [0, 0, 168]
    upper_white = [172, 111, 255]

    filter = cv2.inRange(img_hsv, np.array(lower_white), np.array(upper_white))
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    filter = cv2.morphologyEx(filter, cv2.MORPH_CLOSE, st)
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50), (-1, -1))
    filter = cv2.morphologyEx(filter, cv2.MORPH_OPEN, st)

    contours, hierarchy = cv2.findContours(filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    good_contours = []
    cx = []
    cy = []
    polygon_contours = None
    paper_contours = None
    area_contours = []

    min_contour_area = 12_000
    prox_measure = 20

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area > min_contour_area:
            area_contours.append(contour_area)
            good_contours.append(cnt)
            M = cv2.moments(cnt)
            cx.append(int(M['m10'] / (M['m00'] + 1e-5)))
            cy.append(int(M['m01'] / (M['m00'] + 1e-5)))

    for i in range(len(good_contours) - 1):
        if np.linalg.norm([cx[i] - cx[i + 1], cy[i] - cy[i + 1]]) < prox_measure:
            polygon_contours = good_contours[i]
            good_contours.pop(i + 1)
            good_contours.pop(i)

    idx = area_contours.index(max(area_contours))
    paper_contours = good_contours[idx]

    cv2.drawContours(img, [paper_contours], 0, (255, 0, 0), 8)
    cv2.drawContours(img, [polygon_contours], 0, (0, 255, 0), 8)
    cv2.imwrite(os.path.join(path_to_save, file_name), img)

    return polygon_contours, paper_contours


def find_objects_contours(image_path: str, paper_contours: np.ndarray) -> Optional[np.ndarray]:
    def get_objects_area(image: np.ndarray, paper_contours: np.ndarray) -> np.ndarray:
        margin = 10
        paper_area_right_max_x = max(paper_contours, key=lambda points: points[0][0])[0][0]
        objects_area_left_min_x = paper_area_right_max_x + margin
        return image[objects_area_left_min_x:, ::, ::]

    img = cv2.imread(image_path)
    objects_area = get_objects_area(img, paper_contours)

    return None


def can_objects_fit_in_polygon(polygon_contours: np.ndarray, objects_contours: np.ndarray) -> bool:
    return False
