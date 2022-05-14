from typing import Tuple, Optional

from matplotlib import pyplot as plt
from skimage.morphology import binary_opening
import skimage.measure
import sys
import numpy as np
import cv2


def process(transformations, data: np.ndarray) -> np.ndarray:
    for transformation in transformations:
        data = transformation(data)
    return data


def find_paper_contours_and_polygon_mask(path_to_img: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
        Находит контуры бумаги и маску многоугольника

        :param path_to_img: путь к файлу с фотографией многоугольника и предметов
        :return: контур бумаги и маску многоугольника
    """

    def create_mask_from_contours(contours: np.ndarray) -> np.ndarray:
        """
            Создает маску на основе контура

            :param contours: контур
            :return: созданную маску, внутри контура заполненную значением False
        """
        bbox = cv2.boundingRect(contours)

        x_most_left, width, y_most_bottom, height = bbox[1], bbox[3], bbox[0], bbox[2]

        mask = np.full((width, height), True, dtype=bool)
        for y in range(y_most_bottom, y_most_bottom + height):
            for x in range(x_most_left, x_most_left + width):
                if cv2.pointPolygonTest(contours, (y, x), False) >= 0:
                    mask[x - x_most_left][y - y_most_bottom] = False

        return mask

    img = cv2.imread(path_to_img)

    assert img is not None, 'cannot read image'

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = [0, 0, 168]
    upper_white = [172, 111, 255]

    transformations = [
        lambda img: cv2.inRange(img, np.array(lower_white), np.array(upper_white)),
        lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))),
        lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50), (-1, -1)))
    ]

    img_hsv_transformed = process(transformations, img_hsv)

    contours, hierarchy = cv2.findContours(img_hsv_transformed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

    assert area_contours, 'area contours is empty'

    idx = area_contours.index(max(area_contours))

    assert good_contours, 'good contours is empty'
    assert 0 <= idx < len(good_contours), 'no paper contours'

    paper_contours = good_contours[idx]

    polygon_mask = create_mask_from_contours(polygon_contours)

    return paper_contours, polygon_mask


def find_object_masks_with_areas(path_to_img: str, paper_contours: np.ndarray) -> list[Tuple[np.ndarray, int]]:
    """
        Находит для каждого объекта на изображении маску и ее площадь

        :param path_to_img: путь к файлу с фотографией многоугольника и предметов
        :param paper_contours: контур листа бумаги
        :return: маски объектов и их площади
    """

    def get_objects_area(image: np.ndarray, paper_contours: np.ndarray) -> np.ndarray:
        """
            Находит область где располагаются предметы на изображении
            Область предметов - область на изображении справа от контура листа бумаги

            :param image: изображение с листом бумаги и предметов
            :param paper_contours: контур листа бумаги
            :return: область с предметами
        """
        margin = 10
        paper_area_right_max_x = max(paper_contours, key=lambda points: points[0][0])[0][0]
        objects_area_left_min_x = paper_area_right_max_x + margin

        return image[::, objects_area_left_min_x:, ::]

    def find_objects_common_mask(objects_area: np.ndarray) -> np.ndarray:
        """
            Находит маску всех предметов в области

            :param objects_area: область с предметами
            :return: маска всех предметов в области
        """
        img_hsv = cv2.cvtColor(objects_area, cv2.COLOR_BGR2HSV)

        lower_green = [36, 25, 25]
        upper_green = [86, 255, 255]

        transformations = [
            lambda img: cv2.inRange(img, np.array(lower_green), np.array(upper_green)),
            lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2), (-1, -1))),
            lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))),
            lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN,
                                         cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30), (-1, -1)))
        ]

        img_hsv_transformed = process(transformations, img_hsv)

        objects_common_mask = ~binary_opening(img_hsv_transformed, footprint=np.ones((20, 20)))

        return objects_common_mask

    def truncate_mask_and_find_area(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
            Урезает маску и находит ее площадь

            :param mask: маска площадь которой надо найти
            :return: урезанная маска и ее площадь
        """
        label = skimage.measure.label(mask)
        prop = skimage.measure.regionprops(label)[0]
        bbox = prop.bbox
        area = prop.area
        truncated_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        return truncated_mask, area

    def find_all_objects_masks_with_areas(all_objects_mask: np.ndarray) -> list[Tuple[np.ndarray, int]]:
        """
            Находит маску каждого предмета отдельно и ее площадь

            :param all_objects_mask: маска всех предметов
            :return: маски отдельных предметов и их площади
        """
        labels = skimage.measure.label(all_objects_mask)
        props = skimage.measure.regionprops(labels)
        areas = [prop.area for prop in props]
        min_object_area = 100

        objects_masks_with_areas = []
        for i in range(len(areas)):
            if areas[i] > min_object_area:
                # Так как нумерация связанных областей начинается с 1
                connected_region_val = i + 1
                mask = labels == connected_region_val
                truncated_mask, mask_area = truncate_mask_and_find_area(mask)
                objects_masks_with_areas.append((truncated_mask, mask_area))

        return objects_masks_with_areas

    img = cv2.imread(path_to_img)

    assert img is not None, 'cannot read image'

    objects_area = get_objects_area(img, paper_contours)

    plt.imshow(cv2.cvtColor(objects_area, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.show()

    objects_common_mask = find_objects_common_mask(objects_area)

    all_objects_masks_with_areas = find_all_objects_masks_with_areas(objects_common_mask)

    for mask_with_area in all_objects_masks_with_areas:
        mask, area = mask_with_area
        plt.imshow(mask)
        plt.gray()
        plt.show()

    return all_objects_masks_with_areas


def can_objects_fit_in_polygon(polygon_mask: np.ndarray, object_masks_with_areas: list[Tuple[np.ndarray, int]]) -> bool:
    """
        Проверяет можно ли расположить все предметы внутри заданного многоугольника

        :param polygon_mask: маска многоугольника в который нужно поместить предметы
        :param object_masks_with_areas: маски предметов с их площадями
        :return: True если можно иначе False
    """

    def can_object_fit_in_polygon(polygon_mask: np.ndarray, object_mask: np.ndarray) -> bool:
        """
            Проверяет можно ли расположить данный предмет внутри заданного многоугольника

            :param polygon_mask: маска многоугольника в который нужно поместить предметы
            :param object_mask: маска предмета с его площадью
            :return: True если можно иначе False
        """
        polygon_mask_height, polygon_mask_width = polygon_mask.shape
        object_mask_height, object_mask_width = object_mask.shape
        step = 10

        for y in range(0, polygon_mask_height - object_mask_height, step):
            for x in range(0, polygon_mask_width - object_mask_width, step):
                polygon_mask_cut = polygon_mask[y:y+object_mask_height, x:x+object_mask_width]

                # Области выходящие за маску многоугольника
                # Или накладывающиеся на уже расположенные объекты в многоугольнике
                bad_area = cv2.bitwise_and(polygon_mask_cut.astype(int), object_mask.astype(int))
                if np.sum(bad_area) == 0:
                    # Добавляем на маску многоугольника успешно расположенный объект
                    polygon_mask[y:y + object_mask_height, x:x + object_mask_width] = cv2.bitwise_xor(polygon_mask_cut.astype(int), object_mask.astype(int)).astype(bool)

                    plt.imshow(polygon_mask)
                    plt.show()

                    return True

        return False

    object_masks_with_areas.sort(key=lambda each_tuple: each_tuple[1], reverse=True)

    for object_mask, object_area in object_masks_with_areas:
        if not can_object_fit_in_polygon(polygon_mask, object_mask):
            return False

    return True
