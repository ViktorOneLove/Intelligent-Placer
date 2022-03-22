import lib_impl as lib


def check_image(path_to_img: str) -> bool:
    polygon_contours, paper_contours = lib.find_paper_and_polygon_contours(path_to_img)

    assert polygon_contours is not None
    assert paper_contours is not None

    objects_contours = lib.find_objects_contours(path_to_img, paper_contours)

    assert objects_contours is not None

    return lib.can_objects_fit_in_polygon(polygon_contours, objects_contours)
