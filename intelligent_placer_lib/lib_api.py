import matplotlib.pyplot as plt

import lib_impl as lib


def check_image(path_to_img: str) -> bool:
    paper_contours, polygon_mask = lib.find_paper_contours_and_polygon_mask(path_to_img)

    assert paper_contours is not None, 'cannot find paper'
    assert polygon_mask is not None, 'cannot find polygon on paper'

    plt.imshow(polygon_mask)
    plt.show()

    object_masks_with_areas = lib.find_object_masks_with_areas(path_to_img, paper_contours)

    assert object_masks_with_areas, 'cannot recognize objects'

    return lib.can_objects_fit_in_polygon(polygon_mask, object_masks_with_areas)
