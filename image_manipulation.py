"""This file contains functions for working with images."""


import cv2 as cv
import numpy as np
import cv2.typing
from typing import Union, Literal, Optional
from icecream import ic


# def _color_to_decimal(
#         color: str
#         ) -> int:
#     """Convert a color in #RGB format to its decimal form.
# 
#     Parameters
#     ----------
#     color : str
#         The color in #RGB or #RGBA format (e.g., '#FF0045' for #RGB or
#         '#FF0045FF' for #RGBA) that will be converted.
# 
#     Returns
#     -------
#     int
#         The color written in decimal form.
# 
#     Raises
#     ------
#     ValueError
#         If the color is not in the right format (such as uneven number of
#         characters (excluding '#')).
#     """
#     unconverted_color = color
#     color = color.lower().strip().lstrip("#")
#     if len(color) % 2 != 0:
#         raise ValueError(f"{unconverted_color} is not in the right format!")
# 
#     result = 0
# 
#     for begin_char_index in range(0, len(color), 2):
#         hexadecimal = "0x" + color[begin_char_index: begin_char_index + 2]
#         multiplier = 256 ** (((len(color) - 2) - begin_char_index) / 2)
#         result += int(hexadecimal, 0) * multiplier
# 
#     return result
# 
# 
# def _image_to_decimal(
#         image: cv2.typing.MatLike
#         ) -> np.ndarray:
#     """Convert an image of shape (y, x, n) to (y, x).
# 
#     Parameters
#     ----------
#     image : cv2.typing.MatLike
#         The image to be converted to decimal.
# 
#     Returns
#     -------
#     np.ndarray
#         The image in its decimal form.
#     """
#     n_channels = image.shape[-1]
# 
#     image = sum([
#         image[:, :, channel_num] * (256 ** (n_channels - channel_num - 1))
#         for channel_num in range(n_channels)
#     ])
#     return image


def find_boundary_coords(
        image: cv2.typing.MatLike
        ) -> tuple[int, int, int, int]:
    """Find boundaries of an image according to the color provided.

    Boundaries are first and last x and y coordinates with a different color
    than the color in the first pixel (the top-left corner)).

    Parameters
    ----------
    image : cv2.typing.MatLike
        The image whose boundaries are to be found.

    Returns
    -------
    x1 : int
        The first x coordinate of the boundary.
    y1 : int
        The first y coordinate of the boundary.
    x2 : int
        The second x coordinate of the boundary.
    y2 : int
        The second y coordinate of the boundary.

    Raises
    ------
    ValueError
        If there is no boundary in the image (corner pixels have different
        values).
    """
    # TODO: add the option to pick a color
    # verify there is a boundary
    top_left_corner = image[0][0]
    top_right_corner = image[0][-1]
    bottom_left_corner = image[-1][0]
    bottom_right_corner = image[-1][-1]
    corners = [
        top_left_corner,
        top_right_corner,
        bottom_left_corner,
        bottom_right_corner
    ]
    if not (np.array(corners) == corners[0]).all():
        raise ValueError(
            f"since the corner pixels have different values, "
            f"there is no boundary"
        )

    color = top_left_corner


    # if RGBA
    if image.shape[-1] == 4:
        # transparent pixels cannot create boundary
        # (since they are transparent)
        additional_condition = image[:, :, -1] == 0
        image[additional_condition == True] = [0, 0, 0, 0]

    min_col = np.argmax(image != color, axis=1)

    # if len(min_col[min_col > 0]) == 0:
    #     raise ValueError(
    #         f"color {unconverted_color} was not found in the image"
    #     )

    x1 = min(min_col[min_col > 0])

    min_row = np.argmax(image != color, axis=0)
    y1 = min(min_row[min_row > 0])

    # while computing x2 and y2 we subtract 1, because of offset
    # (if subtrahend is 0,
    # we might a value greater than the coords of the last pixels)
    horizontally_flipped_image = np.flip(image, axis=1)
    max_col = np.argmax(horizontally_flipped_image != color, axis=1)
    x2 = (image.shape[1] - 1) - min(max_col[max_col > 0])

    vertically_flipped_image = np.flip(image, axis=0)
    max_row = np.argmax(vertically_flipped_image != color, axis=0)
    y2 = (image.shape[0] - 1) - min(max_row[max_row > 0])

    return x1, y1, x2, y2


def crop_image(
        image: cv2.typing.MatLike,
        x1: int,
        y1: int,
        x2: int,
        y2: int
        ) -> cv2.typing.MatLike:
    """Crop an image so that it contains only pixels in range.

    The range is defined by (x1, x2) and (y1, y2).

    Parameters
    ----------
    image : cv2.typing.MatLike
        The image to be cropped.
    x1 : int
        The first x coordinate in the cropped image.
    y1 : int
        The first y coordinate in the cropped image.
    x2 : int
        The last x coordinate in the cropped image.
    y2 : int
        The last y coordinate in the cropped image.

    Returns
    -------
    cv2.typing.MatLike
        The cropped image.

    Raises
    ------
    ValueError
        If the coordinate arguments are given in the wrong order or one of the
        coordinate arguments is greater than dimension size of the image.
    """
    if x1 > x2:
        raise ValueError(f"x1({x1}) should be lower or equal than x2({x2})")

    if y1 > y2:
        raise ValueError(f"y1({y1}) should be lower or equal than y2({y2})")

    if x2 > image.shape[1]:
        raise ValueError(
            f"x2({x2}) cannot be greater than width({image.shape[1]})"
        )

    if y2 > image.shape[0]:
        raise ValueError(
            f"y2({y2}) cannot be greater than width({image.shape[0]})"
        )

    return image[y1:y2, x1:x2]


def crop_background(
        image: cv2.typing.MatLike
        ) -> cv2.typing.MatLike:
    """Crop background of an image.

    Parameters
    ----------
    image : cv2.typing.MatLike
        The image, whose background is to be cropped.

    Returns
    -------
    cv2.typing.MatLike
        The image with cropped background.
    """
    boundary_coords = find_boundary_coords(image)
    cropped_image = crop_image(
        image,
        boundary_coords[0],
        boundary_coords[1],
        boundary_coords[2] + 1,
        boundary_coords[3] + 1
    )

    return cropped_image


def pad_image(
        image: cv2.typing.MatLike,
        n: int,
        axis: Literal["col", "row"],
        addition_side: Literal["after", "before"]
        ) -> cv2.typing.MatLike:
    """Pad an image by n first/last rows/columns added to a side.

    Parameters
    ----------
    image : cv2.typing.MatLike
        The image that is to be padded.
    n : int
        The number of rows or columns to be added.
    axis : Literal["col", "row"]
        If axis equals 'col', then a column is added, if axis equals "row",
        then a row is added.
    addition_side : Literal["after", "before"]
        If addition_side equals 'after', the row / column is added to the
        right / bottom.  If addition_side equals 'before', the row / column is
        added to the left / top.

    Returns
    -------
    cv2.typing.MatLike
        The padded image.

    Raises
    ------
    ValueError
        If the addition_side is neither 'after', nor 'before'.
    """

    def get_to_pad(
            image: cv2.typing.MatLike,
            axis: Literal["col", "row"],
            side: Literal["after", "before"]
            ) -> cv2.typing.MatLike:
        """Get row/column that is about to be added on the side of an image.

        Parameters
        ----------
        image : cv2.typing.MatLike
            The image whose column/row is obtained.
        axis : Literal["col", "row"]
            The axis to be padded. If axis equals 'col', first / last column is
            obtained. If axis equals 'row', first / last row is obtained.
        side : Literal["after", "before"]
            The side to be padded. If axis equals 'after', last column / row is
            obtained. If axis equals 'before', first column / row is obtained.

        Returns
        -------
        cv2.typing.MatLike
            The part of image to be padded to the original.
        """
        if axis == "col" and side == "before":
            return image[:, 0:1]
        elif axis == "col" and side == "after":
            return image[:, -1:]
        elif axis == "row" and side == "before":
            return image[0:1, :]
        elif axis == "row" and side == "after":
            return image[-1:, :]

    if n == 0:
        return image

    to_pad = get_to_pad(image, axis, addition_side)

    axis_position = {"row": 0, "col": 1}[axis]
    to_pad = np.concatenate([to_pad]*n, axis=axis_position)

    if addition_side == "before":
        return np.concatenate([to_pad, image], axis=axis_position)
    elif addition_side == "after":
        return np.concatenate([image, to_pad], axis=axis_position)
    else:
        raise ValueError(
            f"The addition_side({addition_side}) is not valid. It must be"
            "either 'before' or 'after'."
        )


def increase_size(
        image: cv2.typing.MatLike,
        size: tuple[Union[int, None], Union[int, None]],
        move_x: int = 0,
        move_y: int = 0,
        bboxes: Optional[np.ndarray] = None
        ) -> cv2.typing.MatLike:
    """Increase the size of an image by padding to a new size.

    Parameters
    ----------
    image : v2.typing.MatLike
        The image whose size will be increased.
    size : tuple[int, int]
        The height and width of a new image (if None, then the dimension will
        not change). If both the height and the width are None, the padded
        image will be square-shaped with the dimension size of the square being the largest dimension size.
    move_x : int, optional
        Determines by how many pixels to the right the original image is to be
        moved. If move_x > 0, it will be moved to the right. If move_x < 0, it
        will be moved to the left.
    move_y : int, optional
        Determines by how many pixels to the top the original image is to be
        moved. If move_x > 0, it will be moved to the top. If move_x < 0, it
        will be moved to the bottom.

    Returns
    -------
    cv2.typing.MatLike
        The padded image with an increased size.

    Raises
    ------
    ValueError
        If move_x or move_y is too small or too large.
    """

    def move_bounding_boxes() -> None:
        bboxes[:, :, 0] *= current_width
        bboxes[:, :, 1] *= current_height

        bboxes[:, :, 0] += to_add_left
        bboxes[:, :, 1] += to_add_up

        bboxes[:, :, 0] /= new_width
        bboxes[:, :, 1] /= new_height

    current_height = image.shape[0]
    current_width = image.shape[1]

    new_height = size[0]
    new_width = size[1]

    if new_height is None and new_width is None:
            new_width = max(current_width, current_height)
            new_height = max(current_width, current_height)
    if new_height is None:
        new_height = current_height
    if new_width is None:
        new_width = current_width

    difference_x = new_width - current_width
    difference_y = new_height - current_height

    to_add_up = difference_y // 2 + difference_y % 2 - move_y
    to_add_down = difference_y // 2 + move_y
    to_add_left = difference_x // 2 + difference_x % 2 + move_x
    to_add_right = difference_x // 2 - move_x

    # check boundaries
    if to_add_up < 0:
        raise ValueError(f"move_y({move_y}) is too large")
    if to_add_down < 0:
        raise ValueError(f"move_y({move_y}) is too small")
    if to_add_left < 0:
        raise ValueError(f"move_x({move_x}) is too small")
    if to_add_right < 0:
        raise ValueError(f"move_x({move_x}) is too large")

    if bboxes is not None:
        move_bounding_boxes()

    if new_height > current_height:
        image = pad_image(image, to_add_up, "row", "before")
        image = pad_image(image, to_add_down, "row", "after")
    if new_width > current_width:
        image = pad_image(image, to_add_left, "col", "before")
        image = pad_image(image, to_add_right, "col", "after")

    return image


def downscale_if_exceeds(
        image: cv2.typing.MatLike, max_size: int,
        ) -> cv2.typing.MatLike:
    """Downscale image if one of its dimensions exceeds max_size.

    The image is downscaled so that the dimension of the highest value will
    have value of max_size. Aspect ratio of the original image is kept.

    Parameters
    ----------
    image : cv2.typing.MatLike
        The image to be downscaled.
    max_size : int
        The largest dimension size after the downscale.

    Returns
    -------
    cv2.typing.MatLike
        The downscaled version of the image.
    """
    height = image.shape[0]
    width = image.shape[1]
    max_dimension = max(height, width)
    if max_dimension > max_size:  # if exceeds
        min_dimension = min(height, width)
        resizer = max_dimension / max_size  # aspect ratio
        max_dimension = max_size
        min_dimension = round(min_dimension / resizer)
        if height > width:
            dsize = (min_dimension, max_dimension)
        else:
            dsize = (max_dimension, min_dimension)
        image = cv.resize(image, dsize=dsize, interpolation=cv.INTER_LANCZOS4)
    return image

def convert_aspect_ratio():
    raise NotImplementedError

if __name__ == "__main__":
    image = cv.imread("test_files/shapes.png")
    if image is None:
        raise FileNotFoundError

    x1, y1, x2, y2 = find_boundary_coords(image)

    # remove background
    image = crop_image(image, x1, y1, x2, y2)

    cv2.imwrite("test_files/shapes_no_background.png", image)

    # Photo by Frederik Sørensen:
    # https://www.pexels.com/photo/photo-of-new-york-city-cityscape-2404843/
    image = cv.imread("test_files/city_frederik_soerensen.jpg")
    if image is None:
        raise FileNotFoundError

    # generate rest of the image by copying lines / columns
    image = increase_size(image, (625, 625))

    cv2.imwrite("test_files/city_frederik_soerensen_square.jpg", image)

    image = cv.imread("test_files/city_frederik_soerensen.jpg")
    if image is None:
        raise FileNotFoundError

    # generate upper lines of the image
    image = increase_size(image, (700, None), move_y=-37)

    cv2.imwrite("test_files/city_frederik_soerensen_top.jpg", image)
