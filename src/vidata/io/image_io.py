import imageio.v3 as iio
import numpy as np

from vidata.registry import register_loader, register_writer


@register_loader("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")
@register_loader("mask", ".png", ".bmp", backend="imageio")
def load_image(file: str):
    data = iio.imread(file)  # automatically handles RGB, grayscale, masks
    return data, {}


@register_writer("image", ".png", ".jpg", ".jpeg", ".bmp", backend="imageio")
@register_writer("mask", ".png", ".bmp", backend="imageio")
def save_image(data: np.ndarray, file: str):
    iio.imwrite(file, data)


# @register_loader("image", ".png", ".jpg", ".jpeg", ".bmp")
# def load_rgb(file: str):
#     """Loads an image from file using OpenCV.
#
#     Args:
#         file (Union[str, bytes]): Path to the image file.
#
#     Returns:
#         np.ndarray: Loaded image as a NumPy array.
#     """
#     data = cv2.imread(file)
#     data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
#
#     return data, None
#
# # TODO How to handle RGB vs Grayscale data
# @register_writer("image", ".png", ".jpg", ".jpeg", ".bmp")
# def save_rgb(data: np.ndarray, file: str):
#     """Saves an image to file using OpenCV.
#
#     Args:
#         data (np.ndarray): Image array to save.
#         file (Union[str, bytes]): Destination file path.
#     """
#     data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(file, data)
#
#
# @register_loader("mask", ".png", ".bmp")
# def load_mask(file: str):
#     """Loads an image from file using OpenCV.
#
#     Args:
#         file (Union[str, bytes]): Path to the image file.
#
#     Returns:
#         np.ndarray: Loaded image as a NumPy array.
#     """
#     data = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#     return data, None
#
#
# @register_writer("image", ".png", ".bmp")
# def save_mask(data: np.ndarray, file: str):
#     """Saves an image to file using OpenCV.
#
#     Args:
#         data (np.ndarray): Image array to save.
#         file (Union[str, bytes]): Destination file path.
#     """
#     cv2.imwrite(file, data)
