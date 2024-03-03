from random import choice
from os import listdir
from scipy import ndimage as nd
from PIL import Image
import numpy as np


def import_images(path: str = "Generate_Noise", filename: bool = 1, find: str = None) -> (Image, str):
    filename = "default_images" if not filename else "result_image"
    image_name = None
    if find is None:
        image_name = choice(listdir(f"../{path}/" + filename + "/"))
    else:
        for i in listdir(f"../{path}/" + filename + "/"):
            if find in i.split(sep="_"):
                image_name = i
                break
        if image_name is None:
            image_name = choice(listdir(f"../{path}/" + filename + "/"))
    name_img = ".".join(image_name.split(sep=".")[:-1])
    print(f"Image name: {name_img}")
    img = Image.open(f"../{path}/" + filename + "/" + image_name)
    return img, name_img


class Filters:
    def __init__(self, path: str = "Generate_Noise", image: Image = None, r: int = 0) -> None:
        image, self.filename = (image, None) if image is not None else import_images(path)
        self.w, self.h = image.size
        self.mode = image.mode
        self.image = np.asarray(image)
        self.r = r

    def linear_filter(self) -> np.ndarray:
        figm_orig = np.zeros_like(self.image)

        def apply_window(channel_idx):
            filtered_image = np.zeros((self.h, self.w))

            for i in range(self.h):
                for j in range(self.w):
                    window = get_window(i, j, channel_idx)
                    filtered_image[i, j] = np.mean(window)
            return filtered_image

        def get_window(i, j, channel_idx) -> np.ndarray:
            i_min = max(0, i - self.r)
            i_max = min(self.h - 1, i + self.r)
            j_min = max(0, j - self.r)
            j_max = min(self.w - 1, j + self.r)
            window = self.image[i_min:i_max + 1, j_min:j_max + 1, channel_idx]
            return window

        for k in range(len(self.mode)):
            figm_orig[:, :, k] = apply_window(k)
        return figm_orig

    def gaussian_filter(self) -> Image:
        figm_orig = np.ndarray(self.image.shape, dtype=self.image.dtype)
        for k in range(len(self.mode)):
            figm_orig[0:self.h, 0:self.w, k] = nd.gaussian_filter(self.image[0:self.h, 0:self.w, k], sigma=self.r)
        return figm_orig

    def median_filter(self) -> np.ndarray:
        figm_orig = np.zeros_like(self.image)
        for k in range(len(self.mode)):
            figm_orig[0:self.h, 0:self.w, k] = nd.median_filter(self.image[0:self.h, 0:self.w, k], size=self.r * 2 + 1)

        # def apply_window(channel_idx):
        #     filtered_image = np.zeros((self.h, self.w))
        #     padded_array = np.pad(self.image, pad_width=((self.r, self.r), (self.r, self.r), (0, 0)), mode='reflect')
        #     for i in range(self.h):
        #         for j in range(self.w):
        #             i_max = i + 2 * self.r + 1
        #             j_max = j + 2 * self.r + 1
        #             window = np.reshape(padded_array[i: i_max, j: j_max, channel_idx], -1)
        #             filtered_image[i, j] = np.median(window)
        #     return filtered_image
        #
        # for k in range(len(self.mode)):
        #     figm_orig[:, :, k] = apply_window(k)

        return figm_orig

    def result_export(self) -> list[Image]:
        return [Image.fromarray(self.image), Image.fromarray(self.gaussian_filter()), Image.fromarray(self.median_filter())]
