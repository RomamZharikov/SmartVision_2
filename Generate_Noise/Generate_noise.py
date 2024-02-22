import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
from PIL import Image
from random import choice
from os import listdir
import time


def import_images(path: str = './default_images/') -> (Image, str):
    images_name = choice(listdir(path))
    name_img = images_name.split(sep=".")[0]
    print(f"Image name: {name_img}")
    img = Image.open(path + images_name)
    return img, name_img


def norm(num: float) -> float:
    if num > 255.0:
        return 255.0
    elif num < 1.0:
        return 0.0
    else:
        return num


def norm_array(arr_num: np.array) -> np.array:
    result = []
    for i in arr_num:
        result.append(norm(i))
    return result


def save_images(np_image: np.ndarray, name_image: str, img_name: str, path: str = './result_image/') -> None:
    img = Image.fromarray(np_image)
    img.save(path + f"{name_image}_{img_name}.jpg")
    return None


def noise_image(s_img: Image, type_noise: str = "normal", skv: float = 20.0, probability: float = 0.1) -> np.array:
    w, h = s_img.size
    img_source = np.asarray(s_img)
    img_orig = np.ndarray(img_source.shape, dtype=img_source.dtype)
    result_image = np.ndarray(img_source.shape, dtype=img_source.dtype)
    rimg = np.ndarray(img_source.shape, dtype=float)
    noise = np.ndarray(img_source.shape, dtype=float)
    mode = s_img.mode
    for j in range(w):
        for i in range(h):
            if type_noise == "normal":
                rimg[i, j] = (256.0 - (skv / 2) * 3) * img_source[i, j] / 256.0 + (((skv / 2) * 3) / 2)
                img_orig[i, j] = np.uint8(rimg[i, j])
                noise[i, j] = [nr.standard_normal() * skv for _ in range(len(mode))]
                rimg[i, j] = rimg[i, j] + noise[i, j]
                rimg[i, j] = norm_array(rimg[i, j])
            elif type_noise == "peper":
                rimg[i, j] = img_source[i, j]
                img_orig[i, j] = np.uint8(rimg[i, j])
                r = nr.random()
                if r < probability:
                    noise[i, j] = [127.0 * np.sin(2 * np.pi * nr.random()) + 128 for _ in range(len(mode))]
                    rimg[i, j] = noise[i, j]
            result_image[i, j] = np.uint8(rimg[i, j])
    return img_orig, noise, result_image


def skv_result(orig: Image, noice: Image, type_noice: str) -> (float, float):
    img1 = np.reshape(orig, -1)
    img2 = np.reshape(noice, -1)
    N = img1.shape[0]
    m1 = np.mean(img1)
    m2 = np.mean(img2)
    s1 = np.std(img1)
    s2 = np.std(img2)
    sigm = np.std(orig*1.0 - noice*1.0)
    print(f"Result for {type_noice}:")
    print(f"s.k.v {type_noice} = {sigm}")
    c = sum((img1 * 1.0 - m1) * (img2 * 1.0 - m2) / (N * s1 * s2))
    print(f"Coefficient Corr. {type_noice} = {c}\n{"=" * 100}")
    return sigm, c


def plot_result(img_orig: np.array, noice: np.array, result_img: np.array, type_noice: str) -> None:
    plt.subplot(1, 3, 1)
    plt.imshow(Image.fromarray(img_orig.astype(np.uint8)))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(Image.fromarray(noice.astype(np.uint8)))
    plt.title('Noise Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(Image.fromarray(result_img.astype(np.uint8)))
    plt.title('Denoised Image')
    plt.axis('off')

    plt.suptitle(f"{type_noice.capitalize()} noice", fontsize=24)
    plt.show()


def plot_dependency(result_list: list, label: str, value_name: str):
    skv_values = [item[0] for item in result_list]
    corr_values = [item[1] for item in result_list]
    conditions = [item[2] for item in result_list]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(conditions, skv_values, label=label, marker='o')
    plt.title(f"Dependency of SKV on {value_name} in {label} noice")
    plt.xlabel(f"{value_name}")
    plt.ylabel("SKV")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(conditions, corr_values, label=label, marker='o')
    plt.title(f"Dependency of Correlation Coefficient on {value_name} in {label} noice")
    plt.xlabel(f"{value_name}")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


if __name__ == '__main__':
    start_time = time.time()
    s_img, name = import_images()
    w, h = s_img.size
    img_source = np.asarray(s_img)
    result_peper = []
    result_normal = []
    print(f"w = {w}; h = {h}; Type: {img_source.dtype}")

    for k in [0.01, 0.05, 0.1, 0.2]:
        img_orig, noise, img_noised = noise_image(s_img, type_noise="peper", probability=k)
        plot_result(img_orig, noise, img_noised, f"peper, probability = {k}")
        save_images(img_noised, name, f"peper_{k}")
        a, b = skv_result(img_orig, img_noised, "peper")
        result_peper.append([a, b, k])

    for i in [1, 5, 10, 25]:
        img_orig, noise, img_noised = noise_image(s_img, type_noise="normal", skv=i)
        plot_result(img_orig, noise, img_noised, f"normal, skv = {i}")
        save_images(img_noised, name, f"normal_{i}")
        c, d = skv_result(img_orig, img_noised, "normal")
        result_normal.append([c, d, i])

    plot_dependency(result_peper, "peper", "probability")
    plot_dependency(result_normal, "normal", "SKV")
    plt.show()
    save_images(img_orig, name, "origin")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Час виконання: {elapsed_time} секунд")


