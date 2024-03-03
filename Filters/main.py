from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Filters import Filters, import_images


def skv_result(orig: Image, noice: Image, type_noice: str) -> [float, float]:
    orig = np.array(orig)
    noice = np.array(noice)
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
    return [sigm, c]


def plot_dependency(result_list: dict, label: str, value_name: str) -> None:
    # Получаем все уникальные названия фильтров
    filter_names = set()
    for value in result_list.values():
        filter_names.update(value.keys())

    # Создаем новый словарь с названиями фильтров в качестве ключей
    filtered_data = {filter_name: [] for filter_name in filter_names}
    # Заполняем значения для каждого фильтра
    for filter_name in filter_names:
        for key in result_list:
            if filter_name in result_list[key]:
                filtered_data[filter_name].append(result_list[key][filter_name])
    conditions = range(6)

    plt.figure(figsize=(10, 6))
    for i in filtered_data.keys():
        plt.plot(conditions, [filtered_data[i][j][0] for j in range(len(filtered_data[i]))], label=i, marker='o')
    plt.title(f"Dependency of SKV on {value_name} in {label}")
    plt.xlabel(f"{value_name}")
    plt.ylabel("SKV")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for i in filtered_data.keys():
        plt.plot(conditions, [filtered_data[i][j][1] for j in range(len(filtered_data[i]))], label=i, marker='o')
    plt.title(f"Dependency of Correlation Coefficient on {value_name} in {label}")
    plt.xlabel(f"{value_name}")
    plt.ylabel("Correlation Coefficient")
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_filters(original_image: Image, noisy_image: Image, salt_and_pepper_image: Image, r: int = 0) -> list:
    original = Filters(image=original_image, r=r).result_export()
    noisy = Filters(image=noisy_image, r=r).result_export()
    salt_and_pepper = Filters(image=salt_and_pepper_image, r=r).result_export()
    skv_res = []

    num_images = len(original)
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 4 * num_images))
    arr_images = [original, noisy, salt_and_pepper]
    arr_images_name = ["Original", "Normal noise", "Salt"]
    filter_name = ["Gaussian", "Median", "Linear"]
    for i in range(len(arr_images)):
        axes[0, i].imshow(arr_images[i][0], cmap='gray')
        axes[0, i].set_title(arr_images_name[i])
        axes[0, i].axis('off')

        axes[1, i].imshow(arr_images[i][1], cmap='gray')
        axes[1, i].set_title('Gaussian Filter')
        axes[1, i].axis('off')

        axes[2, i].imshow(arr_images[i][2], cmap='gray')
        axes[2, i].set_title('Median Filter')
        axes[2, i].axis('off')

        skv_res.append([skv_result(arr_images[0][0], arr_images[i][1], f"{filter_name[0]} Filter {arr_images_name[i]}"),
                        skv_result(arr_images[0][0], arr_images[i][2], f"{filter_name[1]} Filter {arr_images_name[i]}")])
    result_origin = {filter_name[k]: skv_res[0][k] for k in range(len(skv_res[0]))}
    result_noice = {filter_name[k]: skv_res[1][k] for k in range(len(skv_res[1]))}
    result_peper = {filter_name[k]: skv_res[2][k] for k in range(len(skv_res[2]))}
    plt.tight_layout()
    plt.show()
    return [result_origin, result_noice, result_peper]


if __name__ == "__main__":
    original_image, _ = import_images(filename=False)
    noisy_image, _ = import_images(find="normal")
    salt_and_pepper_image, _ = import_images(find="peper")
    origin = {}
    noice = {}
    peper = {}

    for i in range(6):
        correlation_values = apply_filters(original_image, noisy_image, salt_and_pepper_image, i)
        origin[i], noice[i], peper[i] = correlation_values
    plot_dependency(origin, "Original Image", "Window Radius")
    plot_dependency(noice, "Normal Noice", "Window Radius ")
    plot_dependency(peper, "Peper Noice", "Window Radius")
