import cv2
import numpy as np
import os


def resize_image_stretch_old(image_path, output_path, target_size=(32, 32)):
    """
    Resizes an image to the exact target size, ignoring aspect ratio (stretching).

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        target_size (tuple): Target size in pixels (width, height).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return


    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    if aspect_ratio > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    # Directly resize to target size, ignoring aspect ratio
    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA if max(h, w) > max(target_w, target_h) else cv2.INTER_CUBIC
        ),
    )

    cv2.imwrite(output_path, resized_image)
    print(f"Image saved to {output_path}")

def resize_image_stretch(image_path, output_path, target_size=(32, 32)):
    """
    Redimensiona una imagen al tamaño objetivo exacto, ignorando 
    la relación de aspecto (estirando).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Se obtienen las dimensiones originales solo para la interpolación
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Se determina la mejor interpolación
    interpolation = (
        cv2.INTER_AREA if max(h, w) > max(target_w, target_h) else cv2.INTER_CUBIC
    )

    # --- CORRECCIÓN ---
    # Se eliminó el bloque if/else y se usa target_size directamente
    resized_image = cv2.resize(
        image,
        target_size,  
        interpolation=interpolation
    )

    cv2.imwrite(output_path, resized_image)
    print(f"Image saved to {output_path}")


def resize_image_blurred_background(
    image_path, output_path, target_size=(32, 32), blur_sigma=10
):
    """
    Resizes an image maintaining aspect ratio and places it on a blurred version of itself as background.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        target_size (tuple): Target size in pixels (width, height).
        blur_sigma (float): Sigma for Gaussian blur.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    if aspect_ratio > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA if max(h, w) > max(target_w, target_h) else cv2.INTER_CUBIC
        ),
    )

    if new_w < target_w or new_h < target_h:
        # Create a blurred background from the original image
        canvas = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        canvas = cv2.GaussianBlur(canvas, (21, 21), blur_sigma)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image
        final_image = canvas
    else:
        final_image = resized_image

    cv2.imwrite(output_path, final_image)
    print(f"Image saved to {output_path}")


def resize_image_mirror_padding(image_path, output_path, target_size=(32, 32)):
    """
    Resizes an image maintaining aspect ratio and fills extra space with mirrored edges.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        target_size (tuple): Target size in pixels (width, height).
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    h, w = image.shape[:2]
    target_w, target_h = target_size
    aspect_ratio = w / h
    target_aspect = target_w / target_h

    if aspect_ratio > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized_image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA if max(h, w) > max(target_w, target_h) else cv2.INTER_CUBIC
        ),
    )

    if new_w < target_w or new_h < target_h:
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        final_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, cv2.BORDER_REFLECT
        )
    else:
        final_image = resized_image

    cv2.imwrite(output_path, final_image)
    print(f"Image saved to {output_path}")


# Process all images in a folder
input_folder = "images/nuevas"
output_folder_stretch = "images/nuevas_stretched"
output_folder_blurred = "images/nuevas_blurred"
output_folder_mirror = "images/nuevas_mirrored"

os.makedirs(output_folder_stretch, exist_ok=True)
os.makedirs(output_folder_blurred, exist_ok=True)
os.makedirs(output_folder_mirror, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg",".ppm")):
        input_path = os.path.join(input_folder, filename)

        # Stretch method
        output_path_stretch = os.path.join(
            output_folder_stretch, f"stretched_{filename}"
        )
        resize_image_stretch(input_path, output_path_stretch)

        # Blurred background method
        output_path_blurred = os.path.join(output_folder_blurred, f"blurred_{filename}")
        resize_image_blurred_background(input_path, output_path_blurred)

        # Mirror padding method
        output_path_mirror = os.path.join(output_folder_mirror, f"mirrored_{filename}")
        resize_image_mirror_padding(input_path, output_path_mirror)
