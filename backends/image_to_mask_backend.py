import os
from pathlib import Path

import cv2
import numpy as np
import tifffile


def read_tif(img_path):
    """
    Read a 2D/3D TIFF.
    Output is always shaped as (Z, Y, X).
    """
    try:
        with tifffile.TiffFile(img_path) as tif:
            data = tif.asarray()
    except AttributeError as exc:
        if "newbyteorder" not in str(exc):
            raise

        from PIL import Image, ImageSequence

        with Image.open(img_path) as img:
            frames = [np.asarray(frame) for frame in ImageSequence.Iterator(img)]

        if len(frames) == 0:
            raise ValueError(f"No frames found in TIFF: {img_path}") from exc

        data = np.stack(frames, axis=0) if len(frames) > 1 else frames[0]

    print("************** read tif *******************")
    print(f"Original Shape: {data.shape}")
    print(f"Original Data Type: {data.dtype}")

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
        print("Converted to 3D by adding leading dimension.")

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D TIFF, got shape={data.shape}"
        )

    print(f"Processed Shape: {data.shape}")
    print("************** read tif *******************")

    return data


def save_tif(save_path, tif_img):
    tifffile.imwrite(save_path, tif_img)
    print(f"TIFF image saved at: {save_path}")


def enhance_adaptive(img_stack, block_size=25, C=0.1):
    """
    Same adaptive enhancement logic used in the original img2pcd code.
    """
    enhanced_stack = np.zeros_like(img_stack)

    for i in range(img_stack.shape[0]):
        img = img_stack[i]

        if img.dtype != np.uint8:
            img = cv2.normalize(
                img,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

        binary_img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C
        )

        enhanced_stack[i] = binary_img / 255.0

    return enhanced_stack * img_stack


def convert_enhanced_image_to_binary_mask(
    image: np.ndarray,
    threshold: int = 50,
):
    """
    Convert the adaptively enhanced image directly to a binary mask.

    This reproduces the relevant behavior of the original:
        enhanced image
        -> percentile normalization to uint8 when needed
        -> threshold
        -> foreground = 255, background = 0

    No voxel TXT or point cloud is created.
    """
    print("********************* image -> mask *********************")

    image = np.asarray(image)

    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    elif image.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D image, got shape {image.shape}."
        )

    print(
        f"Original image: dtype={image.dtype}, "
        f"max={np.nanmax(image)}, min={np.nanmin(image)}"
    )

    if image.dtype != np.uint8:
        finite_values = image[np.isfinite(image)]

        if finite_values.size == 0:
            raise ValueError("Cannot normalize image: no finite pixel values.")

        lower_value = float(np.min(finite_values))
        upper_value = float(np.percentile(finite_values, 99.9))

        if upper_value <= lower_value:
            image_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            image_work = np.asarray(image, dtype=np.float64)
            image_work = np.nan_to_num(
                image_work,
                nan=lower_value,
                posinf=upper_value,
                neginf=lower_value,
            )
            image_work = np.clip(
                image_work,
                lower_value,
                upper_value
            )
            image_work = (
                (image_work - lower_value)
                / (upper_value - lower_value)
            )
            image_uint8 = np.rint(
                image_work * 255.0
            ).astype(np.uint8)

            print(
                f"Percentile normalization: min={lower_value:.6g}, "
                f"P99.9={upper_value:.6g} -> uint8 [0, 255]"
            )
    else:
        image_uint8 = image.copy()

    foreground = image_uint8 > threshold

    mask = np.zeros_like(image_uint8, dtype=np.uint8)
    mask[foreground] = 255

    print(f"Input shape: {image_uint8.shape}")
    print(f"Threshold: {threshold}")
    print(f"Foreground voxels: {int(foreground.sum())}")
    print("********************* image -> mask *********************")

    return mask


def main():
    # ============================================================
    # User settings
    # ============================================================

    BASE_DIR = Path(
        "data"
    )

    OUTPUT_DIR = BASE_DIR / "processed_output"

    # Same values as your previous script.
    ADAPTIVE_BLOCK_SIZE = 37
    ADAPTIVE_C = 12
    THRESHOLD = 50

    # If False, an existing *_mask.tif will be skipped.
    OVERWRITE_EXISTING_MASK = True

    # ============================================================
    # Process all TIFF files directly under BASE_DIR
    # ============================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imgs_list = sorted(
        [
            f for f in os.listdir(BASE_DIR)
            if f.lower().endswith((".tif", ".tiff"))
        ]
    )

    print(f"Found {len(imgs_list)} TIFF file(s).")

    for i, img_file in enumerate(imgs_list, start=1):
        print("-------------------------------------------------------------------------")
        print(f"------- processing -------- {i}/{len(imgs_list)} ------------------------")
        print("-------------------------------------------------------------------------")

        img_path = BASE_DIR / img_file
        img_name = Path(img_file).stem

        save_mask_path = OUTPUT_DIR / f"{img_name}_mask.tif"

        if save_mask_path.exists() and not OVERWRITE_EXISTING_MASK:
            print(f"[SKIP EXISTING MASK] {save_mask_path}")
            continue

        print(f"Reading image: {img_path}")

        img = read_tif(img_path)

        img_adpt = enhance_adaptive(
            img,
            block_size=ADAPTIVE_BLOCK_SIZE,
            C=ADAPTIVE_C,
        )

        mask = convert_enhanced_image_to_binary_mask(
            img_adpt,
            threshold=THRESHOLD,
        )

        save_tif(save_mask_path, mask)

        print(f"[SAVED MASK] {save_mask_path}")


if __name__ == "__main__":
    main()
