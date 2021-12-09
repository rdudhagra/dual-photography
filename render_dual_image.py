import numpy as np
import skimage.io, skimage.transform
import pickle

from constants import PROJECTOR_H, PROJECTOR_W, CAMERA_H, CAMERA_W

MAX_LEVEL = 6


def main():
    # Load T, blocks, levels
    with open("T.pkl", "rb") as f:
        T = pickle.load(f)
    with open("blocks.pkl", "rb") as f:
        blocks = pickle.load(f)
    with open("levels.pkl", "rb") as f:
        levels = pickle.load(f)

    # Load prime image
    prime_image = skimage.io.imread("prime_image.jpg")
    prime_image = prime_image / prime_image.max()

    prime_image_colvec = np.reshape(prime_image, (-1, 3))

    dual_image = np.zeros_like(prime_image)

    for level, blocks_in_level in levels.items():
        if level > MAX_LEVEL:
            break
        print(f"Processing level {level}...")

        T_mat = np.zeros((2 ** level, 2 ** level, 3), dtype=np.float64)
        for block_num in blocks_in_level:
            block = blocks[block_num]

            # Calculate dual image result for this block
            T_row = np.zeros((1, prime_image_colvec.shape[0]))
            if block_num in T:
                T_block = T[block_num]
                for i, val in T_block.items():
                    T_row[0, i] = val
            result = T_row @ prime_image_colvec

            # Add to T_mat in correct position
            row_num = int(block.min_x / PROJECTOR_H * T_mat.shape[0])
            col_num = int(block.min_y / PROJECTOR_W * T_mat.shape[1])
            T_mat[row_num, col_num, :] = result

        # Resize T_mat to image size using bi-cubic interpolation
        img_at_this_level = skimage.transform.resize(
            T_mat, (CAMERA_H, CAMERA_W, 3), order=3
        )
        if img_at_this_level.max():
            img_at_this_level = img_at_this_level / np.sum(img_at_this_level)
        # skimage.io.imshow(img_at_this_level / img_at_this_level.max())
        # skimage.io.show()
        dual_image += img_at_this_level

    # Normalize dual image
    dual_image = dual_image / np.max(dual_image)

    # Save dual image
    dual_image = dual_image ** (1 / 3)
    dual_image = np.clip(dual_image, 0, 1)
    skimage.io.imsave("ddual_image.jpg", dual_image)


if __name__ == "__main__":
    main()
