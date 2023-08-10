import numpy as np
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.preprocessing.image import image_dataset_from_directory


def tilegen(img, tilesize=(512, 512), overlap=(32, 32), multi_class=False):
    """Generator function to yield square tiles from an image with a defined overlap."""

    if len(tilesize) > 2:
        tilesize = np.squeeze(tilesize)

    # setup:
    shape = img.shape
    offset = (tilesize[0] - overlap[0], tilesize[1] - overlap[1])
    r_tiles = int(math.ceil(shape[0] / (offset[1] * 1.0)))
    c_tiles = int(math.ceil(shape[1] / (offset[0] * 1.0)))

    # generate tiles:
    for i in range(r_tiles):
        for j in range(c_tiles):
            # create tile:
            crop = img[
                offset[1] * i : min(offset[1] * i + tilesize[1], shape[0]),
                offset[0] * j : min(offset[0] * j + tilesize[0], shape[1]),
            ]
            if crop.shape != tilesize:
                crop = np.pad(
                    crop,
                    (
                        (0, tilesize[0] - crop.shape[0]),
                        (0, tilesize[1] - crop.shape[1]),
                    ),
                    "reflect",
                )
            crop = crop / np.amax(crop)
            # crop = crop / 255
            crop = np.reshape(crop, crop.shape + (1,)) if (not multi_class) else crop
            crop = np.reshape(crop, (1,) + crop.shape)
            # crop = np.expand_dims(crop, axis=0)
            print(crop.shape)
            yield crop


def stitch_with_overlap(tiles, original_shape, tile_shape=(512, 512), overlap=(32, 32)):
    """Stitch tiles back together with a defined overlap."""

    # setup:
    offset = (tile_shape[0] - overlap[0], tile_shape[1] - overlap[1])
    r_tiles = int(math.ceil(original_shape[0] / (offset[1] * 1.0)))
    c_tiles = int(math.ceil(original_shape[1] / (offset[0] * 1.0)))

    stitched_shape = (
        r_tiles * tile_shape[1] - ((r_tiles - 1) * overlap[1]),
        (c_tiles * tile_shape[1] - (c_tiles - 1) * overlap[1]),
    )
    stitched = np.zeros(stitched_shape)

    # stitch tiles:
    for i in range(r_tiles):
        for j in range(c_tiles):
            r_min = offset[1] * i
            r_max = min(offset[1] * i + tile_shape[1], stitched_shape[0])
            c_min = offset[0] * j
            c_max = min(offset[0] * j + tile_shape[0], stitched_shape[1])
            stitched[r_min:r_max, c_min:c_max] = tiles[i * c_tiles + j, :, :, 0]

    # crop to original size:
    stitched = stitched[: original_shape[0], : original_shape[1]]

    return stitched


def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1
        new_mask = (
            np.reshape(
                new_mask,
                (
                    new_mask.shape[0],
                    new_mask.shape[1] * new_mask.shape[2],
                    new_mask.shape[3],
                ),
            )
            if flag_multi_class
            else np.reshape(
                new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2])
            )
        )
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(
    batch_size,
    train_path,
    image_folder,
    mask_folder,
    aug_dict,
    image_color_mode="grayscale",
    mask_color_mode="grayscale",
    image_save_prefix="image",
    mask_save_prefix="mask",
    flag_multi_class=False,
    num_class=2,
    save_to_dir=None,
    target_size=(256, 256),
    seed=1,
):
    """
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        shuffle=False,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        shuffle=False,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )
    train_generator = zip(image_generator, mask_generator)
    for img, mask in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)
