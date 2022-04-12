import os
import sys
import shutil
import numpy as np
import configparser
import multiprocessing as mp
import matplotlib.pyplot as plt

from PIL import Image
from eyepy.core.base import Oct
from sklearn.model_selection import train_test_split

config = configparser.ConfigParser()
config.read("config.ini")
paths = config['PATHS']
settings = config['SETTINGS']

def main():
    create_dir(paths.get('IMAGES_PATH'))
    create_dir(paths.get('MASK_PATH'))
    create_dir(paths.get('ANN_PATH'))
    create_dir(paths.get('TRAIN_IMAGES'))
    create_dir(paths.get('TRAIN_MASKS'))
    create_dir(paths.get('VAL_IMAGES'))
    create_dir(paths.get('VAL_MASKS'))
    create_dir(paths.get('TRAIN_IMAGES_CROP'))
    create_dir(paths.get('TRAIN_MASKS_CROP'))
    create_dir(paths.get('VAL_IMAGES_CROP'))
    create_dir(paths.get('VAL_MASKS_CROP'))

    oct_files = get_filenames(paths.get('OCT_PATH'), 'vol')

    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(get_images_masks, oct_files)
    pool.close()
    pool.join()
    split_data(train_size=settings.getfloat('TRAIN_SIZE'))
    read_images()


def split_data(train_size=0.8):
    extention = 'bmp'
    images_path = paths.get('IMAGES_PATH')
    masks_path = paths.get('MASK_PATH')

    train_images_dir = paths.get('TRAIN_IMAGES')
    val_images_dir = paths.get('VAL_IMAGES')
    train_masks_dir = paths.get('TRAIN_MASKS')
    val_masks_dir = paths.get('VAL_MASKS')

    x = get_filenames(images_path, extention)
    y = get_filenames(masks_path, extention)

    X_train, X_val, y_train, y_val = train_test_split(
        x, y, train_size=train_size, shuffle=True
    )

    for i, j in zip(X_train, y_train):
        shutil.copy(os.path.join(i), train_images_dir)
        shutil.copy(os.path.join(j), train_masks_dir)

    for i, j in zip(X_val, y_val):
        shutil.copy(os.path.join(i), val_images_dir)
        shutil.copy(os.path.join(j), val_masks_dir)


def vol_files(name):
    # print(name)
    vol_filename = os.path.join(paths.get('OCT_PATH'), name + '.vol')
    oct_read = Oct.from_heyex_vol(vol_filename)
    return oct_read


def get_images_masks(file):
    try:
        name = os.path.splitext(os.path.split(file)[1])[0]

        oct_read = vol_files(name)

        data = oct_read.bscans[0].scan
        data = np.expand_dims(data, axis=-1)
        dat1 = Image.fromarray(data.squeeze(2))
        dat1.save(paths.get('IMAGES_PATH') + name + ".bmp")

        zeros = np.zeros((data.shape[0], data.shape[1], 3)).astype('uint8')
        data1 = np.add(data, zeros)

        OPL, INL, PR2, PR1, BM, ELM = get_annotations(oct_read)

        # Generate ground truth
        mask = np.zeros((data.shape[0], data.shape[1])).astype('uint8')
        for i in range(OPL.shape[0]):
            data1[INL[i], i, 1] = 255
            data1[OPL[i], i, 2] = 255
            data1[PR2[i], i, 0] = 255
            data1[PR1[i], i, :] = [255, 255, 0]
            data1[ELM[i], i, :] = [0, 255, 255]
            data1[BM[i], i, :] = [125, 200, 125]
            mask[(INL[i]):(OPL[i]), i] = 1 if INL[i] > 0 and OPL[i] > 0 else 0
            mask[(PR1[i]):(PR2[i]), i] = 2 if PR1[i] > 0 and PR2[i] > 0 else 0
            # mask[BM[i], i] = 3 if BM[i] > 0 else 0
            mask[ELM[i]:(PR1[i]), i] = 3 if ELM[i] > 0 and PR1[i] > 0 else 0
        mask1 = Image.fromarray(mask)
        ann1 = Image.fromarray(data1)
        dat1 = Image.fromarray(data.squeeze(axis=2))
        name_file = name + ".bmp"        
        dat1.save(paths.get('IMAGES_PATH') + name_file)
        mask1.save(paths.get('MASK_PATH') + name_file)
        ann1.save(paths.get('ANN_PATH') + name_file)
    except Exception as exc:
        print(exc)
        print(file)


def get_annotations(oct_read):
    # Outer Plexiform Layer: OPL
    OPL = np.round(
        oct_read.bscans[0].annotation["layers"]['OPL']).astype('uint16')
    INL = np.round(
        oct_read.bscans[0].annotation["layers"]['INL']).astype('uint16')
    # Ellipsoide Zone: EZ
    PR2 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR2']).astype('uint16')
    PR1 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR1']).astype('uint16')
    # BM
    try:
        BM = np.round(
            oct_read.bscans[0].annotation["layers"]['BM']).astype('uint16')
    except:
        BM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    # ELM
    try:
        ELM = np.round(
            oct_read.bscans[0].annotation["layers"]['ELM']).astype('uint16')
    except:
        ELM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    return OPL, INL, PR2, PR1, BM, ELM


def crop_overlap(file, image, mask, path_img, path_msk):
    oct_read = vol_files(file)
    OPL, INL, PR2, PR1, BM, ELM = get_annotations(oct_read)
    j = 1
    k = 0
    size = settings.getint('IMAGE_SIZE')
    shift = settings.getint('SHIFT')
    for i in range(size, image.shape[1], shift):
        min_pixel = np.max(INL[INL[k:i]])
        max_pixel = np.max(PR1[k:i])
        if min_pixel != 0 and max_pixel != 0 and max_pixel > min_pixel:
            delta1 = max_pixel - min_pixel
            delta2 = size - delta1
            delta3 = delta2 // 2
            delta4 = min_pixel - delta3
            delta5 = max_pixel + delta3
            if delta2 % 2 != 0:
                delta5 += 1
            if delta4 < 0:
                delta4 = 0
                delta5 = size
            # if delta4-delta5 != size:
            #     delta5 = delta4-delta5
            if delta5 > image.shape[0]:
                delta5 = image.shape[0]
                delta4 = delta5 - size

            img_save = image[delta4:delta5, i - size:i]
            msk_save = mask[delta4:delta5, i - size:i]
            img = Image.fromarray(img_save)
            msk = Image.fromarray(msk_save)
            img.save(path_img + file + f"_{j}.bmp")
            msk.save(path_msk + file + f"_{j}.bmp")
            j += 1
        k = i


def read_images():
    train_images_files = get_filenames(paths.get('TRAIN_IMAGES'), 'bmp')
    train_masks_files = get_filenames(paths.get('TRAIN_MASKS'), 'bmp')
    val_images_files = get_filenames(paths.get('VAL_IMAGES'), 'bmp')
    val_masks_files = get_filenames(paths.get('VAL_MASKS'), 'bmp')

    for img_f, msk_f in zip(train_images_files, train_masks_files):
        image_file = os.path.splitext(os.path.split(img_f)[1])[0]
        image = np.array(Image.open(img_f))
        mask = np.array(Image.open(msk_f))
        crop_overlap(image_file, image, mask,
                     paths.get('TRAIN_IMAGES_CROP'), paths.get('TRAIN_MASKS_CROP'))

    for img_f, msk_f in zip(val_images_files, val_masks_files):
        image_file = os.path.splitext(os.path.split(img_f)[1])[0]
        image = np.array(Image.open(img_f))
        mask = np.array(Image.open(msk_f))
        crop_overlap(image_file, image, mask,
                     paths.get('VAL_IMAGES_CROP'), paths.get('VAL_MASKS_CROP'))


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0


if __name__ == "__main__":
    main()
