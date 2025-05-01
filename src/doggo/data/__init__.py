import os
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@dataclass
class TrainTestData:
    Xtr: NDArray[np.float32]
    ytr: NDArray[np.int32]
    Xte: NDArray[np.float32]
    yte: NDArray[np.int32]


@dataclass
class BreedInfo:
    breed: str
    imgs_path: str
    label: int


class DogBreedImageLoader:
    BREED_DIR_RE = re.compile(r"^[0-9]+-(?P<breed>[\w-]+)$")

    IMG_DIR = "./data/images"
    IMG_MODE = "RGB"

    IMG_LEN = 50
    IMG_SIZE = (IMG_LEN, IMG_LEN)

    # Take 10% as test data
    TEST_SPLIT = 0.10

    def _get_breed_dirs(self) -> list[str]:
        breed_dirs = os.listdir(self.IMG_DIR)
        filtered_breed_dirs = [
            dir_
            for dir_ in breed_dirs
            if os.path.isdir(os.path.join(self.IMG_DIR, dir_))
        ]
        sorted_breed_dirs = sorted(filtered_breed_dirs)
        return sorted_breed_dirs

    def _get_breed_names(self, breed_dirs: list[str]) -> list[str]:
        breeds = []
        for breed_dir in breed_dirs:
            parsed_breed = (
                self.BREED_DIR_RE.match(breed_dir)["breed"].replace("_", " ").lower()
            )
            breeds.append(parsed_breed)
        return breeds

    def _get_breeds(self) -> list[BreedInfo]:
        breed_dirs = self._get_breed_dirs()
        breed_names = self._get_breed_names(breed_dirs)
        breeds = []
        for i, pair in enumerate(zip(breed_dirs, breed_names)):
            breed_dir, breed_name = pair
            imgs_path = os.path.join(self.IMG_DIR, breed_dir)
            breed = BreedInfo(breed_name, imgs_path, i)
            breeds.append(breed)
        return breeds

    def __init__(self):
        self.breeds = self._get_breeds()

    def _load_img_vec(self, img_path: str) -> NDArray[np.float32]:
        img_file = Image.open(img_path).resize(self.IMG_SIZE)
        if img_file.mode != self.IMG_MODE:
            img_file = img_file.convert(self.IMG_MODE)
        img_arr = np.array(img_file).astype("float32")
        img_vec = img_arr.flatten() / 255
        return img_vec

    def _load_img_vecs(self, imgs_path: str) -> NDArray[np.float32]:
        img_vecs = []
        for img_file in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img_file)
            img_vec = self._load_img_vec(img_path)
            img_vecs.append([img_vec])
        img_vecs = np.concatenate(img_vecs)
        return img_vecs

    def _get_train_test_data_for_breed(self, breed: BreedInfo) -> TrainTestData:
        X = self._load_img_vecs(breed.imgs_path)
        N = X.shape[0]
        y = np.full(N, breed.label, np.int32)
        N_test = int(N * self.TEST_SPLIT)
        Xte, Xtr = X[:N_test, :], X[N_test:, :]
        yte, ytr = y[:N_test], y[N_test:]
        data = TrainTestData(Xtr, ytr, Xte, yte)
        return data

    def _get_train_test_data(self) -> TrainTestData:
        Xtr, ytr, Xte, yte = [], [], [], []
        for breed in self.breeds:
            data = self._get_train_test_data_for_breed(breed)
            Xtr.append(data.Xtr)
            ytr.append(data.ytr)
            Xte.append(data.Xte)
            yte.append(data.yte)
        Xtr = np.concatenate(Xtr)
        ytr = np.concatenate(ytr)
        Xte = np.concatenate(Xte)
        yte = np.concatenate(yte)
        data = TrainTestData(Xtr, ytr, Xte, yte)
        return data
