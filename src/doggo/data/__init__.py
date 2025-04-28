import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.ImageFile import ImageFile


@dataclass
class LabeledImagePath:
    label: int
    img_path: str


@dataclass
class LabeledImage:
    label: int
    img_file: ImageFile


@dataclass
class LabeledImageVector:
    label: int
    img_vec: NDArray[np.float32]


class DogBreedImageLoader:
    BREED_DIR_RE = re.compile(r"^(n[0-9]{8}-)(?P<breed>[\w-]+)$")

    IMG_DIR = "./data/images"
    IMG_MODE = "RGB"

    IMG_LEN = 200
    IMG_SIZE = (IMG_LEN, IMG_LEN)
    IMG_VEC_LEN = 3 * IMG_LEN**2

    def _get_breed_dirs(self) -> list[str]:
        breed_dirs = os.listdir(self.IMG_DIR)
        filtered_breed_dirs = [
            dir_
            for dir_ in breed_dirs
            if os.path.isdir(os.path.join(self.IMG_DIR, dir_))
        ]
        sorted_breed_dirs = sorted(filtered_breed_dirs)
        return sorted_breed_dirs

    def __init__(self):
        self.breed_dirs = self._get_breed_dirs()
        self.cpu_count = os.process_cpu_count()
        if not self.cpu_count:
            self.cpu_count = 8

    def _load_labels(self) -> list[str]:
        labels = []
        for breed_dir in self.breed_dirs:
            parsed_breed = (
                self.BREED_DIR_RE.match(breed_dir)["breed"].replace("_", " ").lower()
            )
            labels.append(parsed_breed)
        return labels

    def _get_N_imgs(self, breed_dir: str) -> int:
        breed_path = os.path.join(self.IMG_DIR, breed_dir)
        N_imgs = len(os.listdir(breed_path))
        return N_imgs

    def _load_img(self, labeled_img_path: LabeledImagePath) -> LabeledImage:
        img_file = Image.open(labeled_img_path.img_path)
        img_file.load()
        labeled_img = LabeledImage(labeled_img_path.label, img_file)
        return labeled_img

    def _get_labeled_img_paths_for_dir(
        self, label: int, breed_dir: str
    ) -> list[LabeledImagePath]:
        output = []
        breed_path = os.path.join(self.IMG_DIR, breed_dir)
        for img_file in os.listdir(breed_path):
            img_path = os.path.join(breed_path, img_file)
            labeled_img_path = LabeledImagePath(label, img_path)
            output.append(labeled_img_path)
        return output

    def _get_labeled_img_paths(self) -> list[LabeledImagePath]:
        input_labels = list(range(len(self.breed_dirs)))
        with ThreadPoolExecutor() as executor:
            result = executor.map(
                self._get_labeled_img_paths_for_dir, input_labels, self.breed_dirs
            )
        return list(el for batch in result for el in batch)

    def _load_labeled_imgs(
        self, labeled_img_paths: list[LabeledImagePath]
    ) -> list[LabeledImage]:
        with ThreadPoolExecutor() as executor:
            result = executor.map(self._load_img, labeled_img_paths)
        return list(result)

    def _process_img(self, labeled_img: LabeledImage) -> LabeledImageVector:
        img_file = labeled_img.img_file
        img = img_file.resize(self.IMG_SIZE)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_arr = np.array(img).astype("float32")
        img_vec = img_arr.flatten() / 255
        labeled_img_vec = LabeledImageVector(labeled_img.label, img_vec)
        return labeled_img_vec

    def _process_imgs(
        self, labeled_imgs: list[LabeledImage]
    ) -> list[LabeledImageVector]:
        batch_size = len(labeled_imgs) // self.cpu_count + 1
        with ProcessPoolExecutor() as executor:
            result = executor.map(self._process_img, labeled_imgs, chunksize=batch_size)
        return list(result)

    def _load_X_y(self) -> tuple[NDArray, NDArray]:
        labeled_img_paths = self._get_labeled_img_paths()
        labeled_imgs = self._load_labeled_imgs(labeled_img_paths)
        labeled_img_vecs = self._process_imgs(labeled_imgs)
        X = np.concatenate(
            [[labeled_img_vec.img_vec] for labeled_img_vec in labeled_img_vecs]
        )
        y = np.array([labeled_img_vec.label for labeled_img_vec in labeled_img_vecs])
        return X, y
