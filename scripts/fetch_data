#!/usr/bin/env -S uv run --script

import os
import re
import shutil
import tarfile
from tempfile import mkdtemp
from time import perf_counter

import requests


class DogBreedImageFetcher:
    IMGS_TARBALL_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    RAW_BREED_DIR_RE = re.compile(r"^n[0-9]{8}-(?P<breed>[\w-]+)$")
    BREED_DIR_RE = re.compile(r"^(?P<id>[0-9]{1,3})-[\w-]+$")

    def _get_data_dir(self) -> str:
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.normpath(os.path.join(script_dir, "..", "data"))
        return data_dir

    def _get_img_dir(self, data_dir: str) -> str:
        img_dir = os.path.join(data_dir, "images")
        return img_dir

    def __init__(self) -> None:
        self.data_dir = self._get_data_dir()
        self.img_dir = self._get_img_dir(self.data_dir)

    def _remove_old_dir(self) -> None:
        try:
            shutil.rmtree(self.data_dir)
        except FileNotFoundError:
            # The directory doesn't exist.
            pass

    def _load_imgs_tarball(self, tmpdir: str) -> str:
        res = requests.get(self.IMGS_TARBALL_URL, stream=True)
        tarball = os.path.join(tmpdir, "images.tar")
        with open(tarball, mode="wb") as f:
            for chunk in res.iter_content(chunk_size=8_192):
                f.write(chunk)
        return tarball

    def _untar_imgs(self, tarball_path: str, tmpdir: str) -> str:
        with tarfile.open(tarball_path, mode="r") as tar:
            imgs_dir = tar.getnames()[0]
            tar.extractall(tmpdir, filter="data")
        extracted_data_path = os.path.join(tmpdir, imgs_dir)
        return extracted_data_path

    def _rename_breed_dirs(self, extracted_data_path: str) -> None:
        for i, dirname in enumerate(os.listdir(extracted_data_path)):
            breed_path = os.path.join(extracted_data_path, dirname)
            if not os.path.isdir(breed_path):
                raise NotADirectoryError(f"{breed_path} is not a directory")
            breed = self.RAW_BREED_DIR_RE.match(dirname)["breed"]
            new_dirname = f"{i + 1}-{breed}"
            new_breed_path = os.path.join(extracted_data_path, new_dirname)
            os.rename(breed_path, new_breed_path)

    def _rename_breed_imgs(self, extracted_data_path: str) -> None:
        for breed_dir in os.listdir(extracted_data_path):
            breed_id = self.BREED_DIR_RE.match(breed_dir)["id"]
            breed_path = os.path.join(extracted_data_path, breed_dir)
            for i, img in enumerate(os.listdir(breed_path)):
                _, ext = os.path.splitext(img)
                img_path = os.path.join(breed_path, img)
                new_img_name = f"{breed_id}-{i + 1}" + ext
                new_img_path = os.path.join(breed_path, new_img_name)
                os.rename(img_path, new_img_path)

    def _mv_data(self, extracted_data_path: str) -> None:
        os.renames(extracted_data_path, self.img_dir)

    def _cleanup(self, tmpdir: str) -> None:
        shutil.rmtree(tmpdir)

    def fetch_data(self) -> None:
        self._remove_old_dir()
        tmpdir = mkdtemp()
        tarball_path = self._load_imgs_tarball(tmpdir)
        extracted_data_path = self._untar_imgs(tarball_path, tmpdir)
        self._rename_breed_dirs(extracted_data_path)
        self._rename_breed_imgs(extracted_data_path)
        self._mv_data(extracted_data_path)
        self._cleanup(tmpdir)

        return self.img_dir


if __name__ == "__main__":
    print("Fetching dog breed images")
    start_ts = perf_counter()
    fetcher = DogBreedImageFetcher()
    img_dir = fetcher.fetch_data()
    end_ts = perf_counter()
    print(f"Finished in {end_ts - start_ts:0.2f}s")
    print(f"Downloaded images to {img_dir}")
