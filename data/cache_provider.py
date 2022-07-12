import pickle
from abc import ABC, abstractmethod
from data.constants import CACHE_DIR
from os import makedirs, remove
from os.path import join, normpath
from glob import glob
import time


class Cache(ABC):

    def __init__(self, base_name=""):
        self.base_name = base_name

    @abstractmethod
    def get(self, **kwargs):
        pass

    @abstractmethod
    def store(self, o, **kwargs):
        pass

    @abstractmethod
    def exists(self, **kwargs):
        return False


class FileCache(Cache):

    def __init__(self, base_dir=CACHE_DIR, base_name=""):
        super().__init__(base_name=base_name)
        self.base_dir = base_dir

    def get(self, **kwargs):
        path = self._create_path(kwargs)
        existing = self._get_existing(path)

        if len(existing) > 0:
            return self._read_file(existing[0])
        else:
            return None

    def _get_existing(self, path):
        return sorted(glob(join(path, "*.{}".format(self._get_extension()))), reverse=True)

    def store(self, obj, overwrite=False, **kwargs):
        path = self._create_path(kwargs)

        if overwrite:
            for f in self._get_existing(path):
                remove(f)

        makedirs(path, exist_ok=True)
        new_path = join(path, time.strftime("%Y%m%d-%H%M%S") + "." + self._get_extension())
        self._write_file(obj, new_path)
        return new_path

    def exists(self, **kwargs):
        path = self._create_path(kwargs)
        return len(self._get_existing(path)) > 0

    def _create_path(self, kwargs):
        path = [self.base_dir, self.base_name]
        sorted_keys = sorted(kwargs.keys())
        for key in sorted_keys:
            path.append(str(key))
            path.append(str(kwargs[key]))
        return normpath(join(*path)).lower()

    def _read_file(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _write_file(self, obj, path):
        with open(path, "wb") as f:
            return pickle.dump(obj, f)

    def _get_extension(self):
        return "pkl"





