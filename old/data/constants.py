from datetime import date
from os.path import join
from multiprocessing import cpu_count
import tqdm

# Disable TQDM from subclasses
class _TQDM(tqdm.tqdm):
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = True
        if kwargs.get('disable_override', 'def') != 'def':
            kwargs['disable'] = kwargs['disable_override']
            kwargs.pop("disable_override")
        super().__init__(*argv, **kwargs)


tqdm.tqdm = _TQDM

START_YEAR = 2000
END_YEAR = date.today().year
CURRENT_YEAR = date.today().year
CURRENT_MONTH = date.today().month

PITCH_TYPES = ["FF", "SIFT", "CH", "CUKC", "FC", "SL", "FS"]
CACHE_DIR = join("data", "cache")
NUM_PROCESS = 2
NUM_CPU = cpu_count()
NUM_THREADS = int(1.5 * NUM_CPU / NUM_PROCESS)
