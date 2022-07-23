import pickle
import old.data.constants as c
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join
from pybaseball import cache
from suppliers import Supplier

cache.enable()

def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def _run(supplier_cls):
    s = supplier_cls()
    r = s.run()
    print("Failures", len(s.api_failure))
    with open(join("data", "build", s.name() + ".pkl"), "wb") as f:
        pickle.dump(r, f)


def main():
    with ProcessPoolExecutor(max_workers=c.NUM_PROCESS) as executor:
        futures = [executor.submit(_run, sc) for sc in get_all_subclasses(Supplier)]
        for f in as_completed(futures):
            f.result()
    # for sc in get_all_subclasses(Supplier):
    #     _run(sc)


if __name__ == "__main__":
    main()
