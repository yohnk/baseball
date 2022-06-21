import tarfile
from os.path import join
import time


def main():
    path = join("data", "data-{}.tar.gz".format(int(time.time())))
    with tarfile.open(path, "w:gz") as tar:
        tar.add(join("data", "build"), arcname="build")
        tar.add(join("data", "cache"), arcname="cache")
        tar.add(join("data", "csv"), arcname="csv")
        tar.add(join("data", "raw"), arcname="raw")


if __name__ == "__main__":
    main()
