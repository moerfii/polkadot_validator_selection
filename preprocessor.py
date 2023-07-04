from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    path = "./data/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
