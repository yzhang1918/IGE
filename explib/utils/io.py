import pickle
import os


def savepkl(obj, fname):
    with open(fname, 'wb') as fh:
        pickle.dump(obj, fh)


def loadpkl(fname):
    with open(fname, 'rb') as fh:
        obj = pickle.load(fh)
    return obj


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        raise
