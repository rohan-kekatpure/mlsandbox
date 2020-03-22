import random
import glob
import os
import shutil
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)-0.50s.%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def randomsplit(paths_, label_, sampling_fraction_, dir1_, dir2_):
    for pth in paths_:
        dst_ = dir1_ if random.random() < sampling_fraction_ else dir2_
        shutil.copy(pth, os.path.join(dst_, label_))            

def main():
    RAW_DATA_DIR = "./data/raw"
    TRAIN_DIR = "./data/train"
    TEST_DIR = "./data/test"
    FRACTION_TRAIN = 0.6

    # Make fresh training and testing directories, removing previous copies.                                             
    for parent in [TRAIN_DIR, TEST_DIR]:
        shutil.rmtree(parent, ignore_errors=True)
        logger.info("Removed %s" % parent)
        for tt in ["spam", "ham"]:
            dirpth = os.path.join(parent, tt)
            shutil.os.makedirs(dirpth)
            logger.info("Created %s" % dirpth)
            

    # Create a randomized 60:40 train:test split
    ham_paths = glob.glob(os.path.join(RAW_DATA_DIR, "enron*/ham/*"))
    spam_paths = glob.glob(os.path.join(RAW_DATA_DIR, "enron*/spam/*"))

    logger.info("Creating train:test split")
    randomsplit(ham_paths, "ham", FRACTION_TRAIN, TRAIN_DIR, TEST_DIR)
    randomsplit(spam_paths, "spam", FRACTION_TRAIN, TRAIN_DIR, TEST_DIR)

    # Create a test file with true labels
    logger.info("Creating true labels")
    with open(os.path.join(TEST_DIR, "labels.txt"), "w") as lf:
        for label in ["ham", "spam"]:            
            for fpth in glob.glob(os.path.join(TEST_DIR, label, "*")):                
                lf.write("%s,%s\n" % (os.path.basename(fpth), label))       

if __name__ == '__main__':
    main()
