

_PROJECT_ROOT = ''
_MFCC_FILE = ''
_LABEL_FILE = ''
_CLASS_INDEX_FILE = ''
_HIERARCHY_FILE = ''
_CNN_FILE = ''
_SIFT_SPECTOGRAM_FILE = ''
_TRAJ_FILE = ''
_MBH_FILE = ''
_HOG_FILE = ''
_HOF_FILE = ''



def set_project_root(project_root):
    global _PROJECT_ROOT
    _PROJECT_ROOT = project_root

def project_root():
    return _PROJECT_ROOT

def set_mfcc_file(mfcc_file):
    global _MFCC_FILE
    _MFCC_FILE = mfcc_file

def mfcc_file():
    return _MFCC_FILE


def set_hierarchy_file(hierarchy_file):
    global _HIERARCHY_FILE
    _HIERARCHY_FILE = hierarchy_file

def hierarchy_file():
    return _HIERARCHY_FILE

def set_label_file(label_file):
    global _LABEL_FILE
    _LABEL_FILE = label_file

def label_file():
    return _LABEL_FILE

def set_cnn_file(cnn_file):
    global _CNN_FILE
    _CNN_FILE = cnn_file

def cnn_file():
    return _CNN_FILE

def set_sift_file(sift_file):
    global _SIFT_FILE
    _SIFT_FILE = sift_file

def sift_file():
    return _SIFT_FILE

def set_sift_spectogram_file(sift_spectogram_file):
    global _SIFT_SPECTOGRAM_FILE
    _SIFT_SPECTOGRAM_FILE = sift_spectogram_file

def sift_spectogram_file():
    return _SIFT_SPECTOGRAM_FILE

def set_traj_file(traj_file):
    global _TRAJ_FILE
    _TRAJ_FILE = traj_file

def traj_file():
    return _TRAJ_FILE



def set_mbh_file(mbh_file):
    global _MBH_FILE
    _MBH_FILE = mbh_file

def mbh_file():
    return _MBH_FILE



def set_hog_file(hog_file):
    global _HOG_FILE
    _HOG_FILE = hog_file

def hog_file():
    return _HOG_FILE



def set_hof_file(hof_file):
    global _HOF_FILE
    _HOF_FILE = hof_file

def hof_file():
    return _HOF_FILE


def set_class_index_file(class_index_file):
    global _CLASS_INDEX_FILE
    _CLASS_INDEX_FILE = class_index_file

def class_index_file():
    return _CLASS_INDEX_FILE

