

_PROJECT_ROOT = ''
_MFCC_FILE = ''

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

