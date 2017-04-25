from keras.backend.common import epsilon

_PROJECT_ROOT = ''
_BACKEND = ''
_EPSILON=10e-8



def set_project_root(project_root):
    global _PROJECT_ROOT
    _PROJECT_ROOT = project_root

def project_root():
    return _PROJECT_ROOT


def set_backend(backend):
    global _BACKEND
    _BACKEND = backend

def backend():
    return _BACKEND

def eps():
    return epsilon()

