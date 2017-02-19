import os
import json
from .common import set_project_root
from .common import set_mfcc_file

_mfcc=""
_project_dir = os.getcwd()
set_project_root(_project_dir)

_config = json.load(open(os.path.join(_project_dir,'config.json')))
_mfcc = _config.get('mfcc_features',_mfcc)
set_mfcc_file(os.path.join(_project_dir,_mfcc))