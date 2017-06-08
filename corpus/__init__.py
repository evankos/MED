import json
import logging
import os
import sys
from os.path import expanduser

import pkg_resources

from .common import set_class_index_file
from .common import set_cnn_file
from .common import set_hierarchy_file
from .common import set_hof_file
from .common import set_hog_file
from .common import set_label_file
from .common import set_mbh_file
from .common import set_mfcc_file
from .common import set_project_root
from .common import set_sift_file
from .common import set_sift_spectogram_file
from .common import set_traj_file

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
_home = expanduser("~")
resource_package = __name__
_mfcc=""
_sift=""
_cnn=""
_sift_spectogram=""
_traj=""
_mbh=""
_hog=""
_hof=""
_hierarchy=""
_label=""
_class_index=""
_project_dir = os.getcwd()
set_project_root(_project_dir)

if os.path.isfile(os.path.join(_home,'config.json')):
    logging.info("Loading configuration from user directory")
    _config = json.load(open(os.path.join(_home,'config.json')))
else:
    logging.info("Using default package configuration")
    _config = json.loads(pkg_resources.resource_string(resource_package,'/config.json').decode("utf-8"))


_mfcc = _config.get('mfcc_features',_mfcc)
set_mfcc_file(os.path.join(_project_dir,_mfcc))

_sift = _config.get('sift_features',_sift)
set_sift_file(os.path.join(_project_dir,_sift))

_cnn = _config.get('cnn_features',_cnn)
set_cnn_file(os.path.join(_project_dir,_cnn))


_sift_spectogram = _config.get('sift_spectogram_features',_sift_spectogram)
set_sift_spectogram_file(os.path.join(_project_dir,_sift_spectogram))

_traj = _config.get('traj_features',_traj)
set_traj_file(os.path.join(_project_dir,_traj))

_mbh = _config.get('mbh_features',_mbh)
set_mbh_file(os.path.join(_project_dir,_mbh))

_hog = _config.get('hog_features',_hog)
set_hog_file(os.path.join(_project_dir,_hog))

_hof = _config.get('hof_features',_hof)
set_hof_file(os.path.join(_project_dir,_hof))


_label = _config.get('labels',_label)
set_label_file(os.path.join(_project_dir,_label))

_hierarchy = _config.get('hierarchy',_hierarchy)
set_hierarchy_file(os.path.join(_project_dir,_hierarchy))

_class_index = _config.get('class_index',_class_index)
set_class_index_file(os.path.join(_project_dir,_class_index))