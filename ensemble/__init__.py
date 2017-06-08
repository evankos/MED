import json
import logging
import os
import sys
from os.path import expanduser

import pkg_resources

from .common import set_project_root, set_backend

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
_home = expanduser("~")
_backend=""

resource_package = __name__


_project_dir = os.getcwd()
set_project_root(_project_dir)

if os.path.isfile(os.path.join(_home,'config.json')):
    logging.info("Loading configuration from user directory")
    _config = json.load(open(os.path.join(_home,'config.json')))
else:
    logging.info("Using default package configuration")
    _config = json.loads(pkg_resources.resource_string(resource_package,'/config.json').decode("utf-8"))

_backend = _config.get('backend',_backend)
set_backend(_backend)
if _backend=="numpy":
    from .np_functions import *

else:
    from .tf_functions import *




logging.info("Setting fusion backend to %s"%_backend)