import os
import json
import logging,sys
from .common import set_project_root,set_backend
import pkg_resources
from os.path import expanduser

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
    from . import np_fusions as fusions

else:
    from . import tf_fusions as fusions




logging.info("Setting fusion backend to %s"%_backend)