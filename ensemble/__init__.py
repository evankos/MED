import os
import json
from .common import set_project_root




_project_dir = os.getcwd()
set_project_root(_project_dir)


# _config = json.load(open(os.path.join(_project_dir,'config.json')))
# _embedding_dim = _config.get('embedding_dim',_embedding_dim)
# _glove_path=os.path.join(os.path.dirname(__file__),
#                         "glove/w2v.6B.%dd.txt" %(_embedding_dim))
# set_embedding_dim(_embedding_dim)
# set_glove(_glove_path)
