import os
import re
from glob import glob as ls
from nose.tools import timed, with_setup
import avians.web.main as main
import logging
LOG = logging.getLogger(__name__)

app = None
the_client = None

def setup_app():
    global app
    global the_client
    
    app = main.app
    LOG.debug("=== app ===")
    LOG.debug(app)
    app.testing = True
    the_client = app.test_client()
    LOG.debug("=== the_client ===")
    LOG.debug(the_client)
    
def teardown_app():
    pass

@with_setup(setup_app, teardown_app)
def test_home():
    result = the_client.get('/')
    assert result.status_code == 200
    dir_list = ls(os.path.expandvars("$HOME/Annex/Arabic/*"))
    for d in dir_list:
        if os.path.isdir(d):
            LOG.debug("=== d ===")
            LOG.debug(d)
            LOG.debug("=== type(result.data) ===")
            LOG.debug(type(result.data))
            LOG.debug("=== result.data ===")
            LOG.debug(result.data)
            dd = os.path.basename(d)
            assert result.data.find(bytes(dd, encoding='utf-8')) > 0, "Cannot find: {}".format(d)

@with_setup(setup_app, teardown_app)
def test_view_image_list():
    result = the_client.get('/image_list/working-set-1/')
    LOG.debug("=== result ===")
    LOG.debug(result)
    LOG.debug("=== result.status_code ===")
    LOG.debug(result.status_code)
    assert result.status_code == 200

    file_list = ls(os.path.expandvars("$HOME/Annex/Arabic/working-set-1/*/original.*"))
    image_keys = [re.sub(r".*/working-set-1/(.*)/original.*", r"\1", f) for f in file_list]
    LOG.debug("=== image_keys ===")
    LOG.debug(image_keys)
    
    for k in image_keys:
        assert result.data.find(bytes(k, 'utf-8')) > 0

@with_setup(setup_app, teardown_app)
def test_view_image():
    file_list = ls(os.path.expandvars("$HOME/Annex/Arabic/working-set-1/*"))
    url = '/image/working-set-1/{}/'.format(os.path.basename(file_list[0]))
    LOG.debug("=== url ===")
    LOG.debug(url)
    result = the_client.get(url)
    LOG.debug("=== result ===")
    LOG.debug(result)
    LOG.debug("=== result.status_code ===")
    LOG.debug(result.status_code)
    LOG.debug("=== result.data ===")
    LOG.debug(result.data)
    
    assert result.status_code == 200

@with_setup(setup_app, teardown_app)
def test_view_labelmap():
    result = the_client.get('$HOME/Annex/Arabic/working-set-1/rt-1-img8094/')
    LOG.debug("=== result ===")
    LOG.debug(result)
    LOG.debug("=== result.status_code ===")
    LOG.debug(result.status_code)
    LOG.debug("=== result.data ===")
    LOG.debug(result.data)
    
    assert result.status_code == 200

