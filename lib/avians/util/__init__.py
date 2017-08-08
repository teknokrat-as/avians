
import os
import datetime
import numpy as np 
import numpy.linalg as nl

import logging
LOG = logging.getLogger(__name__)

def get_random_dir(parent): 
    "Returns a fresh random dir under parent"
    ## NB: This assumes a linux/unix path 
    if parent[-1] != '/': 
        parent += '/'
    exists = True
    while exists: 
        d = "{}-{}".format(datetime.datetime.now().strftime("%F-%H-%M-%S"),
                           np.random.randint(100000, 999999))
        complete = os.path.join(parent, d)
        exists = os.path.exists(complete)
    os.mkdir(complete)
    return complete
    
