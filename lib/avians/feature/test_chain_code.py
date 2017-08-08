
import cv2
import numpy as np
import avians.util.image as aui
import avians.feature.chain_code as afcc
import os

def test_chain_code_from_image(): 
    dataset = "/home/iesahin/Annex/Arabic/arabic-component-dataset-518"
    output_dir = "/tmp/chain-code-results"
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    image_files = aui.recursive_file_list(dataset)
    samples = []
    for imgf in image_files:
        img = cv2.imread(imgf, 0)
        cc = afcc.chain_code_from_img(img)
        outfname = "{}/{}.txt".format(output_dir,
                                      os.path.basename(imgf))
        np.savetxt(outfname, cc, fmt='%i')
    
