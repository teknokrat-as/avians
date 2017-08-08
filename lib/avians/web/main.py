import sys
from flask import Flask, render_template, request
import os
from glob import glob as ls
import re
import cv2
from math import floor
from datetime import datetime as dt

import avians.video_search as avs
import avians.util.image as aui
import avians.detect.nm_detect as adnd

import numpy as np

app = Flask(__name__)

ROOT = os.path.expandvars('Arabic')

@app.route('/')
def home():
    raw_root_list = ls("{}/*".format(ROOT))
    app.logger.debug("raw_root_list: {}".format(raw_root_list))
    root_list = [os.path.basename(r) for r in raw_root_list if os.path.isdir(r)]
    app.logger.debug("root_list: {}".format(root_list))
    return render_template('home.html', dirs = root_list)

@app.route('/image_list/<library_dir>/')
def image_list(library_dir):
    image_file_list = (ls("{}/{}/*/original.png".format(ROOT, library_dir)) +
                       ls("{}/{}/*/original.jpg".format(ROOT, library_dir)))
    app.logger.debug(image_file_list)
    images = []
    for f in image_file_list:
        img = {}
        img['key'] = image_key(f)
        img['thumbnail_url'] = thumbnail_file(f)
        images.append(img)
    return render_template('image_list.html',
                           title='Avians ' + os.path.basename(ROOT),
                           library_dir = library_dir,
                           images = images)

"""
The layout of the files in the dataset should be as follows:

ROOT/library_dir/image_key/original.(png|jpg) The original image file
ROOT/library_dir/image_key/thumbnail.png  Thumbnail for the image
ROOT/library_dir/image_key/labelmap.lm2  The labelmap for the image
ROOT/library_dir/image_key/labelmap-backup-2017-02-03-04-05-06.lm2  The backed up labelmap for the image
ROOT/library_dir/image_key/region-[0-9]+.png Text detection results for the image

"""

def library_dir(path):
    "Returns the library dir from the whole file path"
    assert path.startswith(ROOT)
    pat = r'{}/(.*?)/.*'.format(ROOT)
    return re.sub(pat, r'\1', path)

def image_key(path):
    "Returns the image_key from the whole file path"
    assert path.startswith(ROOT)
    pat = r'{}/.*?/(.*?)/.*'.format(ROOT)
    return re.sub(pat, r'\1', path)

def unique_file(p):
    "Returns the path for original image file in given lib dir and image key"
    fl = ls(p)
    if len(fl) == 0:
        raise Exception("No such file found: {}".format(p))
    if len(fl) > 1:
        raise Exception("More than 1 file found: {}".format(str(fl)))
    else:
        return fl[0]

def original_path(library_dir, image_key):
    p = "{}/{}/{}/original.*".format(ROOT, library_dir, image_key)
    return unique_file(p)

def thumbnail_path(library_dir, image_key):
    d = os.path.dirname(os.path.realpath(__file__))
    p = "{}/{}/{}/thumbnail.png".format(ROOT, library_dir, image_key)
    return unique_file(p)

def td_result_path(library_dir, image_key):
    p = "{}/{}/{}/text_detection/image.png".format(ROOT, library_dir, image_key)
    return unique_file(p)

def labelmap_path(library_dir, image_key):
    p = "{}/{}/{}/labelmap.lm2".format(ROOT, library_dir, image_key)
    return unique_file(p)

def copy_labelmap_path(library_dir, image_key):
    d = dt.now().strftime("-%d-%m-%Y-%H-%M-%S")
    p = "{}/{}/{}/labelmap{}.lm2".format(ROOT, library_dir, image_key, d)
    if not os.path.exists(p):
        f = open(p, "w")
    return unique_file(p)

def text_regions_path(library_dir, image_key):
    p = "{}/{}/{}/region-*.png".format(ROOT, library_dir, image_key)
    return ls(p)

def comp_img_paths(library_dir, image_key):
    p = "{}/{}/{}/comp-img-*.png".format(ROOT, library_dir, image_key)
    return ls(p)

def whole_img_paths(library_dir, image_key):
    p = "{}/{}/{}/whole-*.png".format(ROOT, library_dir, image_key)
    return ls(p)

def file_extension(path):
    return os.path.splitext(path)[1][1:]

@app.route('/image/<library_dir>/<image_key>/')
def image(library_dir, image_key):
    img = {}
    img['library_dir'] = library_dir
    img['image_key'] = image_key
    return render_template("image.html",
                           image_key=image_key,
                           library_dir = library_dir)

@app.route('/labelmap/<library_dir>/<image_key>/')
def labelmap_page(library_dir, image_key):
    p = labelmap_path(library_dir, image_key)
    lm = avs.load_labelmap(p)

    if len(whole_img_paths(library_dir, image_key)) == 0:
        whole_img_files, comp_img_files = write_labelmap_images(lm, library_dir, image_key)
    else:
        whole_img_files = sorted(whole_img_paths(library_dir, image_key))
        comp_img_files = sorted(comp_img_paths(library_dir, image_key))

    stats = lm['stats']
    labels = lm['labels']
    els = []
    assert len(whole_img_files) == len(comp_img_files)
    assert len(whole_img_files) == stats.shape[0]
    for i in range(stats.shape[0]):
        el = {}
        el['x'] = stats['x'][i]
        el['y'] = stats['y'][i]
        el['w'] = stats['w'][i]
        el['h'] = stats['h'][i]
        el['cx'] = stats['cx'][i]
        el['cy'] = stats['cy'][i]
        el['area'] = stats['area'][i]
        el['wimg_path'] = whole_img_files[i]
        el['cimg_path'] = comp_img_files[i]
        el['l'] = []
        for j in range(labels.shape[1]):
            el['l'].append("{}: {}".format(labels['label'][i, j], labels['prob'][i, j]))
        els.append(el)
    return render_template('labelmap.html',
                           library_dir = library_dir,
                           image_key = image_key,
                           elements = els)

def write_labelmap_images(lm, library_dir, image_key):
    imgs = lm['images']
    pad_length = len(str(imgs.shape[0]))
    whole_img_files = []
    comp_img_files = []
    output_dir = "{}/{}/{}/".format(ROOT, library_dir, image_key)
    for i in range(imgs.shape[0]):
        whole_img_f = os.path.join(output_dir, "whole-{num:0={pad}d}.png".format(num=i, pad=pad_length))
        comp_img_f = os.path.join(output_dir, "comp-img-{num:0={pad}d}.png".format(num=i, pad=pad_length))
        img = imgs[i]
        whole_img = aui.draw_rect_around_nonzero(img, padding=7, thickness=2)
        comp_img = aui.yank_nonzero(img, padding=3)
        cv2.imwrite(whole_img_f, whole_img)
        cv2.imwrite(comp_img_f, comp_img)
        whole_img_files.append(whole_img_f)
        comp_img_files.append(comp_img_f)

    return (whole_img_files, comp_img_files)

@app.route('/text_detect/<library_dir>/<image_key>/')
def text_detect(library_dir, image_key):
    # model = "$HOME/Annex/Arabic/text-detection-arabic-models/td_trained_arbc2.xml"
    image = original_path(library_dir, image_key)
    # output = "text-detection-result-{}.png".format(dt.now().strftime("%F-%T"))
    td_image = adnd.main_inner(image, library_dir, image_key)
    # image_file_list = (ls("{}/{}/*/original.png".format(ROOT, library_dir)))
    # images = []
    # for f in image_file_list:
    #     img = {}
    #     img['key'] = image_key(f)
    #     img['thumbnail_url'] = thumbnail_file(f)
    #     images.append(img)
    # return render_template('image_list.html',
    #                        title='Avians ' + os.path.basename(ROOT),
    #                        library_dir = library_dir,
    #                        images = images)
    # return "OK. FILENAME: {}".format(td_image)
    # return app.send_static_file(tdi_filename)
    return render_template('text_detect.html',
                           library_dir = library_dir,
                           image_key = image_key)

@app.route('/resolution+/<library_dir>/<image_key>/')
def resolution_plus(library_dir, image_key):
    pass

MAX_THUMBNAIL_DIM = 100

@app.route('/thumbnail/<library_dir>/<image_key>/')
def serve_thumbnail(library_dir, image_key):
    tp = thumbnail_path(library_dir, image_key)
    app.logger.debug("tp: {}".format(tp))
    return app.send_static_file(tp)

def make_thumbnail(filename, outfile):
    img = cv2.imread(filename)
    resize_factor = max(img.shape[0], img.shape[1]) / MAX_THUMBNAIL_DIM
    # Beware: OpenCV dimensions are inverse of NumPy
    dsize = (floor(img.shape[1] / resize_factor), floor( img.shape[0] / resize_factor ))
    img_thumb = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)
    cv2.imwrite(outfile, img_thumb)

def thumbnail_file(filename):
    ld = library_dir(filename)
    ik = image_key(filename)
    tp = thumbnail_path(ld, ik)
    if not os.path.exists(tp):
        make_thumbnail(filename, tp)
    return tp

@app.route('/original/<library_dir>/<image_key>/')
def serve_original(library_dir, image_key):
    op = original_path(library_dir, image_key)
    app.logger.debug("op: {}".format(op))
    return app.send_static_file(op)

@app.route('/text_detection_result/<library_dir>/<image_key>/')
def serve_text_detection_result(library_dir, image_key):
    op = td_result_path(library_dir, image_key)
    app.logger.debug("op: {}".format(op))
    return app.send_static_file(op)

def labelmap_save_process(fn, i, s, l, c):
    # fn: filename
    # i : images
    # s : stats
    # l : labels
    # c : component map
    
    f = open(fn, 'wb') 
    np.savez_compressed(f,
                        images=i,
                        stats=s,
                        labels=l,
                        component_map=c)
    

def edit_labelmap(fo, fc, ltc):
    # fo  : filename of original labelmap
    # fc  : filename of copy labelmap
    # ltc : datas of label to change

    # reading from file
    b = np.load(fo)
    # copying original labelmap file
    labelmap_save_process(fc, b['images'], b['stats'], b['labels'], b['component_map'])
    # removing original labelmap file
    os.remove(fo)
    # changing label's data
    temp = b['labels'].tolist()
    for k, v in ltc.items():
        temp[k][0] = v
        temp[k][1:10] = [(b'', 0.0) for i in range(1, 10)]
        temp_np = np.array(temp, dtype=[('label', 'S64'), ('prob', np.float)])
    # writing to file
    labelmap_save_process(fo, b['images'], b['stats'], temp_np, b['component_map'])

@app.route('/save_labelmap/<library_dir>/<image_key>/',  methods=['POST'])
def save_labelmap(library_dir, image_key):
    r = ""
    i = 1
    data = {}
    while 1:
        if request.form.get(str(i), None) is None:
            break;
        else:
            ex_labelmap = request.form['ex-'+str(i)]
            new_labelmap = request.form[str(i)]
            # r = r + "Counter: " + str(i) + " Ex-Labelmap-"+ str(i) +": " + ex_labelmap + " New-Labelmap-"+ str(i) +": " + new_labelmap + " \n ";
            if ex_labelmap != new_labelmap:
                s = new_labelmap.split(':')
                data[i-1] = tuple((s[0][2:-1].encode(), float(s[1])))
            i = i + 1
    # edit_labelmap('labelmap.lm2', 'labelmap_ex.lm2', {0:(b'l', 1.0), 1:(b'lm',  1.0)})
    edit_labelmap(labelmap_path(library_dir, image_key), copy_labelmap_path(library_dir, image_key), data)
    return render_template('save_labelmap.html', result = str(data)) 

if __name__ == '__main__':
    app.run(port=9999, host='0.0.0.0', debug=True)

