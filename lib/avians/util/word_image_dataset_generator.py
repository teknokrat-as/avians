import datetime
import os
import re
import shutil
import cv2
import argparse

import numpy as np
import numpy.random as nr

import avians.util.arabic_defs as auad
import avians.util.arabic as aua
import avians.util.image as aui

import logging
LOG = logging.getLogger(__name__)

def letter_group_image_dataset_generator(words_file="$HOME/Repository/avians/doc/arabic/arabic-wordlist-65k.txt",
                                         fonts=aua.system_arabic_fonts(),
                                         dpis=list(range(50, 600, 50)),
                                         n_images=100,
                                         output_dir="/tmp/avians-word-image-dataset/"):
    # for k, v in aua.arabic_to_visenc_dict.items():
    #     print("{}: {}".format(k, v))
        
    with open(os.path.expandvars(words_file)) as wf:
        words = [l.strip() for l in wf.readlines()]

    for i in range(n_images):
        word = nr.choice(words)
        letter_groups = divide_into_letter_groups(word)
        font = nr.choice(fonts)
        dpi = nr.choice(dpis)
        for lg in letter_groups:
            create_letter_group_image(lg, font, dpi, output_dir)
        if i % 1000 == 0: 
            print(i)
    return output_dir
        
def create_letter_group_image(lg, font, dpi, output_dir):
    otm = aua.visenc_to_arabic(lg)
    word_image = aui.get_gray_word_image(otm,
                                         fontname=font,
                                         dpi=dpi, 
                                         background=0, 
                                         foreground=255)
    
    components, stats = aui.text_image_segmentation(word_image)
    LOG.debug("=== lg ===")
    LOG.debug(lg)
    LOG.debug("=== components.shape ===")
    LOG.debug(components.shape)
    LOG.debug("=== stats.shape ===")
    LOG.debug(stats.shape)
    # aui.write_image_array(components, lg)
    stats_sorted_area = np.argsort(stats['area'])
    stem_label = aua.delete_visenc_diacritics(lg)
    li = stats_sorted_area[-1]
    largest = components[li]
    largest_dir = os.path.join(output_dir, stem_label)
    largest_file = os.path.join(largest_dir, 
                                "cc-{}-font-{}-dpi-{}-w-{}-h-{}-area-{}.png".format(stem_label,
                                                                                    font.replace(" ", "").lower(),
                                                                                    dpi,
                                                                                    stats[li]["w"],
                                                                                    stats[li]["h"],
                                                                                    stats[li]["area"]))
    LOG.debug("=== stem_label ===")
    LOG.debug(stem_label)
    
    if not os.path.isdir(largest_dir): 
        os.makedirs(largest_dir)
    cv2.imwrite(largest_file, largest)
    if len(stats) == 2:   # if there is a single diacritic
        di = stats_sorted_area[0]
        diacritic = components[di]
        diacritic_label = aua.delete_visenc_stems(lg)
        n_diacritics = (diacritic_label.count('o') + 
                        diacritic_label.count('u') + 
                        diacritic_label.count('i'))
        if n_diacritics == 1: 
            diacritic_label = re.sub(r'^[oui]?(.*)', r'\1', diacritic_label)
            LOG.debug("=== diacritic_label ===")
            LOG.debug(diacritic_label)
            diacritic_dir = os.path.join(output_dir, diacritic_label)
            diacritic_file = os.path.join(diacritic_dir,
                                          "cc-{}-font-{}-dpi-{}-w-{}-h-{}-area-{}.png".format(diacritic_label,
                                                                                              font.replace(" ", "").lower(),
                                                                                              dpi,
                                                                                              stats[di]["w"],
                                                                                              stats[di]["h"],
                                                                                              stats[di]["area"]))
        
            if not os.path.isdir(diacritic_dir): 
                os.makedirs(diacritic_dir)
            cv2.imwrite(diacritic_file, diacritic)
    
def decompose_visenc(visenc): 
    "decompose arabic text into a list of stems and diacritics"
    elements = []
    while len(visenc) > 0:
        visenc = visenc.strip()
        next_el = re.search(r"^" + huod.VISENC_REGEX, visenc)
        if next_el: 
            el = next_el.group(0)
            elements.append(el)
            visenc = visenc[len(el):]
        else:
            LOG.debug("=== Invalid Visenc ===")
            LOG.debug(visenc)
            visenc = visenc[1:]

    return elements

def divide_into_letter_groups(text): 
    "decompose arabic text into a list of stems and diacritics"
    elements = []
    visenc = aua.arabic_to_visenc(text)
    visenc = delete_ornaments(visenc)
    LOG.debug("=== visenc ===")
    LOG.debug(visenc)
    while len(visenc) > 0:
        visenc = visenc.strip()
        next_el = re.search(auad.VISENC_LETTER_GROUP_SPLITTER, visenc)
        LOG.debug("=== next_el ===")
        LOG.debug(next_el)
        
        if next_el: 
            el = next_el.group(0)
            elements.append(el.strip())
            visenc = visenc[len(el):]
        else:
            lg = re.match(auad.VISENC_LETTER_GROUP, visenc.strip())
            if lg: 
                el = lg.group(0)
                elements.append(el)
                visenc = visenc[len(el):]
            else: 
                LOG.debug("=== Invalid Visenc ===")
                LOG.debug(visenc)
                visenc = visenc[1:]
    return elements

def delete_ornaments(visenc): 
    return visenc.replace("*", "")

def flatten(l): 
    return [item for sublist in l for item in sublist]

DEFAULT_FONTS = ["Amiri Quran", "DejaVu Sans Mono", 
                 "Amiri", "Droid Arabic Naskh"
                 "Droid Arabic Kufi", 
                 "FreeSerif", "FreeMono"
                 "Droid Sans Arabic"]

def main():
    parser = argparse.ArgumentParser(description="""
    Arabic Word Image Dataset Generator

    It receives an Arabic text file and produces labeled component images. 

    """,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('doc', help="Document to produce the dataset")
    parser.add_argument('n', help="Number of components to produce")
    parser.add_argument('--output-dir', help="Image output directory.", 
                        default="/tmp/arabic-component-dataset-{}".format(nr.randint(10000)), 
                        required=False)
    parser.add_argument('--fonts', help='Comma separated list of font names',
                        default=None,
                        required=False)
    args = vars(parser.parse_args())
    print(args)

    if args['fonts']: 
        fonts = args['fonts'].split(',')
    else: 
        fonts = aua.system_arabic_fonts()
    letter_group_image_dataset_generator(args['doc'],
                                         n_images=int(args['n']),
                                         output_dir=args['output_dir'],
                                         fonts=fonts)

if __name__ == "__main__": 
    main()

