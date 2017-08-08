
import avians.util.arabic_defs as auad
import unicodedata
import re
import subprocess as sp

import logging
LOG = logging.getLogger(__name__)

visenc_shortcuts = auad.VISENC_SHORTCUT_LIST
visenc_table = auad.VISENC_TABLE

visenc_to_arabic_l1 = {ve: otm for ve, otm in visenc_table if len(ve) == 1}
visenc_to_arabic_l2 = {ve: otm for ve, otm in visenc_table if len(ve) == 2}
visenc_to_arabic_l3 = {ve: otm for ve, otm in visenc_table if len(ve) == 3}
visenc_to_arabic_l4 = {ve: otm for ve, otm in visenc_table if len(ve) == 4}
visenc_to_arabic_l5 = {ve: otm for ve, otm in visenc_table if len(ve) == 5}
arabic_to_visenc_dict = {otm: ve for ve, otm in visenc_table}
visenc_to_arabic_dict = {ve: otm for ve, otm in visenc_table}
visenc_shortcuts_dict = {vs: vl for vs, vl in visenc_shortcuts}

def visenc_to_arabic(visenc): 
    arabic = ""
    while len(visenc) > 0:
        found = False
        for visenc_len in [6, 5, 4, 3, 2, 1]:
            next_el = visenc[:visenc_len]
            # print("next_el for len {}: {}".format(visenc_len, next_el))
            if (not found) and (next_el in visenc_to_arabic_dict):
                # print("Found {} in dict".format(next_el))
                arabic += visenc_to_arabic_dict[next_el]
                # print("Arabic: {}".format(arabic))
                visenc = visenc[visenc_len:]
                # print("Visenc: {}".format(visenc))
                found = True
                break

        if not found: 
#            logging.debug("Cannot convert first char in {}".format(visenc))
            logging.debug("Cannot convert: {} after {} ".format(list(visenc), 
                                                                list(arabic)))
            arabic += visenc[0] if len(visenc) > 0 else ""
            visenc = visenc[1:]
            
    return arabic

def arabic_to_visenc(arabic_str):
    normalized_str = unicodedata.normalize("NFKC", arabic_str)
    visenc_str = ""
    for v in normalized_str:
        if v in arabic_to_visenc_dict:
            visenc_str += arabic_to_visenc_dict[v]
        else:
            try: 
                print("Cannot convert to Visenc: {} ({}) [{}]".format(v, 
                                                                      ord(v), 
                                                                      unicodedata.name(v)))
            except ValueError: 
                print("Cannot convert to Visenc: {} ({}) [#INVALID NAME#]".format(v, 
                                                                                  ord(v)))
            # visenc_str += v
            
    return visenc_str

def convert_visenc_shortcuts(visenc_short):
    visenc_long = visenc_short
    for vs, vl in visenc_shortcuts_dict.items():
        visenc_long = visenc_long.replace(vs, vl)
    return visenc_long

def delete_visenc_diacritics(visenc): 
    return re.sub(auad.VISENC_DIACRITIC_REGEX, "", visenc)

def delete_visenc_stems(visenc): 
    return re.sub(auad.VISENC_STEM_REGEX, "", visenc)

def split_visenc(visenc): 
    "decompose visenc text into a list of stems and diacritics"
    elements = []
    while len(visenc) > 0:
        visenc = visenc.strip()
        next_el = re.search(auad.VISENC_LETTER_GROUP_SPLITTER,
                            visenc)
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

def canonical_diacritics(visenc): 
    diacritics = delete_visenc_stems(visenc)
    diacritics = diacritics.replace("u", "-u").replace("o", "-o").replace("i", "-i")
    for pat, subst in auad.VISENC_CANONICAL_DIACRITICS:
        diacritics = re.sub(pat, subst, diacritics)
    diacritics = "-".join(sorted(diacritics.split("-")))
    return diacritics

def canonical_stem(visenc): 
    stem = delete_visenc_diacritics(visenc)
    for pat, subst in auad.VISENC_CANONICAL_STEMS:
        stem = re.sub(pat, subst, stem)
    return stem

def canonical_visenc(visenc): 
    cstem = canonical_stem(visenc)
    cdiac = canonical_diacritics(visenc)
    return cstem + "#" + cdiac

def is_stem(c): 
    return not is_diacritic(c)

def is_diacritic(c): 
    return re.match(auad.VISENC_DIACRITIC_REGEX, c)
     
smalls_table = "aâābcçdefgğhıiîjklmnoôöprsştuûūüvyzqxw"
capits_table = "AÂĀBCÇDEFGĞHIİÎJKLMNOÔÖPRSŞTUÛŪÜVYZQXW"

s2c_tbl = str.maketrans(smalls_table, capits_table)
c2s_tbl = str.maketrans(capits_table, smalls_table)

def turkish_upper(turkish_str): 
    """Convert turkish letters to uppercase.
    
    e.g.
    i -> İ
    â -> Â
    """
    
    return turkish_str.translate(s2c_tbl)

def turkish_lower(turkish_str):
    "Convert Turkish letters to lowercase"
    return turkish_str.translate(c2s_tbl)


def turkish_capitalize(turkish_str): 
    """Capitalize each word in turkish str
    
    e.g.

    iBraHim -> İbrahim
    
    âSUDE -> Âsude

    """

    lower_str = turkish_lower(turkish_str)
    upper_str = turkish_upper(turkish_str)

    res_str = ""
    for i, c in enumerate(turkish_str):
        previous = turkish_str[i-1] if i > 0 else ""
        # Capitalization rules
        # Beginning of string
        if i == 0: 
            res_str += upper_str[i]
        # After a space or tab or newline
        elif previous in [" ", "\t", "\n"]:
            res_str += upper_str[i]
        else: 
            res_str += lower_str[i]

    return res_str

def system_arabic_fonts(output='family'):
    "output maybe file or family or any other attribute returned by fc-query"
    fontlist_raw = sp.check_output(['/usr/bin/fc-list', '-f', '%{{{}}}\n'.format(output), ':lang=ar'])
    # print(fontlist_raw)
    fontlist = fontlist_raw.decode('ascii').split('\n')
    fontlist = [re.sub(r'(.*?)(,.*)$', r'\1', f) for f in fontlist]
    fontlist = [f for f in fontlist if len(f) > 0]
    # remove duplicates
    fontlist  = list(set(fontlist))
    
    return fontlist
