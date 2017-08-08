
import unicodedata
import logging
LOG = logging.getLogger(__name__)

ARABIC_FONTS = """A_Nefel_Adeti
A_Nefel_Adeti_Qelew
A_Nefel_Botan
A_Nefel_Sereke
Amiri
Amiri Quran
Andalus
Arabic Typesetting
B Arabic Style
B Arshia
B Compset
B Davat
B Elham
B Esfehan
B Fantezy
B Farnaz
B Ferdosi
B Homa
B Kamran
B Kamran Outline
B Koodak
B Koodak Outline
B Majid Shadow
B Nasim
B Roya
B Sina
B Tabassom
B Titr
B Traffic
B Yagut
B Yekan
B Zar
Bold Italic Art
DejaVu Sans
DejaVu Sans
DejaVu Sans
DejaVu Sans Mono
DejaVu Sans Mono
DejaVu Sans,DejaVu Sans Condensed
Diwani Bent
Diwani Letter
Diwani Simple Outline
Diwani Simple Outline 2
Diwani Simple Striped
Droid Arabic Kufi
Droid Arabic Naskh
Droid Naskh Shift Alt
Droid Sans Arabic
Elham
Farsi Simple Outline
FreeMono
FreeSerif
Hacen Digital Arabia LT
Homa
KacstArt
KacstBook
KacstDecorative
KacstDigital
KacstFarsi
KacstLetter
KacstNaskh
KacstOffice
KacstOne
KacstPoster
KacstQurn
KacstScreen
KacstTitle
Koodak
Lateef
Led Italic Font
Mothanna
Nazli
Old Antic Bold
Old Antic Decorative
Old Antic Outline
Old Antic Outline Shaded
PT Bold Arch
PT Bold Broken
PT Bold Heading
PT Bold Mirror
PT Simple Bold Ruled
PakType Naqsh
PakType Tehreer
Pashtu Abdaali
Pashtu Asad
Pashtu Breshnik
Pashtu Kandahar
Pashtu pa' Storee
Riwaj
Roya
Scheherazade
Simple Indust Outline
Simple Indust Shaded
Simple Outline Pat
Simplified Arabic
Simplified Arabic Fixed
Terafik
Thabit
Titr
Traditional Arabic
kacstPen
mry_KacstQurn""".split('\n')

VISENC_REGEX = r'(((e|b|x|d|r|s|z|t|a|f|k|l|m|w|h|y|c)([uoi][0123456789]+){0,4})|(n[01234567890])|(\|\||<>|><|pp|_))|(p[0-9]{1,2})|(j[0-9]{2})'
VISENC_STEM_BEGIN_REGEX = r'(e|b|x|d|r|s|z|t|a|f|k|l|m|w|h|y|c|j)'
VISENC_STEM_CONT_REGEX = r'(b|x|s|z|t|a|f|k|l|m|h|y)'
VISENC_STEM_REGEX = r'(e|b|x|d|r|s|z|t|a|f|k|l|m|w|h|y|c|j)(b|x|s|z|t|a|f|k|l|m|h|y)*'
# H and Y are for h|| and y|| which ends a word
VISENC_STEM_END_REGEX = r'(|e|d|r|w|H|Y|c|n[0123456789])'
VISENC_DIACRITIC_REGEX = r'([uoi][0123456789][49]?)'
VISENC_LETTER_DIACRITICS_REGEX = r'([uo][1237])'
VISENC_BARE_LETTER_REGEX = '%s%s?'.format(VISENC_STEM_REGEX, VISENC_LETTER_DIACRITICS_REGEX)
# In recognition, used to decide whether we can be certain that we are at the end of a word. 
VISENC_WORD_END = r'^.*(b|x|s|z|t|a|f|k|l|m|h|y)([uo][0123456789]+)?$'
VISENC_DIGIT_REGEX = r'n[0123456789]'
VISENC_NUMBER_REGEX = r'(n[0123456789])+'
VISENC_DIVIDER_REGEX = r'\|\||<>|><|pp|_'
VISENC_PUNCTUATION_REGEX = r'p[0-9]'
VISENC_LATIN_REGEX = r'j[0-9]{2}'

VISENC_LETTER_GROUP_SPLITTER = r'^.*?((\s)|(n[0123456789])|((e|d|r|w|c)([uoi][0123456789][49]?)*)|(\|\||<>|><|pp|_))'
VISENC_LETTER_GROUP = r'(e|b|x|d|r|s|z|t|a|f|k|l|m|w|h|y|c)([uoi][0123456789][49]?)*(((b|x|s|z|t|a|f|k|l|m|h|y)([uoi][0123456789][49]?)*))*'

# Following two are determines whether we can be certain that we are at the end of a word
VISENC_POSSIBLE_WORD_END = r'.*(e|d|r|w|h|y|c)([uoi][0123456789][49])?$'
VISENC_CERTAIN_WORD_END = r'.*(b|x|s|z|t|a|f|k|l|m|\|\|)([uoi][0123456789][49])?$'

VISENC_VALID_LETTERS = ["c", "e", "x", "d", "r", "s", "z", "t", "a",
                        "_", "l", "m", "h", "w", "y", "b", "f", "k",
                        "n0", "n1", "n2", "n3", "n4", "n5", "n6",
                        "n7", "n8", "n9", "pp", "<>", "><", "eo6",
                        "eo5", "wo5", "eu5", "bo5", "bu1", "ho2",
                        "bo2", "bo3", "xu1", "xo1", "do1", "ro1",
                        "so3", "zo1", "to1", "ao1", "fo1", "fo2",
                        "lo5", "ko5", "bo1", "bu2", "bu3", "xu3",
                        "ro3", "ko7", "wo5", "lie", "lo5o3", "lo3o5",
                        "ko7o3", "ko3o7", "lilih", "lieo6", "lo5"] + ["j{:02}".format(i) for i in (list(range(1, 27)) + list(range(33, 59)))] + ['p2']

VISENC_POSSIBLE_MERGES = ["c", "e", "x", "d", "r", "s", "z", "t", "a",
                          "_", "l", "m", "h", "w", "y", "b", "f", "k",
                          "n0", "n1", "n2", "n3", "n4", "n5", "n6",
                          "n7", "n8", "n9", "pp", "<>", "><", "||",
                          "eo6", "eo5", "wo5", "eu5", "bo5", "bu1",
                          "ho2", "ho1o1", "bo2", "bo1o1", "bo3",
                          "bo1o2", "bo1o1o1", "bo1o1", "xu1", "xo1",
                          "do1", "ro1", "so3", "so1o1o1", "so1o1",
                          "so2o1", "so1o2", "zo1", "to1", "ao1",
                          "fo1", "fo2", "fo1o1", "lo5", "ko5", "bo1",
                          "bu2", "bu1u1", "bu3", "bu1u1u1", "bu1u2",
                          "bu2u1", "xu3", "xu1u1", "xu2", "xu1u1u1",
                          "xu2u1", "xu1u2", "ro3", "ro1o1o1", "ro1o2",
                          "ro2o1", "ko7", "wo5", "lie", "lo5o3",
                          "lo5o1o1o1", "lo5o1o2", "lo5o2o1", "lo3o5",
                          "lo1o1o1o5", "lo2o1o5", "lo1o2o5", "ko7o3",
                          "ko7o1o1o1", "ko7o1o2", "ko7o2o1", "ko3o7",
                          "ko1o1o1o7", "ko1o2o7", "ko2o1o7", "lilih",
                          "lieo6", "lo5"]

VISENC_EQUAL_DIACRITICS = [("u1u1", "u2"),
                           ("u1u1u1", "u3"),
                           ("u2u1", "u3"),
                           ("u1u2", "u3"),
                           ("o1o1", "o2"),
                           ("o1o1o1", "o3"),
                           ("o2o1", "o3"),
                           ("o1o2", "o3")]

VISENC_CANONICAL_DIACRITICS = [(r"u2", "u1-u1"),
                               (r"u3", "u1-u1-u1"),
                               (r"o2", "o1-o1"),
                               (r"o3", "o1-o1-o1")]

VISENC_CANONICAL_STEMS = [(r"y(.)", "b\g<1>")]

OTAPCHARTABLE = {
    "a5": "\u0627", 
    "a2": "\u0622",
    "a3": "\u0622",
    "e2": "\u0627\u064A",
    "e5": "\u0627",
    "i2": "\u0627\u064A",
    "i4": "\u0627",
    "i3": "\u0627\u064A",
    "i5": "\u0627",
    "o2": "\u0627\u0648",
    "o4": "\u0627",
    "o3": "\u0627\u0648",
    "o4": "\u0627",
    "u2": "\u0627\u0648",
    "u4": "\u0627",
    "u3": "\u0627\u0648",
    "u5": "\u0627",
    "a1": "\u0627",
    "a7": "\u0627",
    "a8": "\u0627\u0648",
    "a0": "\u0647",
    "a6": "\u0670",
    "a9": "\u064A",
    "e2": "\u064A",
    "e0": "\u0647",
    "i1": "\u064A",
    "i6": "\u064A",
    "i6": "\u0649",
    "i8": "\u0647",
    "i9": "\u0648\u064A",
    "i7": "\u064A",
    "o1": "\u0648",
    "o6": "\u0648", 
    "o7": "\u0648",
    "u1": "\u0648",
    "u6": "\u0648",
    "u8": "\u0649",
    "u7": "\u0648",
    "u9": "\u0649",
    "b": "\u0628",
    "b7": "\u0628",
    "c": "\u062C",
    "c7": "\u062C",
    "c2": "\u0686",
    "d": "\u062F",
    "d7": "\u062F",
    "d6": "\u0637",
    "d3": "\u0636", 
    "f": "\u0641",
    "g": "\u0643",
    "g7": "\u0643",
    "g6": "\u0643",
    "g4": "\u063A",
    "h": "\u0647",
    "h3": "\u062D",
    "h5": "\u062E",
    "j": "\u0698",
    "k": "\u0643",
    "k3": "\u0642",
    "l": "\u0644",
    "m": "\u0645",
    "n": "\u0646",
    "n6": "\u06A9",
    "p": "\u067E",
    "r": "\u0631",
    "s": "\u0633",
    "s2": "\u0634",
    "s3": "\u0635",
    "s5": "\u062B",
    "t": "\u062A",
    "t7": "\u0629",
    "t3": "\u0637",
    "v": "\u0648",
    "y": "\u064A",
    "z": "\u0632",
    "z3": "\u0638",
    "z4": "\u0636",
    "z5": "\u0630",
    "'2": "\u0639",
    "'3": "\u0621",
    "'4": "\u0654",
    "w2": "\u0651"
}


VISENC_SHORTCUT_LIST = [("A", "eo6"),
                        ("E", "eo5"),
                        ("B", "bu1"),
                        ("P", "bu3"),
                        ("T", "bo2"),
                        ("C", "xu1"),
                        ("Ç", "xu3"),
                        ("X", "xo1"),
                        ("Z", "ro1"),
                        ("J", "ro3"),
                        ("S", "so3"),
                        ("Ş", "so3"),
                        ("D", "zo1"), 
                        ("Ğ", "ao1"), 
                        ("F", "fo1"), 
                        ("Q", "fo2"), 
                        ("G", "ko7"), 
                        ("N", "bo1"), 
                        ("K", "lo5"), 
                        ("Y", "bu2"),
                        ("H", "h||")]

VISENC_SHORTHANDS = {k: v for k, v in VISENC_SHORTCUT_LIST}

VISENC_TABLE = [("c", "ء"), 
                ("eo6", "آ"), 
                ("e", "ا"),
                ("eo5", "أ"),
                ("eu5", chr(1575)),
                ("eu5", chr(1573)),  
                ("bu1", "ب"), 
                ("bu3", "پ"), 
                ("bo2", "ت"), 
                ("bo3", "ث"), 
                ("xu1", "ج"), 
                ("xu3", "چ"), 
                ("x", "ح"), 
                ("xo1", "خ"), 
                ("d", "د"), 
                ("do1", "ذ"), 
                ("r", "ر"), 
                ("ro1", "ز"), 
                ("ro3", "ژ"), 
                ("s", "س"), 
                ("so3", "ش"), 
                ("z", "ص"), 
                ("zo1", "ض"), 
                ("t", "ط"), 
                ("to1", "ظ"), 
                ("a", "ع"), 
                ("ao1", "غ"), 
                ("fo1", "ف"), 
                ("fo2", "ق"), 
                ("lo5", "ك"), 
                ("k", "ك"), 
                ("k", "ک"),
                ("ko7", "گ"), 
                ("ko3", "ڭ"), 
                ("lo5o3", "ڭ"), 
                ("l", "ل"), 
                ("m", "م"), 
                ("bo1", "ن"), 
                ("w", "و"), 
                ("wo5", "ؤ"), 
                ("h", "ه"), 
                ("ho2", "ة"), 
                ("y", "ی"),   # x6cc
                ("y", "ى"),   # x649
                ("bu2", "ي"), 
                ("yo5", "ئ"), 
                ("bo5", "ئ"), 
                ("n0", "۰"), 
                ("n1", "۱"), 
                ("n2", "۲"), 
                ("n3", "۳"), 
                ("n4", "۴"), 
                ("n5", "۵"), 
                ("n6", "۶"), 
                ("n7", "۷"), 
                ("n8", "۸"), 
                ("n9", "۹"), 
                ("&zwnj;", "\u200C"), 
                ("||", "\u200C"), 
                ("<>", "\u200C"), 
                ("&zwj;", "\u200D"), 
                ("><", "\u200D"), 
                ("&lrm;", "\u200E"), 
                ("&rlm;", "\u200F"), 
                ("&ls;", "\u2028"), 
                ("&ps;", "\u2028"), 
                ("&lre;", "\u202A"), 
                ("&rle;", "\u202B"), 
                ("&pdf;", "\u202C"), 
                ("&lro;", "\u202D"), 
                ("&rlo;", "\u202D"), 
                ("&bom;", "\uFEFF"), 
                ("o4", "َ"), 
                ("u4", "ِ"), 
                ("o9", "ُ"), 
                ("u44", "ٍ"), 
                ("o44", "ً"), 
                ("o99", "ٌ"), 
                ("o8", "ّ"), 
                ("o0", "ْ"), 
                ("o6", "\u0653"), 
                (" ", " "),
                ("p2", "،"),
                ("j01", "A"), 
                ("j02", "B"),
                ("j03", "C"),
                ("j04", "D"),
                ("j05", "E"),
                ("j06", "F"),
                ("j07", "G"),
                ("j08", "H"),
                ("j09", "I"),
                ("j10", "J"),
                ("j11", "K"),
                ("j12", "L"),
                ("j13", "M"),
                ("j14", "N"),
                ("j15", "O"),
                ("j16", "P"),
                ("j17", "Q"),
                ("j18", "R"),
                ("j19", "S"),
                ("j20", "T"),
                ("j21", "U"),
                ("j22", "V"),
                ("j23", "W"),
                ("j24", "X"),
                ("j25", "Y"),
                ("j26", "Z"),
                ("j33", "a"),
                ("j34", "b"),
                ("j35", "c"),
                ("j36", "d"),
                ("j37", "e"),
                ("j38", "f"),
                ("j39", "g"),
                ("j40", "h"),
                ("j41", "i"),
                ("j42", "j"),
                ("j43", "k"),
                ("j44", "l"),
                ("j45", "m"),
                ("j46", "n"),
                ("j47", "o"),
                ("j48", "p"),
                ("j49", "q"),
                ("j50", "r"),
                ("j51", "s"),
                ("j52", "t"),
                ("j53", "u"),
                ("j54", "v"),
                ("j55", "w"),
                ("j56", "x"),
                ("j57", "y"),
                ("j58", "z"),
                ]


VISENC_TO_UNICODE_L1 = {
    "c": "\u0621",
    "e": "\u0627",
    "x": "\u062D",
    "d": "\u062F",
    "r": "\u0631",
    "s": "\u0633",
    "z": "\u0635",
    "t": "\u0637",
    "a": "\u0639",
    "_": "\u0640",
    "l": "\u0644",
    "m": "\u0645",
    "h": "\u0647",
    "w": "\u0648",
    "y": "\u064A",
    "b": "\u066E",
    "f": "\u066F",
    "k": "\u06A9"
}


VISENC_TO_UNICODE_L2 = {

    "o4": "\u064E",
    "o9": "\u064F",
    "u4": "\u0650",
    "o8": "\u0651",
    "o0": "\u0652",
    "o6": "\u0653",
    "o5": "\u0654",
    "u5": "\u0655",

    "n0": "\u0660",
    "n1": "\u0661",
    "n2": "\u0662",
    "n3": "\u0663",
    "n4": "\u0664",
    "n5": "\u0665",
    "n6": "\u0666",
    "n7": "\u0667",
    "n8": "\u0668",
    "n9": "\u0669",

    "pp": "\u066D",
    "<>": "\u200C",
    "><": "\u200D"
}

VISENC_TO_UNICODE_L3 = {

    "eo6": "\u0622",
    "eo5": "\u0623",
    "wo5": "\u0624",
    "eu5": "\u0625",
    "bo5": "\u0626",

    "bu1": "\u0628",
    "ho2": "\u0629",
    "bo2": "\u062A",
    "bo3": "\u062B",
    "xu1": "\u062C",

    "xo1": "\u062E",
    "do1": "\u0630",
    "ro1": "\u0632",
    "so3": "\u0634",
    "zo1": "\u0636",
    "to1": "\u0638",
    "ao1": "\u063A",

    "fo1": "\u0641",
    "fo2": "\u0642",
    "lo5": "\u0643",
    "ko5": "\u0643",
    "bo1": "\u0646",
    "bu2": "\u064A",
    "bu3": "\u067E",
    
    "o44": "\u064B",
    "o99": "\u064C",
    "u44": "\u064D",

    "xu3": "\u0686",
    "ro3": "\u0698",

    "ko7": "\u06AF",

    "wo5": "\uFE86",
    "lie": "\u0644\u0627",

    "j01": "A", 
    "j02": "B",
    "j03": "C",
    "j04": "D",
    "j05": "E",
    "j06": "F",
    "j07": "G",
    "j08": "H",
    "j09": "I",
    "j10": "J",
    "j11": "K",
    "j12": "L",
    "j13": "M",
    "j14": "N",
    "j15": "O",
    "j16": "P",
    "j17": "Q",
    "j18": "R",
    "j19": "S",
    "j20": "T",
    "j21": "U",
    "j22": "V",
    "j23": "W",
    "j24": "X",
    "j25": "Y",
    "j26": "Z",
    "j33": "a",
    "j34": "b",
    "j35": "c",
    "j36": "d",
    "j37": "e",
    "j38": "f",
    "j39": "g",
    "j40": "h",
    "j41": "i",
    "j42": "j",
    "j43": "k",
    "j44": "l",
    "j45": "m",
    "j46": "n",
    "j47": "o",
    "j48": "p",
    "j49": "q",
    "j50": "r",
    "j51": "s",
    "j52": "t",
    "j53": "u",
    "j54": "v",
    "j55": "w",
    "j56": "x",
    "j57": "y",
    "j58": "z",

}

VISENC_TO_UNICODE_L5 = {
    "lo5o3": "\u06AD",
    "lo3o5": "\u06AD",
    "ko7o3": "\u06B4",
    "ko3o7": "\u06B4",
    "lilih": "\uFDF2",
    "lieo6": "\uFEF5"
}

VISENC_TO_UNICODE = {}
VISENC_TO_UNICODE.update(VISENC_TO_UNICODE_L1)
VISENC_TO_UNICODE.update(VISENC_TO_UNICODE_L2)
VISENC_TO_UNICODE.update(VISENC_TO_UNICODE_L3)
VISENC_TO_UNICODE.update(VISENC_TO_UNICODE_L5)

VISENC_NUMERIC_LABELS = [('', 0),
                         ('e', 1),
                         ('b', 2),
                         ('x', 3),
                         ('d', 4),
                         ('r', 5),
                         ('s', 6),
                         ('z', 7),
                         ('t', 8),
                         ('a', 9),
                         ('f', 10),
                         ('k', 11),
                         ('l', 12),
                         ('m', 13),
                         ('w', 14),
                         ('h', 15),
                         ('y', 16),
                         ('c', 17),
                         ('0', 18),
                         ('1', 19),
                         ('2', 20),
                         ('3', 21),
                         ('4', 22),
                         ('5', 23),
                         ('6', 24),
                         ('7', 25),
                         ('8', 26),
                         ('9', 27),
                         ('n0', 28),
                         ('n1', 29),
                         ('n2', 30),
                         ('n3', 31),
                         ('n4', 32),
                         ('n5', 33),
                         ('n6', 34),
                         ('n7', 35),
                         ('n8', 36),
                         ('n9', 37),
                         ('||', 38),
                         ('<>', 39),
                         ('><', 40),
                         ('pp', 41),
                         ('_', 42),
                         (' ', 43),
                         ('u', 44),
                         ('o', 45),
                         ('i', 46)]

VISENC_NUMERIC_DICT = {v: c for v, c in VISENC_NUMERIC_LABELS}
NUMERIC_VISENC_DICT = {c: v for v, c in VISENC_NUMERIC_LABELS}

ARABIC_TO_VISENC_DICT = {v: k for k, v in VISENC_TABLE}

VISENC_TO_ARABIC_DICT = {k: v for k, v in VISENC_TABLE}


class MalformedArabicStringException(Exception):
    def __init__(self, the_string):
        self.value = "Malformed Arabic String ({}) [{}]".format(the_string, ",".join([unicodedata.name(c) for c in the_string]))

    def __str__(self):
        return repr(self.value)

class InvalidVisencException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
