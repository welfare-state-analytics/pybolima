SPECIAL_CHARS = {
    'hyphens': '-‐‑⁃‒–—―',
    'minuses': '-−－⁻',
    'pluses': '+＋⁺',
    'slashes': '/⁄∕',
    'tildes': '~˜⁓∼∽∿〜～',
    'apostrophes': "'’՚Ꞌꞌ＇",
    'single_quotes': "'‘’‚‛",
    'double_quotes': '"“”„‟',
    'accents': '`´',
    'primes': '′″‴‵‶‷⁗',
}

# SPECIAL_CHARS_ESCAPED = {
#     'hyphens': '-\u2010\u2011\u2043\u2012\u2013\u2014\u2015',
#     'minuses': '-\u2212\uff0d\u207b',
#     'pluses': '+\uff0b\u207a',
#     'slashes': '/\u2044\u2215',
#     'tildes': '~\u02dc\u2053\u223c\u223d\u223f\u301c\uff5e',
#     'apostrophes': "'\u2019\u055a\ua78b\ua78c\uff07",
#     'single_quotes': "'\u2018\u2019\u201a\u201b",
#     'double_quotes': '"\u201c\u201d\u201e\u201f',
#     'accents': '`\xb4',
#     'primes': '\u2032\u2033\u2034\u2035\u2036\u2037\u2057',
# }

"""Cretae translations that maps characters to first character in each string"""
SPECIAL_CHARS_GROUP_TRANSLATIONS = {k: str.maketrans(v[1:], v[0] * (len(v) - 1)) for k, v in SPECIAL_CHARS.items()}

# ALL_IN_ONE_TRANSLATION = str.maketrans(
#     *list(map(''.join, zip(*[(v[1:], v[0] * (len(v) - 1)) for _, v in SPECIAL_CHARS.items()])))
# )

ALL_IN_ONE_TRANSLATION = str.maketrans(
    *[
        '‐‑⁃‒–—―−－⁻＋⁺⁄∕˜⁓∼∽∿〜～’՚Ꞌꞌ＇‘’‚‛“”„‟´″‴‵‶‷⁗',
        '----------++//~~~~~~~\'\'\'\'\'\'\'\'\'""""`′′′′′′',
    ]
)


def normalize_characters(text: str, groups: str = None) -> str:

    if groups is None:
        return text.translate(ALL_IN_ONE_TRANSLATION)

    for group in groups.split(","):
        text = text.translate(SPECIAL_CHARS_GROUP_TRANSLATIONS[group])

    return text
