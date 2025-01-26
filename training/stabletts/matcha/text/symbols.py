""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
#_pad = "_"
#_punctuation = ';:,.!?¡¿—…"«»“” '
#_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
#_letters_ipa = (
#    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
#)


pmap = {
        "_": [
            0
        ],
        "^": [
            1
        ],
        "$": [
            2
        ],
        " ": [
            3
        ],
        "!": [
            4
        ],
        "\"": [
            5
        ],
        "(": [
            6
        ],
        ")": [
            7
        ],
        ",": [
            8
        ],
        "-": [
            9
        ],
        ".": [
            10
        ],
        ":": [
            11
        ],
        ";": [
            12
        ],
        "?": [
            13
        ],
        "a0": [
            14
        ],
        "a1": [
            15
        ],
        "b": [
            16
        ],
        "bj": [
            17
        ],
        "c": [
            18
        ],
        "ch": [
            19
        ],
        "d": [
            20
        ],
        "dj": [
            21
        ],
        "e0": [
            22
        ],
        "e1": [
            23
        ],
        "f": [
            24
        ],
        "fj": [
            25
        ],
        "g": [
            26
        ],
        "gj": [
            27
        ],
        "h": [
            28
        ],
        "hj": [
            29
        ],
        "i0": [
            30
        ],
        "i1": [
            31
        ],
        "j": [
            32
        ],
        "k": [
            33
        ],
        "kj": [
            34
        ],
        "l": [
            35
        ],
        "lj": [
            36
        ],
        "m": [
            37
        ],
        "mj": [
            38
        ],
        "n": [
            39
        ],
        "nj": [
            40
        ],
        "o0": [
            41
        ],
        "o1": [
            42
        ],
        "p": [
            43
        ],
        "pj": [
            44
        ],
        "r": [
            45
        ],
        "rj": [
            46
        ],
        "s": [
            47
        ],
        "sch": [
            48
        ],
        "sh": [
            49
        ],
        "sj": [
            50
        ],
        "t": [
            51
        ],
        "tj": [
            52
        ],
        "u0": [
            53
        ],
        "u1": [
            54
        ],
        "v": [
            55
        ],
        "vj": [
            56
        ],
        "y0": [
            57
        ],
        "y1": [
            58
        ],
        "z": [
            59
        ],
        "zh": [
            60
        ],
        "zj": [
            61
        ]
}

symbols = list(pmap.keys())


# Export all symbols:
#symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")
