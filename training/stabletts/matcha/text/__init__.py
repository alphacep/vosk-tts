""" from https://github.com/keithito/tacotron """
from matcha.text import cleaners
from matcha.text.symbols import symbols
import re

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}  # pylint: disable=unnecessary-comprehension


#def text_to_sequence(text, cleaner_names):
#    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
#    Args:
#      text: string to convert to a sequence
#      cleaner_names: names of the cleaner functions to run the text through
#    Returns:
#      List of integers corresponding to the symbols in the text
#    """
#    sequence = []
#
#    clean_text = _clean_text(text, cleaner_names)
#    for symbol in clean_text:
#        symbol_id = _symbol_to_id[symbol]
#        sequence += [symbol_id]
#    return sequence, clean_text


wdic = {}
probs = {}
for line in open("db/dictionary", encoding='utf-8'):
   items = line.split()
   prob = float(items[1])
   if probs.get(items[0], 0) < prob:
       wdic[items[0]] = items[2:]
       probs[items[0]] = prob

from .ru_dictionary import convert

import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("rubert-base")
tokenizer = BertTokenizer.from_pretrained("rubert-base")



def get_bert_embeddings(text):
    with torch.no_grad():
        text = text.replace("+", "")
        inputs = tokenizer(text, return_tensors="pt")
#        print (inputs)
        text_inputs = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
#        print (text_inputs)

        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1).squeeze(0)

        pattern = "[-,.?!;:\"]"
        selected = []
        for i, t in enumerate(text_inputs):
            if t[0] != '#' and not re.match(pattern, t):
#                print (i, t)
                selected.append(i) 
        res = res[selected]
        return res




pattern = "([,.?!;:\"() ])"
def text_to_sequence1(text, cleaners_names):
    embeddings = get_bert_embeddings(text)

    phone_embeddings = [embeddings[0]]
    phonemes = ["^"]
    word_index = 1
    for word in re.split(pattern, text.lower()):
        if word == "":
            continue
        if re.match(pattern, word) or word == '-':
            phonemes.append(word)
            phone_embeddings.append(embeddings[word_index])
        elif word in wdic:
            for p in wdic[word]:
                phonemes.append(p)
                phone_embeddings.append(embeddings[word_index])
        else:
            for p in convert(word).split():
                phonemes.append(p)
                phone_embeddings.append(embeddings[word_index])
        if word != " ":
            word_index = word_index + 1
    phonemes.append("$")
    phone_embeddings.append(embeddings[-1])

    sequence = [_symbol_to_id[symbol] for symbol in phonemes]
    return sequence, phone_embeddings




# Kaldi style word position
def get_pos(x):
    if len(x) == 1:
        return [x[0] + "_S"]
    else:
        res = []
        for i, p in enumerate(x):
            if i == 0:
                res.append(p + "_B")
            elif i == len(x) - 1:
                res.append(p + "_E")
            else:
                res.append(p + "_I")
        return res

def text_to_sequence(text, cleaners_names):
    bert_embeddings = get_bert_embeddings(text)

    phonemes = [("^", [], 0, 0)]

    pattern = "(\.\.\.|- |[ ,.?!;:\"()])"
    text = text.replace("\n", " ")
    text = text.replace(" -", "- ") # Unify dash with other punctuations

    in_quote = 0
    cur_punc = []
    bert_word_index = 1

    for word in re.split(pattern, text.lower()):
        if word == "":
            continue

#        print ("!!!", word)

        if word == "\"":
            if in_quote == 1:
                in_quote = 0
            else:
                in_quote = 1
            continue

        if word == "- " or word == "-":
            cur_punc.append('-')
            continue

        if re.match(pattern, word) and word != " ":
            cur_punc.append(word)
            continue

        if word == " ":
            phonemes.append((' ', cur_punc, in_quote, bert_word_index))
            cur_punc = []
            continue


        if word in wdic:
            cur_punc = []
            for p in get_pos(wdic[word]):
                phonemes.append((p, [], in_quote, bert_word_index))
        else:
            cur_punc = []
            for p in get_pos(convert(word).split()):
                 phonemes.append((p, [], in_quote, bert_word_index))

        bert_word_index = bert_word_index + 1

    phonemes.append((" ", cur_punc, in_quote, bert_word_index))
    phonemes.append(("$", [], 0, bert_word_index))


    last_punc = " "
    last_sentence_punc = " "

    lp_phonemes = []
    phone_bert_embeddings = []
    for p in reversed(phonemes):
        if "..." in p[1]:
            last_sentence_punc = "..."
        elif "." in p[1]:
            last_sentence_punc = "."
        elif "!" in p[1]:
            last_sentence_punc = "!"
        elif "?" in p[1]:
            last_sentence_punc = "?"
        elif "-" in p[1]:
            last_sentence_punc = "-"
        if len(p[1]) > 0:
            last_punc = p[1][0]

        if len(p[1]) > 0:
            cur_punc = p[1][0]
        else:
            cur_punc = "_"

        lp_phonemes.append((_symbol_to_id[p[0]], _symbol_to_id[cur_punc], p[2], _symbol_to_id[last_punc], _symbol_to_id[last_sentence_punc]))
#        lp_phonemes.append((_symbol_to_id[p[0]], _symbol_to_id["!"], p[2], _symbol_to_id["!"], _symbol_to_id["!"]))
        if p[3] >= len(bert_embeddings):
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", orig_text, flush=True)
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        phone_bert_embeddings.append(bert_embeddings[p[3]])
    lp_phonemes = list(reversed(lp_phonemes))
    phone_bert_embeddings = list(reversed(phone_bert_embeddings))

#    for p in lp_phonemes:
#        print (p)
    return lp_phonemes, phone_bert_embeddings



pattern = "([,.?!;:\"() ])"
def text_to_ali(text, cleaners_names):
    phonemes = []
    for word in re.split(pattern, text.lower()):
        if word == "":
            continue
        if re.match(pattern, word) or word == '-':
            phonemes.append(word)
        elif word in wdic:
            phonemes.append("_".join(wdic[word]))
        else:
            phonemes.append("_".join(convert(word).split()))
    return "".join(phonemes)



def text_to_sequence_aligned(orig_text, text):
    bert_embeddings = get_bert_embeddings(orig_text)

    phonemes = [("^", [], 0, 0)]

    pattern = "(\.\.\.|- |[ ,.?!;:\"()])"
    text = text.replace("\n", " ")
    text = text.replace(" -", "- ") # Unify dash with other punctuations

    in_quote = 0
    cur_punc = []
    bert_word_index = 1

    for word in re.split(pattern, text):
        if word == "":
            continue

#        print ("!!!", word)

        if word == "\"":
            if in_quote == 1:
                in_quote = 0
            else:
                in_quote = 1
            continue

        if word == "- " or word == "-":
            cur_punc.append('-')
            continue

        if re.match(pattern, word) and word != " ":
            cur_punc.append(word)
            continue

        if word == " ":
            phonemes.append((' ', cur_punc, in_quote, bert_word_index))
            cur_punc = []
            continue

        for p in get_pos(word.split("_")):
           cur_punc = []
           phonemes.append((p, [], in_quote, bert_word_index))

        bert_word_index = bert_word_index + 1

    phonemes.append((" ", cur_punc, in_quote, bert_word_index))
    phonemes.append(("$", [], 0, bert_word_index))


    last_punc = " "
    last_sentence_punc = " "

    lp_phonemes = []
    phone_bert_embeddings = []
    for p in reversed(phonemes):
        if "..." in p[1]:
            last_sentence_punc = "..."
        elif "." in p[1]:
            last_sentence_punc = "."
        elif "!" in p[1]:
            last_sentence_punc = "!"
        elif "?" in p[1]:
            last_sentence_punc = "?"
        elif "-" in p[1]:
            last_sentence_punc = "-"
        if len(p[1]) > 0:
            last_punc = p[1][0]

        if len(p[1]) > 0:
            cur_punc = p[1][0]
        else:
            cur_punc = "_"

        lp_phonemes.append((_symbol_to_id[p[0]], _symbol_to_id[cur_punc], p[2], _symbol_to_id[last_punc], _symbol_to_id[last_sentence_punc]))
        if p[3] >= len(bert_embeddings):
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", orig_text, flush=True)
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", flush=True)
        phone_bert_embeddings.append(bert_embeddings[p[3]])
    lp_phonemes = list(reversed(lp_phonemes))
    phone_bert_embeddings = list(reversed(phone_bert_embeddings))

#    for p in lp_phonemes:
#        print (p)
    return lp_phonemes, phone_bert_embeddings



def text_to_sequence_aligned1(orig_text, text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    '''
#    print ("!!!!", orig_text, text)
    embeddings = get_bert_embeddings(orig_text)

    phone_embeddings = [embeddings[0]]
    phonemes = ["^"]
    pattern = "([,.?!;:\"() ])"
    word_index = 1
    for word in re.split(pattern, text):
        if word == "":
            continue
        if "_" in word:
            for p in word.split("_"):
                phonemes.append(p)
                phone_embeddings.append(embeddings[word_index])
        else:
            phonemes.append(word)
            phone_embeddings.append(embeddings[word_index])
        if word != " ":
            word_index = word_index + 1
            assert word_index < len(embeddings)
    phonemes.append("$")
    phone_embeddings.append(embeddings[-1])

#    print (len(phonemes), len(phone_embeddings))

    sequence = [_symbol_to_id[symbol] for symbol in phonemes]

    return sequence, phone_embeddings


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
