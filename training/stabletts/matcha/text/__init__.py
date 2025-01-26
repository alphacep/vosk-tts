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
for line in open("checkpoints/dictionary/dictionary", encoding='utf-8'):
   items = line.split()
   prob = float(items[1])
   if probs.get(items[0], 0) < prob:
       wdic[items[0]] = items[2:]
       probs[items[0]] = prob

from .ru_dictionary import convert

import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained("checkpoints/rubert-base")
tokenizer = BertTokenizer.from_pretrained("checkpoints/rubert-base")

def get_bert_embeddings(text):
    with torch.no_grad():
        text = text.replace("+", "")
        text_inputs = tokenizer.tokenize(text)
#        print (text_inputs)
        inputs = tokenizer(text, return_tensors="pt")

        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1).squeeze(0)

        selected = [0]
        for i, t in enumerate(text_inputs):
            if t[0] != '#':
                selected.append(i + 1) # skip BOS token
        selected.append(len(text_inputs) + 1)
        res = res[selected]
        return res

pattern = "([,.?!;:\"() ])"
def text_to_sequence(text, cleaners_names):
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

def text_to_sequence_aligned(orig_text, text):
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
