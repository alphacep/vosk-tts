""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols
import re

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    if symbol not in _symbol_to_id.keys():
      continue
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence

wdic = {}
for line in open("db/dictionary"):
    items = line.split()
    if items[0] not in wdic:
        wdic[items[0]] = items[1:]

from .ru_dictionary import convert

pattern = "([,.?!;:\"() ])"
def text_to_sequence_g2p(text):

    phonemes = ["^"]
    for word in re.split(pattern, text.lower()):
        if word == "":
            continue
        if re.match(pattern, word) or word == '-':
            phonemes.append(word)
        elif word in wdic:
            phonemes.extend(wdic[word])
        else:
            phonemes.extend(convert(word).split())
    phonemes.append("$")

    sequence = [_symbol_to_id[symbol] for symbol in phonemes]
    return sequence


def text_to_sequence_aligned(text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    '''
    phonemes = ["^"]
    for word in re.split(pattern, text):
        if word == "":
            continue
        if "_" in word:
            phonemes.extend(word.split("_"))
        else:
            phonemes.append(word)
    phonemes.append("$")

    sequence = [_symbol_to_id[symbol] for symbol in phonemes]

    return sequence

def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id.keys()]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
