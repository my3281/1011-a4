import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdz", "d": "sefx", "f": "drgc", "g": "fthv",
    "h": "gyjb", "j": "hukn", "k": "jilm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}
SYN_PROB = 0.3
TYPO_PROB = 0.15


def _get_synonym(word):
    if word[0].isupper():
        return None
    synsets = [x for x in wordnet.synsets(word) if x.pos() in ("a", "s", "r")]
    if not synsets:
        return None
    candidates = [l.name() for l in synsets[0].lemmas()
                  if "_" not in l.name() and l.name().lower() != word.lower()]
    if not candidates:
        return None
    return random.choice(candidates)


def _add_typo(word):
    positions = [i for i, c in enumerate(word) if c.lower() in QWERTY_NEIGHBORS]
    if not positions:
        return word
    i = random.choice(positions)
    new_c = random.choice(QWERTY_NEIGHBORS[word[i].lower()])
    if word[i].isupper():
        new_c = new_c.upper()
    return word[:i] + new_c + word[i + 1:]


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    tokens = word_tokenize(example["text"])
    new_tokens = []
    for tok in tokens:
        if tok.isalpha() and len(tok) > 3 and random.random() < SYN_PROB:
            syn = _get_synonym(tok)
            if syn:
                tok = syn
        if tok.isalpha() and len(tok) > 2 and random.random() < TYPO_PROB:
            tok = _add_typo(tok)
        new_tokens.append(tok)
    example["text"] = TreebankWordDetokenizer().detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
