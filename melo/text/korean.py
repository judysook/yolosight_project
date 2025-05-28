import re
from jamo import hangul_to_jamo
from transformers import AutoTokenizer

def text_normalize(text):
    text = text.lower().strip()
    return text

def korean_text_to_phonemes(text):
    text = text_normalize(text)
    return "".join(list(hangul_to_jamo(text)))

def distribute_phone(n_phone, n_word):
    return [1] * n_word if n_word > 0 else [1]

model_id = 'kykim/bert-kor-base'
tokenizer = AutoTokenizer.from_pretrained(model_id)

def g2p(norm_text):
    tokenized = tokenizer.tokenize(norm_text)
    phs = []
    word2ph = []

    for token in tokenized:
        phones = korean_text_to_phonemes(token)
        phs += phones
        word2ph += [len(phones)]

    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

def get_bert_feature(text, word2ph, device='cuda'):
    return None  # BERT 생략




