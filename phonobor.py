#!/usr/bin/env python
# coding: utf-8

# In[22]:


get_ipython().system('pip install --upgrade matplotlib')
get_ipython().system('pip install --upgrade nltk')
get_ipython().system('pip install --upgrade numpy')
get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install --upgrade treetaggerwrapper')


# In[1]:


from collections import defaultdict, namedtuple
import csv
import gzip
from itertools import islice
import re
from matplotlib.pyplot import figure, show, rc, subplots
from mpl_toolkits.mplot3d import Axes3D
import nltk
from nltk.stem.snowball import SnowballStemmer
from numpy import arange, array, meshgrid
from pandas import read_csv
from sklearn.metrics.pairwise import cosine_similarity
import treetaggerwrapper


# Download parallel corpora from 
# http://opus.nlpl.eu/
# 
# OpenSubtitles Moses-format
# 
# https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/fi-ru.txt.zip
# 
# put to directory 'dl' and unzip and gzip unzipped files separately:
# gzip OpenSubtitles.fi-ru.ru
# gzip OpenSubtitles.fi-ru.fi
# 
# Install TreeTagger from:
# https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/#Windows
# https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-windows-3.2.2.zip
# 
# unzip package file to:
# C:\TreeTagger
# 
# Put parameter files for russian and finnish to
# C:\TreeTagger\lib
# from
# https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/#parfiles
# https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/finnish.par.gz
# https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/russian.par.gz
# 
# decompress these files with
# gzip -d finnish.par.gz
# gzip -d russian.par.gz
# 
# Install Perl from:
# 
# http://www.activestate.com/activeperl/

# In[2]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[3]:


def remove_english_ners(sentence):
    ner_tags = {(' '.join(c[0] for c in chunk), chunk.label()) for chunk in nltk.ne_chunk(
        nltk.pos_tag(nltk.word_tokenize(sentence))) if hasattr(chunk, 'label') and chunk.label() != 'GPE'}
    for ner_tag in ner_tags:
        while True:
            index = sentence.find(ner_tag[0])
            if index == -1:
                break
            sentence = sentence[:index] + sentence[index + len(ner_tag[0]) + 1:]
    return sentence


# In[4]:



features = [
    'close_', 'close_mid', 'open_mid', 'open_', 'high', 'mid', 'low', 'front', 'back', 'wide', 'round_',
    'tenuis', 'media', 'sibilant', 'spirant', 'nasal', 'tremulant', 'lateral', 'semivowel',
    'bilabial', 'labiodental', 'pro', 'medio', 'post', 'palatal', 'velar', 'laryng', 'voiced', 'consonant',
    'non_palatalization', 'palatalization']

Phone = namedtuple('Phone', features, defaults=(0,) * len(features))

finnish_phones = {
    'a': Phone(open_=1, low=1, back=1, wide=1),
    'b': Phone(media=1, bilabial=1, voiced=1, consonant=1),
    'c': Phone(tenuis=1, velar=1),
    'd': Phone(media=1, medio=1, voiced=1),
    'e': Phone(close_mid=1, open_mid=1, mid=1, front=1, wide=1),
    'f': Phone(spirant=1, labiodental=1, voiced=0, consonant=1),
    'g': Phone(media=1, velar=1, voiced=1, consonant=1),
    'h': Phone(spirant=1, laryng=1, voiced=0, consonant=1),
    'i': Phone(close_=1, high=1, front=1, wide=1),
    'j': Phone(semivowel=1, palatal=1, consonant=1),
    'k': Phone(tenuis=1, velar=1, voiced=0, consonant=1),
    'l': Phone(lateral=1, pro=1, medio=1, post=1, consonant=1),
    'm': Phone(nasal=1, bilabial=1, voiced=1, consonant=1),
    'n': Phone(nasal=1, pro=1, medio=1, post=1, voiced=1, consonant=1),
    'o': Phone(close_mid=1, open_mid=1, mid=1, back=1, round_=1),
    'p': Phone(tenuis=1, bilabial=1, voiced=0, consonant=1),
    'q': Phone(tenuis=1, velar=1, consonant=1),
    'r': Phone(tremulant=1, pro=1, medio=1, post=1, voiced=1, consonant=1),
    's': Phone(sibilant=1, pro=1, medio=1, voiced=0, consonant=1),
    't': Phone(tenuis=1, pro=1, voiced=0, consonant=1),
    'u': Phone(close_=1, high=1, back=1, round_=1),
    'v': Phone(semivowel=1, labiodental=1, voiced=1, consonant=1),
    'w': Phone(semivowel=1, labiodental=1, consonant=1),
    'x': Phone(tenuis=1, velar=1, consonant=1),
    'y': Phone(close_=1, high=1, front=1, round_=1),
    'z': Phone(sibilant=1, pro=1, medio=1, voiced=1, consonant=1),
    'å': Phone(close_mid=1, open_mid=1, mid=1, back=1, round_=1),
    'ä': Phone(open_=1, low=1, front=1, wide=1),
    'ö': Phone(close_mid=1, open_mid=1, mid=1, front=1, round_=1),
    'ij': Phone(semivowel=1, palatal=1, consonant=1),
    'ng': Phone(nasal=1, velar=1, consonant=1),
    'ts': (Phone(tenuis=1, pro=1, voiced=0, consonant=1), Phone(sibilant=1, pro=1, medio=1, voiced=0, consonant=1)),
    'sh': Phone(sibilant=1, medio=1, wide=1, voiced=0, consonant=1)
}

finnish_capital = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ'
finnish_small   = 'abcdefghijklmnopqrstuvwxyzåäö'
finnish_vowels  = 'aeiouyåäö'

def clean_finnish(
    s):
    return ''.join(filter(lambda c: c in finnish_small, map(lambda l: finnish_small[finnish_capital.find(l)] if l in finnish_capital else l, s)))

russian_capital = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
russian_small   = 'абвгдеёжзийклмнопрcтуфхцчшщъыьэюя'
russian_vowels  = 'аийоуыэяеёю'
russian_capital1 = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
russian_small1   = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
russian_letters = russian_capital + russian_small + russian_capital1 + russian_small1

def check_russian(
    s):
    return ''.join(filter(lambda c: c in russian_letters, map(lambda l: russian_small[russian_capital1.find(l)] if l in russian_capital1 else (russian_small[russian_small1.find(l)] if l in russian_small1 else l), s)))

def clean_russian(
    s):
    return ''.join(filter(lambda c: c in russian_small, map(lambda l: russian_small[russian_capital.find(l)] if l in russian_capital else l, check_russian(s))))

russian_phones = {
    'а': Phone(open_=1, low=1, back=1, wide=1),
    'б': Phone(medio=1, bilabial=1, voiced=1, consonant=1),
    'в': Phone(semivowel=1, labiodental=1, voiced=1, consonant=1),
    'г': Phone(media=1, velar=1, voiced=1, consonant=1),
    'д': Phone(media=1, medio=1, voiced=1, consonant=1),
    'е': Phone(close_mid=1, open_mid=1, mid=1, front=1, wide=1),
    'ё': (Phone(semivowel=1, palatal=1, consonant=1), Phone(close_mid=1, open_mid=1, mid=1, back=1, round_=1)),
    'ж': Phone(sibilant=1, medio=1, wide=1, voiced=1, consonant=1),
    'з': Phone(sibilant=1, pro=1, medio=1, voiced=1, consonant=1),
    'и': Phone(close_=1, high=1, front=1, wide=1),
    'й': Phone(semivowel=1, palatal=1),
    'к': Phone(tenuis=1, velar=1, voiced=0, consonant=1),
    'л': Phone(lateral=1, pro=1, medio=1, post=1, consonant=1),
    'м': Phone(nasal=1, bilabial=1, voiced=1, consonant=1),
    'н': Phone(nasal=1, pro=1, medio=1, post=1, voiced=1, consonant=1),
    'о': Phone(close_mid=1, open_mid=1, mid=1, back=1, round_=1),
    'п': Phone(tenuis=1, bilabial=1, voiced=0, consonant=1),
    'р': Phone(tremulant=1, pro=1, medio=1, post=1, voiced=1, consonant=1),
    'c': Phone(sibilant=1, pro=1, medio=1, voiced=0, consonant=1),
    'т': Phone(tenuis=1, pro=1, voiced=0, consonant=1),
    'у': Phone(close_=1, high=1, back=1, round_=1),
    'ф': Phone(spirant=1, labiodental=1, voiced=0, consonant=1),
    'х': Phone(spirant=1, laryng=1, voiced=0, consonant=1),
    'ц': (Phone(tenuis=1, pro=1, voiced=0, consonant=1), Phone(sibilant=1, pro=1, voiced=0, consonant=1)),
    'ч': (Phone(tenuis=1, pro=1, voiced=0, consonant=1), Phone(sibilant=1, pro=1, medio=1, voiced=0, consonant=1)),
    'ш': Phone(sibilant=1, medio=1, wide=1, voiced=0, consonant=1),
    'щ': Phone(sibilant=1, pro=1, wide=1, voiced=0, consonant=1),
    'ъ': Phone(non_palatalization=1),
    'ы': Phone(close_=1, high=1, front=1, back=1, wide=1),
    'ь': Phone(palatalization=1),
    'э': Phone(close_mid=1, open_mid=1, mid=1, front=1, wide=1),
    'ю': (Phone(semivowel=1, palatal=1, consonant=1), Phone(close_=1, high=1, back=1, round_=1)),
    'я': (Phone(semivowel=1, palatal=1, consonant=1), Phone(open_=1, low=1, back=1, wide=1)),
    'cл': Phone(lateral=1, pro=1, medio=1, post=1, consonant=1),
    'хл': Phone(lateral=1, pro=1, medio=1, post=1, consonant=1),
    'вт': Phone(spirant=1, labiodental=1, voiced=0, consonant=1)
}


# In[10]:


stemmer_russian = SnowballStemmer('russian')
stemmer_finnish = SnowballStemmer('finnish')

def flattened_phone_vector(
    word, phones
):
    pv = []
    index = 0
    while index < len(word):
        phone = None
        if index < len(word) - 1:
            if word[index + 1] in phones:
                next_phone = phones[word[index + 1]]
                current_phone = phones[word[index]]
                if isinstance(next_phone, Phone):
                    if getattr(next_phone, 'non_palatalization') == 1:
                        if isinstance(current_phone, Phone):
                            phone = current_phone._replace(palatal=0)
                        else:
                            phone = list(current_phone)
                            phone[-1] = phone[-1]._replace(palatal=0)
                            phone = tuple(phone)
                        index += 1
                    elif getattr(next_phone, 'palatalization') == 1:
                        if isinstance(current_phone, Phone):
                            phone = current_phone._replace(palatal=1)
                        else:
                            phone = list(current_phone)
                            phone[-1] = phone[-1]._replace(palatal=1)
                            phone = tuple(phone)
                        index += 1
            if not phone:
                letters = word[index:index + 2]
                if letters in phones:
                    phone = phones[letters]
                    index += 1
        if not phone:
            phone = phones[word[index]]
        if isinstance(phone, Phone):
            for value in phone:
                pv.append(value)
        else:
            for a_phone in phone:
                for value in a_phone:
                    pv.append(value)
        index += 1
    return pv

def read_words(
    contexts1,
    contexts2,
    token_tags1,
    token_tags2,
    filename1,
    filename2,
    clean1,
    clean2,
    stemmer1,
    stemmer2,
    phones1, phones2,
    vowels1, vowels2,
    result_filename,
    offset,
    verbose
):
    print('read_words', filename1, filename2)
    found_tokens1 = set()
    found_tokens2 = set()
    count = 0
    number = 1
    word_class_counts = defaultdict(lambda: 0)
    with open(result_filename, mode='at', encoding='utf-8-sig', newline='') as result_file:
        csv_writer = csv.writer(result_file, dialect='excel-tab', quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerow(['COUNT', 'SIM_VALUE', 'SEM_VALUE', 'STEM1', 'STEM2', 'TOKEN1', 'TOKEN2', 'TAG1', 'TAG2', 'LINE1', 'LINE2'])
        with (gzip.open(filename1, mode='rt', encoding='utf-8')
                if filename1.endswith('.gz') else open(filename1, mode='rt', encoding='utf-8')) as file1:
            with (gzip.open(filename2, mode='rt', encoding='utf-8')
                if filename2.endswith('.gz') else open(filename2, mode='rt', encoding='utf-8')) as file2:
                for line1, line2 in zip(islice(file1, offset, None), islice(file2, offset, None)):
                    toks1 = line1.split()
                    toks2 = line2.split()
                    for tok1 in toks1:
                        tok1 = clean1(tok1)
                        if not tok1:
                            continue
                        tok1_found = tok1 in found_tokens1
                        tok1a = tok1
                        tok1 = stemmer1.stem(tok1)
                        if tok1 not in token_tags1:
                            continue
                        tag1 = token_tags1[tok1]
                        for tok2 in toks2:
                            tok2 = clean2(tok2)
                            if not tok2:
                                continue
                            if tok1_found and tok2 in found_tokens2:
                                continue
                            tok2a = tok2
                            tok2 = stemmer2.stem(tok2)
                            if tok2 not in token_tags2:
                                continue
                            tag2 = token_tags2[tok2]
                            if tag1[0] != tag2[0] and tag1 != 'NON-TWOL' and tag2 != 'NON-TWOL':
                                continue
                            similarity_value = similarity(
                                tok1a, tok2a, phones1, phones2, vowels1, vowels2)
                            if not similarity_value:
                                continue
                            if similarity_value < 0.7:
                                continue
                            context_similarity_value = context_similarity(
                                contexts1, contexts2, None, None, tok1, tok2)
                            if context_similarity_value < 0.4:
                                continue
                            count += 1
                            word_class_counts[tag1[0]] += 1
                            csv_writer.writerow([
                                count, similarity_value, context_similarity_value, tok1, tok2, tok1a, tok2a, tag1, tag2,
                                ' ' + line1.strip(), ' ' + line2.strip()])
                            if verbose:
                                print(count, similarity_value, context_similarity_value, tok1, tok2, tok1a, tok2a, tag1, tag2)
                                print(line1.strip())
                                print(line2.strip())
                                print(dict(word_class_counts))
                                print()
                            found_tokens1.add(tok1a)
                            found_tokens2.add(tok2a)
                    if number % 100 == 0:
                        print('\r' + str(number), end='')
                    number += 1
    print('\r' + str(number))

def similarity(
    word1, word2,
    phones1, phones2,
    vowels1, vowels2,
    show=False
):
    word1a = remove_consonants(reduce_diphtongs(remove_doubles(word1), vowels1), vowels1)
    word2a = remove_consonants(reduce_diphtongs(remove_doubles(word2), vowels2), vowels2)
    if abs(len(word1a) - len(word2a)) > 2 or len(word1a) < 2 or len(word2a) < 2:
        return None
    pv1 = flattened_phone_vector(word1a, phones1)
    pv2 = flattened_phone_vector(word2a, phones2)
    if len(pv1) > len(pv2):
        pv2 += ([0] * (len(pv1) - len(pv2)))
    elif len(pv2) > len(pv1):
        pv1 += ([0] * (len(pv2) - len(pv1)))
    sim = cosine_similarity(array(pv1).reshape(1, -1), array(pv2).reshape(1, -1))[0][0]
    if show:
        print(word1, word2, sim)
    return sim

def russian_finnish_similarity(
    word1, word2,
    show=False
):
    tok1 = clean_russian(word1)
    tok2 = clean_finnish(word2)
    return similarity(tok1, tok2, russian_phones, finnish_phones, russian_vowels, finnish_vowels, show)

def remove_doubles(
    word
):
    reduced_word = ''
    for index in range(0, len(word)):
        if index >= len(word) - 1 or word[index + 1] != word[index]:
            reduced_word += word[index]
    return reduced_word

def remove_consonants(
    word,
    vowels
):
    if len(word) < 3:
        return word
    reduced_word = ''
    index = 0
    while index < len(word):
        if index == 0 and index < len(word) - 1 and vowels.find(word[index]) == -1 and vowels.find(word[index + 1]) == -1:
            reduced_word += word[index + 1]
            index += 1
        else:
            reduced_word += word[index]
        index += 1
    return reduced_word

def reduce_diphtongs(
    word,
    vowels
):
    if len(word) < 3:
        return word
    reduced_word = ''
    index = 0
    while index < len(word):
        reduced_word += word[index]
        if index < len(word) - 1 and vowels.find(word[index]) != -1 and vowels.find(word[index + 1]) != -1:
            index += 1
        index += 1
    return reduced_word

def read_contexts(
    filename1,
    filename2,
    clean1,
    clean2,
    stemmer1,
    stemmer2,
    lang1, lang2,
    offset
):
    print('read_contexts', filename1, filename2)
    tagger1 = treetaggerwrapper.TreeTagger(TAGLANG=lang1)
    tagger2 = treetaggerwrapper.TreeTagger(TAGLANG=lang2)
    contexts1 = defaultdict(lambda: defaultdict(lambda: 0))
    contexts2 = defaultdict(lambda: defaultdict(lambda: 0))
    token_tags1 = {}
    token_tags2 = {}
    tok_indices = {}
    tok_index = 0
    count = 1
    tok_indices2 = {}
    with (gzip.open(filename1, mode='rt', encoding='utf-8')
            if filename1.endswith('.gz') else open(filename1, mode='rt', encoding='utf-8')) as file1:
        with (gzip.open(filename2, mode='rt', encoding='utf-8')
                if filename2.endswith('.gz') else open(filename2, mode='rt', encoding='utf-8')) as file2:
            for line1, line2 in zip(islice(file1, offset, None), islice(file2, offset, None)):
                toks1 = line1.split()
                toks2 = line2.split()
                toks = []
                for tok1 in toks1:
                    tok1 = clean1(tok1)
                    if not tok1:
                        continue
                    tok1a = tok1
                    tok1 = stemmer1.stem(tok1)
                    if not tok1:
                        continue
                    if tok1 not in tok_indices:
                        tags1 = tagger1.tag_text(text=tok1a, tagonly=True)
                        tag1 = tags1[0].split('\t')[1]
                        if tag1.startswith('C'):
                            tok_indices[tok1] = -1
                            continue
                        tok_indices[tok1] = tok_index
                        tok_index += 1
                        token_tags1[tok1] = tag1
                    if tok1 in tok_indices and tok_indices[tok1] != -1:
                        toks.append(tok1)
                for tok in toks:
                    context1 = contexts1[tok]
                    for tok in toks:
                        context1[tok_indices[tok]] += 1
                for tok2 in toks2:
                    tok2b = tok2
                    tok2 = clean2(tok2)
                    if not tok2:
                        continue
                    tok2a = tok2
                    tok2 = stemmer2.stem(tok2)
                    if not tok2:
                        continue
                    if not remove_english_ners(tok2b[0] + tok2[1:]):
                        continue
                    if tok2 not in tok_indices2:
                        tags2 = tagger2.tag_text(text=tok2a, tagonly=True)
                        tag2 = tags2[0].split('\t')[1]
                        if tag2.startswith('C'):
                            tok_indices2[tok2] = -1
                            continue
                        tok_indices2[tok2] = 1
                        token_tags2[tok2] = tag2
                    if tok2 in tok_indices2 and tok_indices2[tok2] != -1:
                        context2 = contexts2[tok2]
                        for tok in toks:
                            context2[tok_indices[tok]] += 1
                if count % 100 == 0:
                    print('\r' + str(count), end='')
                count += 1
    print('\r' + str(count))
    return contexts1, contexts2, token_tags1, token_tags2

def save_contexts(
    output_filename,
    contexts,
    token_tags
):
    with gzip.open(output_filename, mode='wt', encoding='utf-8') as output_file:
        for token, context in contexts.items():
            output_file.write(token + '/' + token_tags[token])
            for token_index, frequency in context.items():
                output_file.write(';' + str(token_index) + ':' + str(frequency))
            output_file.write('\n')

def load_contexts(
    input_filename
):
    contexts = {}
    token_tags = {}
    with gzip.open(input_filename, mode='rt', encoding='utf-8') as input_file:
        for line in input_file:
            pairs = line.split(';')
            context = {}
            for index in range(1, len(pairs)):
                pair = pairs[index].split(':')
                context[int(pair[0])] = int(pair[1])
            pair = pairs[0].split('/')
            contexts[pair[0]] = context
            token_tags[pair[0]] = pair[1]
    return contexts, token_tags

def context_similarity(
    contexts1,
    contexts2,
    stemmer1,
    stemmer2,
    tok1,
    tok2
):
    if stemmer1:
        tok1 = stemmer1.stem(tok1)
    if stemmer2:
        tok2 = stemmer2.stem(tok2)
    if tok1 not in contexts1:
        print(tok1, 'not found in contexts1')
        return 0.0
    if tok2 not in contexts2:
        print(tok2, 'not found in contexts2')
        return 0.0
    context1 = contexts1[tok1]
    context2 = contexts2[tok2]
    if not context1:
        print(tok1, 'context1 empty')
        return 0.0
    if not context2:
        print(tok2, 'context2 empty')
        return 0.0
    min_index1 = min(context1)
    max_index1 = max(context1)
    min_index2 = min(context2)
    max_index2 = max(context2)
    min_index = min(min_index1, min_index2)
    max_index = max(max_index1, max_index2)
    count = max_index - min_index + 1
    vector1 = [0] * count
    for tok_index, frequency in context1.items():
        vector1[tok_index - min_index] = frequency
    vector2 = [0] * count
    for tok_index, frequency in context2.items():
        vector2[tok_index - min_index] = frequency
    return cosine_similarity(array(vector1).reshape(1, -1), array(vector2).reshape(1, -1))[0][0]


# In[26]:


russian_finnish_similarity('хлеб', 'leipä', show=True)
russian_finnish_similarity('конь', 'koni', show=True)
russian_finnish_similarity('молоко', 'maito', show=True)
russian_finnish_similarity('машина', 'kone', show=True)
russian_finnish_similarity('знать', 'snaijaa', show=True)
russian_finnish_similarity('меcто', 'mesta', show=True)
russian_finnish_similarity('бал', 'bailut', show=True)
russian_finnish_similarity('хочет', 'hotsittaa', show=True)
russian_finnish_similarity('кабак', 'kapakka', show=True)
russian_finnish_similarity('лавка', 'lafka', show=True)
russian_finnish_similarity('cлужить', 'lusia', show=True)
russian_finnish_similarity('завтрак', 'safka', show=True)
russian_finnish_similarity('Карни', 'carney', show=True)
russian_finnish_similarity('большевик', 'bolshevikki', show=True)
russian_finnish_similarity('дача', 'datsha', show=True)
russian_finnish_similarity('кафтан', 'kaftaani', show=True)
russian_finnish_similarity('сказка', 'kasku', show=True)
russian_finnish_similarity('копейка', 'kopeekka', show=True)
russian_finnish_similarity('клеймо', 'leima', show=True)
russian_finnish_similarity('матрёшка', 'maatuska', show=True)
russian_finnish_similarity('махорка', 'mahorkka', show=True)
russian_finnish_similarity('веретено', 'värttinä', show=True)
russian_finnish_similarity('квас', 'kvassi', show=True)


# In[12]:


offset = 0


# In[ ]:


print('begin')
#contexts_russian, contexts_finnish, tags_russian, tags_finnish = read_contexts('dl/OpenSubtitles.fi-ru.ru.gz', 'dl/OpenSubtitles.fi-ru.fi.gz', clean_russian, clean_finnish, stemmer_russian, stemmer_finnish, 'ru', 'fi', offset)
print('end')


# In[75]:


print('begin')
print(len(contexts_russian))
#save_contexts('russian.txt.gz', contexts_russian, tags_russian)
print('end')


# In[8]:


contexts_russian, tags_russian = load_contexts('russian.txt.gz')
print(len(contexts_russian), len(tags_russian))


# In[76]:


print('begin')
print(len(contexts_finnish))
#save_contexts('finnish.txt.gz', contexts_finnish, tags_finnish)
print('end')


# In[9]:


contexts_finnish, tags_finnish = load_contexts('finnish.txt.gz')
print(len(contexts_finnish), len(tags_finnish))


# In[13]:


print('begin')
read_words(contexts_russian, contexts_finnish, tags_russian, tags_finnish, 'dl/OpenSubtitles.fi-ru.ru.gz', 'dl/OpenSubtitles.fi-ru.fi.gz', clean_russian, clean_finnish, stemmer_russian, stemmer_finnish, russian_phones, finnish_phones, russian_vowels, finnish_vowels, 'ldp-results.csv', offset, False)
print('end')


# In[14]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'хлеб', 'baari')


# In[16]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'бар', 'baari')


# In[45]:


context_similarity(contexts_russian1, contexts_finnish1, stemmer_russian, stemmer_finnish, 'бар', 'baari')


# In[46]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'хлеб', 'leipä')


# In[17]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'капитализм', 'kapitalismi')


# In[50]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'веретено', 'värttinä')


# In[69]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'большевик', 'bolshevik')


# In[52]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'кабак', 'kapakka')


# In[27]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'дача', 'datsha')


# In[28]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'квас', 'kvassi')


# In[54]:


remove_consonants(reduce_diphtongs('клеймо', russian_vowels), russian_vowels)


# In[55]:


reduce_diphtongs('tuoli', finnish_vowels)


# In[18]:


data = read_csv('ldp-results.csv', sep='\t', quoting=csv.QUOTE_NONNUMERIC)


# In[19]:


data.head()


# In[20]:


plot_data = data.copy()
plot_data['SEM_VALUE'] = (data['SEM_VALUE'] * 1000).astype('int32')
plot_data['SIM_VALUE'] = (data['SIM_VALUE'] * 1000).astype('int32')
print(plot_data.head())
z = []
for sem_value in range(600, 1001):
    a = []
    for sim_value in range(800, 1001):
        number = len(plot_data[(plot_data['SEM_VALUE'] == sem_value) & (plot_data['SIM_VALUE'] == sim_value)])
        a.append(number)
    z.append(a)
print('end')


# In[21]:


x = arange(800, 1001)
y = arange(600, 1001)
x, y = meshgrid(x, y)
fig = figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.set_zlim(0,20)
ax.set_xlabel('phon')
ax.set_ylabel('emb')
ax.set_zlabel('count')
ax.plot_surface(X=x, Y=y, Z=array(z))
show()


# In[22]:


proper_name_count = len(data[data['TAG2'].str.startswith('N_Prop')])
common_name_count = len(data[data['TAG2'].str.startswith('N')]) - proper_name_count
verb_count = len(data[data['TAG2'].str.startswith('V')])
numeral_count = len(data[data['TAG2'].str.startswith('Num')])
interjection_count = len(data[data['TAG2'].str.startswith('Interj')])
adverb_count = len(data[data['TAG2'].str.startswith('Adv')])
adjective_count = len(data[data['TAG2'].str.startswith('A')]) - adverb_count
adverb_count


# In[23]:


rc('font', size=20)
height = [proper_name_count, common_name_count, verb_count, numeral_count, interjection_count, adverb_count, adjective_count]
fig, ax = subplots(figsize=(14,12))
rects = ax.bar(x=arange(len(height)), height=height, tick_label=['Proper', 'Common', 'Verb', 'Numeral', 'Interjection', 'Adverb', 'Adjective'])
ax.set_title('Word classes')
ax.set_ylabel('Count')
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects)
fig.tight_layout()
show()


# In[25]:


tagger = treetaggerwrapper.TreeTagger(TAGLANG='fi')
stemmer = SnowballStemmer('finnish')
with open('ldp-results.csv', 'rt', encoding='utf-8') as f:
    results = f.read()
gold_count = 0
found_count = 0
with open('dl/gold_list.txt', 'rt', encoding='utf-8') as g:
    for word in g:
        tokens = word.split()
        if len(tokens) > 1:
            continue
        stem = stemmer.stem(tokens[0])
        tags = tagger.tag_text(word.strip())
        tag = tags[0].split('\t')[1]
#        if tag == 'NON-TWOL' or stem not in tags_finnish:
#            print('0', word.strip(), stem)
#            continue
        gold_count += 1
        result = re.match(r'.*\t"(' + stem + r'.*?)"\t', results, re.DOTALL)
        if result:
            print(result.group(1))
            print('+', word.strip(), stem, tags[0])
            found_count += 1
        else:
            print('-', word.strip(), stem, tags[0])
        print()
print('gold_count', gold_count)
print('found_count', found_count)


# In[65]:


stemmer_russian.stem('большевиков')


# In[66]:


stemmer_finnish.stem('bolshevikit')


# In[67]:


russian_finnish_similarity('большевиков', 'bolshevikit')


# In[34]:


print('begin')
read_words(contexts_russian, contexts_finnish, tags_russian, tags_finnish, 'dl/r1.txt', 'dl/f1.txt', clean_russian, clean_finnish, stemmer_russian, stemmer_finnish, russian_phones, finnish_phones, russian_vowels, finnish_vowels, 'rf.csv', offset, False)
print('end')


# In[75]:


tagger.tag_text('datsha')


# In[30]:


russian_tagger = treetaggerwrapper.TreeTagger(TAGLANG='ru')


# In[31]:


russian_tagger.tag_text('квас')


# In[32]:


stemmer_russian.stem('квас')


# In[35]:


stemmer_finnish.stem('kvassia')


# In[36]:


print('begin')
contexts_russian1, contexts_finnish1, tags_russian1, tags_finnish1 = read_contexts('dl/r1.txt', 'dl/f1.txt', clean_russian, clean_finnish, stemmer_russian, stemmer_finnish, 'ru', 'fi', offset)
print('end')


# In[ ]:




