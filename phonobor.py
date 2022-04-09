#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install --upgrade nltk')
get_ipython().system('pip install --upgrade numpy')
get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install --upgrade treetaggerwrapper')


# In[68]:


from collections import defaultdict, namedtuple
import csv
import gzip
from itertools import islice
from nltk.stem.snowball import SnowballStemmer
from numpy import array
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

# In[69]:



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
russian_vowels  = 'аиоуыэяеёю'
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


# In[70]:


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
                if isinstance(next_phone, Phone):
                    if getattr(next_phone, 'non_palatalization') == 1:
                        phone = phones[word[index]]._replace(palatal=0)
                        index += 1
                    elif getattr(next_phone, 'palatalization') == 1:
                        phone = phones[word[index]]._replace(palatal=1)
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
    filename1,
    filename2,
    clean1,
    clean2,
    stemmer1,
    stemmer2,
    phones1, phones2,
    vowels1, vowels2,
    lang1, lang2,
    result_filename,
    offset,
    verbose
):
    print('read_words', filename1, filename2)
    tagger1 = treetaggerwrapper.TreeTagger(TAGLANG=lang1)
    tagger2 = treetaggerwrapper.TreeTagger(TAGLANG=lang2)
    found_tokens1 = set()
    found_tokens2 = set()
    count = 0
    word_class_counts = defaultdict(lambda: 0)
    with open(result_filename, mode='at', encoding='utf-8-sig', newline='') as result_file:
        csv_writer = csv.writer(result_file, dialect='excel-tab', quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerow(['COUNT', 'SIM_VALUE', 'TOKEN1', 'TOKEN2', 'TAG1', 'TAG2', 'LINE1', 'LINE2'])
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
                        for tok2 in toks2:
                            tok2 = clean2(tok2)
                            if not tok2:
                                continue
                            if tok1 in found_tokens1 and tok2 in found_tokens2:
                                continue
                            similarity_value = similarity(tok1, tok2, phones1, phones2, vowels1, vowels2)
                            if not similarity_value:
                                continue
                            if similarity_value > 0.8:
                                context_similarity_value = context_similarity(contexts1, contexts2, stemmer1, stemmer2, tok1, tok2)
                                if context_similarity_value > 0.6:
                                    tags1 = tagger1.tag_text(text=tok1, tagonly=True)
                                    tags2 = tagger2.tag_text(text=tok2, tagonly=True)
                                    tag1 = tags1[0].split('\t')[1]
                                    tag2 = tags2[0].split('\t')[1]
                                    if not tag1.startswith('Np') and tag2 != 'NON-TWOL' and not tag2.startswith('N_Prop') and tag1[0] == tag2[0]:
                                        count += 1
                                        word_class_counts[tag1[0]] += 1
                                        csv_writer.writerow([
                                            count, similarity_value, tok1, tok2, tag1, tag2,
                                            ' ' + line1.strip(), ' ' + line2.strip()])
                                        if verbose:
                                            print(count, similarity_value, tok1, tok2, tag1, tag2)
                                            print(line1.strip())
                                            print(line2.strip())
                                            print(dict(word_class_counts))
                                            print()
                                        found_tokens1.add(tok1)
                                        found_tokens2.add(tok2)

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
    offset
):
    print('read_contexts', filename1, filename2)
    contexts1 = defaultdict(lambda: defaultdict(lambda: 0))
    contexts2 = defaultdict(lambda: defaultdict(lambda: 0))
    tok_indices = {}
    tok_index = 0
    count = 1
    with (gzip.open(filename1, mode='rt', encoding='utf-8')
            if filename1.endswith('.gz') else open(filename1, mode='rt', encoding='utf-8')) as file1:
        with (gzip.open(filename2, mode='rt', encoding='utf-8')
                if filename2.endswith('.gz') else open(filename2, mode='rt', encoding='utf-8')) as file2:
            for line1, line2 in zip(islice(file1, offset, None), islice(file2, offset, None)):
                toks1 = line1.split()
                toks2 = line2.split()
                toks = []
                for tok2 in toks2:
                    tok2 = clean2(tok2)
                    if not tok2:
                        continue
                    if stemmer2:
                        tok2 = stemmer2.stem(tok2)
                    toks.append(tok2)
                    if tok2 not in tok_indices:
                        tok_indices[tok2] = tok_index
                        tok_index += 1
                for tok in toks:
                    context2 = contexts2[tok]
                    for tok in toks:
                        context2[tok_indices[tok]] += 1
                for tok1 in toks1:
                    tok1 = clean1(tok1)
                    if not tok1:
                        continue
                    if stemmer1:
                        tok1 = stemmer1.stem(tok1)
                    context1 = contexts1[tok1]
                    for tok in toks:
                        context1[tok_indices[tok]] += 1
                print('\r' + str(count), end='')
                count += 1
    print()
    return contexts1, contexts2

def context_similarity(
    contexts1,
    contexts2,
    stemmer1,
    stemmer2,
    tok1,
    tok2
):
    tok1 = stemmer1.stem(tok1)
    tok2 = stemmer2.stem(tok2)
    if tok1 not in contexts1 or tok2 not in contexts2:
        print('!! ', tok1, tok2, 'not found')
        return 0
    context1 = contexts1[tok1]
    context2 = contexts2[tok2]
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


# In[71]:


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


# In[33]:


offset = 0


# In[34]:


print('begin')
contexts_russian, contexts_finnish = read_contexts('dl/OpenSubtitles.fi-ru.ru.gz', 'dl/OpenSubtitles.fi-ru.fi.gz', clean_russian, clean_finnish, stemmer_russian, stemmer_finnish, offset)
print('end')


# In[16]:


print('begin')
read_words(contexts_russian, contexts_finnish, 'dl/OpenSubtitles.fi-ru.ru.gz', 'dl/OpenSubtitles.fi-ru.fi.gz', clean_russian, clean_finnish, stemmer_russian, stemmer_finnish, russian_phones, finnish_phones, russian_vowels, finnish_vowels, 'ru', 'fi', 'ldp-results.csv', offset, False)
print('end')


# In[35]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'хлеб', 'baari')


# In[36]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'бар', 'baari')


# In[37]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'хлеб', 'leipä')


# In[38]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'капитализм', 'kapitalismi')


# In[39]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'веретено', 'värttinä')


# In[40]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'большевик', 'bolshevikki')


# In[41]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'кабак', 'kapakka')


# In[ ]:


context_similarity(contexts_russian, contexts_finnish, stemmer_russian, stemmer_finnish, 'махорк, 'kapakka')


# In[72]:


remove_consonants(reduce_diphtongs('клеймо', russian_vowels), russian_vowels)


# In[ ]:




