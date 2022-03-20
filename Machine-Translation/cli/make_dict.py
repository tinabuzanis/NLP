import json
import string
import re
from pickle import dump
from unicodedata import normalize
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a loaded document into sentences
def to_sentences(doc):
	return doc.strip().split('\n')
 

# save a list of clean sentences to file
def save_clean_sentences(en, ru):
    with open('out.json', 'a') as f:
        cnt = 0
        while cnt < len(en) and cnt < len(ru):
            tmpdict = {"id" : str(cnt), "translation" : { "en" : en[cnt], "ru" : ru[cnt] }}
            json.dump(tmpdict, f, ensure_ascii=False) 
            f.write('\n')
            cnt += 1
 
# load English data
filename = 'corpus.en_ru.1m.en'
doc = load_doc(filename)
en_sentences = to_sentences(doc)
for i in range(10):
	print(en_sentences[i])
 
# load Russian data
filename = 'corpus.en_ru.1m.ru'
doc = load_doc(filename)
ru_sentences = to_sentences(doc)
# spot check
for i in range(10):
	print(ru_sentences[i])

save_clean_sentences(en_sentences, ru_sentences)
print('done')
