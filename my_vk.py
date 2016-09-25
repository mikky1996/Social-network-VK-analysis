#!/usr/bin/env python
#-*- coding: utf-8 -*-
import vk
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


def download_list_from_file(path_to_file):
    f = open(path_to_file,'r')
    str_ = f.read()
    list_ = str_.split('\n')
    list1_ = []
    for i in list_:
        list1_.append(unicode(i, "utf-8"))
    return list1_

def beautiful_text(text):
    stemmer_ru = SnowballStemmer("russian", ignore_stopwords=True)
    stemmer_en = SnowballStemmer("english", ignore_stopwords=True)
    text = BeautifulSoup(text,'lxml').get_text()
    text = re.sub(r'[.|-|,|!|?|@|#|:|/]',' ',text)
    text = re.sub(ur'[^а-яА-Яa-zA-Z \n]', '', text)
    text = (text.lower()).split()
    #text = [stemmer_ru.stem(w) for w in text if not ((w in stemmer_en.stopwords)or(w in stemmer_ru.stopwords))]
    my_stop_words = download_list_from_file('my_stop_words.txt')
    text = [stemmer_ru.stem(w) for w in text if not ((w in stemmer_en.stopwords)or(w in stemmer_ru.stopwords)or(w in my_stop_words))]
    #text = [w for w in text if not ((w in stemmer_en.stopwords)or(w in stemmer_ru.stopwords))]
    text = " ".join(text)
    return text
    
def gen_text_from_id(id):
    session = vk.Session()
    api = vk.API(session)
    text = ''
    try:
        list_of_posts = api.wall.get(owner_id = id,count = 100)
    except:
        return 'VK ACCESS DENIED'
    for i in range(len(list_of_posts)):
        if ((i != 0)and(list_of_posts[i]['text'] != '')):
            text += ' '
            text += list_of_posts[i]['text']
    return text

def get_docs(id, type_):
    print ("Connecting to the server")
    session = vk.Session()
    api = vk.API(session)
    if (type_ == 'Friends'):
        print("Getting friends ids")
        list_of_id = api.friends.get(user_id = id)
        list_of_id.append(id)
    if (type_ == 'Sub'):
        print("Getting subscriptions ids")
        list_of_id = (api.users.getSubscriptions(user_id = id))['users']['items']
        list_of_id += api.friends.get(user_id = id)
    counter = 0
    list_of_pairs = []
    print("Getting text from friends walls")
    for friend_id in list_of_id:
        text = gen_text_from_id(friend_id)
        if ((text == '') or (text == 'VK ACCESS DENIED')):
            text = ''
            counter += 1
        list_of_pairs.append([friend_id,text])
    print("Number of 'bad wall' users -",counter)
    print("Starting text preprocessing")
    common_text = []
    new_list = []
    for pair in list_of_pairs:
        pair[1] = beautiful_text(pair[1])
        if (pair[1] != ''):
            common_text.append(pair[1])
            new_list.append([pair[0],pair[1]])
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(common_text)
    vocab = vectorizer.get_feature_names()
    print('Sucess!')
    return new_list, data, vocab, len(common_text), counter

def create_voc(vocab):
    g = open('vocab.friends.txt','w')
    for w in vocab:
        g.write(w.encode('utf-8'))
        g.write('\n')
    g.close()

def create_doc(vectors,vocab,D):
    f = open('docword.friends.txt','w')
    D = str(D)
    w = str(len(vocab))
    nnz = str(vectors.nnz)
    f.write(D)
    f.write('\n')
    f.write(w)
    f.write('\n')
    f.write(nnz)
    f.write('\n')
    v = vectors.toarray()
    d_num,v_num = v.shape
    for i in range(d_num):
        for j in range(v_num):
            if (v[i][j] != 0):
                f.write(str(i))
                f.write(' ')
                f.write(str(j+1))
                f.write(' ')
                f.write(str(v[i][j]))
                f.write('\n')
    f.close()
def print_list(doc):
    for i in range(len(doc)):
        print(doc[i])

def create_model(T,P):
    import os
    import artm
    import numpy as np
    topic_n = T
    batch_vectorizer = artm.BatchVectorizer(data_format='bow_uci',
                                        data_path='',
                                        collection_name='friends',
                                        target_folder='friends')
    while(1):
        model = artm.ARTM(num_topics= topic_n)
        if not os.path.isfile('friends/dictionary.dict'):
            model.gather_dictionary('dictionary', batch_vectorizer.data_path)
            model.save_dictionary(dictionary_name='dictionary', dictionary_path='friends/dictionary.dict')
        model.load_dictionary(dictionary_name='dictionary', dictionary_path='friends/dictionary.dict')
        model.initialize(dictionary_name='dictionary')
        model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))
        model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1))
        model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))
        model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15))
        model.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary_name='dictionary'))
        while(1):
            model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=5, num_document_passes=1)
            last_val = (model.score_tracker['PerplexityScore'].value)[-1]
            mean_ = np.mean((model.score_tracker['PerplexityScore'].value)[-5:])
            if abs(mean_ - last_val) < .05*last_val:
                break
        theta_matrix = model.fit_transform()
        A = []
        for i in range((theta_matrix.shape)[1]):
            A.append(theta_matrix[i].sum())
        L = np.array(A)
        if ((L==0).sum()):
            break
        else:
            topic_n += 1
    
    model = artm.ARTM(num_topics= int(P*topic_n))
    if not os.path.isfile('friends/dictionary.dict'):
        model.gather_dictionary('dictionary', batch_vectorizer.data_path)
        model.save_dictionary(dictionary_name='dictionary', dictionary_path='friends/dictionary.dict')
    model.load_dictionary(dictionary_name='dictionary', dictionary_path='friends/dictionary.dict')
    model.initialize(dictionary_name='dictionary')
    model.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))
    model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1))
    model.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))
    model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15))
    model.scores.add(artm.PerplexityScore(name='PerplexityScore',dictionary_name='dictionary'))
    while(1):
        model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=5, num_document_passes=1)
        last_val = (model.score_tracker['PerplexityScore'].value)[-1]
        mean_ = np.mean((model.score_tracker['PerplexityScore'].value)[-5:])
        if abs(mean_ - last_val) < .05*last_val:
            break
    return topic_n,model

def check_list(k,list_,max_,z):
    if (k < max_):
        list_.append(k)
    else:
        check_list(int(z[k-max_][0]),list_,max_,z)
        check_list(int(z[k-max_][1]),list_,max_,z)

def get_leaves(k,max_,z):
    list_ = []
    check_list(k,list_,max_,z)
    return list_

def get_left_leaves(k,max_,z):
    index = int(z[k-max_][0])
    list_ = get_leaves(index,max_,z)
    return list_

def get_right_leaves(k,max_,z):
    index = int(z[k-max_][1])
    list_ = get_leaves(index,max_,z)
    return list_

def get_nodes(k,list_,max_,z):
    if (k < max_):
        list_.append(k)
    else:
        list_.append(k)
        get_nodes(int(z[k-max_][0]),list_,max_,z)
        get_nodes(int(z[k-max_][1]),list_,max_,z)

def get_all_nodes(k,max_,z):
    list_ = []
    get_nodes(k,list_,max_,z)
    return list_

def alg(l,M):
    list_ = []
    for i in l:
        for j in l:
            list_.append((np.array(((M.T[i] - M.T[j])**2).sum())**(.5)))
    a = np.array(list_)
    return a.sum()

def alg1(l,r,M):
    list_ = []
    for i in l:
        for j in r:
            list_.append((np.array(((M.T[i] - M.T[j])**2).sum())**(.5)))
    a = np.array(list_)
    return a.sum()

def test(k,z,M,threshold):
    l = get_left_leaves(k,M.shape[0],z)
    r = get_right_leaves(k,M.shape[0],z)
    if ((len(l) == 1) and (len(r) == 1)):
        return 1
    f = float(1)
    S1 = (f/(len(l)**2))*alg(l,M)
    S2 = (f/(len(r)**2))*alg(r,M)
    f1 = float(S1+S2)
    S3 = (f/(len(l)*len(r)))*alg1(l,r,M)
    ans = (1-(f1/(2*S3)))
    if (ans < threshold):
        return 1
    else:
        return 0
    
def create_colors(z,M,threshold):
    colors = []
    m_colors = ['g','r','c','m','y','k']
    seen_nodes = []
    needed,rubish = M.shape
    head = needed + len(z)
    for i in range(head):
        node_num = head-i-1
        if node_num not in seen_nodes:
            if ((node_num >= needed) and (test(node_num,z,M,threshold))):
                c = int(6*random.random())
                l = get_all_nodes(node_num,needed,z)
                colors.append([l,m_colors[c]])
                seen_nodes += l
    return colors