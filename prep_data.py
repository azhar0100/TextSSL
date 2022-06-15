from collections import defaultdict
from nltk.corpus import stopwords
import nltk
from os.path import join, exists
import re


def clean_data(dataset):
    clean_text_path = join(get_data_path(), 'corpus', dataset + '_sentences_clean.txt')
    if not exists(clean_text_path):
        docs_list = []
        old_name = dataset
        if "no_hashtag" in dataset:
            dataset = '_'.join(dataset.split('_')[:-2])
        with open(join(get_data_path(), 'corpus', dataset + '_sentences.txt')) as f:
            for line in f.readlines():
                docs_list.append(line.strip())
        dataset = old_name
        word_counts = defaultdict(int)
        for doc in docs_list:
            temp = clean_doc(doc, dataset)
            words = temp.split()
            for word in words:
                word_counts[word] += 1
        clean_docs = clean_documents(docs_list, word_counts, dataset)
        corpus_str = '\n'.join(clean_docs)
        f = open(clean_text_path, 'w')
        f.write(corpus_str)
        f.close()
    f = open(clean_text_path, 'r')
    lines = f.readlines()
    min_len = 10000
    aver_len = 0
    max_len = 0
    for line in lines:
        line = line.strip()
        temp = line.split()
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    f.close()
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))


def clean_documents(docs, word_counts, dataset):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    print(stop_words)
    ret = []
    for doc in docs:
        doc = clean_doc(doc, dataset)
        words = doc.split()
        words = [word for word in words if word not in stop_words]
        doc = ' '.join(words).strip()
        if doc != '':
            ret.append(' '.join(words).strip())
        else:
            ret.append(' ')
    return ret


def clean_doc_ap(string):
    string = re.sub(r"http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+", " ", string)
    string = re.sub(r"[^A-Za-z0-9()_+,!?:\'\`]", " ", string)  # replace all non alpha numeric characters
    string = re.sub(r"(?<!HASHTAG)_", " ", string)
    string = re.sub(r"(?<!EASTASIA)\+ | (?<!VIRUS)\+", " ", string)
    string = re.sub(r"\+", "_", string)
    string = re.sub(r"HASHTAG_EASTASIA_VIRUS(?!(\s))", "HASHTAG_EASTASIA_VIRUS ", string)
    string = re.sub(r"HASHTAG_EASTASIA(?!(\s|_))", "HASHTAG_EASTASIA ", string)
    string = re.sub(r"HASHTAG_VIRUS(?!(\s|_))", "HASHTAG_VIRUS ", string)
    string = re.sub(r"HASHTAG_VIRUS_OTHERCOUNTRY(?!(\s))", "HASHTAG_VIRUS_OTHERCOUNTRY ", string)
    string = re.sub(r"HASHTAG(?!([\s|_]))", "HASHTAG ", string)
    if "no_hashtag" in dataset:
        string = re.sub(r"HASHTAG_EASTASIA_VIRUS", " ", string)
        string = re.sub(r"HASHTAG_EASTASIA", " ", string)
        string = re.sub(r"HASHTAG_VIRUS", " ", string)
        string = re.sub(r"HASHTAG_VIRUS_OTHERCOUNTRY", " ", string)
        string = re.sub(r"HASHTAG", " ", string)
    return string


def clean_doc(string, dataset):
    if 'twitter_asian_prejudice' in dataset:
        string = clean_doc_ap(string)
    else:
        pass
    string = re.sub(r"^\"", "", string)
    string = re.sub(r"\"$", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


import argparse
import itertools
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
from os.path import join, exists
import re
from stemming.porter2 import stem
import string

def text_normalize(line):
    words=[]


    ''' ### Convert to the lower case email contents### '''
    line=line.lower()
    #print(line)

    ''' ### Strip the HTML tags ### '''
    regex=re.compile("[<\[^<>\]+>]")
    line=regex.sub('',line)
    #print(line)

    ''' ### Process Numbers ### '''
    regex=re.compile('[0-9]+')
    line=regex.sub(' number ',line)
    #print(line)

    ''' ### Process URL ### '''
    regex=re.compile('(http|https)://[\S]*')
    line=regex.sub(' httpaddr ',line)
    #print line

    ''' ### Process Email Address ### '''
    regex=re.compile('[\S]+@[\S]+')
    line=regex.sub(' emailaddr ',line)
    #print(line)

    ''' ### Process Dollar Sign ### '''
    line=re.sub('[$]+',' dollar ',line)
    #print(line)

    ''' ### remove Puntuaution ### '''
    line=line.translate(str.maketrans('','',string.punctuation))
    #print( line)

    ''' ### TOkenize the list ### '''
    words+=line.split(' ')
    #print words

    ''' ### Remove Non alpha Numeric ### '''
    words=map(lambda x:re.sub('[^a-zA-z0-9]','',x),words)
    #print(words)
    words=filter(None,words)

    ''' ### Stem the Strings ### '''
    # words=map(lambda x:stem(x),words)

    return words

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Take an imagejson pkl and make a dataset that can be input to the rest of the pipeline')
    parser.add_argument('input_imagejson_pkl',nargs='+')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    input_imagejson_pkl = args.input_imagejson_pkl
    output_folder = args.output_folder

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    pkl_data = []
    for input_imagejson_pkl_single in input_imagejson_pkl:
        with open(input_imagejson_pkl_single, 'rb') as f:
            pkl_data.extend( pickle.load(f) )

    vocab = []
    docs = []
    output_names = []
    word_counts = defaultdict(int)
    for data_item in tqdm(pkl_data):
        
        # Gather all the text, the 'sentences' in block_texts
        block_nums = list(set(data_item['block']))
        block_nums.sort()
        block_texts = []
        # print('WORDS ARE')
        # print(data_item['words'])
        for block_num in block_nums:
            block_text = " ".join([w for w,b in zip(data_item['words'],data_item['block']) if b == block_num])
            block_text = block_text.replace('.',' ')
            block_text+= '. '
            block_texts.append( block_text )
        
        final_text = ' '.join(block_texts)
        # print(final_text)
        final_text = clean_doc(final_text, 'english_data')
        # print(final_text)
        words = final_text.split()
        for word in words:
            word_counts[word] += 1
        # Find the ouput filename
        filename = Path(data_item['image_path'])
        output_name = filename.stem + '.txt'
        output_path = output_folder / output_name
        output_names.append(output_path)
        docs.append(final_text)

    docs = clean_documents(docs,word_counts,'english_data')
    doc_text = '\n'.join(docs)
    
    with open(output_folder/'corpus_english_data.clean.txt','w') as f:
        f.write(doc_text)
    
    with open(output_folder/'vocab_english_data.clean.txt','w') as f:
        vocab = '\n'.join(list(set(doc_text.split())))
        f.write(vocab)