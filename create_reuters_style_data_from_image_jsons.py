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
    parser.add_argument('input_imagejson_pkl')
    parser.add_argument('output_folder')
    args = parser.parse_args()
    input_imagejson_pkl = args.input_imagejson_pkl
    output_folder = args.output_folder

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    with open(input_imagejson_pkl, 'rb') as f:
        pkl_data = pickle.load(f)

    for data_item in tqdm(pkl_data):

        
        # Gather all the text, the 'sentences' in block_texts
        block_nums = list(set(data_item['block']))
        block_nums.sort()
        block_texts = []
        for block_num in block_nums:
            block_text = " ".join([w for w,b in zip(data_item['words'],data_item['block']) if b == block_num])
            block_text = block_text.replace('.',' ')
            block_text+= '. '
            block_texts.append( block_text )
        
        final_text = '\n'.join(block_texts)

        doctype = data_item['doc_type']
        # Find the ouput filename
        filename = Path(data_item['image_path'])
        output_name = filename.stem + '.txt'
        output_path = output_folder / doctype / output_name
        output_path.parent.mkdir(exist_ok=True, parents=True )
        with open(output_path, 'w') as f:
            f.write(final_text)