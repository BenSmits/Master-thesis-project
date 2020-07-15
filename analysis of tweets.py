
# coding: utf-8

# In[1]:


import os
import pandas as pd
from numpy import random

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import tweepy
import re

import matplotlib.pyplot as plt
import seaborn as sns

from langdetect import detect
from langid import classify
from spellchecker import SpellChecker

from textblob.translate import Translator
from textblob import TextBlob
from textblob_nl import PatternTagger, PatternAnalyzer

import time as t

import spacy

twitter_c = [0.11, 0.63, 0.95]
pd.set_option('display.max_colwidth', -1) # normally 50
data = 'DATA/'
pickles = 'pickles/'


# # Twitter Data
# 

# ## Functions

# In[2]:


def replace_url(txt, sub = '||U||'):
    """Replace URLs found in a text string with ||U|| 
    (i.e. it will remove the URL from the string).

    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.

    Returns
    -------
    The same txt string with url's replaced.
    """
    return " ".join(re.sub("((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*", sub, txt).split())

def replace_tags(txt, sub = "||T||"):
    return " ".join(re.sub('(@[a-zA-Z0-9]+)', sub, txt).split())

def check_inst(txt, inst):
    for word in txt.split():
        if word in inst:
            return 1
    return 0


def nl_sent(text):
    ''' 
    text: string in Dutch
    returns 
    '''
    return TextBlob(text, analyzer = PatternAnalyzer()).sentiment[0]

def en_sent(text):
    '''
    text: string in English
    returns: polarity sentiment of the given text. the text is translated from dutch to english
    '''
    return TextBlob(text).sentiment.polarity

def remove_punct(text):
    '''
    removes all punctuation from a string
    '''
    return re.sub('([^\sA-Za-z0-9])', '', text)

def remove_small(text):
    '''
    removes small words of size smaller than 3
    '''
    return re.sub('(\\W*\\b\\w{1,2}\\b)', '', text)

stopwords_l = set(stopwords.words('dutch'))

def remove_stopwords(text, stopwords = stopwords_l):
    '''
    text: string
    stopwords: (array-like, optional) a list containing stopwords. default list is dutch.
    removes stopwords from the given text
    '''
    word_list = text.split()
    return ' '.join([word for word in word_list if word not in stopwords])

def small_tweet(text):
    '''
    returns true when the given text is smaller than 3 words
    '''
    length = len(text.split())
    if length < 3:
        return False
    else:
        return True
    
def contains_text(text):
    '''
    if text contains text True is returned False otherwise'''
    if re.search('([a-zA-Z0-9]+)', text):
        return True
    else:
        return False
    
def detect_lang(x):
    '''
    x: a string of text
    returns the language and None if there is no language detected'''
    try:
        lang = detect(x)
        return lang
    except:
        return None
    
def sentence_checker(sentence, speller):
    '''
    sentence: a string
    speller: a SpellChecker object
    returns the same text but corrected for spelling'''
    word_list = sentence.split()
    for i, word in enumerate(word_list):
        new = word_checker(word, speller)
        word_list[i] = new
    return ' '.join(word_list)

def word_checker(word, speller):
    '''
    word: a string containing a single word
    speller: a SpellChecker object
    returns: the corrected word'''
    if word in speller:
        return word
    else:
        cor = speller.correction(word)
#         print('5', cor)
        return cor

def sent_cat(sent, threshold = 0.05):
    '''
    Sets sentiment score to 1 if positive, -1 if negative and 0 if neutral
    theshold (float) default 0.05 and -(0.05) for negative side
    '''
    if sent > threshold:
        return 1
    elif sent < -threshold:
        return -1
    else:
        return 0

def is_pos(x, threshold = 0.05):
    if x > threshold:
        return 1
    else:
        return 0

def is_neg(x, threshold = 0.05):
    if x < -threshold:
        return 1
    else:
        return 0
    
def margins(freq, margin = 0.05):
    '''
    freq: a integer representing the frequency of a word occuring in the text
    margin: the margin to consider similar word frequencies default 0.05
    returns a tuple with lower and upperbound'''
    upper = freq*(margin+1)
    lower = freq*(1-margin)
    return (lower, upper)


# # First analysis of the twitter data

# ## Check spelling checkers
# The spellingschecker is checked on spelling of words and whether weird things happen. 

# In[7]:


# years of data 
years = ['2014', '2015', '2016', '2017', '2018', '2019']

path = os.getcwd()
data_folder = '15 km radius'

year_data = {}

for year in years:
    print('year: '+year)
    files = [file for file in os.listdir(path+os.sep+data_folder) if year in file] # pay attention where your data is stored
    temp = pd.DataFrame(data = None, columns = ['date', 'username', 'to', 'replies', 'retweets', 'favorites', 'text',
                                                'geo', 'mentions', 'hashtags', 'id', 'permalink'])

    for file in files:
        print(file)
        temp = temp.append(pd.read_csv(path+os.sep+data_folder+os.sep+file), ignore_index = True)
        print(temp.shape)

    start = t.time()
    print('select data')
    tweets = temp
    tweets = tweets.drop_duplicates()
    print(tweets.shape)
    year_data[year] = tweets


# In[8]:


# getting the total number of tweets in each year before pre-processing
tweets = pd.DataFrame(data = None, columns = ['date', 'username', 'to', 'replies', 'retweets', 'favorites', 'text',
                                            'geo', 'mentions', 'hashtags', 'id', 'permalink'])
for year in years:
    temp = year_data[year]
    print(year+' ', year_data[year].shape)
    tweets = tweets.append(temp)


# In[9]:


random_state = 10
temp = tweets.sample(n = 5000, random_state= random_state)
temp.loc[:,'langdetect'] = temp.loc[:,'text'].apply(lambda x: detect_lang(x))


# In[10]:


distance = 1 # sets the Levenshtein distance for the spellingchecker

nl_spell = SpellChecker(language = None, distance = distance)
nl_spell.word_frequency.load_dictionary('nl_NL.json')

nl_ext = ['ff', "'s", 'A2', 'file', 'cda', 'IT', 'CDA', 'mengen', 'kraan', 'NL']
nl_spell.word_frequency.load_words(nl_ext)

test_nl = temp.loc[temp.loc[:,'langdetect'] == 'nl', :].sample(n = 100, random_state = random_state)

test_nl.loc[:, 'corrected'] = test_nl.loc[:,'text'].apply(lambda x: sentence_checker(x, nl_spell))

test_nl.loc[:,'gelijk'] = test_nl.loc[:, ['text', 'corrected']].apply(lambda x: x['text'] == x['corrected'], axis = 1)
test_nl.loc[test_nl.loc[:,'gelijk'] == False, ['text', 'corrected']].head(20)


# In[11]:


# setting up the english spellingchecker
en_spell = SpellChecker(distance = distance)

en_ext = ["i'm", "we're", "I'll", "i'd", "you're"]
en_spell.word_frequency.load_words(en_ext)

test_en = temp.loc[temp.loc[:,'langdetect'] == 'en', :].sample(n = 100, random_state = random_state)

test_en.loc[:, 'corrected'] = test_en.loc[:,'text'].apply(lambda x: sentence_checker(x, en_spell))

test_en.loc[:,'gelijk'] = test_en.loc[:, ['text', 'corrected']].apply(lambda x: x['text'] == x['corrected'], axis = 1)
test_en.loc[test_en.loc[:,'gelijk'] == False, ['text', 'corrected']].tail(20)


# # Pre processing the Twitter data for emotions and sentiment
# This was originally used to load all data and apply multiple steps at once. Once this was run the data would be stored in 6 pre-processed files. One for each year.

# The first cell does only pre processing. After this step all files are saved to "tweets pre=processed 2014.csv". After the previous step the spellingchecker and sentiment analysis can be done. After this step all files are saved to "tweets+year+.csv". 
# This was done to not do all the steps again if something went wrong. 

# In[12]:


### SpellingChecker objects 
## may take one hour to run
distance = 1 # sets the Levenshtein distance for the spellingchecker
# setting up the Dutch spellingchecker
nl_spell = SpellChecker(language = None, distance = distance)
nl_spell.word_frequency.load_dictionary('nl_NL.json')

nl_ext = ['ff', "'s", 'A2', 'file', 'cda', 'IT', 'CDA', 'mengen', 'kraan', 'NL']
nl_spell.word_frequency.load_words(nl_ext)

# setting up the english spellingchecker
en_spell = SpellChecker(distance = distance)

en_ext = ["i'm", "we're", "I'll", "i'd", "you're"]
en_spell.word_frequency.load_words(en_ext)

# years of data 
years = ['2014', '2015', '2016', '2017', '2018', '2019']

### Pipeline for Selection, pre-processing, Language detection, spelling checker and Sentiment analysis
# for year in years:
path = os.getcwd()
data_folder = '15 km radius'

for year in years:
    print('year: '+year)
    files = [file for file in os.listdir(path+os.sep+data_folder) if year in file]
    temp = pd.DataFrame(data = None, columns = ['date', 'username', 'to', 'replies', 'retweets', 'favorites', 'text',
                                                'geo', 'mentions', 'hashtags', 'id', 'permalink'])

    for file in files:
        temp = temp.append(pd.read_csv(path+os.sep+data_folder+os.sep+file), ignore_index = True)
    print('shape: ',temp.shape)
    ### # analysis starts here
    ### Select data
    start = t.time()
    print('select data')
    tweets = temp
    tweets = tweets.drop_duplicates()

    p2000 = temp.loc[temp.loc[:,'username'] == 'P2000013', 'id'] # Removing this account, since it only posts emergency calls reportings
    tweets = tweets.loc[~tweets.loc[:,'id'].isin(p2000),:]
    
    ### Pre-processing
    print('Pre-processing')
    # remove nans in column text
    tweets = tweets.dropna(axis = 0, subset = ['text'])
    
    #remove urls
    tweets.loc[:,'text'] = tweets.loc[:,'text'].apply(lambda x: replace_url(x, sub = ''))

    # removes tweets without text
    tweets.loc[:,'contains text'] = tweets.loc[:,'text'].apply(lambda x: contains_text(x))
    tweets = tweets.loc[tweets.loc[:,'contains text'],:]
    tweets = tweets.drop(labels = 'contains text', axis= 'columns')

    # removes tweets that have less than 3 words
    tweets.loc[:,'long'] = tweets.loc[:,'text'].apply(lambda x: small_tweet(x))
    tweets = tweets.loc[tweets.loc[:,'long'], :]
    tweets = tweets.drop(labels = 'long', axis= 'columns')
    
    ### language detection
    print('Language detection')
    tweets.loc[:,'langdetect'] = tweets.loc[:,'text'].apply(lambda x: detect_lang(x))
    # only Dutch and English tweets
    tweets = tweets.loc[(tweets.loc[:,'langdetect'] == 'nl') | (tweets.loc[:,'langdetect'] == 'en'), :] 
    print('Final shape: ', tweets.shape)
    print('Finished\nTime: {}'.format(t.time() - start))

    tweets.to_csv(data+'tweets pre-processed '+year+'.csv', index = False)


# In[13]:


# tag tweets talking about institutions maually considered 
institutions = ['#FNV', 'Stadskantoor', 'MWB', 'Burgemeester', 'Burgernet', 'TBV','#MWB', 'beleid', 'gemeentehuis', 
                'provinciehuis', 'verkiezingen', 'raadslid', '@MVO_NL', 'campagneleider', '@raadtilburg', '@gemeentetilburg', 
                'minister', 'campagne','bestuursakkoord', '@D66Brabant', 'SP', 'LST', '@GLTilburg', '@SPTilburg', 'VVD', 
                '@CDATilburg', '@PvdATilburg', '@50pluspartij', '@LokaalTilburg', '#pvv', '@VoorTilburg', '#CDA', 
                '@Onderzoeksraad', '@IFVtweet', 'debat', '#tilburginbeeld', 'europa', '@D66Tilburg', '#D66Tilburg', 
                'coalitieakkoord', '@fontys', '@stationTilburg', 'Theresialyceum', '@Brabant', '#Spoorzone', '#lochal', 
                '#locomotiefhal', '#TilburgU', '#tilburguniversity', '#tiu', '#uvt', '@TilburgUniversity', '#starterslift', 
                '#topinkomens', '@NatuurmuseumBra', 'cultuurbudget', '@uvt_tilburg', '@BerkelEnschot', '@Avanshogeschool', 
                'Midpoint', 'BOM', '@BOMBrabant', '@TiwosTilburg', '@WonenBreburg', '@MidpointBrabant', '#midpointbrabant', 
                '@starterslift', '@nvrewin', '@IFVtweet','regelgeving', '#onderwijs', 'onderwijs', 'debat',
                'Europa','#CDA7','#demonstratie', '@minlnv']

years = ['2014', '2015', '2016', '2017', '2018', '2019']

for year in years:
    start = t.time()
    tweets = pd.read_csv(data+'tweets pre-processed '+year+'.csv')
    tweets.loc[:,'institutions'] = tweets.loc[:,'text'].apply(lambda x: check_inst(x, institutions))

    ### spellingchecker
    print('spellingchecker\nTime: {}'.format(t.time() - start))
    tweets.loc[(tweets.loc[:,'langdetect'] == 'nl') & (tweets.loc[:,'institutions'] == 1), 'text'] = tweets.loc[(tweets.loc[:,'langdetect'] == 'nl')& (tweets.loc[:,'institutions'] == 1), 'text'].apply(lambda x: sentence_checker(x, nl_spell))
    tweets.loc[(tweets.loc[:,'langdetect'] == 'en') & (tweets.loc[:,'institutions'] == 1), 'text'] = tweets.loc[(tweets.loc[:,'langdetect'] == 'en')& (tweets.loc[:,'institutions'] == 1), 'text'].apply(lambda x: sentence_checker(x, en_spell))

    ### sentiment analysis
    print('sentiment analysis\nTime: {}'.format(t.time() - start))
    tweets.loc[(tweets.loc[:,'langdetect'] == 'nl') & (tweets.loc[:,'institutions'] == 1), 'sentiment'] = tweets.loc[(tweets.loc[:,'langdetect'] == 'nl') & (tweets.loc[:,'institutions'] == 1),'text'].apply(lambda x: nl_sent(x))
    tweets.loc[(tweets.loc[:,'langdetect'] == 'en') & (tweets.loc[:,'institutions'] == 1), 'sentiment'] = tweets.loc[(tweets.loc[:,'langdetect'] == 'en') & (tweets.loc[:,'institutions'] == 1),'text'].apply(lambda x: en_sent(x))
    tweets.to_csv(data+'tweets '+year+'.csv', index = False)


# ## Read tweets pre-processed for sentiment analysis

# In[14]:


tweets = pd.DataFrame(data = None, columns = ['date', 'text', 'id'])
years = ['2014', '2015', '2016', '2017', '2018', '2019']

for year in years:
    tweets = tweets.append(pd.read_csv(data+'tweets '+year+'.csv', dtype = {'id':'object'}), ignore_index = False)
    print(tweets.shape)
print('read files')

tweets.loc[:,'date'] = pd.to_datetime(tweets.date, dayfirst = True, infer_datetime_format = True)


# # Robustness check of Random wordlist
# ## Sentiment analysis
# Correlate you monthly sentiment with the entrepreneurial output. 
# Correlate the sentiment in tweets with randomly chosen words. 

# In[15]:


# counts the occurrence of all words in all tweets so that we can take similar words from this list compared to our own list 
# of words. 
# takes about 8 secs # Get the frequency of all words
texts_df = pd.DataFrame(tweets.loc[:,'text'].apply(lambda x: re.sub('[\\\'!?,.:";]', '', x.lower())))
word_count = texts_df.loc[:,'text'].str.split(expand=True).stack().value_counts()
word_count_df = pd.DataFrame(word_count, columns = ['frequency'])

# get the frequency of my own words
institutions_freq = {}
for key in institutions:
    try:
        institutions_freq[key] = word_count[key.lower()]
    except:
        print(key, '-')

# Get the random words based on my own list of words with a margin of 5% 
random.seed(seed = 10)
rand_bow = []
for word in institutions_freq.keys():
    lower, upper = margins(institutions_freq[word])
    potential = list(word_count_df.loc[(word_count_df.loc[:,'frequency'] >= lower) & (word_count_df.loc[:,'frequency'] <= upper) ,:].index)
    chosen = False
    while not(chosen):
        chosen_word = potential.pop(random.randint(len(potential)))
        if re.search('[0-9-()$%&?!<>/\\\.]', chosen_word):
            continue
        else:
            rand_bow.append(chosen_word)
            chosen = True
            
# put the word list in this list so we have 2 columns which indicates where the tweet belongs to. 
tweets.loc[:,'random'] = tweets.loc[:,'text'].apply(lambda x: check_inst(x, rand_bow))
tweets.loc[:, 'institutions'] = tweets.loc[:,'text'].apply(lambda x: check_inst(x, institutions))


# In[16]:


tweets = tweets.loc[:,['date', 'favorites', 'geo', 'hashtags', 'id', 'institutions','langdetect', 'mentions', 'permalink', 
                       'replies', 'retweets', 'sentiment', 'text', 'to', 'username', 'random',]]


# In[17]:


tweets.loc[tweets.loc[:,'langdetect'] == 'nl', 'sentiment'] = tweets.loc[tweets.loc[:,'langdetect'] == 'nl','text'].apply(lambda x: nl_sent(x))
tweets.loc[tweets.loc[:,'langdetect'] == 'en', 'sentiment'] = tweets.loc[tweets.loc[:,'langdetect'] == 'en','text'].apply(lambda x: en_sent(x))


# # Wordcount of new word list

# ## quaterly and monthly results

# In[18]:


tweets.loc[:,'quarter'] = tweets.date.dt.to_period('Q')
tweets.loc[:,'month'] = tweets.date.dt.to_period('M')
tweets.loc[:,'year'] = tweets.date.dt.to_period('Y')

threshold = 0.05

tweets.loc[:,'pos'] = tweets.loc[:, 'sentiment'].apply(lambda x: is_pos(x, threshold = threshold))
tweets.loc[:,'neg'] = tweets.loc[:, 'sentiment'].apply(lambda x: is_neg(x, threshold = threshold))


# #### A checkpoint for the data to avoid running previous analysis

# In[16]:


#################################################################################################################################

tweets.to_pickle(pickles+'tweets 12-6-2020.pkl')

#################################################################################################################################
# Dataframe with sentiment of all tweets and indication for random words, institutions old and newest word list. Ready to be 
# calculated in quarter and month and correlate with 


# In[8]:


tweets = pd.read_pickle(pickles+'tweets 12-6-2020.pkl')


# ## An analysis of the pre-processed tweets

# In[19]:


tweets.loc[tweets.loc[:,'langdetect'] == 'en', 'year'].value_counts()


# In[20]:


tweets.loc[tweets.loc[:,'langdetect'] == 'nl', 'year'].value_counts()


# In[21]:


def has_sent(pos, neg):
    if pos == 1 or neg == 1:
        return 1
    return 0

tweets.loc[:,'has_sent'] = tweets.loc[:,:].apply(lambda x: has_sent(x['pos'], x['neg']), axis = 'columns')


# In[22]:


time = 'month'
monthly_sent = tweets.loc[tweets.loc[:,'institutions'] == 1, [time, 'sentiment', 'pos', 'neg']].groupby(time).mean()
has_sent_df = tweets.loc[tweets.loc[:,'institutions'] == 1, [time, 'has_sent']].groupby(time).sum()


# In[26]:


temp = tweets.loc[:,['date','id',  'institutions', 'langdetect','sentiment', 'text', 'username', 'random','quarter', 'month','pos','neg', 'has_sent']]
temp.to_csv(data+'Final sentiment.csv')


# # Emotions analysis

# In[27]:


import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize.casual import TweetTokenizer

import spacy
import string
from collections import Counter

langs = ['en', 'nl']
# You have to instll the right lemmatizers in order to lemmatize your text.
lemm_dict = {'nl': spacy.load('nl_core_news_sm'), 'en':spacy.load("en_core_web_sm")}

text = tweets.loc[:,['text', 'langdetect', 'date', 'id', 'username']]
text.loc[:,'old text'] = text.loc[:,'text']


# In[28]:


# tokenize and lowercase
tknzr = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False)
text.loc[:, "text"] = text.loc[:, "text"].apply(lambda x: tknzr.tokenize(x))

# remove punctuation
punct_to_remove = string.punctuation
text.loc[:, "text"] = text.loc[:, "text"].apply(lambda txt: [x for x in txt if x not in punct_to_remove])


# In[29]:


# remove stopwords
stop_dict = {'en':stopwords.words('english'), 'nl': stopwords.words('dutch')}
for lang in langs:
     text.loc[text.loc[:,'langdetect'] == lang, "text"] = text.loc[text.loc[:,'langdetect'] == lang, 'text'].apply(lambda txt: [x for x in txt if x not in stop_dict[lang]])


# In[30]:


# translating the emoticons to their textual equivalent
vert = {'happy':'blij', 'wink':'knipoog', 'sad':'verdrietig', 
        'cheeky':'ondeugend', 'crying':'huilen', 'annoyed':'geirriteerd'}
emoticons_en = {':)': 'happy', 
                ';)': 'wink', 
                ':(': 'sad', 
                ":p": 'cheeky', 
                ";-)": 'wink', 
                ":-)": 'happy', 
                ":D": 'happy', 
                "(:":'happy', 
                "]:":'sad', 
                ":')":'crying', 
                ':-/':'annoyed', 
                ':-p': 'cheeky',
                ':-(':'sad', 
                '):': 'sad', 
                ":'(": 'crying'}

emoticons_nl = {}
for emoticon in emoticons_en.keys():
    emoticons_nl[emoticon] = vert[emoticons_en[emoticon]]
emoticons = {'nl':emoticons_nl, 'en': emoticons_en}

def trans_emoticon(text, emoticon_dict):
    '''Changes the emoticon into the word'''
    try:
        for i, word in enumerate(text):
            if word in emoticon_dict.keys():
                text[i] = emoticon_dict[word]
    except:
        print(text)
    return text
            
for lang in langs:
    text.loc[text.loc[:,'langdetect'] == lang, "text"] = text.loc[text.loc[:,'langdetect'] == lang, "text"].apply(lambda txt: trans_emoticon(txt, emoticons[lang]))


# In[31]:


# lemmatization --> may take 60 mins
def lemmatize(txt, lemmatizer):
    return [word.lemma_ for word in lemmatizer(txt)]

for lang in langs:
    text.loc[text.loc[:,'langdetect'] == lang, "text"] = text.loc[text.loc[:,'langdetect'] == lang, "text"].apply(lambda txt: lemmatize(' '.join(txt), lemm_dict[lang]))


# # Applying the emotion lexicon
# derived emotions:
# - dominant emotion = 
#     - valence approach -> what is the most frequent emotion positive or negative
#     - cognitive appraisal -> what is the most frequent high or low controllability and certainty
# - conflicting emotion 
#     - valence approach -> what is the least frequent emotion positive or negative
#     - cognitive appraisal -> what is the least frequent high or low controllability and certainty
# - mixed emotions (C --> Conflicting, D --> Dominant)
#     - $$5*(C+1)^p - (D+1)^{1/C}$$ --> p is less than 1 (0.5) and 
#     
# | emotion| Valence| Controlability  | Certainty|
# | - |-| -|-|
# | Joy (Happiness) | positive | High | High |
# | Fear (fear) | Negative | Low | Low |
# | Anticipation (hope) | Positive | Low | Low |
# | Anger (anger) | Negative | High | High |

# ## Checkpoint in between again

# In[32]:


text.to_pickle(pickles+'pre-processed.p')


# In[33]:


text = pd.read_pickle(pickles+'pre-processed.p')


# In[34]:


# emotions are detected in the text
emo_lex = pd.read_csv('NRC/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations - corrected.csv', encoding = 'iso8859_15')

fear_nl = list(emo_lex.loc[emo_lex.loc[:,'Fear'] == 1, 'dutch new'])
anger_nl = list(emo_lex.loc[emo_lex.loc[:,'Anger'] == 1, 'dutch new'])
Surprise_nl = list(emo_lex.loc[emo_lex.loc[:,'Surprise'] == 1, 'dutch new'])
joy_nl = list(emo_lex.loc[emo_lex.loc[:,'Joy'] == 1, 'dutch new'])
fear_en = list(emo_lex.loc[emo_lex.loc[:,'Fear'] == 1, 'English (en)'])
anger_en = list(emo_lex.loc[emo_lex.loc[:,'Anger'] == 1, 'English (en)'])
Surprise_en = list(emo_lex.loc[emo_lex.loc[:,'Surprise'] == 1, 'English (en)'])
joy_en = list(emo_lex.loc[emo_lex.loc[:,'Joy'] == 1, 'English (en)'])

emo_lex_dict = {'nl': {'Fear': fear_nl, 'Anger': anger_nl, 'Surprise': Surprise_nl, 'Joy': joy_nl},
                'en': {'Fear': fear_en, 'Anger': anger_en, 'Surprise': Surprise_en, 'Joy': joy_en}}


# In[35]:


emo_lex.loc[(emo_lex.loc[:, 'Surprise'] == 1) & (emo_lex.loc[:, 'Surprise'] == 1), :].count()
emo_lex.loc[(emo_lex.loc[:, 'Surprise'] == 1),'English (en)'].count()
emo_lex.loc[(emo_lex.loc[:, 'Surprise'] == 1),'English (en)'].count()


# In[36]:


def find_emo(txt, emo_lookup):
    '''
    counts the occurence of each emotion in the text
    txt: a list of words
    emo_lookup: a dataframe where words can be looked up in the index
    returns: a list with all emotions added up
    '''
    if type(emo_lookup) != type(list()):
        raise TypeError('"emo_lookup" should be type: list')
    emo_count = 0
    for word in txt:
        if word in emo_lookup:
            emo_count += 1
    
    return emo_count

langs = ['nl', 'en']
        
for lang in langs:
    print(lang)
    for emo in emo_lex_dict[lang].keys():
        print(emo)
        text.loc[text.loc[:, 'langdetect'] == lang,emo] = text.loc[text.loc[:, 'langdetect'] == lang,'text'].apply(lambda txt: find_emo(txt, emo_lex_dict[lang][emo]))


# In[37]:


text = text.loc[:, ['text', 'langdetect', 'date', 'username', 'Fear', 'Anger', 'Joy', 'Surprise']]


# In[38]:


text.to_csv(data+'Final emotion.csv')

