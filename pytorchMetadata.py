import tweepy
import math
from time import sleep
import pandas as pd
import json
import sys
import re
import fields
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji
from sentiment_Analyzer import score_paragraph
import torch
import nltk
#from autocorrect import Speller
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
import random

nltk.download('stopwords')
def remove_emoticons(text):
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_urls(text):
    result = re.sub(r"http\S+", "", text)
    return (result)


def remove_twitter_urls(text):
    clean = re.sub(r"pic.twitter\S+", "", text)
    return (clean)


def give_emoji_free_text(text):
    return emoji.get_emoji_regexp().sub(r'', text)


def get_hour(date):
    return int(date.split(' ')[3].split(':')[0])


def hashtag_count(hashtagList):
    return len(hashtagList)

def count_words(text):
    word_list=text.split(' ')
    return len(word_list)


# convert to lowercase to facilitate search process in COVID dictionary
def lowercase(text):
    lower = text.lower()
    return lower

# misspelling corrector needs external library
def autoCorrect(text):
    # spell = Speller(lang='en')
    # correctVal = spell(text)
    textBlb = TextBlob(text)  # Making our first textblob
    correctVal = textBlb.correct()  # Correcting the text
    return str(correctVal)

def relevance_score(para):
    # para = "covid Cooool Mussage Survice hte a in http://www.fff.com ? vaccine"
    # print('testing string:')
    # print(para)
    # 1- Remove all irrelevant characters such as any non alphanumeric characters, urls
    # urls = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', "", rawstring)
    # symbols = re.sub(r'[^\w]', ' ', urls)
    # print('removing URLs')
    # print(urls)
    # print('removing Symbols')
    # print(symbols)
    # 5- Convert all characters to lowercase
    # lower = symbols.lower()
    # print('To lower case')
    # print(lower)
    # 1- misspelling correcter - normalization
    # spell = Speller(lang='en')
    # correctVal = spell(para)
    # print('spelling correct')
    # print(correctVal)
    # 2- tokenization
    # print('tokens are:')
    # print(tokens)
    # 3- remove stopwords
    tokens = nltk.word_tokenize(para)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # 7- check in COVID-19 Dictionary
    relevanceScore = 0
    for search_word in para:
        if search_word in open('Covid_dictionary_low.txt').read():
            relevanceScore += 1
    return relevanceScore


# def classify_retweet2(retweet_count):
#     if(retweet_count<3):
#         return 1,0,0
#     elif(3<=retweet_count<5):
#         return 0,1,0
#     else:
#         return 0,0,1
def classify_retweet(retweet_count):
    if(retweet_count==0):
        return 0
    elif(0<retweet_count<20):
        return 1
    else:
        return 2

def metadata_parse(input_data, api,preprocess, idcolumn=None, mode=None):
    '''

    :param input_data: a tensor
    :param output_data: a tensor
    :param keyfile:
    :param idcolumn:
    :param mode:
    :return:
    '''

    hydration_mode = mode
    tweet_df=pd.DataFrame();
    ids = input_data.data.tolist()
    # print('total ids: {}'.format(len(ids)))
    # features_number=8
    tweet_batch_number=100
    start = 0
    end = tweet_batch_number
    limit = len(ids)
    i = int(math.ceil(float(limit) / tweet_batch_number))

    try:
        # with open(output_file, 'a') as outfile:
        for go in range(i):
            # print('currently getting {} - {}'.format(start, end))
            # sleep(6)  # needed to prevent hitting API rate limit
            id_batch = ids[start:end]
            start += tweet_batch_number
            end += tweet_batch_number
            backOffCounter = 1
            while True:
                try:
                    if hydration_mode == "e":
                        tweets = api.statuses_lookup(id_batch, tweet_mode="extended")
                    else:
                        tweets = api.statuses_lookup(id_batch)
                    break
                except tweepy.TweepError as ex:
                    print('Caught the TweepError exception:\n %s' % ex)
                    sleep(30 * backOffCounter)  # sleep a bit to see if connection Error is resolved before retrying
                    backOffCounter += 1  # increase backoff
                    continue

            for tweet in tweets:
                if(len(tweet_df.columns)<=1):
                    tweet_df=pd.json_normalize(tweet._json)
                else:
                    tweet_df=tweet_df.append( pd.json_normalize(tweet._json))
                # json.dump(tweet._json, outfile)
                # outfile.write('\n')
    except:
        print('exception: continuing to zip the file')


    # tweet_df2=tweet_df.loc[:, tweet_df.columns.isin(fields.fields)]
    # tweet_df = pd.DataFrame(columns=fields.fields)
    # for index, twe in tweet_df2.iterrows():
    #     if(twe['retweet_count']<10 and random.random()<0.00001):
    #         tweet_df=tweet_df.append(twe)
    #     elif (twe['retweet_count'] < 100 and random.random() < 0.005):
    #         tweet_df = tweet_df.append(twe)
    #     elif(twe['retweet_count']>=100):
    #         tweet_df = tweet_df.append(twe)

    tweet_df['text'] = tweet_df['text'].str.replace('\n','')
    tweet_df['text'] = tweet_df['text'].str.replace('\r','')

    if preprocess == 'p':
        tweet_df['text'] = tweet_df['text'].apply(lambda x : remove_urls(x))
        tweet_df['text'] = tweet_df['text'].apply(lambda x : remove_twitter_urls(x))
        tweet_df['text'] = tweet_df['text'].apply(lambda x : remove_emoticons(x))
        tweet_df['text'] = tweet_df['text'].apply(lambda x : remove_emoji(x))
        tweet_df['text'] = tweet_df['text'].apply(lambda x : give_emoji_free_text(x))
        tweet_df['text'] = tweet_df['text'].apply(lambda x: lowercase(x))
        # tweet_df['text'] = tweet_df['text'].apply(lambda x: autoCorrect(x))
        tweet_df['tweet_length']=tweet_df['text'].apply(lambda x: count_words(x))
        tweet_df['time_hour'] = tweet_df['created_at'].apply(lambda x: get_hour(x))
        tweet_df['hashtag_count']=tweet_df['entities.hashtags'].apply(lambda x: hashtag_count(x))
        tweet_df['sentiment_score']=tweet_df['text'].apply(lambda x : score_paragraph(x))
        # tweet_df['retweet_class0'], tweet_df['retweet_class1'], tweet_df['retweet_class2'] = zip(*tweet_df['retweet_count'].map(classify_retweet2))
        tweet_df['retweet_class'] = tweet_df['retweet_count'].apply(lambda x: classify_retweet(x))
        tweet_df['relevance_score'] = tweet_df['text'].apply(lambda x: relevance_score(x))

    # filtered_df=tweet_df[['sentiment_score','relevance_score','tweet_length','time_hour','hashtag_count','user.friends_count',
    #                       'user.verified','retweet_class0','retweet_class1','retweet_class2']]

    # filtered_df = tweet_df[
    #     ['sentiment_score', 'relevance_score', 'tweet_length', 'time_hour', 'hashtag_count', 'user.friends_count',
    #      'user.verified', 'retweet_class0', 'retweet_class1', 'retweet_class2']]
    filtered_df = tweet_df[
        ['sentiment_score', 'relevance_score', 'tweet_length', 'time_hour', 'hashtag_count', 'user.followers_count',
         'user.verified', 'retweet_count']]
    tensors=torch.tensor(filtered_df.values.astype(np.float64))

    #input fetures: 1.sentiment_score, 2.relevance_score, 3.tweet_length, 4. time_hour, 5. hashtag_count 6. user.friends_count,
    # 7. user.verified, output feature: retweet_class
    return tensors



