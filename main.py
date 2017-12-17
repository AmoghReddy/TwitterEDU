"""
Read the output of a zipped twitter archive from:
https://archive.org/details/twitterstream
"""
import bz2
import datetime
import json
import os
import profile
# import psycopg2
from pyspark.context import SparkContext
from pyspark.accumulators import AccumulatorParam
from pprint import pprint
import numpy as np
from scipy import spatial
import pandas as pd
import re
import operator
import csv

# with open("postgresConnecString.txt", 'r') as f:
#     DB_CONNECTIONSTRING = f.readline()
# DB_CONNECTIONSTRING = "host='localhost' dbname='postgres' user='postgres' password='admin'"
# conn = psycopg2.connect(DB_CONNECTIONSTRING)
CACHE_DIR = "D:\TwitterDatastream\PYTHONCACHE_SMALL_TEST"
EDU_DATA = 'merged.csv'
TRAIN_FEAT_CSV = 'testFeat.csv'
TRAIN_LABS_CSV = 'testLabs.csv'
TRAIN_FEAT_LABS_CSV = 'testFeatLabs.csv'
FEATURE_NAMES_CSV = 'featureNames.csv'
sc = SparkContext('local', 'test')
# location_data = pd.read_csv('new_merged.csv')

class WordsSetAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return set()
    def addInPlace(self, acc1, acc2):
        return acc1.union(acc2)

class WordsDictAccumulatorParam(AccumulatorParam):
    def zero(self, v):
        return dict()
    def addInPlace(self, acc1, acc2):
        for key in acc2.keys():
            try:
                acc1[key] += acc2[key]
            except:
                acc1[key] = acc2[key]
        # acc1.update(acc2)
        return acc1

# vocabulary = sc.accumulator(set(), WordsSetAccumulatorParam())
vocabulary = sc.accumulator(dict(), WordsDictAccumulatorParam())

location_data = pd.read_csv(EDU_DATA)
area_dict = dict(zip(location_data['city'], location_data[['fips', 'without_hsd','with_hsd', 'somecollege', 'bachelors']].values.tolist()))
county_dict = dict(zip(location_data['county'], location_data[['fips', 'without_hsd','with_hsd', 'somecollege', 'bachelors']].values.tolist()))
coord_dict = {tuple(x[:2]):x[2] for x in location_data[['lat', 'lng', 'county']].values}

# latlon = np.zeros(shape=(38120,2))
latlon = list()
# ind = 0
for index, row in location_data.iterrows():
    # latlon[ind] = [location_data['lat'][index], location_data['lng'][index]]
    latlon.append([location_data['lat'][index], location_data['lng'][index]])
    # ind+=1

latlon = np.array(latlon)
latlonKDT = spatial.KDTree(latlon)

def mapToCounty(place, location, coordinates):
    # coordr_dict = {tuple(x[:2]):x[2] for x in location_data[['lat_r', 'lng_r', 'county']].values}
    if place:
        place = (place.split(",")[0]).lower()
        # country = (place.split(",")[1]).lower()
        try:
            if area_dict[place]: return area_dict[place]
        except: None
    if location:
        location = (location.split(",")[0]).lower()
        try:
            if area_dict[location]: return area_dict[location]
        except: None
    if coordinates:
        closestLoc = spatial.KDTree(latlon).query(coordinates, k=1, distance_upper_bound=9)[1]
        try:
            closest = latlon[closestLoc]
        except:
            return None
        # closest = spatial.KDTree(latlon).query(coordinates, k=1, distance_upper_bound=9)
        # print("Amogh",closest,"Reddy")
        # if closest[0] != float('inf') and latlon[closest[1]][0] != 0. and latlon[closest[1]][1] != 0.:
        #     print(coordinates, closest, latlon[closest[1]])
        # return closest[0], closest[1]
        if coord_dict[closest[0], closest[1]]:
            county_k = coord_dict[(closest[0], closest[1])]
            return county_dict[county_k]

    return None

# def mapToCounty(place, location, coordinates):
#     for index, rows in location_data.iterrows():
#         if place:
#             place = place.split(",")[0]
#             if location_data.city[index] == place:
#                 education = [location_data.with_hsd[index], location_data.without_hsd[index], location_data.somecollege[index],location_data.bachelors[index]]
#                 return education
#         if location:
#             location = location.split(",")[0]
#             if location_data.city[index] == location:
#                 education = [location_data.with_hsd[index], location_data.without_hsd[index], location_data.somecollege[index], location_data.bachelors[index]]
#                 return education
#         if coordinates:
#             temp = np.array(coordinates).flatten()
#             if len(temp) == 2:
#                 #print(temp)
#                 if abs(round(location_data.latitude[index],3)-round(temp[0],3))<0.00000001 and abs(round(location_data.longitude[index],3)-round(temp[1],3))<0.00000001:
#                     education = [location_data.with_hsd[index], location_data.without_hsd[index], location_data.somecollege[index], location_data.bachelors[index]]
#                     return education
#                 elif round(location_data.latitude[index],0) == round(temp[0],0) and round(location_data.longitude[index],0) == round(temp[1],0):
#                     education = [location_data.with_hsd[index], location_data.without_hsd[index], location_data.somecollege[index], location_data.bachelors[index]]
#                     return education
#             if len(temp) == 1:
#                 temp1 = temp[0]
#                 if abs(round(location_data.latitude[index],3)-round(temp1[0],3))<0.00000001 and abs(round(location_data.longitude[index],3)-round(temp1[1],3))<0.00000001:
#                     education = [location_data.with_hsd[index], location_data.without_hsd[index], location_data.somecollege[index], location_data.bachelors[index]]
#                     return education
#                 elif round(location_data.latitude[index],0) == round(temp1[0],0) and round(location_data.longitude[index],0) == round(temp1[1],0):
#                     education = [location_data.with_hsd[index], location_data.without_hsd[index], location_data.somecollege[index], location_data.bachelors[index]]
#                     return education
#     return None

def load_bz2_json(filename):
    """ Takes a bz2 filename, returns the tweets as a list of tweet dictionaries"""
    # with open(filename, 'rb') as f:
    with bz2.open(filename, 'rt') as f:
        lines = str(f.read()).split('\n')
        # print(filename, type(s))
        # lines = str(bz2.decompress(s))
        # print("len_lines",len(lines))
        # print("lines",len(lines),lines)
        # lines = lines.split('\n')
    num_lines = len(lines)
    # print("num_lines", num_lines)
    # print("line 1", lines[0])
    tweets = []
    for line in lines:
        try:
            if line == "":
                num_lines -= 1
                continue
            # print(len(json.loads(line)), type(json.loads(line)))
            tweets.append(json.loads(line))
        except: # I'm kind of lenient as I have millions of tweets, most errors were due to encoding or so)
            continue
    # print("len tweets", len(tweets))
    return tweets


def load_tweet(tweet, tweets_saved):
    """Takes a tweet (dictionary) and upserts its contents to a PostgreSQL database"""
    try:
        # tweet_id = tweet['id']
        tweet_text = tweet['text']
        tweet_user_id = tweet['user']['id']
        tweet_user_location = tweet['user']['location']
        tweet_user_lang = tweet['user']['lang']
        try: tweet_coordinates = tweet['coordinates']['coordinates']
        except: tweet_coordinates = None
        try: tweet_place = tweet['place']['full_name']
        except: tweet_place = None
        map_to_county = mapToCounty(tweet_place, tweet_user_location, tweet_coordinates)
        if map_to_county:
            tweet_county = int(map_to_county[0])
            tweet_education_level = tuple(map_to_county[1:])
        else:
            tweet_county = None
            tweet_education_level = None
            # created_at = tweet['created_at']
    except KeyError:
        return {}, tweets_saved

    data = {'tweet_text': tweet_text,
            # 'tweet_id': tweet_id,
            'tweet_user_id': tweet_user_id,
            # 'tweet_user_location': tweet_user_location,
            'tweet_user_lang': tweet_user_lang,
            # 'tweet_place': tweet_place,
            # 'tweet_coordinates': tweet_coordinates,
            'tweet_county': tweet_county,
            'tweet_education_level': tweet_education_level}
            # 'date_loaded': datetime.datetime.now(),
            # 'tweet_json': json.dumps(tweet)}

    tweets_saved += 1
    return data, tweets_saved

wordPattern = re.compile(r"\b[A-Za-z_.,!\"']+\b", re.IGNORECASE)
httpPattern = re.compile(r"^RT |@\S+|http\S+", re.IGNORECASE)

def parseTweetText(tweet):
    text = tweet['tweet_text']
    text = httpPattern.sub(r"", text)
    words = wordPattern.findall(text)
    tweet['tweet_text'] = words #list(zip(words, [1]*len(words)))
    # print(tweet)
    return tweet

def combineWordLists(x ,y):
    global vocabulary
    if isinstance(x, dict):
        wordDict = x
        xny = y
    else:
        wordDict = dict()
        xny = x + y
    for w in xny:
        # vocabulary +=[w]
        vocabulary += {w: 1}
        try:
            wordDict[w] += 1
        except:
            wordDict[w] = 1

    return wordDict
    # return [(w, wordDict[w]) for w in wordDict]

def genVocabulary(x):
    global vocabulary
    arr = x[1]
    if isinstance(arr, dict):
        return x
    else:
        wordDict = dict()
        for w in arr:
            vocabulary += {w: 1}
            try:
                wordDict[w] += 1
            except:
                wordDict[w] = 1
        x = (x[0],wordDict)
        return x

def handle_file(filename):
    """Takes a filename, loads all tweets into a PostgreSQL database"""
    tweets = load_bz2_json(filename)
    tweet_dicts = []
    tweets_saved = 0
    for tweet in tweets:
        tweet_dict, tweets_saved = load_tweet(tweet, tweets_saved)  # Extracts proper items and places them in database
        if tweet_dict:
            tweet_dicts.append(tweet_dict)
            # print(tweet_dict, "\n")

    # tup = [(d['tweet_id'], d['tweet_text'], d['tweet_user_id'],
    #         d['tweet_coordinates']) for d in tweet_dicts]
    # print("tup = ",tup)
    # for y in tup:
    #     print(y,"\n")

    # args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s)", x) for x in tup)
    # cur.execute("INSERT INTO tweets_test (tweet_id, tweet_text, tweet_locale, created_at_str, date_loaded, tweet_json) VALUES " + args_str)
    return tweet_dicts

def filterTweets(tweet):
    # location = tweet['tweet_user_location']
    # coordinates = tweet['tweet_place']
    # place = tweet['tweet_coordinates']
    text = tweet['tweet_text']
    lang = tweet['tweet_user_lang']
    education = tweet['tweet_education_level']
    county = tweet['tweet_county']
    # if location or coordinates or place: ret = True
    # else: return False
    if not text or text == []: return False
    if lang != 'en': return False
    if education is None or county is None: return False

    return True

def storeResults(traindata, vocab):
    # trainFeat = np.zeros((len(traindata), len(vocab)))
    # trainLabs = np.zeros((len(traindata), 4))
    columnIdx = {vocab[voc][0]: voc for voc in range(len(vocab))}
    # print(columns)

    with open(TRAIN_FEAT_CSV, 'wt') as trainFeatFile, open(TRAIN_LABS_CSV, 'wt') as trainLabsFile, open(TRAIN_FEAT_LABS_CSV, 'wt') as trainFeatLabsFile:
        trainFeatwriter = csv.writer(trainFeatFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        trainLabswriter = csv.writer(trainLabsFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        trainFeatLabswriter = csv.writer(trainFeatLabsFile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        for row in traindata:
            # print(row[0], "###", row[1])
            edu = row[0][1]
            featDict = row[1]
            feats = np.zeros(len(columnIdx))
            for key in featDict:
                try:
                    feats[columnIdx[key]] = featDict[key]
                except:
                    continue
            trainFeatwriter.writerow(feats.tolist())
            trainLabswriter.writerow(list(edu))
            combList = list(edu) + feats.tolist()
            trainFeatLabswriter.writerow(combList)

    # with open(FEATURE_NAMES_CSV, 'wt') as featNamesFile:
    #     featNameswriter = csv.writer(featNamesFile)
    #     featNameswriter.writerows([columnIdx])


def main():
    files_processed = 0
    fileNames = sc.parallelize([])
    for root, dirs, files in os.walk(CACHE_DIR):
        subFileNames = sc.parallelize(files).map(lambda file: os.path.join(root, file))
        fileNames = sc.union([fileNames, subFileNames])
        # for file in files:
        #     files_processed += 1
        #     filename = os.path.join(root, file)
        #     # cur = conn.cursor()
        #     print('Starting work on file ' + str(files_processed) + '): ' + filename)
        #     handle_file(filename)
        #     # if files_processed % 10 == 0:
        #     #     # conn.commit()
        #     # if files_processed == 1000:
        #     #     break
        # if files_processed == 1000:
        #     break
    tweetsRdd = fileNames.flatMap(lambda file: handle_file(file)).filter(lambda tweet: filterTweets(tweet))
    wordsRdd = tweetsRdd.map(lambda tweet: parseTweetText(tweet)).filter(lambda tweet: filterTweets(tweet))
    countyEduRdd = wordsRdd.map(lambda tweet: ((tweet['tweet_user_id'], tweet['tweet_education_level']), tweet['tweet_text']))
    countyEduRdd = countyEduRdd.reduceByKey(lambda x, y: combineWordLists(x, y)).map(lambda z: genVocabulary(z))
    tempRes = countyEduRdd.collect()
    print(tempRes, "\n", len(tempRes))
    vocabRDD = sc.parallelize(vocabulary.value.items())
    vocabRDD = vocabRDD.filter(lambda voc: True if voc[1] > 1 else False)
    # print("vocabulary = ", sorted(vocabulary.value.items(), key=operator.itemgetter(1)))
    vocab = sorted(vocabRDD.collect(), key=operator.itemgetter(1), reverse=True)
    print("vocabulary = ", vocab)
    print("vocabulary size = ", len(vocab))
    storeResults(tempRes, vocab)

if __name__ == "__main__":
    pprint('Starting work!')
    main()
    # profile.run('main()', sort='tottime')
    # Changed logging on table to off
    # conn.close()
else:  # If running interactively in interpreter (Pycharm):
    filename = r"D:\TwitterDatastream\PYTHONCACHE\2015\03\29\00\00.json.bz2"