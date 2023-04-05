#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install -q snscrape


# In[31]:


import os
import pandas as pd
from datetime import date
import numpy as np


# In[4]:


#today = date.today()
#end_date = today
#print(end_date)


# In[5]:


#search_term = "Imran Khan"
#from_date = "2022-03-08"


# In[9]:


#os.system("snscrape --jsonl --max-results 500 --since 2022-03-08 twitter-search 'imran khan until:2020-04-24' > text-query-tweets.json")


# In[10]:


#tweets_df = pd.read_json('text-query-tweets.json', lines=True)


# In[13]:


#import snscrape.modules.twitter as sntwitter
#import pandas as pd

# Creating list to append tweet data to
#tweets_list2 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
#for i,tweet in enumerate(sntwitter.TwitterSearchScraper('imran khan since:2022-03-08 until:2022-04-24').get_items()):
#    if i>1000:
#        break
#    tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
#tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


# In[14]:


#tweets_df2.head


# In[16]:


#tweets_df2


# In[11]:


import snscrape.modules.twitter as sntwitter
import pandas as pd

# Creating list to append tweet data to
tweets_list3 = []

# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('pakistan since:2022-04-03 until:2022-04-11').get_items()):
    if i>999:
        break
    tweets_list3.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
tweets_df3 = pd.DataFrame(tweets_list3, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])


# In[12]:


tweets_df3


# In[13]:


tweets_cleaned = tweets_df3.drop(columns=['Datetime', 'Tweet Id', 'Username'])


# In[14]:


tweets_cleaned


# In[15]:


import string
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

#add punctuation char's to stopwords list
stop_words += list(string.punctuation) # <-- contains !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

#add integers and search term variations
stop_words += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Pakistan', 'pakistan', 'Pakistani', 'pakistani', 'Imran', 'imran', 'Khan', 'khan']


# In[16]:


from nltk import word_tokenize

def tokenize_lowercase(text):
    tokens = word_tokenize(text)
    stopwords_removed = [token.lower() for token in tokens if token.lower() not in stop_words]
    return stopwords_removed

tweets_cleaned['Text'].apply(tokenize_lowercase)


# In[17]:


tweets_cleaned['Text'] = tweets_cleaned['Text'].str.replace(r"http\S+", "")


# In[18]:


tweets_cleaned


# In[19]:


all_words = [word for tokens in tweets_cleaned['Text'] for word in tokens]
tweet_lengths = [len(tokens) for tokens in tweets_cleaned['Text']]
vocab = sorted(list(set(all_words)))

print('{} words total, with a vocabulary size of {}'.format(len(all_words), len(vocab)))
print('Max tweet length is {}'.format(max(tweet_lengths)))


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (15,8))
sns.countplot(tweet_lengths)
plt.title('Tweet Length Distribution', fontsize = 18)
plt.xlabel('Words per Tweet', fontsize = 12)
plt.ylabel('Number of Tweets', fontsize = 12)
plt.show()


# In[21]:


from nltk.probability import FreqDist

#iterate through each tweet, then each token in each tweet, and store in one list
flat_words = [item for sublist in tweets_cleaned['Text'] for item in sublist]

word_freq = FreqDist(flat_words)

word_freq.most_common(30)


# In[22]:


#retrieve word and count from FreqDist tuples

most_common_count = [x[1] for x in word_freq.most_common(30)]
most_common_word = [x[0] for x in word_freq.most_common(30)]

#create dictionary mapping of word count
top_30_dictionary = dict(zip(most_common_word, most_common_count))
top_30_dictionary


# In[23]:


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(tweets_cleaned['Text'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()


# In[44]:


import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'pakistan', 'imran', 'khan', 'imrankhan', 'imrankhanpti', 'ki', 'ko', 'hai', 'IK', 'u', 'nhi', 'jo'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]
data = tweets_cleaned.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])


# In[45]:


import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])


# In[46]:


from pprint import pprint
# number of topics
lst = list(np.arange(1,10+1))
#num_topics = k
# Build LDA model
#lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                      num_topics=5)
#for l in lst:
# Print the Keyword in the different number of topics
#    num_topics = l
#    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)
#    pprint(lda_model.print_topics())

    
    
#doc_lda = lda_model[corpus]


# In[53]:


def get_lda_topics(model, num_topics, top_n_words):
     word_dict = {}
     for i in range(num_topics):
         word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in model.show_topic(i, topn = top_n_words)];
 
     return pd.DataFrame(word_dict)
get_lda_topics(lda_model,5,10)


# In[57]:


lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                      num_topics=5)


# In[58]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis


# In[ ]:




