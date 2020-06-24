#!/usr/bin/env python
# coding: utf-8

# !pip install textsearch
# !pip install contractions
# !pip install textsearch
# !pip install --user gensim

# In[706]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import datetime
from datetime import datetime


# In[280]:


#add path from your computer
df = pd.read_csv('Internship_Assignment.csv',encoding= 'unicode_escape',dtype=str)
df.head()


# In[281]:


df_new = df[['NAME', 'TAGLINE', 'TAGS', 'LAUNCH DATE']]
df_new.TAGLINE.fillna('', inplace=True)
df_new.TAGS.fillna('', inplace=True)
df_new['text'] = df_new['TAGLINE'].map(str) + ' ' + df_new['TAGS']
df_new.dropna(inplace=True)
df_new.info()


# In[282]:


df_new.head(10)


# In[283]:


import nltk
import re
import numpy as np
import contractions

#for removing stopwords
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    doc = contractions.fix(doc)
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    #filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

norm_corpus = normalize_corpus(list(df_new['text']))
len(norm_corpus)


# In[285]:


from gensim.models import FastText

tokenized_docs = [doc.split() for doc in norm_corpus]
ft_model = FastText(tokenized_docs, window=4, min_count=1, workers=4, sg=1, iter=100)


# In[286]:


len(tokenized_docs)


# In[287]:


tokenized_docs


# In[288]:


def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)


# In[289]:


doc_vecs_ft = averaged_word2vec_vectorizer(tokenized_docs, ft_model, 100)
doc_vecs_ft.shape


# In[702]:


#for knowing optimal cluters and plotting the cluster plot using pca and tsne
tfidf = TfidfVectorizer(
    min_df = 0.01,
    stop_words = 'english'
)
tfidf.fit(df_new.TAGLINE)
text = tfidf.transform(df_new.text)


# In[709]:


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(text, 8)


# In[695]:


NUM_CLUSTERS = 5
km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=10000, n_init=100, random_state=42).fit_predict(doc_vecs_ft)
km.shape


# In[637]:


print(km)


# In[704]:


# for visualisation 
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000, replace=False)
    
    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(data[max_items,:].todense()))
    
    
    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')
    
    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    
plot_tsne_pca(text, km)


# In[698]:


df['kmeans_cluster'] = km
df.head()
df.to_csv("clustering.csv")


# In[638]:


df_new['kmeans_cluster'] = km
df_new.head()


# In[641]:


df.groupby(df_new['kmeans_cluster']).count()


# In[642]:


# cluster wise company name
for cluster_num in range(NUM_CLUSTERS):
    company = comp_clusters[comp_clusters['kmeans_cluster'] == cluster_num]['NAME'].values.tolist()
    print('CLUSTER #'+str(cluster_num+1))
    print('Popular :', company)
    print('-'*80)


# In[643]:


#cluster1
arr=np.where(km==0) ## index of docs in cluster 1
km[arr]
cluster1=df_new.iloc[arr]  ## all docs in cluster 1
#print cluster1['text', 'tags','kmeans_cluster', 'LAUNCH DATE'])
print(cluster1.columns)   
cluster1


# In[645]:


#as the tokenised words or documents within cluster has no related meaning with each other
cluster1['type']=cluster1['kmeans_cluster'].apply(lambda x: 'unclassified' if x == 0 else 'bug')
cluster1.head()


# In[646]:


arr=np.where(km==1) ## index of docs in cluster 2
km[arr]
cluster2=df_new.iloc[arr]  ## all docs in cluster 2
#print cluster1['text', 'tags','kmeans_cluster', 'LAUNCH DATE'])
print(cluster2.columns)   
cluster2


# In[691]:


# as all these are related to the finance and risk for which entity is not defined
cluster2['type']=cluster2['kmeans_cluster'].apply(lambda x: 'unclassified' if x == 1 else 'bug')
cluster2.head()


# In[647]:


arr=np.where(km==2) ## index of docs in cluster 3
km[arr]
cluster3=df_new.iloc[arr]  ## all docs in cluster 3


# In[660]:


# as the tokenised words are similar to schools education, friends and similar tags
cluster3['type']=cluster3['kmeans_cluster'].apply(lambda x: 'schools/University' if x == 2 else 'bug')
cluster3.head()


# In[649]:


arr=np.where(km==3) ## index of docs in cluster 4
km[arr]
cluster4=df_new.iloc[arr]  ## all docs in cluster 4


# In[658]:


arr=np.where(km==4) ## index of docs in cluster 5
km[arr]
cluster5=df_new.iloc[arr] 
## all docs in cluster 5

cluster5.head()


# In[659]:


#as the tokenised words are related to the public,research and similar tags
cluster5['type']=cluster5['kmeans_cluster'].apply(lambda x: 'Government/Non-profit' if x == 4 else 'bug')
cluster5.head()


# In[651]:


#define cluster no for which cluster you want the results of tokinize
cluster5=str(cluster5[['text']])
# removes punctuation and returns list of words
tokenizer = RegexpTokenizer(r'\w+')
zen_no_punc = tokenizer.tokenize(cluster5)#initialize cluster no for which you want tokens


# In[652]:


#hash_map for tokenized word
from collections import Counter
word_count_dict = Counter(w.title() for w in zen_no_punc if w.lower() not in stopwords.words())
word_count_dict.most_common()


# In[653]:


df_new.head()


# In[681]:


df_classify=pd.concat([cluster1,cluster4], ignore_index=True)


# In[682]:


#creating new column in datetime format for classification of startup and mature companies
df_classify['date']=pd.to_datetime(df_new['date'],errors='coerce')
df_new.head()


# In[683]:


some_date='1990-01-01' #classification for companies for mature and startup companies based on launch date


# In[684]:


date_before= datetime.strptime(some_date,'%Y-%m-%d')


# In[685]:


df_classify['type']=np.where((df_classify['date'] < some_date), 'mature_company', 'startup_company')


# In[686]:


df_classify.head(25)


# In[687]:


df_classify['type'].value_counts()


# In[692]:


df_final=pd.concat([cluster2,cluster3,df_classify,cluster5], ignore_index=True)


# In[693]:


#final dataframe with type as entity being categorized
df_final.head(20)


# In[694]:


#final result count
df_final['type'].value_counts()


# In[ ]:




