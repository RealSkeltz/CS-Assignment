#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Packages
import pandas as pd 
import numpy as np
import json
from collections import Counter
import string
from random import randint, random, sample
import collections
from tqdm import tqdm
import itertools
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle


# In[34]:


# Necessary functions

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def EucDist(vector1, vector2):
    a = list(vector1)
    b = list(vector2)
    
    c = [a_i - b_i for a_i, b_i in zip(a, b)]
    dist = np.linalg.norm(c)
    
    return dist

def makeDataSet(data):
    models = list(data.keys())
    N = len(models)
    i = 0
    data_list = []

    while i < N:
        if len(data[models[i]]) == 1:
            data_list.append(data[models[i]][0])

        else:
            for duplicate in data[models[i]]:
                data_list.append(duplicate)

        i+=1
    
    return data_list

def weightsED(bvm, pairs):

    diff_cols = []

    for pair in pairs:
        v0 = bvm.iloc[pair[0]]  
        v1 = bvm.iloc[pair[1]]  
        v2 = v1 - v0
        l = [i for i,k in enumerate(v2) if k != 0]
        diff_cols.extend(l)

    x = Counter(diff_cols)
    x = x.most_common()

    OldMax = x[0][1]
    OldMin = x[-1][1]
    NewMax = 1
    NewMin = -1

    new_weights = []
    for i in range(len(bvm.columns)):
        if i in diff_cols:
            OldValue = [k for k in x if k[0] == i][0][1]
            NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin   
            new_weights.append(NewValue)
        else:
            new_weights.append(1)
    
    return new_weights

def calculateTruePairs(data_list):
    pairs = []

    for i in range(len(data_list)):
        for j in range(len(data_list)):
            if (data_list[i]['modelID'] == data_list[j]['modelID']) & (i != j):
                if (j,i) not in pairs:
                    pairs.append((i,j))
                else:
                    None
    return pairs  


# In[35]:


# Import data

# Opening JSON file
f = open('data.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)

# More workable data
data_list = makeDataSet(data)


# In[36]:


# Step 1: Creating binary vector matrix

def binaryVectorMatrix(data, threshold):
    # Step 1.1: List of words from titles    
    models = list(data.keys())
    N = len(models)
    i = 0
    words = []

    while i < N:
        if len(data[models[i]]) == 1:
            words.extend(data[models[i]][0]['title'].split())
        else:
            for duplicate in data[models[i]]:
                words.extend(duplicate['title'].split())
        i+=1

    #Step 1.1: Clean list of words
    words = [w.lower() for w in words]
    words = [w.replace('"', '-inch') for w in words]
    for char in string.punctuation:
        words = [w.strip(char) for w in words]

    x = Counter(words)
    x = x.most_common()
    filtered_x = []

    for tup in x:
        if tup[1] >= threshold:
            filtered_x.append(tup[0])

    filtered_x.remove("")

    #Step 1.2: Clean titles
    models = list(data.keys())
    N = len(models)
    i = 0
    titles = []
    shops = []

    while i < N:
        if len(data[models[i]]) == 1:
            title = data[models[i]][0]['title'].lower()
            title = title.replace('"', '-inch')
            titles.append(title)

        else:
            for duplicate in data[models[i]]:
                title = duplicate['title'].lower()
                title = title.replace('"', '-inch')
                titles.append(title)

        i+=1
    
    #Step 1.3: Fill dataframe
    df = pd.DataFrame(columns=filtered_x)
    
    new_row = []
    for title in titles:
        for feature in filtered_x:
            if feature in title:
                new_row.append(1)
            else:
                new_row.append(0)

        df.loc[len(df)] = new_row
        new_row = []
    
    return df


# In[37]:


# Step 2: Create signature matrix with Minhashing
# approximate permutations

def minHash(data, N):
    dft = data.transpose()

    signmatrix = np.full((len(dft.columns), N), np.inf)
    hash_values = []

    np.random.seed(1)
    for row in range(len(dft)):
        hash_row = []
        for i in range(N):
            hash_value = (randint(0, N) + randint(1, N) * (row+1)) % 1063
            hash_row.append(hash_value)
        hash_values.append(hash_row)
        for column in dft:
            if (dft.iloc[row][column] == 1):
                for i in range(len(hash_values[row])):
                    value = hash_values[row][i]
                    if value < signmatrix[column][i]:
                        signmatrix[column][i] = value
        
    return signmatrix


# In[38]:


# Step 2: Create signature matrix with Minhashing
# find first 1 in row

def test_minHash(data, N):
    dft = data.transpose()
    signmatrix = np.full((N, len(dft.columns)), np.inf)
    hash_values = []

    for i in range(N):
        dft = shuffle(dft)      
        
        for product in dft:
            value = list(dft[product]).index(1)
            signmatrix[i][product] = value
        
    return signmatrix.T


# In[39]:


# Step 3.1: Create matrix with all bucket assignments per band for each observation
def test_LSH(M, b, r):
    
    n, d = M.shape
    assert(d==b*r)

    bucketmatrix = np.full((b, n), 0)    

    k=0
    for band in range(b):
        signature_list = []

        for product in range(n):
            partial_signature = M[product, k:r+k]
            mod_par_sig = list(partial_signature % 10)
            
            if mod_par_sig not in signature_list:
                signature_list.append(mod_par_sig)
            else:
                None

            bucket = signature_list.index(mod_par_sig)
            bucketmatrix[band][product] = bucket    
        
        k = k + r
        
    return bucketmatrix


# In[40]:


# Step 3: Create candidate pairs with LSH

# Step 3.1: Create matrix with all bucket assignments per band for each observation
def LSH(M, b, r):
    
    n, d = M.shape
    assert(d==b*r)

    bucketmatrix = np.full((b, n), 0)    

    k=0
    for band in range(b):
        signature_list = []

        for product in range(n):
            partial_signature = list(M[product, k:r+k])

            if partial_signature not in signature_list:
                signature_list.append(partial_signature)
            else:
                None

            bucket = signature_list.index(partial_signature)
            bucketmatrix[band][product] = bucket    
        
        k = k + r
        
    return bucketmatrix

# Step 3.2: Create candidate pairs with the bucket matrix
def candidatePairs(bm, b, r):
    candidate_pairs = set() 
    hashbuckets = collections.defaultdict(set)

    for band_id in range(b):
        band = list(bm[band_id])
        for bucket in set(band):
            for index in range(len(band)):
                if band[index] == bucket:
                    hashbuckets[bucket].add(index)
                else:
                    None

        candidate_pairs = set() 
        for bucket in hashbuckets.values():
            if len(bucket) > 1:
                for pair in itertools.combinations(bucket, 2):
                    candidate_pairs.add(pair)

    return candidate_pairs

# Step 3.3: Create set of LSH-pairs that satisfy threshold
def LSHpairs(cp, sm, b, r, jac_threshold):
    
    lsh_pairs = set()
    for (i, j) in cp:
        if jaccard_similarity(sm[i], sm[j]) > jac_threshold:
            lsh_pairs.add((i, j))
            
    return lsh_pairs


# In[41]:


# Step 3.2: Create candidate pairs with the bucket matrix
def test_candidatePairs(bm, b, r):
    candidate_pairs = set() 
    hashbuckets = collections.defaultdict(set)

    for band_id in range(b):
        band = list(bm[band_id])
        for bucket in set(band):
            for index in range(len(band)):
                if band[index] == bucket:
                    hashbuckets[bucket].add(index)
                else:
                    None

        candidate_pairs = set() 
        for bucket in hashbuckets.values():
            if len(bucket) > 1:
                for pair in itertools.permutations(bucket, 2):
                    candidate_pairs.add(pair)

    return candidate_pairs


# In[42]:


# Step 4: Calculate similarity measure based on Euclidean distance and predict duplicates
def predDups(bvm, obs, threshold, lshp, weights, weighted=False):
    shop = data_list[obs]['shop']
    
    # Search within LSH-pairs
    pairs = []
    
    for val in lshp:
        if val[0] == obs:
            # Filter products from the same webshop
            if data_list[val[1]]['shop'] != shop:
                pairs.append(val[1])
            else:
                None
        else:
            None
        
    nb_dist = []    
    for pair in pairs: 
        if weighted == True:
            dist = EucDist(weights[obs] * bvm.iloc[obs], weights[pair] * bvm.iloc[pair])
        else:
            dist = EucDist(bvm.iloc[obs], bvm.iloc[pair])
            
        if dist < threshold:
                nb_dist.append(pair) 
    
    return nb_dist


# In[163]:


# Set parameters
min_occurences = 1
hash_fn = 50 #50
b = 10 #10
r = 5
#jac_threshold = 0.
jac_threshold = (1/b)**(1/r)
dist_threshold = 3
print('Parameters set')

# Do functions
data_list = makeDataSet(data)
pairs = calculateTruePairs(data_list)
weights = weightsED(bvm, pairs)

bvm = binaryVectorMatrix(data, min_occurences)
sm = minHash(bvm, hash_fn)
bm = test_LSH(sm, b, r)
cp = test_candidatePairs(bm, b, r)
lshp = LSHpairs(cp, sm, b, r, jac_threshold)

# Check predictions
preds = pd.DataFrame(columns=['product','predicted_duplicates','true_duplicates'])

print('Ready for blast-off')
for product in tqdm(range(len(data_list))):
    true_duplicates = []
    predicted_duplicates = predDups(bvm, product, dist_threshold, lshp, weights)
    
    for item in range(len(data_list)):
        if (data_list[item]['modelID'] == data_list[product]['modelID']) & (item!=product):
            true_duplicates.append(item)
    
    preds = preds.append({'product':product, 'predicted_duplicates':predicted_duplicates, 'true_duplicates':true_duplicates}, 
                 ignore_index = True)


# In[144]:


# Step 5: Evaluation

# Step 5.1: Evaluation of whole algorithm

# Set parameters
min_occurences = 1
hash_fn = 50 
b = 10
r = 5
jac_threshold = 0.
dist_threshold = 10

# Do functions
data_list = makeDataSet(data)
bvm = binaryVectorMatrix(data, min_occurences)
sm = minHash(bvm, hash_fn)
bm = test_LSH(sm, b, r)
cp = test_candidatePairs(bm, b, r)
lshp = LSHpairs(cp, sm, b, r, jac_threshold)


# In[171]:


def bootstrap(pairs, data_list, bvm, dist_threshold, lshp, bootstraps):
    
    duplicates = set()
    for pair in pairs:
        duplicates.add(pair[0])
        duplicates.add(pair[1])

    total_duplicates = len(duplicates)

    # Set up bootstrap
    for i in tqdm(range(bootstraps)):

        total = range(0,1624)
        train = sample(range(0, 1623), int(1624*0.63))
        test = [x for x in total if x not in randomlist]

        train_datalist = []
        for num in train:
            train_datalist.append(data_list[num])

        test_datalist = []
        for num in test:
            test_datalist.append(data_list[num])

        train_pairs = calculateTruePairs(train_datalist)
        weights = weightsED(bvm, train_pairs)

        # Make predictions
        total_predictions = 0
        correct_predictions = 0

        for product in test:
            predicted_duplicates = predDups(bvm, product, dist_threshold, lshp, weights, weighted=True)
            total_predictions = total_predictions + len(predicted_duplicates)

            for pd in predicted_duplicates:
                if ((pd, product) in pairs) | ((product, pd) in pairs):
                    correct_predictions = correct_predictions + 1
        
        PC = correct_predictions*2 / total_duplicates
        PQ = correct_predictions*2 / total_predictions
        F1 = (2*PC*PQ)/(PC+PQ)
        
        print(len(lshp))
        
        print(PC, PQ, F1)


# In[173]:


bootstrap(pairs, data_list, bvm, 3, lshp, 5)


# In[51]:


duplicates = set()
for pair in pairs:
    duplicates.add(pair[0])
    duplicates.add(pair[1])

total_duplicates = len(duplicates)


# In[114]:


def PCandPQ(cp, sm, b, r, pairs):
    l_cp = []
    l_comp_made = []
    
    for t in range(1,11,1):
        print(t)
        t = t/10
        lshp = LSHpairs(cp, sm, b, r, t)
        l_comp_made.append(len(lshp))
        correct_pairs = 0
        for pair in lshp:
            if pair in pairs:
                correct_pairs = correct_pairs + 1
                
        l_cp.append(correct_pairs)
    return l_cp, l_comp_made

