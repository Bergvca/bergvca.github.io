---
layout: post
title:  "Super Fast String Matching in Python"
date:   2017-10-14
---

Traditional approaches to string matching such as the [Jaro-Winkler](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) or [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) distance measure are too slow for large datasets. Using TF-IDF with N-Grams as terms
 to find similar strings transforms the problem into a matrix multiplication problem, which is computationally much cheaper. Using this approach made it possible to search for near duplicates in a set of 663,000 company names in 42 minutes using only a dual-core laptop. 

*Update: run all code in the below post with one line using [string_grouper](/2019/01/02/string-grouper.html*:

`match_strings(companies['Company Name'])`

## Name Matching

A problem that I have witnessed working with databases, and I think many other people with me,
 is name matching. Databases often have multiple entries that relate to the same entity, for example a person 
 or company, where one entry has a slightly different spelling then the other. This is a problem, and 
 you want to de-duplicate these. A similar problem occurs when you want to merge or join databases 
 using the names as identifier. 

The following table gives an example:

|Company Name|
|--------|
|Burger King|
|Mc Donalds|
|KFC|
|Mac Donald's|

For the human reader it is obvious that both *Mc Donalds* and *Mac Donald's* are the same company.
 However for a computer these are completely different making spotting these nearly identical strings difficult. 
 
One way to solve this would be using a string similarity measures 
like [Jaro-Winkler](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) or 
the [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) distance measure. 
The obvious problem here is that the amount of calculations necessary grow quadratic. 
Every entry has to be compared with every other entry in the dataset, in our case this 
means calculating one of these measures 663.000^2 times. In this post I will explain how this can be done faster 
using TF-IDF, N-Grams, and sparse matrix multiplication. 

## The Dataset

I just grabbed a random dataset with lots of company names 
from [Kaggle](https://www.kaggle.com/dattapiy/sec-edgar-companies-list). It contains all company 
names in the SEC EDGAR database. I don't know anything about the data or the amount of duplicates in this
 dataset (it should be 0), but most likely there will be some very similar names. 


```python
import pandas as pd

pd.set_option('display.max_colwidth', -1)
names =  pd.read_csv('data/sec_edgar_company_info.csv')
print('The shape: %d x %d' % names.shape)
names.head()
```

    The shape: 663000 x 3





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: center;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th></th>
      <th>Line Number</th>
      <th>Company Name</th>
      <th>Company CIK Key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>!J INC</td>
      <td>1438823</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>#1 A LIFESAFER HOLDINGS, INC.</td>
      <td>1509607</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>#1 ARIZONA DISCOUNT PROPERTIES LLC</td>
      <td>1457512</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>#1 PAINTBALL CORP</td>
      <td>1433777</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>$ LLC</td>
      <td>1427189</td>
    </tr>
  </tbody>
</table>
</div>



## TF-IDF

TF-IDF is a method to generate features from text by multiplying the frequency of a term (usually a word) in a 
document (the *Term Frequency*, or *TF*) by the importance (the *Inverse Document Frequency* or *IDF*) of the same 
term in an entire corpus. This last term weights less important words (e.g. the, it, and etc) down, and words that 
don't occur frequently up. IDF is calculated as: 

```IDF(t) = log_e(Total number of documents / Number of documents with term t in it).```

An example (from [www.tfidf.com/](http://www.tfidf.com/)):

Consider a document containing 100 words in which the word cat appears 3 times. The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.

TF-IDF is very useful in text classification and text clustering. It is used to transform documents into numeric 
vectors, that can easily be compared. 

## N-Grams

While the terms in TF-IDF are usually words, this is not a necessity. In our case using words as terms wouldn't help us much, as most company names only contain one or two words. This is why we will use *n-grams*: sequences of *N*  contiguous items, in this case characters. The following function cleans a string and generates all n-grams in this string:


```python
import re

def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

print('All 3-grams in "McDonalds":')
ngrams('McDonalds')
```

    All 3-grams in "McDonalds":

    ['McD', 'cDo', 'Don', 'ona', 'nal', 'ald', 'lds']



As you can see, the code above does some cleaning as well. Next to removing some punctuation (dots, comma's etc) it removes the string " BD". This is a nice example of one of the pitfalls of this approach: some terms that appear very infrequent will result in a high bias towards this term. In this case there where some company names ending with " BD" that where being identified as similar, even though the rest of the string was not similar. 

The code to generate the matrix of TF-IDF values for each is shown below. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer

company_names = names['Company Name']
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(company_names)
```

The resulting matrix is very sparse as most terms in the corpus will not appear in most company names. Scikit-learn deals with this nicely by returning a sparse [CSR  matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html).

You can see the first row ("!J INC") contains three terms for the columns 11, 16196, and 15541.


```python
print(tf_idf_matrix[0])

# Check if this makes sense:

ngrams('!J INC')
```

      (0, 11)	0.844099068282
      (0, 16196)	0.51177784466
      (0, 15541)	0.159938115034

    ['!JI', 'JIN', 'INC']



The last term ('INC') has a relatively low value, which makes sense as this term will appear often in the 
corpus, thus receiving a lower IDF weight.

## Cosine Similarity 

To calculate the similarity between two vectors of TF-IDF values the *Cosine Similarity* is usually used. 
The cosine similarity can be seen as a normalized dot product. 
For a good explanation see: [this site](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/). 
We can theoretically calculate the cosine similarity of all items in our dataset with all other items 
in scikit-learn by using the cosine_similarity function, however the Data Scientists 
at ING found out this has [some disadvantages](https://medium.com/@ingwbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618):

- The sklearn version does a lot of type checking and error handling.
- The sklearn version calculates and stores all similarities in one go, while we are only interested in the most similar ones. Therefore it uses a lot more memory than necessary.

To optimize for these disadvantages they created their [own library](https://github.com/ing-bank/sparse_dot_topn) which stores only the top N highest matches in each row, and only the similarities above an (optional) threshold. 


```python
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))
```

The following code runs the optimized cosine similarity function. It only stores the top 10 most similar items, and only items with a similarity above 0.8:


```python
import time
t1 = time.time()
matches = awesome_cossim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, 0.8)
t = time.time()-t1
print("SELFTIMED:", t)
```

    SELFTIMED: 2718.7523670196533


The following code unpacks the resulting sparse matrix. As it is a bit slow, an option to look at only the first *n* values is added. 


```python
def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()
    
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]
    
    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size
    
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    
    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]
    
    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})
        
```

Lets look at our matches:


```python
matches_df = get_matches_df(matches, company_names, top=100000)
matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
matches_df.sample(20)
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>left_side</th>
      <th>right_side</th>
      <th>similairity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41024</th>
      <td>ADVISORY U S EQUITY MARKET NEUTRAL OVERSEAS FUND LTD</td>
      <td>ADVISORY US EQUITY MARKET NEUTRAL FUND LP</td>
      <td>0.818439</td>
    </tr>
    <tr>
      <th>48061</th>
      <td>AIM VARIABLE INSURANCE FUNDS</td>
      <td>AIM VARIABLE INSURANCE FUNDS (INVESCO VARIABLE INSURANCE FUNDS)</td>
      <td>0.856922</td>
    </tr>
    <tr>
      <th>14978</th>
      <td>ACP ACQUISITION CORP</td>
      <td>CP ACQUISITION CORP</td>
      <td>0.913479</td>
    </tr>
    <tr>
      <th>54837</th>
      <td>ALLFIRST TRUST CO NA</td>
      <td>ALLFIRST TRUST CO NA                         /TA/</td>
      <td>0.938206</td>
    </tr>
    <tr>
      <th>89788</th>
      <td>ARMSTRONG MICHAEL L</td>
      <td>ARMSTRONG MICHAEL</td>
      <td>0.981860</td>
    </tr>
    <tr>
      <th>54124</th>
      <td>ALLEN MICHAEL D</td>
      <td>ALLEN MICHAEL J</td>
      <td>0.928606</td>
    </tr>
    <tr>
      <th>66765</th>
      <td>AMERICAN SCRAP PROCESSING INC</td>
      <td>SCRAP PROCESSING INC</td>
      <td>0.858714</td>
    </tr>
    <tr>
      <th>44886</th>
      <td>AGL LIFE ASSURANCE CO SEPARATE ACCOUNT VA 27</td>
      <td>AGL LIFE ASSURANCE CO SEPARATE ACCOUNT VA 24</td>
      <td>0.880202</td>
    </tr>
    <tr>
      <th>49119</th>
      <td>AJW PARTNERS II LLC</td>
      <td>AJW PARTNERS LLC</td>
      <td>0.876761</td>
    </tr>
    <tr>
      <th>16712</th>
      <td>ADAMS MICHAEL C.</td>
      <td>ADAMS MICHAEL A</td>
      <td>0.891636</td>
    </tr>
    <tr>
      <th>96207</th>
      <td>ASTRONOVA, INC.</td>
      <td>PETRONOVA, INC.</td>
      <td>0.841667</td>
    </tr>
    <tr>
      <th>26079</th>
      <td>ADVISORS DISCIPLINED TRUST 1329</td>
      <td>ADVISORS DISCIPLINED TRUST 1327</td>
      <td>0.862806</td>
    </tr>
    <tr>
      <th>16200</th>
      <td>ADAMANT TECHNOLOGIES</td>
      <td>NT TECHNOLOGIES, INC.</td>
      <td>0.814618</td>
    </tr>
    <tr>
      <th>77473</th>
      <td>ANGELLIST-SORY-FUND, A SERIES OF ANGELLIST-SDA-FUNDS, LLC</td>
      <td>ANGELLIST-NABS-FUND, A SERIES OF ANGELLIST-SDA-FUNDS, LLC</td>
      <td>0.828394</td>
    </tr>
    <tr>
      <th>70624</th>
      <td>AN STD ACQUISITION CORP</td>
      <td>OT ACQUISITION CORP</td>
      <td>0.855598</td>
    </tr>
    <tr>
      <th>16669</th>
      <td>ADAMS MARK B</td>
      <td>ADAMS MARY C</td>
      <td>0.812897</td>
    </tr>
    <tr>
      <th>48371</th>
      <td>AIR SEMICONDUCTOR INC</td>
      <td>LION SEMICONDUCTOR INC.</td>
      <td>0.814091</td>
    </tr>
    <tr>
      <th>53755</th>
      <td>ALLEN DANIEL M.</td>
      <td>ALLEN DANIEL J</td>
      <td>0.829631</td>
    </tr>
    <tr>
      <th>16005</th>
      <td>ADA EMERGING MARKETS FUND, LP</td>
      <td>ORANDA EMERGING MARKETS FUND LP</td>
      <td>0.839016</td>
    </tr>
    <tr>
      <th>97135</th>
      <td>ATHENE ASSET MANAGEMENT LLC</td>
      <td>CRANE ASSET MANAGEMENT LLC</td>
      <td>0.807580</td>
    </tr>
  </tbody>
</table>
</div>



The matches look pretty similar! The cossine similarity gives a good indication of the similarity between the two company names. *ATHENE ASSET MANAGEMENT LLC* and *CRANE ASSET MANAGEMENT LLC* are probably not the same company, and the similarity measure of 0.81 reflects this. When we look at the company names with the highest similarity, we see that these are pretty long strings that differ by only 1 character:


```python
matches_df.sort_values(['similairity'], ascending=False).head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: center;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>left_side</th>
      <th>right_side</th>
      <th>similairity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77993</th>
      <td>ANGLE LIGHT CAPITAL, LP - ANGLE LIGHT CAPITAL - QUASAR SERIES I</td>
      <td>ANGLE LIGHT CAPITAL, LP - ANGLE LIGHT CAPITAL - QUASAR SERIES II</td>
      <td>0.994860</td>
    </tr>
    <tr>
      <th>77996</th>
      <td>ANGLE LIGHT CAPITAL, LP - ANGLE LIGHT CAPITAL - QUASAR SERIES II</td>
      <td>ANGLE LIGHT CAPITAL, LP - ANGLE LIGHT CAPITAL - QUASAR SERIES I</td>
      <td>0.994860</td>
    </tr>
    <tr>
      <th>81120</th>
      <td>APOLLO OVERSEAS PARTNERS (DELAWARE 892) VIII, L.P.</td>
      <td>APOLLO OVERSEAS PARTNERS (DELAWARE 892) VII LP</td>
      <td>0.993736</td>
    </tr>
    <tr>
      <th>81116</th>
      <td>APOLLO OVERSEAS PARTNERS (DELAWARE 892) VII LP</td>
      <td>APOLLO OVERSEAS PARTNERS (DELAWARE 892) VIII, L.P.</td>
      <td>0.993736</td>
    </tr>
    <tr>
      <th>66974</th>
      <td>AMERICAN SKANDIA LIFE ASSURANCE CORP VARIABLE ACCOUNT E</td>
      <td>AMERICAN SKANDIA LIFE ASSURANCE CORP VARIABLE ACCOUNT B</td>
      <td>0.993527</td>
    </tr>
    <tr>
      <th>66968</th>
      <td>AMERICAN SKANDIA LIFE ASSURANCE CORP VARIABLE ACCOUNT B</td>
      <td>AMERICAN SKANDIA LIFE ASSURANCE CORP VARIABLE ACCOUNT E</td>
      <td>0.993527</td>
    </tr>
    <tr>
      <th>80929</th>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND III (EURO B), L.P.</td>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND II (EURO B), L.P.</td>
      <td>0.993375</td>
    </tr>
    <tr>
      <th>80918</th>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND II (EURO B), L.P.</td>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND III (EURO B), L.P.</td>
      <td>0.993375</td>
    </tr>
    <tr>
      <th>80921</th>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND III (DOLLAR A), L.P.</td>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND II (DOLLAR A), L.P.</td>
      <td>0.993116</td>
    </tr>
    <tr>
      <th>80907</th>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND II (DOLLAR A), L.P.</td>
      <td>APOLLO EUROPEAN PRINCIPAL FINANCE FUND III (DOLLAR A), L.P.</td>
      <td>0.993116</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

As we saw by visual inspection the matches created with this method are quite good, as the strings are very similar. 
The biggest advantage however, is the speed. The method described above can be scaled to much larger
 datasets by using a distributed computing environment such as Apache Spark. This could be done by broadcasting 
 one of the TF-IDF matrices to all workers, and parallelizing the second (in our case a copy of the TF-IDF matrix) 
 into multiple sub-matrices. Multiplication can then be done (using Numpy or the sparse_dot_topn library) 
 by each worker on part of the second matrix and the entire first matrix. 
 An example of this is described [here](https://labs.yodas.com/large-scale-matrix-multiplication-with-pyspark-or-how-to-match-two-large-datasets-of-company-1be4b1b2871e).
