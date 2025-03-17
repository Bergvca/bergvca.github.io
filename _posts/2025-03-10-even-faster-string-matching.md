---
layout: post
title:  "Even Faster String matching in Python"
date:   2025-01-02
---

## TL;DR

[String Grouper](https://github.com/Bergvca/string_grouper) is now _87% faster_ (than 5 years ago).

## String Grouper

A few years ago I wrote a post about a method of [String Matching](2017/10/14/super-fast-string-matching.md) were we use
**tf-idf** and **cosine-similarity** to find similar strings in large datasets. The blogpost mentioned a time of
`45 minutes` to find all potential duplicates in a dataset with 663 000 different company names. A few years later (2020)
I took some time to put the code together with some grouping functions in a module and upload it to [PyPI](https://pypi.org/project/string-grouper/): 
the python repository. [String Grouper](https://github.com/Bergvca/string_grouper) was born. 

## 5 Years Later

To my surprise, the module gained traction and active contributors. Or, maybe not so surprising, since [Record Linkage](https://en.wikipedia.org/wiki/Record_linkage), 
is a frequently and extensively employed process. Any piece of code that can make this task easier and faster
could be useful to many industries. After a while, the module ran into some issues with the v3 version of `Cython`, became hard
to install, and as such was pretty much _dead_. TThe strength of open source shone through when recently a new contributor
came along and updated code to be compliant with Cython v3 again. 

## 87% faster

Since this new version is now _"presentable"_ again (it works by simply `pip install`-ing it), it seems like a good time to
write another blog post on it. The latest version has new functionalities that can be found in the
new [documentation](https://bergvca.github.io/string_grouper/). The most striking improvement, however, is the same
string matching exercise described in [Super Fast String Matching](2017/10/14/super-fast-string-matching.md), on the 
exact same hardware (_yes my laptop is old!_) now runs in **5:34** minutes, or an _87%_ increase in speed. 

## Coincidence

The main reason for this huge improvement was discovered as a bit of a coincidence: a user was noticing `OverflowError`'s 
due to large arrays getting created. Even though the algorithm works on sparse matrices, there is
still an array created with the number of rows in the matrix you are trying to compare with 
(equal to the length of the pandas Series/Dataframe). These can be huge and cause OverFlow errors. Therefore, an option
was added to split the matrices in several smaller matrices, which do not need to instantiate arrays of the same length 
as when using the original matrices. It turned out that this could drastically improve the performance. See also the [Performance](https://bergvca.github.io/string_grouper/performance/)
section of the documentation.

## Blocks

The idea behind this partitioning of the matrices (actually, just 1 matrix) works as follows: We go from the regular dot 
product:
<img src="/media/img/regular matrix mult.svg" style="height: 100%;width: 100%;">

To partitioning the second matrix in multiple blocks and performing the multiplication on all the blocks sequentially:

<img src="/media/img/regular partitioned.svg" style="height: 100%;width: 100%;">

So: we split the second matrix into many small matrices and perform multiplication of the big matrix with all the small
matrices. At the end we merge these matrices together into a single matrix again. 

In practice the B matrices are so much smaller than the original, they will fit in a CPU's cache (as opposed to the 
computer's RAM memory). As the values in the B matrices are accessed much more often this results in a tremendous boost in access speed.

In [tests](https://github.com/Bergvca/string_grouper/issues/93#issuecomment-2599842979) we've estimated that a block size
of +/- 4000 (e.g. each partitioned _B_ matrix has around 4000 rows) seems to be a good default size. 
For the above-mentioned company name dataset it means
the blocks are of shape 3994 by 34 835. While this still seems huge, remember that the matrices are stored in _sparse
[csr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html)_ format. The actual amount
of floating point numbers for our dataset in each B matrix (_or: number of non-zero elements_) is around 75k. When using
64 bit floating point numbers this is around 585 kilobytes, or slightly more than my laptop's L2 cache (512KB). 
The number of floating point numbers in the A matrix is 11.7 million, or 89 MB, which does not fit in my CPU cache at all (the L3 cache is 4MB). 


Now, I'm by no means an expert on CPU's and their cache but to me, it sounds like a plausible explanation for this major speed improvement.  
