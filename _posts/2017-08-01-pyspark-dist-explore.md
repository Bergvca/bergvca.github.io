---
layout: post
title:  "PySpark Dist Explore"
date:   2017-08-01
---

# PySpark Dataframe Distribution Explorer

I found myself using some half baked, quickly written functions to do data exploration in PySpark, 
every time using a similar but modified version of the same function. So I decided to create a more
structural solution. The result is: [pyspark_dist_explore](https://github.com/Bergvca/pyspark_dist_explore)

## Create histograms as you would in Matplotlib

Creating an histogram is as easy as:

```python
hist(ax, dataframe, **kwargs)
```

Where ax is a matplotlib Axes object. dataframe a PySpark DataFrame, and kwargs all the kwargs you would 
use in the matplotlib hist function.

## Other options

I've added some other options I found myself using a lot as well:

* __distplot(ax, x, **kwargs).__ Combines a normalized histogram of each column in x with a density plot of the same column.

* __pandas_histogram(x, bins=None, range=None).__ Creates histograms for all columns in x and converts this to a Pandas DataFrame

See for more info:

[https://github.com/Bergvca/pyspark_dist_explore](https://github.com/Bergvca/pyspark_dist_explore)