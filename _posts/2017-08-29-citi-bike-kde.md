---
layout: post
title:  "1 Day of Citi Bike availability "
date:   2017-08-29
---

After moving to New York from the Netherlands I was relieved to find out that biking in
Manhattan is actually pretty do-able. It's not really as common as it is in the Netherlands, where
it's often the only means of transportation. Biking is much faster then traveling by car, and more 
enjoyable then the subway. Add the small risk of being hit by a cab (mirrors and side windows are objects
the drivers chose to ignore), and you have a bit of a "thrill-seeking" experience.

<img src="/media/img/citibike.jpg" style="height: 100%;width: 100%;">

Whereas in the Netherlands we have at least three bikes 
(one racing, one old for in the city, and one broken), owning your own bike in Manhattan
is not very practical due to the lack of storage space. Luckily for us Dutchies there are the 
Citi Bikes: a bike-sharing program with over 10.000 bikes and 600+ stations (and growing). 

### Where do all the bikes go?
After using the bike sharing system for a while, you start noticing the main problem of
 bike sharing systems: _everybody wants to be at the same place at the same time._ E.g. people go to
 work in the morning - taking a bike from their living area or the place they come off their commute to 
 the location of their workspace. The same happens in the afternoon, which results in some docks always 
 being empty after 08:00 AM and some docks always being full after 08:00 am. Although Citi-Bike employees 
 are working hard to re-stock some of the empty docks, it is mission impossible to do this fast enough
 during rush hour. 
 
So where are the bikes going during the rush hours? Using the app you get a feeling of certain area's
being depleted completely during certain times, while other areas are fully stocked. However this
is just a snapshot in time. For a better overview I decided to 
plot the bike distribution using a [Kernel Denisity Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) 
of the probability function.  By creating one plot for every minute during the day, and using these
 as frames in a movie, I was able to get a good visualization of bike availability throughout the day. 
 
### One day of Citi Bikes in NYC

Next to the density plot, I've added some markers for the location of the docks as well, where the
color denotes the percentage of bikes available in that dock (0% means an empty dock, and 100% means
a completely full dock). And a plot that displays the total availability throughout the day. 

See for yourself where all the bikes go:

<iframe src="https://player.vimeo.com/video/231933066" width="800" height="700" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

### What do we see?

Studies show that Citi Bikes are mostly [used for commuting](http://www.nydailynews.com/new-york/new-yorkers-citi-bikes-cut-time-commutes-article-1.2998166),
and I believe this animation shows that as well. During the night most bikes are in the upper east and 
upper west, near Penn station. The highest density by far is on the lower east side. 

<img src="/media/img/NYCB20170804030000.png" style="height: 100%;width: 100%;">

During the morning rush hour the bikes move quickly to midtown, the financial district, and other work-areas.

<img src="/media/img/NYCB20170804113100.png" style="height: 100%;width: 100%;">

During the afternoon rush hour, the bike stations in these working areas are depleted again. 
Note that the data used in this animation was on a Friday, which means the rush hour started early. 

<img src="/media/img/NYCB20170804230000.png" style="height: 100%;width: 100%;">

### How it's made


This visualisation was made in Python, using [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/)
[Seaborn](https://seaborn.pydata.org/index.html), and [Basemap](https://matplotlib.org/basemap/).

For the full Jupter Notebook with code click [here](https://github.com/Bergvca/bergvca.github.io/tree/master/_notebooks/nyccitibikeclean.md)




