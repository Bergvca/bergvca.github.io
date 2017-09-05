---
layout: post
cvdbtitle: 1 Day of New York Citi Bikes
layout: page_noheader
---

# 1 Day of New York Citi Bikes

The code used to create the plot

#### Imports


```python
import pandas as pd
import matplotlib  
import matplotlib.pyplot as plt
import json
import seaborn as sns
import urllib.request
import numpy as np
import os
import shapefile
import matplotlib.cbook as cbook

from datetime import datetime
from time import sleep
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from sklearn.neighbors import KernelDensity
import matplotlib.dates as dates
from scipy.misc import imread

%matplotlib inline
```

### Get the data

First we need to get the data, the data used is loaded from the Citi Bike API every minute and written to a CSV file. This code is not very robust but it gets the job done.


```python
station_link = 'https://gbfs.citibikenyc.com/gbfs/en/station_status.json'
station_info_link = 'https://gbfs.citibikenyc.com/gbfs/en/station_information.json'
station_csv = 'station_status.csv'


def get_station_df(station_info_df, station_status_link):
    
    with urllib.request.urlopen(station_status_link) as url:
        data = json.loads(url.read().decode())
        station_status_df = pd.DataFrame(data['data']['stations']).set_index('station_id')
    
    station_df = pd.merge(station_status_df, 
                      station_info_df, 
                      left_index=True, 
                      right_index=True)
    station_df = station_df.assign(ts=pd.Series([datetime.now()] * len(station_df)).values)
    return station_df


with urllib.request.urlopen(station_info_link) as url:
    data = json.loads(url.read().decode())
    station_info_df = pd.DataFrame(data['data']['stations']).set_index('station_id')

while True:
    station_df = get_station_df(station_info_df, station_link)
    station_df.to_csv(station_csv, mode='a', header=False)
    sleep(60)
```

### Functions to create the plot


```python
# The KDE Plot requires a long list of x, y coordinates. This function takes a dataframe with x, y coordinates 
# and a number of available bikes, and creates a much longer dataframe with one row for each available bike

def create_expanded_df(all_status_df, timestamp):
    station_status_df = all_status_df[all_status_df.ts == timestamp]
    expanded_df = pd.DataFrame()

    for sid in station_status_df.index:
        num_bikes_available = station_status_df.loc[sid, ['num_bikes_available']][0]
        if num_bikes_available > 0:
            expanded_sub_df = pd.concat([pd.DataFrame(station_status_df.loc[sid, 
                                                                 ['lat', 'lon', 'num_bikes_available']]).T] 
                                    * num_bikes_available)
            expanded_df = pd.concat([expanded_df, expanded_sub_df])
    return expanded_df

```


```python
# This is the main function to create a plot.The function loads in a background image, plots the kde plot
# over it, and plots a scatter plot over that again. 

def create_citi_bike_graph(expanded_df, station_info_df, non_expanded_df, ts, all_status_df):

    title = 'New York Citi Bike - Bike Availability' 
    ts_rounded = str(pd.Timestamp(ts).round('min'))
    descripton = 'Kernel Density Estimation of New York Citi Bike Availability during one day \n' \
                  + 'Author: Chris van den Berg \n' \
                  + 'Data: openstreetmap.org, citibikenyc.com'
    imgfile = './img/{}.png'.format('NYCB - %s' % ts_rounded)
    fontcolor='#666666'
    datafile = './img/mapofnewyork.png'
    
    # Read the image and add it to an Axis object. 
    img = imread(datafile)
    fig = plt.figure(figsize=(28, 20))
    ax = fig.add_subplot(111)
    
    fig.suptitle(title, fontsize=50, y=.94, color=fontcolor)
    
    # Set the axis to conform with the area in which citi bike stations are available. 
    x0 = station_info_df['lon'].min()
    x1 = station_info_df['lon'].max()
    y0 = station_info_df['lat'].min()
    y1 = station_info_df['lat'].max()
    
    ax.imshow(img, extent=[x0, x1, y0, y1]) 

    # Plot the KDE plot 
    sns.kdeplot(expanded_df[['lon', 'lat']], bw=0.003, shade=True,
                gridsize = 200,
                n_levels=60, cmap='Purples', ax=ax, shade_lowest=True)

    # The standard KDE plot doesn't have transparent backgrounds. The following code adds a level
    # of transparency to each level in the plot. 
    num_collections = len(ax.collections)
    alphas = np.arange(0, 100, 100/num_collections)
    alphas = np.round(np.append(alphas, 100), decimals=1)
    i = 0
    for col in ax.collections:
        col.set_alpha(float(alphas[i]/100))
        
        i += 1
    
    # Don't plot anything outside the boundaries of the image
    ax.set_ylim(y0, y1)
    _ = ax.set_xlim(x0, x1)
    # min lon, max lon, min lat, max lat

    
    # Add the scatterplot
    non_expanded_df = (non_expanded_df.
                       assign(perc_bikes_available = pd.Series(
                           (non_expanded_df.num_bikes_available / non_expanded_df.capacity)
                                                              *100)).
                       fillna(0)
                      )

    latlon = non_expanded_df[['lat', 'lon', 'perc_bikes_available']].as_matrix().astype(float)
    sax = ax.scatter(latlon[:, 1], latlon[:, 0], c=latlon[:, 2], zorder=3, cmap="RdBu_r", vmin=0, vmax=101)
    plt.axis('off')

    cax = fig.add_axes([0.18, 0.6, 0.1, 0.25])  # left, bottom, width, height
    cax.axis('off')
    cbar = fig.colorbar(sax, ticks=[0, 50, 100], ax=cax, orientation='vertical')
    cbar.set_label('% of spots used', rotation=270, size=16, labelpad=15)
    
    ax.annotate(descripton, xy=(.50, 0.005), size=16, xycoords='axes fraction', color=fontcolor)
    
    ax.annotate(ts_rounded, xy=(.37, 1.01), size=30, xycoords='axes fraction', color=fontcolor)
    
    # Add the # avaiable bikes plot
    naax = fig.add_axes([0.28, 0.25, 0.2, 0.1])
    
    status_df_till_ts = all_status_df[all_status_df.ts <= ts]
    status_df_till_ts.groupby('ts')['num_bikes_available'].sum().plot(ax=naax)
    
    naax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    naax.patch.set_alpha(0.5)
    naax.xaxis.label.set_visible(False)
    naax.set_xlim(all_status_df.ts.min(), all_status_df.ts.max())
    naax.set_ylim(all_status_df.groupby('ts')['num_bikes_available'].sum().min(),
                 all_status_df.groupby('ts')['num_bikes_available'].sum().max())
    naax.set_title('# Of Bikes Available', size=16)
    
    
    # Save the image
    plt.savefig(imgfile, bbox_inches='tight', pad_inches=0)
#     plt.show()
 
```

### The code to create the different frames

First we need to create a map of new york. I've used the Basemap libary for this, with data from openstreet maps. There are Citi bikes in the state of New York and in the state of New Jersey,  so we need to load the OSM files for this. I've used the shapefiles that can be found here: https://download.geofabrik.de/north-america.html. Running this code is slow, thats why I save the image only once and reload the image for every frame. 


```python
# Load data created before. I ran in some formatting and memory issues, so didn't load everything

station_csv = 'station_status_2.csv'
station_info_link = 'https://gbfs.citibikenyc.com/gbfs/en/station_information.json'

with urllib.request.urlopen(station_info_link) as url:
    data = json.loads(url.read().decode())
    station_info_df = pd.DataFrame(data['data']['stations']).set_index('station_id')
    print('nr invalid lines %d' % station_info_df[(station_info_df.lat == 0) | (station_info_df.lon == 0)].
         shape[0])
    # remove invalid lines
    station_info_df = station_info_df[(station_info_df.lat != 0) & (station_info_df.lon != 0)]

all_status_df = pd.read_csv(station_csv, error_bad_lines=False,  nrows=2500000, encoding = "ISO-8859-1")
all_status_df['ts'] = pd.to_datetime(all_status_df.ts)
```


```python
imgfile = './img/mapofnewyork.png'
shpfile = os.path.expanduser('osm_roads_free_1') # State of NY
fontcolor='#666666'
jerseyshpfile = os.path.expanduser('./newjersey/osm_roads_free_1') # State of NJ

fig = plt.figure(figsize=(28, 20))
ax = fig.add_subplot(111, axisbg='w')

sf = shapefile.Reader(shpfile)
njsf = shapefile.Reader(jerseyshpfile)

x0 = station_info_df['lon'].min()
x1 = station_info_df['lon'].max()
y0 = station_info_df['lat'].min()
y1 = station_info_df['lat'].max()

latlon = expanded_df[['lat', 'lon']].as_matrix().astype(float)

cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

m = Basemap(llcrnrlon=x0, 
            llcrnrlat=y0, urcrnrlon=x1, urcrnrlat=y1, lat_0=cx, lon_0=cy, resolution='c', 
            projection='cyl', ax = ax)

# Avoid border around map.
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)


m.readshapefile(shpfile, 'metro', linewidth=.15)
m.readshapefile(jerseyshpfile, 'metro', linewidth=0.15)

plt.savefig(imgfile, bbox_inches='tight', pad_inches=0)
```

### Load the CSV file


```python
#filter data for 1 day, and set correct datatypes

all_status_df = all_status_df[(all_status_df.ts >= '2017-08-04 00:00:01') 
                             & (all_status_df.ts <= '2017-08-04 23:59:59')
                             ]
all_status_df.loc[all_status_df['station_id'].isnull(), 'station_id'] = 999

all_status_df['station_id'] = all_status_df['station_id'].astype(np.int32)
all_status_df['num_bikes_available'] = all_status_df['num_bikes_available'].astype(np.int32)

all_status_df = all_status_df.set_index('station_id')
```

### Create the plot


```python
# Create one plot for every timestamp. As there is one timestamp per minute, this creates a plot for every
# minute of the day

unique_ts = all_status_df.ts.unique()

unique_ts = unique_ts[unique_ts > np.datetime64('2017-08-04 13:06:00')]

for current_ts in unique_ts:
    expanded_df = create_expanded_df(all_status_df, current_ts)
    non_expanded_df = all_status_df[all_status_df.ts == current_ts]
    create_citi_bike_graph(expanded_df, station_info_df, non_expanded_df, current_ts, all_status_df)
    plt.close('all')
```
