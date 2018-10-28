import pandas as pd
from pathlib import Path
import json
import itertools
import random
import time
import numpy as np
import os


#######
#######
# Extract scraped articles (not in Github) into pandas dataframes.
# config_no == 1 for TRAIN (and test, 'familiar' publications), config == 2 for TEST ('novel' publications)
# BEWARE, each call yields a quasi-random subset of all articles.
# In GITHUB, don't uses this code, use resultant dataframe pickle.
# Currently: 'gdata_01_config_01_testrun.pkl' (train), 'tdata_01_config_01_testrun.pkl' (test)
# See footer for details on scraped data.

# Output PD columns: leaning, source, 'date_publish', path, 'title', 'authors', 'text'
# For new data subsets, MAKE NEW CONFIG.
def gross_pd(config_no = 1, save_pickle = 1):
    print('Using config_no ' + str(config_no) + ', save_pickle = ' + str(save_pickle))
    # Train set.
    if config_no == 1:
        article_num = 300
        print('Making training set...')
        print('\'leaning\', \'source\', \'path\', \'title\', \'text\' FOR up to 300 articles from ' + \
              '\'theblaze.com\', \'spectator.org\', \'nationalreview.com\', \'infowars.com\', \'dailywire.com\', \'westernjournal.com\', \'frontpagemag.com\', ' + 
              '\'huffpost.com\', \'salon.com\', \'thinkprogress.org\', \'newrepublic.com\', \'vox.com\', \'politicususa.com\' , \'mediamatters.org \' ...')
        source_list = ['theblaze.com_conservative_good','spectator.org_conservative_good','nationalreview.com_conservative_good','infowars.com_conservative_good','dailywire.com_conservative_good','westernjournal.com_conservative_good','frontpagemag.com_conservative_good',\
                   'huffpost.com_liberal_good','salon.com_liberal_good','thinkprogress.org_liberal_good','newrepublic.com_liberal_good_checkdups','vox.com_liberal_good','politicususa.com_liberal_good','mediamatters.org_liberal_good_notitles']
                   #'reuters.com_neutral_good']  
    # Test set. 
    elif config_no == 2:
        article_num = 50
        print('Making test set...')
        print('\'leaning\', \'source\', \'path\', \'title\', \'text\' FOR up to 50 articles from ' + \
              '\'alternet.org\', \'rawstory.com\', \'theintercept.com\',\'pjmedia.com\',\'city-journal.org\'')
        source_list = ['alternet.org_liberal_good4test', 'rawstory.com_liberal_good4test', 'theintercept.com_liberal_good4test',\
                       'pjmedia.com_conservative_good4test', 'city-journal.org_conservative_good4test'] 
        
    else:
        raise Exception('Put in config_no argument, dope!')

    gdata_pd = pd.DataFrame(columns=['leaning','source','date_publish','path','title','authors','text'])
       
    for source_dir in source_list: #itertools.islice(source_list,0,1):
        
        if source_dir == 'theblaze.com_conservative_good':
            # take only files starting with 'news_2018'
            pathlist = Path('/Users/seanwoodward/Documents/datascience_stuff/newspaper_data/20180901-20181003').glob('theblaze.com_conservative_good/news_2018*.json')
            #print([path for path in pathlist])
        elif source_dir == 'nationalreview.com_conservative':
            # exclude files starting with 'photos'
            pathlist = [fn for fn in Path('/Users/seanwoodward/Documents/datascience_stuff/newspaper_data/20180901-20181003').glob('nationalreview.com_conservative_good/*.json') if not os.path.basename(fn).startswith('photos')]
        elif source_dir == 'westernjournal.com_conservative_good':
            # exclude files starting with 'wc_'
            pathlist = [fn for fn in Path('/Users/seanwoodward/Documents/datascience_stuff/newspaper_data/20180901-20181003').glob('westernjournal.com_conservative_good/*.json') if not os.path.basename(fn).startswith('wc_')]
        
        else:        
            pathlist = Path('/Users/seanwoodward/Documents/datascience_stuff/newspaper_data/20180901-20181003').glob(source_dir + '/*.json')
        
        # Randomize pathlist, as we will want no more than 300 articles from each source
        pathlist=list(pathlist)
        random.shuffle(pathlist)
        t = time.time()
        article_cnt = 0
        article_index = 0
        while (article_cnt < article_num) and (article_index < len(pathlist)):
            path = pathlist[article_index]
            with open(path) as data_file:
                article = json.load(data_file)
            leaning = source_dir.split(sep='_')[1]
            source = source_dir.split(sep='_')[0]
            # path = path
            for item,value in article.items():
                if item == 'authors':                  
                    # !!! need to find comprehensive proper noun list for filtering
                    # another option: wordnet.synsets('michael')
                    #authors = list 
                    #for name in value:
                    #    print(name)
                    #    print(any([pos_tag([word])[0][1] =='NNP' for word in name.split(' ')]))
                    #    if any([pos_tag([word])[0][1] =='NNP' for word in name.split(' ')]):
                    #        authors += [name]
                    #print(authors)         
                    authors = value                               
                elif item == 'date_publish':
                    date_publish = value
                elif item == 'title':
                    title = value
                elif item == 'text':  
                    if (source_dir == 'reuters.com_neutral_good') and ('(Reuters)' in value):
                        text = value.split('(Reuters) - ')[1].replace('\n',' ')          
                    elif (source_dir == 'nationalreview.com_conservative_good'):  
                        text = value.replace('IN THE NEWS: \‘[WATCH]','').replace('\n', ' ')
                    elif (source_dir == 'westernjournal.com_conservative_good'):
                        text = value.replace('We are committed to truth and accuracy in all of our journalism. Read our editorial standards.','').replace('\n', ' ')
                        #text = text.replace('Completing this poll entitles you to The Wildcard updates free of charge. You may opt out at anytime. You also agree to our Privacy Policy and Terms of Use You\'re logged in to Facebook. Click here to log out.','')
                    elif (source_dir == 'salon.com_liberal_good'):        
                        text = value.replace('Today\'s hottest topics\nCheck out the latest stories and most recent guests on SalonTV.','').replace('\n',' ')
                    elif (source_dir == 'politicususa.com_liberal_good'):
                        text = value.replace('SHARES Facebook Twitter Google Whatsapp Pinterest Print Mail Flipboard','').replace('\n',' ')
                    elif (source_dir == 'mediamatters.org_liberal_good_notitles'):
                        if value is None:
                            text = None
                        else:
                            text = value.replace('Media Matters',' ').replace('\n',' ') 
                    else:
                        if value is None:
                            text = None
                        else:
                            text = value.replace('\n',' ')
            if ((title in np.array(gdata_pd['title'])) and (date_publish in np.array(gdata_pd['date_publish']))) or (text is None) or (title is None):
                # article redundant, not counted
                pass
            else:
                gdata_pd = gdata_pd.append({'leaning':leaning,'source':source,'date_publish':date_publish,'path':path,'title':title,'authors':authors,'text':text}, ignore_index=True)
                article_cnt += 1
            article_index += 1
        
        print(str(article_cnt) + ' articles from ' + str(path) + ' added in time ' + str(time.time() - t))
        t = time.time()                
                        
    if save_pickle == 1:
        if config_no == 1:
            gdata_pd.to_pickle('gdata_01_config_01_testrun.pkl')
        if config_no == 2:
            gdata_pd.to_pickle('tdata_01_config_01_testrun.pkl')
        
    return gdata_pd
          
# Make train/familiar-publication test data set.           
gdata_pd=gross_pd(config_no = 1, save_pickle = 1)  
# Make novel publication test set.
tdata_pd=gross_pd(config_no = 2, save_pickle = 1)


#####
#####
# Extraction comment block.


# Conservative

# theblaze.com_conservative_good, 871 articles
# authors (good?), date_publish (good), title (good)
# choose only json's starting with "news_2018"
# "visit his channel on TheBlaze and listen live to “The Morning Blaze with Doc Thompson” weekdays 6 a.m. – 9 a.m. ET, only on TheBlaze Radio Network" at article end
# actually, "news_2018" filter should account for most of problems
# ** scan complete
# ** extraction adjusted (excluding quote)

# spectator.org_conservative_good, 294 articles
# authors (none), date_publish (good), title (good)
# very good, sometimes author information in footer articles, not big deal...
# see if you can get more articles?
# ** scan complete
# ** extraction adjusted (could still use more articles...)

# nationalreview.com_conservative_good, 312
# authors (meh, false positives), date_publish (good), title(good)
# possibly axe "Advertisement" from NationalReview.com and thinkprogress.org?
# somtimes conspicuous photo header at TEXT beginning, with ref to Reuters or other news sources
# Senator Elizabeth Warren (D, Mass.) delivers a major policy speech on “Ending corruption in Washington” at the National Press Club, Washington, D.C., August 21, 2018. (Yuri Gripas/Reuters)... usually parenthetical source constant
# also, in the footer, sometimes stuff of the form "/nIN THE NEWS: ‘[WATCH]..."
# ** scan complete
# ** extraction adjusted (left Reuters references, cross-referencing could be important, left Advertisement... )

# infowars.com_conservative_good, 307
# authors (good), date_publish (good), title(good)
# at end of some: ”\nTwitter: Follow @WhiteIsTheFury" ... very frequent ... filter?
# ** scan complete
# ** extraction_adjusted (left in Twitter thing...)

# dailywire.com_conservative_good, 500-1000+ (duplicates)
# authors (none), date_publish (good), title(good)
# Sometimes at end: \nWATCH: ...not that frequent
# ** scan complete
# ** extraction_adjusted (left in WATCH...)

# westernjournal.com_conservative_good, ~1000
# authors (meh, false positives), date_publish (good), title(good)
# cut from article ends: \nWe are committed to truth and accuracy in all of our journalism. Read our editorial standards.
# lots of bold heading titles like, "TRENDING\n", "RELATED\n", "BREAKING\n"
# ignore "wc_" titles...sports
# stuff like this (problem?): es No\nCompleting this poll entitles you to The Wildcard updates free of charge. You may opt out at anytime. You also agree to our Privacy Policy and Terms of Use You're logged in to Facebook. Click here to log out.\
# ** scan complete
# ** extraction complete (left in poll, bold headings...)

# frontpagemag.com_conservative_good, ~257 articles
# authors(none), date_publish(good), title(good)
# ** scan complete
# ** extraction complete

#####
#####
# Liberal

# huffpost.com_liberal_good (also huffingtonpost), 115 articles
# authors (can be weird), date_publish (good), title
# 
# salon.com_liberal_good, 985 articles
# authors (none), date_publish (good), title (good)
# sometimes: \nToday's hottest topics\nCheck out the latest stories and most recent guests on SalonTV."
# some irrelevant articles, no indicators...
# ** scan complete
# ** extraction complete (nothing to do about irrelevant articles?)

# thinkprogress.org_liberal_good, 418 articles
# authors (none), date_publish (good), title (good)
# some "Advertisement" words...
# ** scan complete 
# ** extraction complete (left Advertisement, consistent with NationalReview)

# newrepublic.com_liberal_good_checkdups, ~250 articles
# authors (good), date_publish (good), title(good)
# ** scan complete
# ** extraction complete

# vox.com_liberal_good, 888 articles
# authors (BAD), date_publish (good), title (good)
# ** scan complete
# ** extraction complete

# politicususa.com_liberal_good, 481 articles
# authors (none), date_publish (good), title(good)
# SHARES Facebook Twitter Google Whatsapp Pinterest Print Mail Flipboard
# ** scan complete
# **extraction complete

# mediamatters.org_liberal_good_notitles, ~500+ articles
# text beginning: Melissa Joskow / Media Matters\n
# ...Wasko / Media Matters\nA
# ** scan complete
# **extraction complete

#####
#####
# Neutral

# reuters.com_neutral_good, 936 articles
# authors (none), date_publish (good), title (good)
 
# bbc.com_neutral_maybe
 
#####
#####
#
# For testing...

# Liberal
# alternet.org_liberal_good4test, ~15 articles
# rawstory.com_liberal_good4test, ~15 articles
# theintercept.com_liberal_good4test, 30+ articles
# currentaffairs (excluded for now)

# Conservative
# pjmedia.com_conservative_good4test, ~23 articles
# city-journal.org_conservative_good4test, ~90 articles

# Others -- excluded for bad quality
# nydailynews.com_liberal_maybe ... lots of irrelevant news
# redstate.com ... insufficient 

# need to remove social media header in TEXT
# Also somtime author information in bottom...

# Perhaps use dictionary?

# /Users/seanwoodward/Documents/datascience_stuff/newspaper_data/20180901-20181003/
