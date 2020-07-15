
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

data = 'DATA/'
pickles = 'pickles/'
month_dict = {'april': '04', 
              'augustus': '08', 
              'december': '12', 
              'februari': '02', 
              'januari': '01', 
              'juli': '07', 
              'juni': '06', 
              'maart': '03', 
              'mei': '05', 
              'november': '11', 
              'oktober': '10', 
              'september': '09'}
quarter_dict = {'januari': 'Q1'
                ,'februari': 'Q1'
                ,'maart': 'Q1'
                ,'april': 'Q2'
                ,'mei': 'Q2'
                , 'juni': 'Q2'
                ,'juli': 'Q3'
                ,'augustus': 'Q3'
                ,'september': 'Q3'
                ,'oktober': 'Q4'
                ,'november': 'Q4'
                ,'december': 'Q4'}


# In[2]:


def is_pos(x, threshold = 0.05):
    '''
    x: polarity of the tweet
    returns 1 when above threshold and 0 otherwise'''
    if x > threshold:
        return 1
    else:
        return 0

def is_neg(x, threshold = 0.05):
    '''
    x: polarity of the tweet
    returns 1 when below threshold and 0 otherwise'''
    if x < -threshold:
        return 1
    else:
        return 0


# In[3]:


# loading the data
sent = pd.read_csv(data + 'Final sentiment.csv')
emot = pd.read_csv(data + 'Final emotion.csv')
wf = pd.read_csv(data+'calculated workforce.csv')
st = pd.read_csv(data+'starters percentage and calculated.csv')


# ## preparing emotions

# In[4]:


def calc_emotion(emotions, approach, dc):
    '''
    emotions: a list of emotions in the order fear anger joy and surprise
    approach: a string either valence or cognitive
    dc: a string either dominant or conflict
    returns the value based on the type of derived emotions must be calculated'''
    emotions_l = [emotions['Fear'], emotions['Anger'], emotions['Joy'], emotions['Surprise']]
    if approach == 'valence':
        pairs = [(2,3),(0,1)] # Joy & Surprise and Fear & Anger 
    elif approach == 'cognitive':
        pairs = [(0,3), (1,2)] # Fear & Surprise and Anger & Joy
    else:
        raise NameError('approach unknown')
    e1 = emotions_l[pairs[0][0]] + emotions_l[pairs[0][1]]
    e2 = emotions_l[pairs[1][0]] + emotions_l[pairs[1][1]]
    if dc == 'D': # dominant
        if e1 >= e2:
            return e1
        else:
            return e2
    elif dc == 'C': # conflict
        if e1 <= e2:
            return e1
        else:
            return e2

def calc_mixed(D, C, p = 0.5):
    '''
    D: float representing the dominant emotion
    C: float representing the conflicting emotion
    p: the threshold value p by default 0.5
    returns the calculated mixed emotion based on the GTM formula'''
    mixed = 5.0*(C+1.0)**p-(D+1.0)**(1.0/(C+1.0))
    return mixed


# In[5]:


emot.loc[:,'date'] = pd.to_datetime(emot.date, dayfirst = True, infer_datetime_format = True)
emot.loc[:,'quarter'] = emot.date.dt.to_period('Q')
emot.loc[:,'month'] = emot.date.dt.to_period('M')


# In[6]:


# calculates the derived emotions
monthly_emot = emot.loc[:, ['month','username','Fear', 'Anger', 'Joy','Surprise']].groupby(['month', 'username']).mean()
monthly_emot = monthly_emot.loc[(monthly_emot.loc[:, 'Fear'] > 0 ) |
                               (monthly_emot.loc[:, 'Anger'] > 0 ) |
                               (monthly_emot.loc[:, 'Joy'] > 0 ) |
                               (monthly_emot.loc[:, 'Surprise'] > 0 ), :]

monthly_emot.loc[:, '(valence, Dominant)'] = monthly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'valence', 'D'), axis = 'columns')
monthly_emot.loc[:, '(valence, Conflict)'] = monthly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'valence', 'C'), axis = 'columns')
monthly_emot.loc[:, '(valence, Mixed)'] = monthly_emot.loc[:,['(valence, Dominant)', '(valence, Conflict)']].apply(lambda x: calc_mixed(float(x[0]), float(x[1])), axis = 'columns')

monthly_emot.loc[:, '(cognitive, Dominant)'] = monthly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'cognitive', 'D'), axis = 'columns')
monthly_emot.loc[:, '(cognitive, Conflict)'] = monthly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'cognitive', 'C'), axis = 'columns')
monthly_emot.loc[:, '(cognitive, Mixed)'] = monthly_emot.loc[:,['(cognitive, Dominant)', '(cognitive, Conflict)']].apply(lambda x: calc_mixed(float(x[0]), float(x[1])), axis = 'columns')


# In[7]:


#aggregated monthly 
monthly_emot = monthly_emot.reset_index().groupby('month').mean()
monthly_emot = monthly_emot.reset_index()
monthly_emot.loc[:,'month'] = monthly_emot.loc[:, 'month'].astype(str)


# In[8]:


# calculates the derived emotions
quarterly_emot = emot.loc[:, ['quarter','username','Fear', 'Anger', 'Joy','Surprise']].groupby(['quarter', 'username']).mean()
quarterly_emot = quarterly_emot.loc[(quarterly_emot.loc[:, 'Fear'] > 0 ) |
                               (quarterly_emot.loc[:, 'Anger'] > 0 ) |
                               (quarterly_emot.loc[:, 'Joy'] > 0 ) |
                               (quarterly_emot.loc[:, 'Surprise'] > 0 ), :]

quarterly_emot.loc[:, '(valence, Dominant)'] = quarterly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'valence', 'D'), axis = 'columns')
quarterly_emot.loc[:, '(valence, Conflict)'] = quarterly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'valence', 'C'), axis = 'columns')
quarterly_emot.loc[:, '(valence, Mixed)'] = quarterly_emot.loc[:,['(valence, Dominant)', '(valence, Conflict)']].apply(lambda x: calc_mixed(float(x[0]), float(x[1])), axis = 'columns')

quarterly_emot.loc[:, '(cognitive, Dominant)'] = quarterly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'cognitive', 'D'), axis = 'columns')
quarterly_emot.loc[:, '(cognitive, Conflict)'] = quarterly_emot.loc[:,['Fear', 'Anger','Joy', 'Surprise']].apply(lambda x: calc_emotion(x, 'cognitive', 'C'), axis = 'columns')
quarterly_emot.loc[:, '(cognitive, Mixed)'] = quarterly_emot.loc[:,['(cognitive, Dominant)', '(cognitive, Conflict)']].apply(lambda x: calc_mixed(float(x[0]), float(x[1])), axis = 'columns')


# In[9]:


#aggregated quarterly 
quarterly_emot = quarterly_emot.reset_index().groupby('quarter').mean()
quarterly_emot = quarterly_emot.reset_index()
quarterly_emot.loc[:,'quarter'] = quarterly_emot.loc[:, 'quarter'].astype(str)


# # Workforce
# - convert everything to montlhy
# - merge with the workforce dataset

# In[10]:


# getting both sentiments based on random word list and the institutions based word list

temp = sent.loc[sent.loc[:,'institutions'] == 1, ['month','username', 'sentiment']].groupby(by = ['month', 'username']).mean()
threshold = 0.05
temp.loc[:,'pos'] = temp.loc[:, 'sentiment'].apply(lambda x: is_pos(x, threshold = threshold))
temp.loc[:,'neg'] = temp.loc[:, 'sentiment'].apply(lambda x: is_neg(x, threshold = threshold))
temp = temp.loc[:,['pos', 'neg']].groupby(by = ['month']).mean()

temp1 = sent.loc[sent.loc[:,'random'] == 1, ['month','username', 'sentiment']].groupby(by = ['month', 'username']).mean()
threshold = 0.05
temp1.loc[:,'pos'] = temp1.loc[:, 'sentiment'].apply(lambda x: is_pos(x, threshold = threshold))
temp1.loc[:,'neg'] = temp1.loc[:, 'sentiment'].apply(lambda x: is_neg(x, threshold = threshold))
temp1 = temp1.loc[:,['pos', 'neg']].groupby(by = ['month']).mean()

monthly_sent = pd.merge(temp, temp1, how = 'left',on = 'month', suffixes = [' institutions', ' random'])


# In[11]:


# the monthly sentiment and monthly emotions are merged with the workforce dataset
temp = pd.merge(wf.loc[:,['Perioden', 'WA Workforce']], monthly_sent, how = 'left', left_on = 'Perioden', right_on = 'month')
temp = pd.merge(temp, monthly_emot, how = 'left', left_on = 'Perioden', right_on = 'month')
wf_final = temp.drop(labels = ['month'], axis = 'columns')


# In[12]:


wf_final.loc[:, 'Workforce 3M'] = wf_final.loc[:,'WA Workforce'].shift(-3)
wf_final.to_csv(data+'final workforce.csv')


# # Starters

# In[13]:


temp = sent.loc[sent.loc[:,'institutions'] == 1, ['quarter','username', 'sentiment']].groupby(by = ['quarter', 'username']).mean()
threshold = 0.05
temp.loc[:,'pos'] = temp.loc[:, 'sentiment'].apply(lambda x: is_pos(x, threshold = threshold))
temp.loc[:,'neg'] = temp.loc[:, 'sentiment'].apply(lambda x: is_neg(x, threshold = threshold))
temp = temp.loc[:,['pos', 'neg']].groupby(by = ['quarter']).mean()

temp1 = sent.loc[sent.loc[:,'random'] == 1, ['quarter','username', 'sentiment']].groupby(by = ['quarter', 'username']).mean()
threshold = 0.05
temp1.loc[:,'pos'] = temp1.loc[:, 'sentiment'].apply(lambda x: is_pos(x, threshold = threshold))
temp1.loc[:,'neg'] = temp1.loc[:, 'sentiment'].apply(lambda x: is_neg(x, threshold = threshold))
temp1 = temp1.loc[:,['pos', 'neg']].groupby(by = ['quarter']).mean()

quarterly_sent = pd.merge(temp, temp1, how = 'left',on = 'quarter', suffixes = [' institutions', ' random'])


# In[14]:


# there were some rows with the total of each year which is not needed, so these are removed
remove = ['2013','2014', '2015', '2016', '2017','2018','2019']
st = st.loc[~st.loc[:,'perioden'].isin(remove),['Til starter dig','Til starter all','perioden']]
st.loc[:,'quarter'] = st.loc[:,'perioden'].apply(lambda x: x.split()[0]+'Q'+x.split()[1][0])


# In[15]:


# the quaterly sentiment and quaterly emotions are merged with the starters dataset
temp = pd.merge(st.loc[:,['quarter','Til starter dig', 'Til starter all']], quarterly_sent, how = 'left',left_on = 'quarter', right_index = True)
temp = pd.merge(temp, quarterly_emot, how = 'left', on = 'quarter')
st_final = temp


# In[16]:


# st_final = st_final.loc[4:,:]
st_final.loc[:, 'Til starter dig 1Q'] = st_final.loc[:,'Til starter dig'].shift(-1)
st_final.loc[:, 'Til starter all 1Q'] = st_final.loc[:,'Til starter all'].shift(-1)
st_final.to_csv(data+'final starters.csv')

