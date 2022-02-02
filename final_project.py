#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:34:11 2022

@author: martinamanno
"""

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib as mpl
from operator import index
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
import subprocess
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score)
from wordcloud import WordCloud, STOPWORDS
import nltk
import csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords



#Import the survey file
data_survey = pd.read_excel('data survey.xlsx')
data_survey.head()

df = data_survey[['Have you got insomnia problems?','How many days you think you are able to stay without the smartphone?',
'How do you think COVID19 has affected your social life?','Have you ever realized to use your smartphone for too many hours?',
'If you ever realized to use your smartphone for too many hours, how many of them?','Have you ever looked at your phone to find out something (eg. the time) but got distracted and looked at something else forgetting the reason why you had looked at the phone?',
'Have you ever thought about deleting your social account?','If you ever though about deleting your social account, have you ever actually done?',
'How would you describe yourself? (please, give three words separate by a comma)','What is your zodiac sign?',
'What would you say is your best quality?','What do you do when you’re going through a personal crisis?',
'Do you consider yourself a stressed person? Answer from 1 to 7 (1 minimum level, 7 maximum level)','Are you anxious?']]

df.rename(columns={'Have you got insomnia problems?':'Insomnia','How many days you think you are able to stay without the smartphone?':'out_smart',
'How do you think COVID19 has affected your social life?':'Covid','Have you ever realized to use your smartphone for too many hours?':'hours_smart',
'If you ever realized to use your smartphone for too many hours, how many of them?':'hours_many',
'Have you ever looked at your phone to find out something (eg. the time) but got distracted and looked at something else forgetting the reason why you had looked at the phone?':'dis_smart',
'Have you ever thought about deleting your social account?':'no_social','If you ever though about deleting your social account, have you ever actually done?':'del_social',
'How would you describe yourself? (please, give three words separate by a comma)':'yourself',
'What is your zodiac sign?':'Zodiac','What would you say is your best quality?':'Quality',
'What do you do when you’re going through a personal crisis?':'Crisis',
'Do you consider yourself a stressed person? Answer from 1 to 7 (1 minimum level, 7 maximum level)':'stressed',
'Are you anxious?':'Y'},inplace=True)

#PRE PROCESSING 

df['hours_many']
for i in df['hours_many']:
    if i == 'I haven\'t realized to use my smartphone for too many hours':
        df['hours_many']=df['hours_many'].replace(i,0)
df['hours_many'].astype('int')

df['del_social']
for i in df['del_social']:
    if i == "I haven't never thought about deleting my social account":
        df['del_social']=df['del_social'].replace(i,"No")
df['del_social']

df['out_smart']
for i in df['out_smart']:
    if i == '1-3 days':
        df['out_smart'].replace(i,'Low',inplace=True)
    elif i == '2-3 weeks':
        df['out_smart'].replace(i,'Medium',inplace=True)
    elif i == '1 month or more':
        df['out_smart'].replace(i,'High',inplace=True)
df['out_smart']

df['Zodiac'] = df['Zodiac'].str.lower()
for i in df['Zodiac']:
    if i == 'scorpion':
        df['Zodiac']=df['Zodiac'].replace(i,'scorpio')
        
    elif i == 'fishes':
        df['Zodiac']=df['Zodiac'].replace(i,'pisces')
        
    elif i =='aries ':
        df['Zodiac']=df['Zodiac'].replace(i,'aries')
        
    elif i =='sagittarius ':
        df['Zodiac']=df['Zodiac'].replace(i,'sagittarius')
     
    elif i =='ahahahahah ':
        df['Zodiac']=df['Zodiac'].replace(i,np.NaN)
        
nan_rows = df[df['Zodiac'].isnull()]
x = df['Zodiac'].value_counts().idxmax()
df['Zodiac']=df['Zodiac'].replace(np.NaN,x)

Y_dummy = pd.get_dummies(df['Y'])
dummy_vars = ['Insomnia','hours_smart', 'dis_smart','no_social', 'del_social']
other_dummies = pd.get_dummies(df[dummy_vars])
dummy_df = pd.concat([Y_dummy, other_dummies], axis=1)

df = pd.concat([df,dummy_df],axis=1)
df.drop(dummy_vars,axis=1,inplace=True)

df.drop(['Insomnia_No','hours_smart_No','dis_smart_No','no_social_No','del_social_No'],axis=1,inplace=True)
df.drop(['Y','No'],axis=1,inplace=True)
df=df.rename(columns={'Yes':'Y', 'Insomnia_Yes': 'insomnia', 'hours_smart_Yes':'hours_smart',
'dis_smart_Yes': 'dis_smart', 'no_social_Yes':'no_social','del_social_Yes':'del_social'})

df.to_excel('Data_preprocessed.xlsx')

#DATA VISUALIZATION 

df = pd.read_excel('Data_preprocessed.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)


#HISTOGRAMS:
#insomnia
df_hist=df['insomnia']
for a in df_hist:
    if a>0.5:
        df_hist.replace(a,'Yes',inplace=True)
    else:
        df_hist.replace(a,'No',inplace=True)

fig = px.histogram(df_hist,x="insomnia",title="Insomnia",
nbins=4,text_auto=True)
fig.show()

#out_smart
fig = px.histogram(df,x="out_smart",title="OutSmart",
nbins=4,text_auto=True,category_orders=dict(out_smart=["Low", "Medium", "High"]))
fig.show()

#no_social
plt.title('Have you ever thought about deleting your social account?')
plt.hist(df['no_social'], rwidth=0.8)
plt.xlabel('0 = No, 1 = Yes')
plt.ylabel('number of people')
plt.yticks(np.arange(0,22,step=3))
plt.xticks(np.arange(0,1,step=1))
plt.show()

#del_social
plt.title('If you ever though about deleting your social account, have you ever actually done?')
plt.hist(df['del_social'], rwidth=0.8)
plt.xlabel('0 = No, 1 = Yes')
plt.ylabel('number of people')
plt.yticks(np.arange(0,22,step=3))
plt.xticks(np.arange(0,1,step=1))
plt.show()

#hours_many
plt.title('Hours on smartphone')
plt.hist(df['hours_many'], rwidth=0.8)
plt.xlabel('hours of smartphone usage per days')
plt.ylabel('number of people')
plt.xticks(range(0,11))
plt.show()

#Hours_smart
plt.title('Realizing too many hours on smartphone')
plt.hist(df['hours_smart'], rwidth=0.8)
plt.xlabel('0 = No, 1= Yes')
plt.ylabel('number of people')
plt.xticks(range(0,2))
plt.show()

#Dis_smart
plt.title('Got distracted')
plt.hist(df['dis_smart'], rwidth=0.8)
plt.xlabel('0 = No, 1 = Yes')
plt.ylabel('number of people')
plt.xticks(range(0,2))
plt.show()


#this plot shows the number of students (on y axis) that are anxious or not divided by zodiac sign 
sns.catplot(x='Zodiac',hue='Y',data=df,kind="count", palette=sns.color_palette("pastel"))
plt.xticks(rotation = 45)
plt.yticks(np.arange(0,5,step=1))
plt.show()

#this plot shows the number of students (on y axis) that are anxious or not basing on the fact that they thought about deleting their social account
sns.catplot(x='no_social',hue='Y',data=df,kind="count", palette=sns.color_palette("pastel"))
plt.yticks(np.arange(0,13,step=1))
plt.show()

#this plot shows the number of students (on y axis) that are anxious or not basing on the fact that deletes their social account
sns.catplot(x='del_social',hue='Y',data=df,kind="count", palette=sns.color_palette("pastel"))
plt.yticks(np.arange(0,16,step=1))
plt.show()

#this plot shows the number of students (on y axis) that thought (or not) about deleting their social accountand then delete it (or not)
sns.catplot(x='no_social',hue='del_social',data=df,kind="count", palette=sns.color_palette("pastel"))
plt.yticks(np.arange(0,15,step=1))
plt.show()

#viceversa
sns.catplot(x='del_social',hue='no_social',data=df,kind="count", palette=sns.color_palette("pastel"))
plt.yticks(np.arange(0,15,step=1))
plt.show()

#Bar chart
bar_data = df.groupby(['out_smart'])['Y'].sum().reset_index()
bar_data
bar_data.sort_values(by=['Y'],inplace=True)
bar_data
fig = px.bar(bar_data,x="out_smart",y="Y", title="Out VS Y")
fig.show()

#Bubble chart
bub_data = df.groupby('stressed')['Y'].sum().reset_index()
bub_data
fig = px.scatter(bub_data,x="stressed",y="Y",size="Y",
hover_name="Y",title="stressed over Y",size_max=60)
fig.show()

bub_data1 = df.groupby('hours_many')['Y'].sum()
bub_data1 = bub_data1.reset_index()
fig = px.scatter(bub_data1,x="hours_many",y="Y",size="Y",
hover_name="Y",title="stressed over Y",size_max=60)
fig.show()


#Pie chart
fig = px.pie(df,names='out_smart',title='Out_smart')
fig.show()

df_out = df.groupby('out_smart')['Y'].sum()
df_out = df_out.reindex(index=['Low','Medium','High'])
df_out = pd.DataFrame(df_out)
df_out = df_out.reset_index()
df_out = df_out.reset_index()
df_out['index']=df_out['index']+1
colors = ['green', 'yellow', 'orange']
explode=[0,0.1,0.1]
df_out['Y'].plot(kind='pie',
                        figsize=(10,8),
                        autopct='%1.1f%%',
                        startangle=90,
                        shadow=True,
                        labels= None,
                        pctdistance = 1.12, 
                        colors = colors,
                        explode = explode)
plt.title('out_smart', y=1.12)
plt.axis('equal')
plt.legend(labels=df_out['index'], loc='lower right')
plt.show()

fig = px.pie(df,names='stressed',title='Stressed')
fig.show()

df_stressed = df.groupby('stressed')['Y'].sum()
df_stressed = df_stressed.reset_index()
colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink', 'blue']
explode_list = [0.1, 0.1, 0, 0.1, 0, 0.1, 0] 

df_stressed['stressed'].plot(kind='pie',
                            figsize=(10,8),
                            autopct='%1.1f%%',
                            startangle=90,
                            shadow=True,
                            labels= None,
                            pctdistance = 1.12, 
                            colors = colors_list,
                            explode = explode_list)

plt.title('stressed', y=1.12)
plt.axis('equal')
plt.legend(labels=(df['stressed'].index)+1, loc='lower right')
plt.show()

#Sunburst chart
fig = px.sunburst(df,path=['out_smart','insomnia'],values='Y')
fig.show() # si parte da Y=1 poi passi a medium e vedi se insomnia si o no

fig = px.sunburst(df,path=['out_smart','stressed'],values='Y')
fig.show()  # solo 1 'Y' nel High-> 3^ 'stressed'


#Waffle chart
df_group = df.groupby('out_smart')['Y'].sum()
df_group
df.index.values

def create_waffle_chart(categories, values, height, width, colormap, value_sign=''):
    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

    # print out number of tiles per category
    for i, tiles in enumerate(tiles_per_category):
        print (df_group.index.values[i] + ': ' + str(tiles))
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1
            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]
    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )
    plt.show()

width = 5 # width of chart
height = 3 # height of chart
categories = df_group.index.values # categories
values = df_group # correponding values of categories
colormap = plt.cm.coolwarm # color map class
create_waffle_chart(categories, values, height, width, colormap)
# di 15 persone che sono ansiose (1 high 12 Low e 2 Medium)

#Italy map
df_map = pd.read_excel('data survey.xlsx')  
df_map.head()

df_map = df_map[['Where did you graduate?','How old are you?']]
df_map = df_map.groupby('Where did you graduate?')['How old are you?'].count()
df_map = pd.DataFrame(df_map)
df_map
df_map = df_map.reset_index()
df_map = df_map.reset_index()
df_map

latitudine =[41.92464912968205,41.86211212467428,41.90395422195313,44.533998472058194,43.772003314695354,40.826498738852415,40.84676825062206,45.47943526491773]
longitudine=[12.493906611035115,12.479509553362414,12.514459855210495,11.353725483219163,11.250006090486822,14.174784792237169,14.256897699349086,12.255437640002375]
df2 = pd.DataFrame({"Latitudine":latitudine,"Longitudine":longitudine})
df2
df2=df2.reset_index()
df2

df_unito = pd.merge(left=df_map,right=df2)
df_unito

df_unito.drop(['How old are you?',"index"],axis=1,inplace=True)
df_unito.rename(columns={"Where did you graduate?":"Where"},inplace=True)
df_unito

italia_map = folium.Map(location=[43.764611662392895, 12.7740200464022], zoom_start=8)

fea = folium.map.FeatureGroup()
for lat,lng in zip(df_unito.Latitudine,df_unito.Longitudine):
    fea.add_child(
        folium.features.CircleMarker(
            [lat,lng],
            radius=5,
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )

latitudes = list(df_unito.Latitudine)
longitudes = list(df_unito.Longitudine)

labels = list(df_unito.Where)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(italia_map)

italia_map.add_child(fea).save('Italia.html')

#Covid text analysis
df = pd.read_excel('Data_preprocessed.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
text= df[['Covid','yourself','Zodiac','Quality','Crisis','Y']]
text.shape
sr = text['Covid']
text=sr.to_string()
text = ''.join([i for i in text if not i.isdigit()])
#Sentence Tokenization.Sentence tokenizer breaks text paragraph into sentences.
tokenized_text=sent_tokenize(text)
print(tokenized_text)
#Word Tokenization. Word tokenizer breaks text paragraph into words.
tokenized_word=word_tokenize(text)
print(tokenized_word)
useful_word = [ word for word in tokenized_word if len(word) >= 5 ]
#Frequency Distribution
fdist = FreqDist(useful_word)
print(fdist)
fdist.most_common(3)
# Frequency Distribution Plot
fdist.plot(30,cumulative=False)
plt.show()

#Zodiac wordcloud
df = pd.read_excel('Data_preprocessed.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
text= df[['Covid','yourself','Zodiac','Quality','Crisis','Y']]
zodiac= df.groupby("Zodiac")
zodiac.describe().head()
text = df.Zodiac[0:32]
text=text.to_string()
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=25, background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Yourself wordcloud
df = pd.read_excel('Data_preprocessed.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
text= df[['Covid','yourself','Zodiac','Quality','Crisis','Y']]
text = df.yourself[0:32]
text=text.to_string()
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud = WordCloud(max_font_size=30, max_words=25, background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Correlation matrix
df = pd.read_excel('Data_preprocessed.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
corr = df.corr()
plt.title('Correlation matrix')
cormap = sns.heatmap(corr, annot=True)
cormap.figure.savefig('corr_matrix.jpg', transparent=False, bbox_inches='tight', dpi = 300)
plt.show()



# MODEL DEVELOPMENT 

df = pd.read_excel('Data_preprocessed.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop(['Covid','yourself','Zodiac','Quality','Crisis'], axis = 1, inplace = True)
df.head()

df['out_smart']
for i in df['out_smart']:
    if i == 'Low':
        df['out_smart'].replace(i,1,inplace=True)
    elif i == 'Medium':
        df['out_smart'].replace(i,2,inplace=True)
    elif i == 'High':
        df['out_smart'].replace(i,3,inplace=True)
df['out_smart']

df.drop(['dis_smart','hours_smart'], axis=1, inplace=True)

#split in test and train
x = df.loc[:,df.columns != 'Y']
x = sm.add_constant(x)
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25,random_state=5488)

#Logit model
logit = sm.Logit(y_train,X_train).fit()
print(logit.summary())

#prediction on train
yhat_train = logit.predict(X_train)
prediction_train = list(map(round, yhat_train))
print('Actual values train:', list(y_train.values))
print('Predictions train:', prediction_train)

#prediction on test
yhat_test = logit.predict(X_test)
prediction_test = list(map(round, yhat_test))
print('Actual values test:', list(y_test.values))
print('Predictions test:', prediction_test)

#confusion matrix train
cm = confusion_matrix(y_train, prediction_train)
print ("Confusion Matrix train: \n", cm)
plt.title('Confusion Matrix (train)')
sns.heatmap(cm, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.show()

# accuracy score
print('Train accuracy = ', accuracy_score(y_train, prediction_train))

#confusion matrix test
cm = confusion_matrix(y_test, prediction_test)
print ("Confusion Matrix test: \n", cm)
plt.title('Confusion Matrix (test)')
sns.heatmap(cm, square=True, annot=True, cmap='Oranges', fmt='d', cbar=False)
plt.show()

# accuracy score
print('Test accuracy = ', accuracy_score(y_test, prediction_test))

#plot Y and Stressed
x_plot = df['stressed']
logit_plot = sm.Logit(y,x_plot).fit()
pred_input = np.linspace(1,7,32) #stressed ha min 1 e max 7, e  32 osservazioni
predictions = logit_plot.predict(pred_input)
plt.scatter(x_plot,y)
plt.plot(pred_input,predictions,c='red')
plt.show()


#ARE YOU ANXIOUS?

#Logit model
model = sm.Logit(y,x).fit()
par = model.params 


a = False

while a == False:

    out_smart = int(input('How many days you think you are able to stay without the smartphone? [Answer should be between 1 and 3] = '))
    
    if out_smart == int(1):
        print('\nAnswer saved')
        a = True
    elif out_smart == int(2):
        print('\nAnswer saved')
        a = True
    elif out_smart == int(3):
        print('\nAnswer saved')
        a = True
    else:
        print('\nIncorrect answer, let\'s have another try')
        a = False
    

a = False

while a == False:

    hours_many = int(input('If you ever realized to use your smartphone for too many hours, how many of them? [Answer should be between 1 and 10] = '))
    
    if hours_many == int(1):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(2):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(3):
       print('\nAnswer saved')
       a = True
    elif hours_many == int(4):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(5):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(6):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(7):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(8):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(9):
        print('\nAnswer saved')
        a = True
    elif hours_many == int(10):
       print('\nAnswer saved')
       a = True
    else:
        print('\nIncorrect answer, let\'s have another try')
        a = False
        
a = False

while a == False:

    stressed = int(input('Do you consider yourself a stressed person? Answer from 1 to 7 (1 minimum level, 7 maximum level) = '))
    
    if stressed == int(1):
        print('\nAnswer saved')
        a = True
    elif stressed == int(2):
        print('\nAnswer saved')
        a = True
    elif stressed == int(3):
        print('\nAnswer saved')
        a = True
    elif stressed == int(4):
        print('\nAnswer saved')
        a = True
    elif stressed == int(5):
        print('\nAnswer saved')
        a = True
    elif stressed == int(6):
        print('\nAnswer saved')
        a = True
    elif stressed == int(7):
        print('\nAnswer saved')
        a = True
    else:
        print('\nIncorrect answer, let\'s have another try')
        a = False
        
a = False

while a == False:

    insomnia = str(input('Have you got insomnia problems? [type yes or no (lowercase)] = '))
    
    if insomnia == 'yes':
        print('\nAnswer saved')
        a = True
    elif insomnia == 'no':
        print('\nAnswer saved')
        a = True
    else:
        print('\nIncorrect answer, let\'s have another try')
        a = False
        

a = False

while a == False:

    no_social = str(input('Have you ever thought about deleting your social account? [type yes or no (lowercase)] = '))
    
    if no_social == 'yes':
        print('\nAnswer saved')
        a = True
    elif no_social == 'no':
        print('\nAnswer saved')
        a = True
    else:
        print('\nIncorrect answer, let\'s have another try')
        a = False
        

a = False

while a == False:

    del_social = str(input('If you ever though about deleting your social account, have you ever actually done? [type yes or no (lowercase)] = '))
    
    if del_social == 'yes':
        print('\nAnswer saved')
        a = True
    elif del_social == 'no':
        print('\nAnswer saved')
        a = True
    else:
        print('\nIncorrect answer, let\'s have another try')
        a = False


Dict_input = {
'const':1,
'out_smart':out_smart,
'hours_many':hours_many,
'stressed':stressed,
'insomnia':insomnia,     
'no_social':no_social,  
'del_social':del_social}

df_input = pd.DataFrame([Dict_input])
df_input.shape

for i in df_input['no_social']:
    if i == 'yes':
        df_input['no_social'].replace(i,1,inplace=True)
    elif i == 'no':
        df_input['no_social'].replace(i,0,inplace=True)

for i in df_input['insomnia']:
    if i == 'yes':
        df_input['insomnia'].replace(i,1,inplace=True)
    elif i == 'no':
        df_input['insomnia'].replace(i,0,inplace=True)

for i in df_input['del_social']:
    if i == 'yes':
        df_input['del_social'].replace(i,1,inplace=True)
    elif i == 'no':
        df_input['del_social'].replace(i,0,inplace=True)
        

yPredicted = model.predict(df_input)
yPredicted = yPredicted.iloc[0]

if yPredicted >= 0.5:
    print('\nYou are an anxious person')
else:
    print('\nYou are not an anxious person')
        

    
    
    

