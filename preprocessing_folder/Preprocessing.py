#1

import pandas as pd 
import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt 


#In this file we're gonna try to make data as insightful as possible for our project.

df= pd.read_csv('/Users/Dani/archive/titanic.csv')  #If you check the dataset, you'll realize we have lots of NaN values in the age column 
# and some of them in the 'embarked' and 'embarked town' columns. 

#So, we obviously need to fill this gaps with actual and what we're gonna try is to study the relationship between the values of these columns
#and the values present in other columns in the rows to fill these gaps accurately. For instance, Southampton may be the city where the most people
#embarked, so we'll just plot in southampton in all filling values. Nevertheless, probably most men travelling in second class embarked in cherbourg 
#so we'll fit these with this value. That's what we're gonna try right now

#First of all we're gonna store our NaN rows into a variable to better work with them.

missing_age=df.loc[df['age'].isna()]

print(missing_age)

#And now we have it, we're gonna start by studying the relationship between the different values to see whether we can spot a pattern or not.
#We're gonna be like a model. With the data we have, we're gonna study it and come to precise conclusions, thus what we need to do is to start 
#looking for relationships between columns, and let's use some graphs to visualize it better. 

#The first thing that comes to my mind are the sibsp and parch columns, these stand for silbing and spounses on board for sibsp,
# or parents or children on board for parch. Since when you're traveling alone to the new world you're normally not a children,
# and probably either a grandpa. Let's see if that's true: 

#Firstly, we're gonna work with the data we already have. 
full_values= df.loc[df['age'].notna()]

#Then we're gonna merge the sibsp and parch column into a new family member column and then plot a graph. 

full_values['family members']= full_values['parch']+full_values['sibsp']

labels=['parch', 'sibsp']
full_values=full_values.drop(labels=labels, axis=1)

#Now we have done that, let's study the relationship that exists between the age of the passenger and other variables

print(full_values)

#We could plot a correlation matrix, but most of our columns are not numerical or boolean, so it would only work for colmns such as family members
#or fare, hence it is gonna be cleaner a probably more ordenated if we just display grafics for the different columns.

#Graphic for survival depending on age: 
#First of all we group the ages into groups: 


full_values['age']=full_values['age'].apply(lambda x: int(x/10)*10) #Each age is converted into a float that is converted into an integrer which 
#then is multiplied by ten, that's how we round to the nearest



#In first place, I've tried to group both sexes into one variable, but then I've realised that there is much difference in survival, depending
#on whether you were a male or a female, so I've decided to plot different lines for them

grouped= full_values.groupby(['age','sex'])['survived'].mean()

female=grouped.xs('female', level='sex')
male=grouped.xs('male',level='sex')

#From these lines, we can extract a meaningful principle of how groupby works. Groupby returns a Series, a list of
#values with an index. That can be clear when you only pass in one column to group, but if you pass more than one, groupby will still return an index. 

#The difference is that index will be hierarchical, that index will be a multi-index, with multiple levels. So to acceed to a value you have to 
#refer to multiple levels at the same time. 

#This poses a problem, as we can't directly acceed to the values we want, just based on one condition, or level of index. For instance, if I want 
#to acceed only to the female survival rate, I can't, for that we need to filter it.

#We can filter these roes by writing the next lines: grouped.xs('female', level=sex) This tells the programm, go to level sex and give me the 
#values of the rows where female is True. 




print(grouped)
#So, we create an smaller dataset, that contains that values we want 
#to compare, the age of the passenger its sex, and its probability of surviving

sns.lineplot(x=female.index, y=female.values, label= 'Females', marker='o')
#Here we're writing the two different lines that are gonna
#define the different trends, the first will be for women, and the second for males. Then plot.show() will show both lines

sns.lineplot(x=male.index,y=male.values, label='Males', marker='o')

plt.savefig('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/Survival_rate_by_age.png', dpi=300)

#As we can see now most men of all ages died during the incident, while most women achieved to scape. There is though some little children that 
#survived, so for most men that survived we're gonna say they were children, as it is the most plausible option. 

#For women we can see this graph is not really useful as the majority of them survived not matter the age. 

#Now we need to think in another relationship, and this is gonna be the relationshp between the age and the number of family members on board,
#since as more family members on board the most likely it is that the person will be whether a child or a mature adult. 

grouped=full_values.groupby(['age','sex'], as_index=False)['family members'].mean()
#As_index= False, we transfrom the index into a column, if we need to work with dataframe instead that with Series as it 
#is the case when we need to plot an histogram

#In a first place, I didn't tink that I would need to separate women, from men in this instance, but I have realised that women tend to have more
#family members than men, so this could lead to errors. 

# Imagine that we suposse that all people older than 20 travels with less of a family member in average. So as soon as we see a female, with more
#of a family member, we would assume she is a child, when in fact she could also be twenty, thirty, or forty. So we need to plot the values for each 
#gender, in this example and so on, given the strong differences between sexes back in time. 


print(grouped)


plt.figure(figsize=(10,6))
sns.barplot(data=grouped, x=grouped['age'],y=grouped['family members'],hue=grouped['sex'])

#What the hue parameter does, is to split the x axis on another different condition normally a color, but it can also be as right now the sex.


plt.savefig('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/Family_members_per_age.png')

#With this graphic we can observe that women tend to travel a lot more with their families, as well as kids, while young and mature men travel,
#most times alone. 

#So till right now we have that, if someone is dead is a men and is traveling alone, is more probably to be between 20-50 years, while if someone
#is alive and traveling with some family members she is probably a little girl or a mature woman. Nevertheless, we can still stablishing relation
#ships to deduce more patrons and make our data more exact. 

#The next thing we're gonna study is the class in which the passenger traveled, according to its age and sex. So let's repeat the same process again




full_values = df.loc[df['age'].notna()] 
#As now the age column won't be in the x-axis we should better work with the exact values of the age column
#rather than rounded valued. 


#And we add the family columns and supprime the sibsp and parch columns:

full_values['family members'] = full_values['parch'] + full_values['sibsp']

full_values.drop(['parch', 'sibsp'], axis=1, inplace=True)



print(full_values)


plt.figure(figsize=(10,6))
sns.boxplot(data= full_values, x='pclass', y='age', hue='sex')


plt.savefig('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/Pclass_according_to_age_and_sex.png')

#By observing the columns we can see that the older the people, the more propense there is for them to be in first or second class. 
#So now, we have stablished multiple patterns, we're gonna work with them to fill the more accurately possible our missing age values. 

plt.show()

#These are the questions, we're going to ask to come to a conclusion: 
#Is the passenger a man or a woman? 
#Traveled the passenger alone or not?: If they didn't there more probably young adults in both cases, or old women. 
#Did they die?: If they did, and they were women, they were more probably children, if they didn't I they were men, they were also probably children
# 
#Which was their class?: The more luxurious the class, the older the individual should be. 


def estimate_age(row):
    if row['sex']=='female':   

        if row['family members']>=1: 

            if row['pclass']==1: 

                if row['survived']==0:  
                    
                    return 7
                if row['survived']==1: 
                    return 35 #
            if row['pclass']==2: 

                if row['survived']==0:
                    return 7
                
                if row['survived']==1: 

                    return 30 
                
            if row['pclass']==3 :

                if row['survived']==0: 
                    return 7
                
                if row['survived']==1: 

                    return 22  

        if row['family members']==0: 

            if row['pclass']==1: 
                return 35

            if row['pclass']==2: 
                return 30
            
            if row['pclass']==3 :
                return 22  

    if row['sex']== 'male': 

        if row['family members']>=1: 

            if row['survived']==1: 
                return 7 
            
            if row['survived']==0: 
                
                if row['pclass']==1: 
                    return 40 
                
                if row['pclass']==2:
                    return 30
                
                if row['pclass']==3: 
                    return 25
        
        if row['family members']==0: 

            if row['pclass']==1:
                return 40 
            
            if row['pclass']==2: 
                return 30 
            
            if row['pclass']==3: 
                return 25
    
    #The logic behind this sequence of ifs is the following. If the individual is a woman, or a man, we're gonna work with the values of the proper
    #sex, which changes dramatically, between ones or others. for instance on the mean age for each class or the survival rate. 
    #After it, we're gonna question if the passenger was accompagnied or not, if she or he is, they can be either a child or an adult.

    #We're gonna guess if the passenger is an adult or a child observing the tendencies of the survival rates. Little girls died more than adult women, 
    #while little boys, were more likely to survive than adult men. So if she is accompagnied and died, we're gonna say she was a child between 0-14
    #, because of the rounded values, while if she survived, we're gonna assign the mean age of each class for that femenin passenger. 

    #With men, we're gonna work similar, but inversely, if they were in compagny and survived, we're gonna trait them as kids, while if they died
    #we're gonna assume they were adults, assigning it age according to their mean class age. 

    #For adults or row['family members']==0, their age is gonna be again determined by the mean age of their class. 

    #Note that for more accuracy, you could split the family members column into its original silbings and spouses and parents or board. I didn't do
    #it because having one column makes things easier, and simpler at the cost of accuracy, since if you know that you have parents on board, you're 
    #definetly in front of a child, but as this is a personal project and nothing proffesional, I've decided not to overcomplicate things with more
    #parameters




df['family members'] = df['parch'] + df['sibsp']

df.drop(['parch', 'sibsp'], axis=1, inplace=True)

print(f'this is df {df}')


df.loc[df['age'].isna(), 'age']= df[df['age'].isna()].apply(estimate_age, axis=1) #Here, what I'm doing is to locate those rows in which the column age is 
#equal to Nan, and asigning them the values we get when we apply our chain of ifs to those rows in which 'age' is again a NAN. Our parameter in the 
#function was rows, as we're gonna then specify that we'll be working with the axis 1

#Now we filled the age column, we need to still fill three columns. The deck, embarked, and embarked town. But no worry as these are pretty easy tasks. 
# First of all there are only two missing values in embark town and embarked, so we can fill them manually, and then the first class mostly occupied
# the decks A,B,C while the rest of them were reparted among the second and third class. 

def fill_town(row):

    return 'Southampton'

df.loc[df['embarked'].isna() & df['embark_town'].isna(), ['embarked', 'embark_town']] = 'Southampton' #In the first instance, we're selecting the 
#rows in which we want to work, and then we specify in which columns of these rows do we want to insert the changes. 

#And now the last one. We need to fill the values for deck when df['pclass']==1, df['pclass']==2, df['pclass']==3 
#To achieve that, we're gonna use np.select. that is a numpy function that realizes some choices based on previous conditions we stablish, analizing
#if these are true or false. 
mask= df['deck'].isna()
conditions= [
    df.loc[mask, 'pclass']==1,
    df.loc[mask, 'pclass']==2,
    df.loc[mask, 'pclass']==3
]
Choices= ['A','D','F']

df.loc[mask, 'deck']=np.select(conditions,Choices, default='Unknown')
#So we select those rows, in which deck== NaN, then we specify we only want to change NaN values in the deck column, and then with np.select we take our conditions, and if the first one is true, then realize the first choice, 
#if the second one is true, execute the second action and so on. Note that the conditions are where deck is NaN and pclass==1 that way we get returned 
#the same amount of values as we need. 

print(df)

df.to_csv('/Users/Dani/Programación/scikit_learn/Titanic_model/Processed_titanic_file.csv', index=False)