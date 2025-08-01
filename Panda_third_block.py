
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"  #Path of the desired file
df = pd.read_csv(url) #We transform the path into a dataframe 

df.to_csv("/Users/Dani/titanic.csv", index=False)# We save the dataframe with the name we want into de directory we want

#Just to know sibsp stands for: number of silbings or spouses aboard, and sibsp: number of parents or children aboard. 


print(df) #We can already work with it 

#31.	Create a pivot table: average of numeric column by two categorical levels
#With pivot_table, we can change and modify the structure of our dataframe, so that it shows only some data, we might be interested in. 
#What we can do is to put the column we want as the index of our df. By doing that, each unique value of the column x will be set as a part of 
#the index. 

#In second place we can modify the columns, in the same way as we did with the rows. We choose a column and each unique of its values, will be 
#a new column. 

#That's how we form our new structure, then in order to fill it we need values, that are going to be these other columns of our data frame. But these
#values will be condensated, and will only show those in accordance with the data in our table. 

#For instance, in my example, I've chosen the column age as a value, so what it is going to do, is not to pick all values of age, but only those 
#of female in the first class, male in the first class... You get it. Then when it picks all these values, we call the method -aggfunc to tell 
#the programm what to do with these lists of values, in my case, I've said the programm to calculate the mean with the values of each corresponding
#frame.

pivoted_df=df.pivot_table(index='sex',columns='pclass',values= ['fare','age'], aggfunc={'fare':'mean', 'age': 'mean'})

print(pivoted_df)
#As you can see the results make sense as the age of the passenger increases as we approach the first class, as well as the cost of the fare.

#32.	One-hot encode a categorical column (use pd.get_dummies())
#One-hot encoding is a technique used in machine learning to make data easier for models to diggest. We're going to use it, when we're facing 
#categorical data type, that's to say objects or strings, and what this hot encoding technique is going to do is help our model to difference, 
#between strings types of data, telling it where one is founded with a one, and representing the absence of it with a zero.

one_hot_encoded= pd.get_dummies(df, columns=['embark_town'])

print(one_hot_encoded)
#For instance what we're getting here is that our model has splitted the column embark_town into all the unique values we have on it, and created 
#a column with each one of them. This column is no longer filled with strings, but with booleans, that's to say 1 or 0, and they show True when 
#the passenger did indeed embark in that town, and false when it did in other. 

#33.	Normalize a numeric column to [0,1]
#Normalization is a pretty useful concept that serves us for making easier to see where a piece of numeric data lies in comparison with the other. 

#In the practice, it consists on taking all the numeric values of a column and mapping them from 0 to 1. We achieve with the next formule: 
#To be sure that all our results fall between 0 and 1 the easiest thing to do is to set a proportion,division, in which the element we're gonna be 
#comparing is the max value of our column, that way, no matter which element we pick it will be never be bigger than the max value, 1, and inferior to 0. 

#Nevertheless, we realize that this system is not hundred percent correct as if we pick the number 50 and the number max is 100, 50 will be closer
#to 0 or to 1, depending in which the minimum is. If the minimum of that list is 50, 50 will be 0, but if it is 0, fifity will be 0,5. So in some 
#way we need to introduce the minimum value in the ecuation to take into account the overall dataframe. We'll do it by substracting the minimal value
#to both terms to map correctly the size of the dataframe

#So, we're left with: x-xmin/xmax-xmin. 

age=df['age']

normalized_age=(age-age.min())/(age.max()-age.min())
print(normalized_age)


#34.	Standardize a numeric column (z-score)
#Standardization is the process by which we some values, and we transform it mean and standard deviation into 0 and 1 respectively. We do this in 
#purpouse to be able to study our data in a more generic way, and interpret it more intuitively. 

#In order to do this, we take all x values and substract them by the mean, this is gonna make that our set of values move in the axis x to the 
#point of zero. The substraction or adding in a plane entails an horizontal movement in this one. Thereby, we ensure that our mean is 0, since 
#mean-mean=0.

#Then we need to think how to make our standard deviation to 1, given that we have a group of values set appart by a distance of x, then if we 
#divide this set of values by x, the distance will be 1. Values set appart by x/x--> Values set appart by 1. So in conclusion, we're left with 
#this formula: 
# x-µ/standard deviation
print(df.dtypes)

mean=df['fare'].mean()
standard_deviation=  np.sqrt(np.sum((df['fare']-df['fare'].mean())**2))/len(df['fare'])

standarized_column= (df['fare']-mean)/standard_deviation
standarized_column=standarized_column.round(3)
                         


print(standarized_column)

#35.	Use .apply() to apply a custom function to a column
#With the function apply(), we can evidently apply a function or some operation to the hole column. Here, I'm multiplicating all the column  by 98,97 since that's gonna give us how much would have cost the ticket for the titanic nowadays. 1 pound back in 1912 was equivalent to 98.97 nowadays

#Here I've declared a lambda function. Lambda functions are used when we need to define simple functions, that we're only gonna use once, since they are easier to write and appear only once in our code. One important detail, is that we can't call them

nowadays_fare=df['fare'].apply(lambda x: x*98.97) 

print(nowadays_fare)
#36.	Apply a function to each row with .apply(axis=1)
#As I only have two columns with numerical number, there is no much room to apply some cool math function or relationship. I've tought on the relationship that exists between age and a bigger fare, but as the relationship coefficient shows there is no relationship between these two values. So instead, we' re going to do something way more unprofessional and exact, but that at least we can do. 

#We're gonna study the relationship between the fare and the age of the passenger, so the bigger the ratio of the fare/age the richer the passenger is. 


ratio_fare_age=df[['age','fare']].apply(lambda row: row['fare']/row['age'], axis=1)
print(f'\nThis is the ratio for age of the titanic:  (The bigger the richer the persone) \n {ratio_fare_age}')

#37.	Bin a numeric column into categories (e.g., age → young/adult/old)
#To divide a column into categories we're gonna use the cut() function. If we don't apply any methods to it, the cut () function will simply 
#divide our numerical data into n categories of equal size, in the next way: pd.cut(df['measures], bins=3) And it will divide the column measures
#into three equal groups 

#Nevertheless, if we want, we can customize a little bit the structure of ur groups. In this example, we're gonna change, firstly the range of the
#groups and then we're gonna label them accordingly. 

#We're gonna divide the age of the passengers into four different groups: child, young, adult, old. For an efficient measurement, we're gonna first 
#calculate the mean, and then standard deviaton, to have a grasp of how our values are ordered within our dataset. 

#In order to prevent the standard deviation to return an Error I have to fill the column with the mean value. The ddof= 0 gives you a result for 
# every single member of the population so you assume you have data for everyone of them. As I'm gonna use it, I'm gonna fill all the missing values
# with the mean. 

#However if you don't want to fill you df with not accurate values you can proporcionate some degree of freedom and assume you lack some values 
# representing that in the formula with ddof=n. (n normally is set to 1). 

#Mathematically what you are doing is picking the formule of the standard deviation, and substracting one to the -n, the total number of values, 
#what gives you that degree of freedom. 
print(df['age'].dtypes)

df_age_full_column= df['age'].fillna(df['age'].mean())

print(df_age_full_column)

print(f"The mean age in the titanic was of {df['age'].mean()}")
#As we can see the average age of the Titanic was of 29.699 years, pretty young age for affording such a luxurious embarcation. This can be a reflec
#tion of how the youth of the time emigrated towards the new world full of new opportunities.

print(f"The standard deviation of the df is of {df['age'].std(ddof=0)}")
#Having into account that we're working with years, we see that the boath was mostly filled with passengers of a young or mature age, what's completely 
#normal being left with rare cases of children or elderly people travelling. These results make sense, because probably they were young emigrating
#without a family and nothing to lose or mature business man who left their families at home to take charge of the affairs.
print(df_age_full_column.max())

bins=[0,13,19,59,80] #We've created a variable that when introducised in the function will create four different groups: 0-13, 14-19, 30-59, 60-80
labels=['Kid','Young','Adult','Elderly'] #These are the names our groups are gonna have

df['age_groups']= pd.cut(df_age_full_column,bins=bins, labels=labels)

print(df)

#38.	Create a new column with a condition (np.where() or df[col] > x)
# Here is the basic structure for the np.where or df[col]>x function: df['col'] = (value_if_false).where(condition, value_if_true)
#It stablishes a quick else-if condition to modify values accordingly. 



#While str.replace() replaces values, replace() replaces integrers.

df['alive']=np.where(df['alive'] == 'no', 'Dead', 'Survived')
df['sex'].astype(str)    

print(df.loc[(df['pclass']==3) & (df['sex']=='male') & (df['age'] < 15 )]) #Making a little check on how many people dead or survived depending on 
#conditions. This one here is for example for male kids in the third class. Pretty sad as you can see. 

#39.	Group by multiple columns and aggregate with multiple functions
#What groupby does is to pick the columns we specify as arguments, and create a dataframe in which each unique value of the column is set as some 
#sort of index. If we have multiple columns as arguments then we have a row for each unique value compounded by the two columns. That is to say 
#the combination of all unique values in each column. As here we have sex: male or female, and pclass: 1,2,3. Tha't to say 1.Male 1. Female 
#2.Male 2.Female, 3.Male 3.Fenale

group_by_representation=df.groupby(['pclass','sex'])['age'].mean()
#After having grouped our values, we can relate some numerical value to it, applying after the function some sum, multiply, mean, standard deviation
#function. In our example, what our programm is doing is to collect all the values of each sort them into groups according with our indexes (1.Male 1. Female 
#2.Male...)  and then picking that all already-sorted values and calcukate its mean. 

print(group_by_representation)
#40.	Calculate correlation matrix between numeric columns
#The correlation is a statistical measures that establishes a relationship between two values. This relationship is stablished with a number fron
#-1 to 1, being -1 a negative relationship, that's to say when one occurr in a greater extension, the second occur in a smaller on, and 1 a positive
#relationship, in which when one happens more the other the same. The more the result approaches to zero, the less grade of relationship exists. 

#What wh're going to do then is first, calculate the correlation that exists between all the numerical column, and then display a heatmatrix that 
#help us visualize it. 

df['pclass']= df['pclass'].astype(int)
df['survived']= df['survived'].astype(int)
df['sibsp']= df['sibsp'].astype(int)
df['parch']= df['parch'].astype(int)

df_int= df[['pclass','survived','sibsp','parch']]

corr= df_int.corr()

sns.heatmap(corr,annot=True, cmap='coolwarm') #Sns is the library we're using to create the heatmap 
plt.show() #And plt the one we're using to display it 

#So if we analyze the table, we can observe that there aren't much relationships between the different values, except from two columns. 
#We can note there is a negative relationship between the class and whether the passenger survived or not, what makes sense, because the bigger
#the class, 2 or 3, the less possibilities probably you have to escape. 

#After it we have the relationship between parch and sibsp what is obvious, since the more parents or children you have on board, the more 
#probable it is that you also have silbing or spouses on board.

#41.	Plot a histogram of a numeric column (with matplotlib or seaborn)
sns.histplot(data=df, x=df['age'], kde=True)
plt.show()
#42.	Plot a bar chart of category counts


sns.countplot(df,x=df['pclass'],hue=df['pclass'], palette='pastel' )
plt.show()

#In these exercises we're just using a library call seaborn to represent the diferent tables and charts asked. The first method is the data, 
# then we have the x axis in which the values are stores, and then we can also choose colors, choosing which one to color with hue and how to color
#them with palette, among other useful methods
#43.	Find the row with the maximum value in a column

highest_value= df['age'].max()
index_highest_value=df['age'].idxmax()

print(f'The highest age value is {highest_value} in the row {index_highest_value}')

#44.	Export summary statistics or cleaned data to Excel
df_summary=df.describe()
print(df_summary)
df_summary.to_csv('/Users/Dani/titanic sumary.csv', index=True)

#It is important to finsih the name of the file with csv, or your computer won't recognize it as that type of file, then we set the index to true
#as we want to describe which statistic is each one.

#45.	Load only a subset of rows from a very large CSV (with nrows= or chunksize)
#When, we're working with large amounts of data, not the case yet, we may want to acced to an specific piece of data without calling the whole data
#frame, here I'm speaking about files that weight 600 mb and have millions of rows. To prevent that there are some techniques. For instance, we can 
#either slice the data frame: 

new_df=df[100:200] 

print(new_df)#Print rows from 100 to 199

#Or either we can use a more advanced method called chunksize. This will divide our datafile in chunks of our preference and then, we'll be able
#to call this chunks individually. 

for chunked_df in pd.read_csv(url, chunksize=25): #What this will do is to read 25 rows of df, per iteration, so if we have 900 rows we'll have 36 
# chunks. 
   print(chunked_df) #And each time the for loop passes will print chunked_df, that will contain the 25 rows of the iteration.

#Thereby we have printed our dataframe divided in different chunks, but normally you won't do it if what you want to do is to prevent calling the 
#whole dataframe. 

#So we'll normally store the 'chunked' df in a variable
chunked_df=pd.read_csv(url, chunksize=25)

#Here we're using an iterator, an iterator is an object that goes trough an object or an element one at a time, or at the ratio we tell it to do. 
#That is what the loop for do for instance. However, it is important to note that an iterator don't go entirely through an object but that it only
#completes one iteration unless we specify it to continue

#How do we tell him to continue? With the function next. The function next tell the iterator to complete another itheration through the object or element
#we're iterating, and gives that value in retun 

#Here we can appreciate how with .islice() we're specifying the iterator where to start and when to begin the iteration. It must start at four, and then 
#with next it does another iteration and ends at five, so we'll print the fifth chunk.

fifth_chunk = next(itertools.islice(chunked_df, 2, 5))
print(fifth_chunk)