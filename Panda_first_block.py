#🔹 BLOCK 1: Essentials — Data Exploration & Manipulation
import pandas as pd
import numpy as np 
#Note in pandas, the category and object type stands for the string type

#📌 Goal: Load a dataset, inspect it, and work with columns and rows.

#1.	Load a CSV file and display the first 10 rows

df=pd.read_csv("/Users/Dani/archive/bitcoin_price_Training - Training.csv")  #Read_csv("file") picks the file and read it, while nrows
#only display the first 10 rows of our dataset

print(df[0:10])

#2.	Print the number of rows and columns

print(df.shape)

#3.	Show all column names
 
print(df.columns) 


for i in df.columns:
    print(i)

#In the first example we get a list while in the second the console just prints the letters of the columns


#4.	Display the data types of each column
print(df.dtypes)


#5.	Show basic statistics for numeric columns

print(df.describe(exclude=[object])) 

#When printing this information you are gonna different values. Count, stands for the number of rows there are, mean, known as the average, min 
# and max, pretty evident what they mean, then some percentages,that stand for the values, in which I would like to look a bit deeper. 

#The percentages make reference to the percentil which is really hepful for having a general intuition of how the data is spread along our dataset
#For instance, 75% porcent of the times bitcoin has opened at a smaller amount than 662.35$ procent, while only 25% of the times the opening value 
#has been smaller than 254.288$

#Finally, the statistic in which I would like to profundize the most is the std, or the standard deviation, for it may be possible that not every-
#knows or understands it. The standard deviation shows us how much in average is our data far away from the mean. This entails that a big standard devia-
#tion will imply a scattered data while a small standard deviation will signify that our data is compacted. How do we calculate it? In the next way: 

#The formula for calculating the standard deviation is the next one--> sqrt(∑((xi-µ))^2/n). Where x1 stands for all the values in our dataset, µ
#for the mean, and n for the quantity of different values. A complete explanation of the formula would go something like this. 

#By substracting x1 to µ we are calculating the difference between all the values and our mean then we add all of them,∑, and divide them by the 
#number of values. That way we would obtain the standar deviation from the origin, except from one detail, negative numbers. These will going 
#to ruin our calculation, since distance from one point will always be positive not negative, but by substracting smaller numbers to the mean
#we are going to get negative numbers. In few words, we need to get rid of them. One way of doing it is by adding to this inital formula the 
#absolute value, ending with something like this, ∑||xi-µ||/n. 

#Nevertheless, working with the absolute value is something tricky, I don't know why, but so say the experts, so we need another way to get rid 
#from negative values. Hence, we'll just square the substraction between xi and µ, to them make the sqrt of the result we get, ending with the formula 
#we started with. 


#6.	Access a specific column and show the first 5 values
#This can be done by just knowing how to slice rows and columns. Basically, we're picking one column and the first 5 rows of it nothing more

print(df["High"][0:4])


#7.	Filter rows greater than a value
print(df[df.select_dtypes(include='number')> 1000].dropna(how='all'))


#8.	Count how many unique values exist in a categorical column
print(df['Low'].value_counts())

#9.	Create a new column adding two numeric columns
mean_rate=(df.Low+df.High)/2 #Here we just select the two columns and create the data for a new one


df.insert(4,'mean',mean_rate) #And down here we just insert the new column in the index 4, with the name 'mean', and whit the values of mean_rate 


print(df)

#10.Replace missing values in a column with the median
#For this exercise we needed to find the gaps in the columns so for that, we'll use the function df.fillna(), which serves this specific purpouse
#The thing is that these gaps in the columns must be of numeric type, not categoric so we'll replace those for truly NaN, not a number, just a gap
#that must be filled by a float. 

#To achieve that we'll pick all the '', ' ', 'NaN', 'nan', that englobe all the gaps of the columns be them numerical or not and replace them by
#np.nan, which is an array that represents the absence of a number. 


#df['Volume']=df['Volume'].replace(['', ' ', 'NaN', 'nan'], np.nan, inplace=True) I've previously done this but this is a mistake since .replace(inplace)
#returns none, so basically, I'm just saying df['Volume]= None.  Beneath the correct way of doing it is showed, since we'll only set to None those 
#places in which we find '', ' ', 'Nan', 'nan'. 

df['Volume'].replace(['', ' ', 'NaN', 'nan'], np.nan, inplace=True)

df['Volume']=df['Volume'].fillna(mean_rate)

print(df['Volume'][1330:])


#🔹 11. Sort the dataset by a numerical column in descending order

modified_df = df.sort_values(by=['Open','High','Low','Close'], ascending=[False,False,False,False] ) #What we are doing here is sorting in descending
#order the values in the columns ['Open','High','Low','Close']. We have set ascending to false, since sort_values will by default sort values 
#in an ascending order, so we can inverse this by setting ascending to false. We need to indicate the order of sorting in all columns as we can
#observe




print(modified_df[['Open','High','Low','Close']]) #I haven't directly convert these new data set into the original one, since it is going to mess
#all up and, since values are ordered following time not increasing or decreasing valkue

#🔹 12. Group the dataset by a categorical column and compute the mean of a numerical column for each group

print(df.groupby("Date")['High'].mean()) #Here I would like to pick some column that makes the result more interesting, since what df.grouby does
#is agruping all the categorical, string, values of a columns into unique groups if they are repeated, and then uniting to it a numeric column 
#we'll do the mean for all the different numeric values for a same category in different entries, but as the dates, volume, and market cap, are usua
#lly unique, doing the mean is practically pointless.

#🔹 13. Drop all rows that contain missing values


print(df['Volume'].dropna())


#🔹 14. Rename a column
df= df.rename(columns={'Market Cap':'Value'})

print(df)



##🔹 15. Save the modified dataset to a new CSV file

df.to_csv('Modified_bitcon_price',index=False)