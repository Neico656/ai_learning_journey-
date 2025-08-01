import pandas as pd
import numpy as np

#We're gonna first pick another data frame. To vary a little bit. 
df= pd.read_csv("/Users/Dani/electricity.csv")

print(df.shape) #We see more or less the dimensions of the data frame we are working with 

print(df.dtypes)
#And we see it is giant, but no problem, we'll work with it as we have done previously. 

#Something to note is the columns name and some values that may not be all clear. 
#Nsw stands for new south wales, vic for victoria. 

#Finally, to set things clear, I would like to explain how each columns work

#Date: date between 7 May 1996 to 5 December 1998. Here normalized between 0 and 1
#Day: day of the week (1-7).
#Period: time of the measurement (1-48) in half hour intervals over 24 hours. Here normalized between 0 and 1
#NSWprice: New South Wales electricity price, normalized between 0 and 1
#NSWdemand: New South Wales electricity demand, normalized between 0 and 1
#VICprice: Victoria electricity price, normalized between 0 and 1
#VICdemand: Victoria electricity demand, normalized between 0 and 1
#transfer: scheduled electricity transfer between both states, normalized between 0 and 1
#Class, up or down

# 16. Change the data type of a column (e.g., from object to int)
#If we want to change the data type from one to another we need to have into account different factors. Firstly, the data
#type we're working with. If we are in front of integrer or floats that we want to change into integrer floats, or even strings
#there is no problem, as we'll just use the astype() function that does the work for us.

#On the other hand, if we're dealing with strings, there is a much complicated process, since we have to manually pick the
#strings values, and assign ourselves an integrer value to them, since the computer can't interpret that. This is a kinda
#difficult procress, I'm gonna explain just down here, with an example.

#Here what I'm going to do, is to pick an object column, say it: the column class, that have two different values b'UP
#or b'down. Then, what we have to do is to assign to these different values, a integrer that conveys the same meaning as the
#string, so we have to comprend our data

#Obviously, what our data is trying to say with Up or Down, is a sa binary decision, yes or no, or integrers, 1 or 0.
#So, we have to convert these b'UP and b'DOWN to 1 and 0. Here is how we do it:

df['class']=df['class'].str.replace("b'",'').str.replace("'",'') #We take the column we are gonna change and we supprime these
#values we don't need, b' in this case.
print(df['class'])
#Then when, we're left, with only what we needed, Up or Down, we convert those into our integrers by using map. (We could
#also haven't deleted the b' and associate the full expression b'Up or b'Down to one value, but this would be obviously less
#stylish)

df['class']=df['class'].map({'UP':1, 'DOWN':0})
print(df)

# 17. Convert a column to datetime format
#Notice that the DateTime data is always in the format YYYY-MM-DD

#Datetime and timestamp refer to an specif date while timedelta refers to a duration which can be expressed in different ways


start_date=pd.Timestamp('1970-01-01')
finish_date=pd.Timestamp('1989-12-30')
total_days=(finish_date-start_date).days

print(total_days)

df['date']=pd.to_datetime(start_date+pd.to_timedelta(total_days*df['date'], unit= 'D'))
#you can add a timedelta to a timestamp to get another date

print(df['date'])

# 18. Filter rows where  string column contains a specific word
df['class']=df['class'].map({1: 'UP', 0: 'DOWN'}) #We reverse what we've done in exercise 16 remake the column 'class' so that it remains the 
#same as it was in the origin, turning the 0 and 1 into Ups and Downs respectively. 

df['class']=df['class'].astype(str) #We change the values in this column to be specifically strings since 'Up' and 'Down' are objets, and although 
#similar we need to change those to strings so that we can later use str.contains() which only work with strings.

selected_rows=df[df['class'].str.contains("DOWN")] #Here we are creating a new data frame, in which we'll only see the columns where df['class],
#contains, the word 'DOWN'. So 'df' is remaining unchanged
print(selected_rows)



#	19.	Replace all values in a column using a dictionary

#For replacing values we have the method or function -map. So we just have to apply it to a column and it's done
df['day']=df['day'].astype(str)

df['day']=df['day'].str.replace("b'",'').str.replace("'",'')
#df['class']=df['class'].str.replace("b'",'').str.replace("'",'')
df['day']=df['day'].map({'1':'Monday', '2':'Tuesday', '3':'Wednesday', '4':'Thursday', '5':'Friday', '6':'Saturday', '7':'Sunday'})

print(df['day'])

#	20.	Drop a column

eight_columns_df=df.drop(columns='vicprice')

print(eight_columns_df)
#	21.	Set a column as the index of the DataFrame
#When working with a dataframe we can have three different indeces, that of the columns, that of the rows, and finally that of the dataframe itself.
#An index stablish a direct relation between two elements and is useful when we want to acceed to an element, as we just have to call the other.

#With columns, the indexes are the name of those, when speaking about rows, they are usually the numbers those that serve as indices of these last. 
#And for the dataframe the index of this is the whole index of the gathering of rows. So for changing it we just have to replace all this 
#line of numbers by a column with the same lenght. 

df=df.set_index('date')



print(df)

#	22.	Reset the index
df=df.reset_index()

print(df)
#	23.	Select multiple columns at once
float_columns= df[['date','period','nswprice','nswdemand','vicprice','vicdemand','transfer']]

print(float_columns)
#	24.	Select rows using iloc[] and loc[]
#Loc[] and iloc[] are pretty similar methods, but that have slight differences. Both serve to call an specific row given some previous parameters.

#The difference between them is that loc[] takes in label parameters, that's to say strings, while iloc[] takes in the index of the row as the 
#parameter to seek for.

print(df.loc[(df['day']=='Wednesday')])  #What we're doing here is to say. Ok look for this rows that meet the next condition. df['day']=='Wednesday'
#That's to say, pick this columns in which the date is equal to Wednesday and print them. 

#As a fact .loc[] can also takes in indexes so we could do: print(df.loc[0:1500,(df['day']=='Wednesday')]) and it would print these rows in which 
# date equals wednesday and in addition are founded between the indexes 0 and 1500


print(df.iloc[1500:3500,2:5]) #And as said, iloc, slices our data depending on numeric indexes from both, rows and columns. 


#	25.	Filter rows with multiple conditions (using &, |)
selected_rows=df[(df['date']> '1975-04-28 00:00:00.000000000') & ((df['day'].isin(['Wednesday','Thursday','Friday','Saturday'])) | (df['nswprice']>0.05))]
#So here what I guess, I'm doing is to only select those rows whose 'date' column is bigger than 1975-04-28 and at the same time or well its day 
#column matches the days there defined or the nwsprice are bigger than 0.03



print(selected_rows)

#	26.	Find duplicated rows
#To find duplicated rows we can easily use the function df.duplicated(), this will go through all the rows and return true when it indeed find 
#a duplicated row. 
#Nevertheless, if we want to properly identify where these duplicated rows are, the procress is somewhat more complex.

duplicated_rows=df.duplicated() #First of all, we detect all the duplicated rows

duplicated_indexes=df.index[duplicated_rows] #Then we get the indexes of duplicated_rows exists, or with other words where duplicated_rows==True
#This will return the list of indexes of these duplicated rows

if len(duplicated_indexes)>0: #Then if that list is longer than 0, that means we have some duplicate rows, 

    for i in duplicated_indexes:
        print(f'There are duplicated rows at indexes {i}')
else:
    print('There are not duplicated rows')



#	27.	Drop duplicated rows
#I have no duplicated rows but in case there were it would be solve the next way:
df=df.drop_duplicates() 

#	28.	Merge two DataFrames on a common column (like SQL JOIN)
#As I don't have a dataframe to merge with this I'm working right now, I'm gonna create a two new ones. In this two dataframes, we have a dataframe 
#of students, and another for the teachers, and we want to merge these two into a big one dataframe, that relates each teacher with their student.

#Merging consists on picking two different dataframes and merging these rows that have the same value on the columns specified. We specify which
#columns must serve as a guide with the argument -on, if not it will merge these columns with the same name. In our case we need to specify the co-
#lumns we want to work with, since otherwise, columns that we don't want to be influenced by such as traits will influence our results. 

#In order for two dataframes to merge, their row in index n of the column we are using as guide must contain the same value in both.

left_df= pd.DataFrame({'Professors': ['Lucia','Marcos','Diego','Sebastian','Andrea'],
                       'Class': ['1ºD','1ºE','1ºB','1ºC','1ºA'],
                       'Traits': ['Outgoing','Agile','Reflective','Honest','Kind'],
                       'Optative': ['Psicology','Chemistry','French','PE','Technical draw']
                       })
right_df=pd.DataFrame({'Students': ['Javier', 'Carla','Manuel','Sara','Laura'],
                       'Class':['1ºD','1ºE','1ºB','1ºC','1ºA'],
                       'Traits': ['Unusual','Virtous','Reflective','Idk','Outgoing'],
                       'Optative': ['Psicology','Chemistry','French','PE','Technical draw']
                       })
                        
class_planning= pd.merge(left_df,right_df, suffixes=['_professors','_students'], on='Class')
#The argument -how determines, how the data is merged. There are four different options: Inner, outer, left, right.
#If we select inner, our new dataframe will be composed by only these rows in which the values of our objective column coincide. All other data di-
#ssapear. 
#With the argument outer, no data is lost and all the rows, of both columns are represented in the new data frame, be the values of the key rows
#the same or not. 
#By using left, the resultant dataframe will be composed by the rows that contain the same value, and by all the data stored in the left dataframe. ##
#The first one. #
#The argument right does the same, but this time with the right dataframe, the second one.

print(class_planning)
#	29.	Concatenate two DataFrames vertically (row-wise)
#When concatening, what we're doing is to add all columns all together and display all of its values. If one column has the same name as other, 
#the data of the first one will be displayed up, and the data of the second one, down. If there are columns that do not exist in both data frames
#a NaN will occupy the place in which there is no data available to fill it.

correction_time=(['The same day','Two weeks','Three days','One week', 'The same day'
])
left_df['Correction time']=correction_time

favorite_subject=(['Biology','Maths','Arts','English','Arts'])
right_df['Favorite_subject']=favorite_subject

concatened_df= pd.concat([left_df,right_df],ignore_index=False)
print(concatened_df)
#As we can see for the professors data frame the columns of students and favorite subjects are fillled with NaN values, as these columns don't exist
#in its data frame. The same but at the inverse with the left data frame

#   30.	Fill missing values using the .fillna(method='ffill') or .bfill()
#Here I should use the ffill or bfill method, but I'm jsust gonna use  fillna, since it makes more sense to use this kind of fill methof with this 
#dataframe, as you can fill the NaN values with whatever value you want, be it string, float, or an integrer. 
#On the other hand the ffill and bfill methods work the next way. They pick the last not NaN number and fill the NaN values with that value. The 
#value of reference is that up if we're using ffill, and down if we instead choose bfill. 

#As you can see, in the column of professors the values will be filled with ffill)(), Andrea, and in that of the students, it will be filled with 
#Javier using the bfill method. With the fillna method we can fill the dataset with custom values that make more sense.

concatened_df=concatened_df.fillna({
    'Professors': 'Not a professor',
    'Correction time': 'Do not correct',
    'Students': 'Not a student',
    'Favorite_subject': 'Do not have a favorite subject'
})


print(concatened_df)

