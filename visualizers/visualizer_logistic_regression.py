#5

#Now for a better understanding, we'll make a visualizer for observing how much each column influenced the overall output of our model. In order 
#to do that, we need to get the coefficients of how much each one influenced, the bigger the coefficient the bigger the influence, 
# and then plot them into a column with its respective column and just plot that. So really simple and easy to follow.

import joblib #We'll bring our model direct to this file with joblib.load
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


model=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/Logistic_model.pkl')
X=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_logistic_model.pkl')

#So now we have our model and the variables we need we can start working with them.

feature_names= X.columns

coefficients=model.coef_ [0]#It returns how much influenced each value to the final decission. And if we print it we'll see that the output is seemingly
#the same as if we printed model.coef_. But that's not true, as model.coef_[0] has a shape of (18,) or in other words is a 1D array so we can operate with it 
#while model.ceof_ has shape [1,18], as this function returns one cofficient for each column of the df, in that exact shape. The column and its coefficient

#If you do the experiment you'll see how they are different.

#Feature names is really easy to understand. It just picks the name of all columns, which are the features and store them in a column.

#After it, we got coefs, that needs some deeper explanation. 
#First of all we need to know that model.coef creates a 2D array in which we have as columns the name of the features and then in one row all the 
#different coefficients for how much each column influences the decission of the model, so when typing [0], we're creating a 1D array, and telling 
#the program: hey! give me only the coefficient, the row,  nd let the columns for another one. So we have something like that [0.4,-1.5,1.2...] stored
#in our variable. 

#Remember that after doing pd.get dummies the categorical columns get converted into as many columns as unique values there are, so we'll not have 
#the coefficient exactly for each column, but for each numeric column, and each unique value of categorical columns that now form new columns due to
# get dummies. 

df=pd.DataFrame({'Features': feature_names, 'Coefficient values': coefficients})

#We can observe as positive as negative values, this is that way, as the negative values influence negatively, that's to say, they mean dead, while
#the possitive values, lead to live. 

#We're gonna add anyways, a column with absolute values, so that we can directly see wich factors influence the most the model
df['Absolute values']= np.abs(df['Coefficient values'])

#This one is optional
df=df.sort_values('Absolute values', ascending=False)  #We take the column absolute values from df and set it to be sorted in a descending order 

plt.figure(figsize=(10,6))
plt.barh(df['Features'], df['Coefficient values'], color='darkviolet') #The first argument goes into the y axis and the second into the x axis.

plt.xlabel('Coefficient Value')
plt.title('Feature Importance - Logistic Regression')

#These three lines, are used for twisting the figure and modifying it. For instance plt.gca() get the axis of the figure and then invert the y axis.
#With inverting the y axis, we're saying that instead of starting from its bottom in a descending way, we pick the highest value and place it there. 

#Then we apply a grid to the table, we set the style of the line to be not continous '--' as here, and then we make them not completely visible. 

#After all plt.tight_layout brings all together, and space automatically, the different elements of the graph, such as labels, titles, visual elements...


plt.gca().invert_yaxis()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout() #This doesn't changes that much the style of the chart, but itsn't bad to use

plt.savefig('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/Logistic_regression_visualizer.png', dpi=300)
#Figures must be saved before we show it. Otherwise, we'll get a blank image.

plt.show()
#As you can see the model takes much importance on the sex and class of the passenger followed by its fare and age, if they are either a baby 
# or an old person, or someone that isn't poor. 

