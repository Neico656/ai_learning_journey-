#7

#Altough we've already done the visualizers for both models, we're gonna apply some tools even more detailled, that are gonna give us even more insights
#of how our models work. These tools are Shap and LIME two models that are going to tell us why our model made the decission it made for each individual, 
#showing us how each feature contributed to the final outcome
import shap
import joblib
import numpy as np

model=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/Logistic_model.pkl')
X=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_logistic_model.pkl')

X=X.astype(float)
#It is really really IMPORTANT that you pass your variable to float otherwise, you will have a problem


explainer=shap.Explainer(model, X) 
shap_values= explainer(X) 


#First of all, we need to understand what shap specifically does, before diving into these two lines. So well, what Shap does, is it takes some reference
#values, called background values, and with them it calculates a base value. What is this base value, this base value is the probability of the event
#we're studying to happen. 
#For instance, it would take in the values we've given it, and with them it would calculate how probable it was to die back in the titanic, after it,
#it would compare the predictions of the model, see how they variate for each passenger from the mean, and explain how the different features has led to 
#this result. 

#As an example, the probability of surviving in the titanic was about 35%, and some passenger has 80% of surviving, and she is a young first class woman. 
#So shap would say to us: Hey, this passenger has this probability of surviving because being a woman in first class gives you 40% more chance of surviving.
#Simply explained this is how it works.

#So this is what is happening with these two lines here. In the first one we create an object called explainer that's going to recognize the model
#we're using and, inspectionate our data. It is gonna set our dataset as background values, calculate a base value, and how much does each feature 
#influences to the final output.

#Then in the second column, we're just saying: Hey, now you've learnt, and you know this dataset, please apply what you've learn to each passenger/row/sample
#of the variable. Note that it isn't necessary to select the same samples in order to te explainer to explain them. Nevertheless, if we want the model to 
#to explain another samples they must have the same structure, columns, and types of data, as the previous one. For instance, we could train the explainer
#with X_train and then ask it to explain X_test. 

shap.summary_plot(shap_values, X, plot_type='violin')
#From this line we can observe how we get a violing graph. You'll always obtain a violin graph from it but, if you have few values, and data as me 
#I recommend you specifying you want that type of form in your graph, thereby, you'll get that waves, and it is gonna look more similar to what it has 
#to look similar. 

#First of all the color represents the feature value, that's to say which value has each feature on each passenger. The aggrupations of red and blue, are aggrupations 
#of passengers where this feature is present, red, or absent, blue. The wave doesn't stand in one unique place, as for each passenger each feature 
#or the absence of it has a different influence. 

#In the graph we can observe some dots, these are isolated samples in which a feature was really determinant to them. Such as the case of the family 
#members if it results to be really high. 

#From the graph we can conclude that being or not being a male in the thirc class was really determinant, that not being a female in the first clas
#was also pretty negative for the subject and then that as long as you didn't have much family you were right.


shap.plots.bar(shap_values)

#And in this more simple graph we can see, how the features are ordered from this that influence the most till the last one. Note that these are 
#absolute values, and that it is our task to separe these that have a negative impact from this than not. 

#As a fun fact if you want to acceed to how much each feature influenced in each passenger, you can do shap.plots.bar(shap_values[0])
