#LIME is a tool used in ML to understand why the model makes certain decission not respecting to a complete dataset but respecting an specific sample
#of it. So it will tell us why the model predicts certain outcomes to certain passengers. The code is as always pretty straightforward and doesn't 
#require that much. 

import joblib
import pandas as pd
import lime 
import lime.lime_tabular #This is the part of lime in charge of working with tabular data, while others work either with text or images. 


X=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_logistic_model.pkl')
model=joblib.load('scikit_learn/Titanic_model/Joblib/Logistic_model.pkl')

#First of all we need to know that LIME works with array, as it is much more easier to operate with them, as it is with a dataframe, in which
#you have more objects, and things to look at, of different types. Thus, we need to convert our X to an array, if it isn't already with the next 
#line. 

X_array= X.values if isinstance(X, pd.DataFrame) else X  #Here, we're transforming X into an array if it isn't already. The thing is that, isinstance
#is a function that compares the object type with that it has being said to compare it with, then if they are not equal it equals X to X, but if 
#X is a dataframe it will equal X to X.values, these are the values inside the dataframe, that are already an array of 2D. So thereby, we just convert
#our x into an array



explainer=lime.lime_tabular.LimeTabularExplainer(
    training_data=X_array,
    feature_names=X.columns,
    class_names=['Survived', 'Died'],
    mode='classification'
)
#These are the five more important lines of the file, as it creates the object that is gonna explain why the model has acted in an specific way 
#with that sample. But first, we have to understand how LIME truly works. 

#Well what LIME does, is to focuss on the data we want it to study, and create similar passengers or samples to this very same data. Then after it
#has created these similar samples, it will create a local surrogate model, which will study the predictions of the original model. As it studies its predictions 
#it will extract patterns: such as if I change male for women the model predicts an higher survival rate, so with these patterns it goes and trains 
#an interpretable model such as a logistic regression or random forest, so that it approximates the behaivour of the black box model and we can see 
#more or less why it makes some predictions or which features have more importance. 

#The models, we're using right now, are already white box models, so LIME is not truly necessary, but we're gonna do it anyway, as we'll have to learn 
#it sooner or later.

#Now that we know how it works, let's explain the different lines. First of all, you have the training_data, the local surrogate_model will be trained
#on an specific point of this data using the information of these same rows to create new samples. Secondly, the name of the columns so we can see it 
#when we display them, the class names, so we know to which thing each probability is refering, and the mode of the model , classification, not regression.


idx=30



exp=explainer.explain_instance(
    X_array[idx],
    lambda x:model.predict_proba(pd.DataFrame(x, columns=X.columns)),
    num_features=len(X.columns)
)    

#Here lies a really important line of the code and that is that lambda function. First we need to know that LIME needs the X_array sample to generate
#similar examples so that the model can classify it, since LIME operates easier and faster if it uses arrays. But then, in order for the model to 
#do the prediction it needs, the data in a Dataframe format, not into array, so all the new samples that LIME is going to generate are going to be store in 
#x, and then that will be convert into a dataframe that our "black box" model can interpretate. Then we get the prediction that will be able to see
#later on. Num_features stands for the number of features that are displayed in the table of our html file. 

exp.save_to_file('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/lime_logistic_explanation.html')

#So to see this file you'll have to find it on your laptop and open it with your browser then you will see on the upper left corner, the prediction 
#values for both probable outcomes, and then you'll see a violin graph, and a table with the values of the passenger on each feature. As we can see
#øur passenger was a male on the first class what gave him 15% more probability of surviving, while being classified as a mid-low-class and having 
#embarked on Cherboug makes 9% more possible for him to die, so we add the positive as the negative features and we end approximately with a 
#55% of surviving and a 45% of dying. 

#Note two things, first that we have been able here to witness how altough we trie to label our data correctly, we can still make logical mistakes 
#as it is impossible to fit all the data for all the samples. In this example, we've set the fare-rich class to 100 as if we didn't people, travelling 
#on third class but being much people would have been labeled as rich, so here, we have the other side of the coin, rich people travelling alone that
#didn't paid 100 pounds for an unique bedroom that are labeled as mid-low fare, because of travelling alone. But you know, sometimes we have to decide
#we can't get all right, so we need to try to get the best we can. Nevertheless, this is an insightful example, of how many things can go wrong 
#and how things aren't neither as perfect nor logical as we expect them to be. 

#Then, be sure you label first your positive class and not vice versa as I did, or you'll get an error in your first try:     
#class_names=['Survived', 'Died']. Class_names=[name for 1,name for 0]



