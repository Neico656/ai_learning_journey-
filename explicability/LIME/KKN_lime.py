#11

#The very same thing for the KNN. We import, we convert to array, we create the explainer, we explain the example, turning the array into dataframes
#after having create those 'fake' samples, and we save the file
import joblib
import pandas as pd 
import lime
import lime.lime_tabular

model= joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/KKN_model.pkl')
X= joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_KNN_model.pkl')

#I've discovered that here we have something especial, that is gonna change the code a little bit respecting the two other files. As when we used 
#X=scaler.fit_transform(X) in line 67 of the file KNN we converted our X to an array, so now we need to convert our array into a dataframe so that 
#we can name our columns. Yes, just for that, but we need to do it 

#Another discoverement that I've made is that my X variables from the other models are not equal to this one, as for the KNN model I transformed 
#the categorical columns of age and fare into numeric columns, so the model could operate better, and hence now the variables have not the same shape
#and I need to get these 16 names of columns, if I don't want to type them manually.

#I already got this variable with the column names, so lets load it and complete the file. 
X_columns=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_variable_for_KNN_lime.pkl')

X_array= X

X=(pd.DataFrame(X_array, columns=X_columns.columns))


explainer= lime.lime_tabular.LimeTabularExplainer(
    training_data=X_array,
    feature_names=X.columns,
    class_names= ['Survived','Died'],
    mode='classification'
)
idx=30

explanation= explainer.explain_instance(
    X_array[idx],
    lambda x: model.predict_proba(x),
    num_features=len(X.columns)
)

explanation.save_to_file('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/lime_graph_KNN.html')

#In this file, we're not transforming our X_array into a dataframe, as this type of model doesn't need it. KKN is better train on arrays, so so be it 
#You can also train it into an array, and let everything as in the other files, but you'll receive a warning that KNN wasn't train that way. The 
#model will work equally fine, but you'll have there that warning.

#So let's then interpret the prediction of the KNN model for this passenger, which is pretty positive onu must say.

#On my personal opinion, this model seems to understand much less what is happening or how each variable influences each other. For interpreting 
#the graph we must remember that the KNN model calculated distances, so the values for the features stands for the relative distance from our sample
#to that very feature. For instance if your distance to one feature is negative, that means that you probably don't that feature, and vice versa. 

#We've been working with the passenger 30, but I would like to explain a case that clearly explains how messy this model seems to be. So, if you've 
#arrived till here, please set the index to be 275, and let's take a look to the data. 

#We can see how the model predicts a 100% probability of this sample dying, and if we look carefully we can see that sex_female is positive, so we'
#re in front of a woman. The surprise comes when we realize that the model has given 0.25 more probability of dying to the sample, because she is a woman,
#and her value for this feature, is higher than -0.74, so for this model, women die and men survive. 

#After it we can observe how our lady has a negative value respecting the deck E, what's good, since if deck E=<0.19 we get a 0.45 
#chance of surviving, something quite exagerate on my opinion. However, our sample, altough haven this really good value, keeps dying with a 100%
#probability, what doesn't make much sense.

#The interesting fact is that this model got an 0.83 accuracy when tested, so I can't really explain what has happened here. 

#The most plausible hipotesis is that as KNN is a black-box model, LIME hasn't been quite good, trying to replicate its behaviour, so it has trained 
#a linear or tree model, that didn't quite understand the data, and has therefore given some confuse results.

#For the sake of the project knows that the 30 passenger didn't survive, so altough it has been really close between two participants, the unique model 
#that has guessed the fate of the passenger has been the random_forest. It must be said, that altough both models had known that this was a rich 
#passenger the random forest, would still have won the competition, as it has predicted a 0.51 dying, and 0.49 surviving, while the logistic regre
#ssion a 0.55 surviving, and a 0.45 dying.