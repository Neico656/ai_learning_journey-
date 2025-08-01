#10

#So the other the process for the models is pretty much the same, we just load the different models, and X altough we could use the same as it is 
#indeed the same, but as it is already created, let's use it. Nothing more to say about this, but that we're going to be using the same sample to 
#compare results and analysis between models.


import joblib
import pandas as pd
import lime
import lime.lime_tabular

model=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/Random_forest_model.plk')
X=joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_random_forest.plk')

X_array= X.values if isinstance(X, pd.DataFrame) else X



explainer= lime.lime_tabular.LimeTabularExplainer(
    training_data=X_array,
    feature_names= X.columns,
    class_names=['Survived','Died'],
    mode='classification'
)

idx=30

explanation= explainer.explain_instance(

    X_array[idx],
    lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns)),
    num_features= len(X.columns)
)

explanation.save_to_file('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/lime_graph_random_forest.html')

#This model for instance is predicting the death of the passenger, as we can see how it gives less importance to the fact of being on first class. 
#Then being poor, which is not the case, but it is an inevitable mistake, is not that important as it should have been, while having embarked in 
#Cherbourg is a stronger indicator of dying, 0.03, for the model than the first class is for surviving, 0.02. Nevertheless it must be said, that 
#if the passenger had been correctly labeled as rich the model would have predicted its survivance, as the logistic regression did. 