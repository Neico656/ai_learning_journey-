#3

#We don't really know which type of model will better fit for our dataset, thus we need to test different types of them. 

#In this file, we're going to try with the random forest classifier. This "forest" is formed by multiple different decission trees that analyzes
#each one a small part of the dataset, and then come together to a conclusion. For those who don't know, a decission tree is a type of algorithm 
#that gathers pattrons from the data and stablishes a road map for classifing and predecting data, that follows specific patterns. 

#The problem with unique trees are that they tend to overfitting, or in other words, that they tend to overcomplicate, so that their algorithm 
#is perfect, and don't make mistakes. That wouldn't be a problem if that didn't entail more messy trees, makes them hyper-sensible to slight varia
#tions of data among others. 

#In these cases, the random forest kicks in. The random forest select a subset of the dataframe for each tree and then in each node of the tree 
#it looks for an specific part of the column, so you end with multiple trees that cover the entire dataset, but each one specializes in an specific
#part of it. 

#It is obvious that a single tree doesn't know anything about the whole dataset, but the forest yes, as it comes to a conclusion, based on what 
#the majority of trees have considered to be the correct response. 

#So we'll import it and see how it performs

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import pandas as pd

df=pd.read_csv('/Users/Dani/Programación/scikit_learn/Titanic_model/Preprocessing_folder/Processed_titanic_file.csv')

X= pd.get_dummies(df[['pclass','sex','age','fare','embark_town','alone','family members', 'deck']],
                    columns=['sex','embark_town','deck'])

y=df['survived']

X_train,X_test,y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=29) #Random states don't have to coincide, as they carry complete
#different tasks. 
#In train_test_split it determines which data goes into test or train, while in the random forest it controls how trees are formed according to 
#random paterns.


model=RandomForestClassifier(n_estimators=1000, random_state=54) #N_estimators stands for the ensemble of trees.

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

print("Classification Report: \n", classification_report(y_test, y_pred))

#It's time know to also improve our model with some changes we made in the logisitc regression.py, and some others. 

#First thing we're gonna change is the hyperparameter of the decision trees, that's to say, some features that are gonna improve the general performance
#of our model. If we want to understand this, we need before to know how a decision tree works, and the problems that may affect its performance. 

#A decission tree takes in data and then it tries to split in into different categories by posing questions that are answered with True or False.
#Thereby the data is classifies and pass trough different decission nodes, till the model elaborates a final answer. The problem with this system 
#is that models are trained to obtain max accuracy so decision trees, may overfit the question or categories for each piece of data in ordert to 
#give its best performance. 

#The results of this attitude are classes such 'age'=23 embark town= Southampton  fare=7.35 result dead. And this is correct but it is no real pattern
#so when the model receives new data it won't know where to classify it and therefore, which the outcome may be, as we won't find another passenger
#of 'age'=23 embark town= Southampton  fare=7.35  and if we do, they won't be necessarily dead. 

#We want our model to generalize, to be able to face new data, to come to the conclusion that male third class, alone= dead probably. 
#So we're going to play with these parameters to oblige our model to generalize. 

#Then something that may also occur, is that the model concentrates itself more into the bigger categories, as there lies the more possible "points"
#for increasing accuracy. Nevertheless, we want our model to equally focus on the survivors, even if they are less than the deaths. In order to do this 
#we include the class_weight argument in our model.


model= RandomForestClassifier(
    n_estimators=200,  #More trees, lead to a bigger variety and a bigger knwoledge about the dataset
    max_depth=5,    #Each tree can't profundize more than 5 layers of questions so that the categories don't become too specific
    min_samples_split=5, #In order for a node to divide into other two it needs to at least have 5 samples
    min_samples_leaf=2, #A leaf is a final node, and it needs to have at least two values. 
    random_state=42,
    class_weight='balanced'
)
 


#After this we're just going to do a little bit of feature engineering with the columns pclass and sex together as we did in the logistic regression, and
#then, we're gonna include another more feature, that is binning the fare and age column so that the model can easier recognize pattterns such as 
#younger and richer=alive, older and poorer=dead. 

df['sex_pclass']= df['sex']+'_'+df['pclass'].astype(str)

bins_age=[0,5,15,64,df['fare'].max()]  #We use the survival_rate png to group the ages approximately by survival rate.
labels_age=['baby', 'child', 'adult', 'elderly']
df['age']= pd.cut(df['age'],bins=bins_age, labels=labels_age)


bins_fare=[df['fare'].min(), 10, 30, 100, df['fare'].max()]
labels_fare=['Poor', 'Mid-low-class','Mid-high-class', 'High class']
df['fare']= pd.cut(df['fare'],bins=bins_fare, labels=labels_fare)

X= pd.get_dummies(df[['sex_pclass','age','fare', 'embark_town','family members']],
                  columns=['sex_pclass','age', 'fare', 'embark_town'])

y=df['survived']

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=11)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))


#We're gonna save our model and X variable so that we can use it in the visualizer file
import joblib 

joblib.dump(model, '/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/Random_forest_model.plk' )
joblib.dump(X,'/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_variable_random_forest.plk')