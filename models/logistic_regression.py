#2

# Training and using models is honestly really simple and straightforward when you already kmow the functions and code to use. 

#First of all, know that that we're going to use the scikit_learn library to make our predictions. Nevertheless, the root model of scikit_learn
#don't include all the models and functions we need, so we'll need to pick specific functions and even model from some concrete parts of scikit. 

#Here beneath, we can see how we're picking the train_test_split function, our model LogisticRegression, among others.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import pandas as pd



df=pd.read_csv('/Users/Dani/Programación/scikit_learn/Titanic_model/Processed_titanic_file.csv')

#Something important to know is that our model can only understand numerical value, so we'll transform our categorical columns into numeric ones 
#using the function get_dummies. This function works by selecting all the unique values of a categorical column and creating new columns refering to these.

#For instance, if I have a column of the days of the week, called week days, the model will create seven diferent columns: week days_Monday, 
#week days_Tuesday, week days_Wednesday... Then it will fill the entire columns with 0 except, from these rows in which the value Monday, Tuesday
#or whatever depending of the column is True. Then it will plot a 1. 

#Note that inside get_dummies, I've writted all the (unique) columns of the df, and then beneath other columns in the argument columns. I've done it
#thereby, since the columns we put into the method columns if we call it, are those that are going to be modified, the others not. You could be wondering: 
#Ok why to use an method? Why not to do it directly? The answer is that it is much more easier and simplier this way to store all these values into a variable.

#Otherwise, I would have to store the 'dummied' columns into a variable and then join this variable, with the other columns, that should also have been 
#stored in a different variable, and merge both into X. So this way is much easier, as we just tell our programm: Pick all the columns I put here 
#get the dummies of these in the method columns, and store everything into X. Is that not way easier and elegant??


#After this we have the X and y variables. When we're training a model, we have two tipes of data , the first one is the data, we want our model
#to use to make predictions and then the data corresponding to those predictions we want our model to make. Thereby, X will be the data upon which 
#our model must realize predictions, and Y will store the data that represents these predictions we want our model to do. 




X= pd.get_dummies(df[['deck', 'pclass','embark_town','sex','family members', 'age', 'fare']], #Try to include the column alive and see what happens and why. Response beneath
                      columns= ['deck','embark_town', 'sex'], drop_first=True)

y=df['survived']

#Then, it's time to split our data into two different sets, the first one will be the data our model will see during training, and then the test data
#is this that's going to be shown to our model when the time of the test come. Train size corresponds to how much of our dataset we want to use 
#to train, and how much we want to use to test. The normal test_size is 0.2, that's to say. 80% for training 20% for testing. The random state just 
#refers to the random sequence in which the model will select the values to train, as it won't pick the first rows for training, and the last rows for 
#testing, it will pick them randomly, so to obtain always the same results we'll set a random_state.

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=32 )

#For this specific test, we're using the LogisticRegression model. For all these ml projects it is important to grasp that a model is only an algorithm 
#that follows an specific concept or functions to predict results, or uncover patterns. For instance the LogisticRegression model, is a model that 
#come to its conclusions by just using the logistic regresssion formula. It is formed by the sigmoid function and a linear combination of the features. 

model=LogisticRegression(max_iter=1000) #We call the model and give it 1000 as limit of iterations, since these are the attemps the model its gonna
#have to 'draw a kind of strategy' by giving more importance to ones or others parameters. 

model.fit(X_train, y_train) #We train our model with the train_data

y_pred=model.predict(X_test) #We take the data test and we say our model to make its predictions, which we are gonna store in y_pred

print("Accuracy: ", accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#The confusion matrix is just a matrix that compares the real data with the model 
#predictions, in order to do that, in one column we find the real data in another the predictions and then we compare. 

print("Classification Report:\n", classification_report(y_test,y_pred)) 

#If we were to include the column alive, our model will get a 100% score, not because it understands that the column alive refers to if someone survived or not, 
#but because during its study of the data it has realized that it exists a correlation coefficient of 1 between these two columns, so they must 
#represent the same. Don't think that this model understands text or string data. 

#These are the results we get with the first attemp:
#Accuracy:  0.7988826815642458
#Confusion Matrix:
# [[93 15]
# [21 50]]
#Classification Report:
#               precision    recall  f1-score   support

#           0       0.82      0.86      0.84       108
#          1       0.77      0.70      0.74        71

#    accuracy                           0.80       179
#   macro avg       0.79      0.78      0.79       179
#    weighted avg       0.80      0.80      0.80       179


#Altough the results are not bad, we're gonna try to improve them a little bit, hence we're gonna make things easier for the model.
#When a model confronts new data, it doesn't know anything about it, and it needs some training, to stablish patterns and relationships that are 
#relevant for the final output. For instance, one really important relationship is the connection between sex and pclass that deeply influences the final output.

#As an example, the probabilities of surviving will radically change depending if the subject of study is a men going in the first or the third class, 
#or a women going in second class, versus a man also in second class. We need our model to grasp this relationship, we want to make it comprehend 
#how much these two variables together could change the faith of a person. Thus, we 're gonna make it easier for him to see this pattern, and 
#we're gonna merge both columns 

#Thereby the model will see that a man going in the third class almost always dies,while a rich woman probably survived. If we don't do it 
#the model when it saw a man dying it could think that it was because of any other variable such as the embark town or the deck, this way we're
#joining these two probabilities alltogether

#Let's then join this two columns

df['sex_pclass']= df['sex']+'_'+df['pclass'].astype(str)  #We first transform the pclass column to string as we can't add an integrer to an stringç

#Now, we'll change both to integrers with pd.get dummies, and after it, repeat the process and train again the model with the new data. 

#In this case, we'll not use the deck column, as it has been mostly filled with values that propably don't represent the real world. It has been filled with three
#different decks, while there were 9 different decks in the titanic. So sometimes, less is more and we'll supprime that column, so that our model don't 
#get confused by it. 

#From here the same process as always, we declare X and Y, convert string columns to integrers, we fit the model, put it into action and we compare 
#results.

X= pd.get_dummies(df[['sex_pclass','age','fare', 'embark_town','family members']], 
                   columns=['sex_pclass','embark_town',] )



Y= df['survived']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=14)


model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test,y_pred))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

print("Classification report: \n", classification_report(y_test, y_pred))

#We see, we have slightly improved the results of the model, but now I have another idea to make it go a little bit further. 
#The logistic regression projects, as I said before, only returns True or False values, but it isn't always, in fact, hardly never, 100% sure of 
#its decission, so it uses thresholds to choose one or another option. For instance, if it is 0.69 sure that 0 is true and 0.31 sure that 1 is true
#it will choose 0. 

#The thing here, is that the threshold is by default set at 0.5. But our model is performing better at guessing who died than who survived, 
# so maybe, if we set down this threshold a little bit it will predict better who survived. For instance for those who it is not really sure, 
#if they died or not, let say 0.45 surviving against 0.55 dying, as it tends to fail when predicting survivors, we're gonna balance the prediction 
#slightly towards them. We're gonna prove different thresholds, to see which one works better. 

#For knowing when to increase or when to reduce our threshold, we need to understand two key principels: Precission and Recall, which give us a grasp
#of how our model is predicting the results. Precission stands for the times our model said one person died or survived and in fact it was true. 
#For instance, if it said that 100 people died and FROM these 100 people only 87 died it would have a 0.87 prediction rate. Note that it can also be
#that it has guessed 87 deaths, but that there are 130, then it would still having 0.87 precission, no matter how much cases it lefted.

#When it comes to Recall it is the pourcentage of real cases the model has guessed. As an examle, if we have as before 130 cases and the model predicts
#87, our recall rate would be of only 0.66. 

#Therefore, when it comes to changing the threshold value, we have to change it accordingly with some rules. Genereally, if your precision rate is higher,
#than the recall rate you may increase the threshold, as if the model has made 1 guess and it was correct, the precision will be of 1 but the recall will 
#be near to 0. Otherwise, if the recall rate is much higher than the precision rate, you may reduce the threshold for your variable, as the model 
#may have correctly predicted the whole number of survivors, but it can also said that everyone survived, so its precision rate will be much smaller. 

#That said, let's dive into work: 

y_probs= model.predict_proba(X_test)[:,1]#This line is going to return a whole array of the probabilities each passenger had of being alive or dead. 
#So from these array, we're gonna first pick all the rows [:] and then only select the column for the probability of surviving [1], so we then have
#[:,1]
threshold=0.5 #We define our threshold 

y_pred_custom=(y_probs >= threshold).astype(int) #This (y_probs >= threshold) will return a boolean Series, which we'll later convert to integrers
#having made that way our custom predicting values, as all probability = or > than 0.45 will be a 1. 

#Now we check the model:
print("Accuracy: \n", accuracy_score(y_test,y_pred_custom))
print("Confusion matrix: \n", confusion_matrix(y_test, y_pred_custom))
print("Classification report: \n", classification_report(y_test, y_pred_custom))

#So I've tried with different thresholds and the better doesn't get better At least we tried. 

#In conclusion our logistic regression model has an accuracy of 0.082 if set to a 0.5 threshold, joining the pclass and sex columns and removing the
#deck category, what didn't influence that much on the final result. That's a real good result for a first try.

#I'm gonna implement into this model another feature I implemented into the random_forest_classifier, which is to label the columns age, and fare. 
#It has worked pretty well on the other model, so now let's try it on this one and see if we achieve to reinfore relationships in this model such as 
#the younger and the richer, the more probabilities it has of surviving, while the older and the poorer the most probable it is that they are dead. 

#Here is the code, I literally pasted it from the other file. 

bins_age=[0,5,15,64,df['fare'].max()]  
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

import joblib #Now, we're gonna save the model, and X variables to use them in the visualizeer with the joblib library.

joblib.dump(model, '/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/Logistic_model.pkl')  #We save the model as Logistic_model.pkl
joblib.dump(X, '/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_logistic_model.pkl')