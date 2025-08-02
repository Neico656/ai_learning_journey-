#4

# We're gonna try now with a third model. This model is the KNN, the k-nearest-neighbour, and it will surely obtain worse results as the others two, 
#since it is not adapted to solve this kind of problems where categorical, and nuanced data is included instead of just numeric columns. 

#I use this model, as I want to show that not all the models perform well on all tasks. 

#We're gonna start by fitting this model as we did with the previous ones, that's to say with the labeled columns and the adaptations we need to do
#for this specific model, as we did with the logistic_regression with the threshold, although it wasn't useful, or as with the parameter with the random 
#tree. 

#Hence, how does this model work? The objective of this model is again classifying the data given into a categorie or other. It achieves that by 
#using a dimensional space where it transforms the data given into a point in this space. For instance if person 1= 180 height, 85 weight, its point
#in the space will be (180,85) or (85,180). Then KNN calculates the distance from this point to its neighbours, the neighbours are those which found
#themselves at the smallest distance from the point we're studying. After it, based on the k=n we have set. It will pick n neighbours as reference f
# for classifying the point into one categorie or another. 

# This works great for numerical data as we can observe with this example. I have a dataset of heights and weights of persons, and I want to label them
#into child or adult, so if I were to pick the former point, I would select the closest neighbours that would probably have similar values if my df has enough
#samples. Then the model would see that these people are labeled as adults and would predict that the person we have fed it with is an adult. Being 
#it the correct decision.

#Nevertheless,things wouldn't work that way if we take the classes of the titanic model. Think about the class of the embark town. If our sample 
#is from Southampton, when we're comparing it to Queenstown or CHeerbourg which one is near to our passenger? The answer, there isn't, as categorical
#columns results in categorical values, that are or not are. We can get closer to them. We are them or we aren't 

#Luckily the model does perfom better when it comes to bolean clases, such as sex, as if you are a woman, you distance to other woman is 0, while 
#the distances from men are of 1, and vice versa. 

#After this little introduction we are going to pass to train and test the model as we did with the others

from  sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd 

df=pd.read_csv('/Users/Dani/Programación/scikit_learn/Titanic_model/Preprocessing_folder/Processed_titanic_file.csv')

bins_age=[0,5,15,64,df['fare'].max()]  
labels_age=['baby', 'child', 'adult', 'elderly']
df['age']= pd.cut(df['age'],bins=bins_age, labels=labels_age)


bins_fare=[df['fare'].min(), 10, 30, 100, df['fare'].max()]
labels_fare=['Poor', 'Mid-low-class','Mid-high-class', 'High class']
df['fare']= pd.cut(df['fare'],bins=bins_fare, labels=labels_fare)


X= pd.get_dummies(df[['pclass','sex','age','fare','embark_town','family members', 'deck']],
                    columns=['sex','embark_town','deck','age','fare'])

y=df['survived']

X_train,X_test,y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=76) 

model=KNeighborsClassifier(n_neighbors=5) #The number of nearest points we take as reference

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

print("Classification Report: \n", classification_report(y_test, y_pred))
    
#We should expect bad performance from this model and in this case, we converted the age and fare columns into categorical columns, so it won't be
#easy for it to properly address this data
    
#Ok these results have completely stunned me. I didn't expect by no means that it would get a 0.838 of accuracy with everything against it. Let's
#tune now the model, so that it can receive numerical data, and better grasp the distance between samples. 

#A thing that we're gonna also do is to scale our data, so that the distances between different points do not depend on unique columns with extreme 
#values, for instance the fare class. For a better understanding, if we don't scale the distances between points, the model could believe that a 
#man and a woman were much similar than someone who paid 7 pounds compare to someone that paid 10, given that the difference between the first two examples is from one, while the difference between the second is from 3. 

from sklearn.preprocessing import StandardScaler

#We take in the original df with the original columns 

df=pd.read_csv('/Users/Dani/Programación/scikit_learn/Titanic_model/Preprocessing_folder/Processed_titanic_file.csv')

scaler= StandardScaler()

X= pd.get_dummies(df[['pclass','sex','age','fare','embark_town','family members', 'deck']],
                    columns=['sex','embark_town','deck'])

X_columns= pd.get_dummies(df[['pclass','sex','age','fare','embark_town','family members', 'deck']],
                 columns=['sex','embark_town','deck']) #If you haven't reach yet the KNN_lime file, don't worry about this. Just keep going.
#Nevertheless, if you come from the KKN_lime file, now we're gonna store this variable with the names of the columns and bring it to our file, 
#and use it for the columns of our X dataframe.  


X=scaler.fit_transform(X)

y=df['survived']

X_train,X_test,y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=76) 


model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

print("Classification Report: \n", classification_report(y_test, y_pred))


#Ok and now I'm not getting any improvement. Something I didn't expect, but life is as it is, and I won't try to enhance it further, as I already
# tried with the most common technique, and I don't want things to get messy so let's call it a day, pick the better model, and make some cool graphs and that stuff to show how our model perform, and why it takes the decision it takes.

#Now i'm going to save the model as the X variable, so that I can use it in another file, with the joblib library:
import joblib

joblib.dump(model, '/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/KKN_model.pkl' )
joblib.dump(X, '/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_KNN_model.pkl')
joblib.dump(X_columns,'/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_variable_for_KNN_lime.pkl')