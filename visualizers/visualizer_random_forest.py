#6

#As I have seen that making the visualizer for the logistic regression model was quite easy, I've decided to do the same for the random_forest_model.
#The visualizer thing can't be done with the KKN model as it doesn't weights the variables, study them, or give them any importance. It just 
#arrives and apply a formula. 

#Nevertheless, the visualizer of the random forest, works a little bit different. It doesn't give a coefficient to each column depending how much 
#it has influenced, but give rather an importance to each one, depending on how much it has helped to take the right decission during the training. 

#The sum of all importances will give as a result 1. Pretty much the same thing as the logistic_regression but with that little detail.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib 

model= joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/Random_forest_model.plk')
X= joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_random_forest.plk')

features= X.columns
importance= model.feature_importances_  #With random forest we don't need to put [0] as it will just return a LIST of importances each one according
#to one column of the original df

df=pd.DataFrame({'Features': features, 'Importance': importance}) #No need for absolute values here
df=df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(df['Features'], df['Importance'], color='cornflowerblue')

plt.xlabel('Importance value')
plt.title('Features importance')

plt.gca().invert_yaxis()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

plt.savefig('/Users/Dani/Programación/scikit_learn/Titanic_model/Graphs/Random_Forest_visualizer.png', dpi=300)

plt.show()
#If you've seen the logistic regression graph you may notice that these two are more or less similar. Both give more importance to the sex and the class
#and then to the age and the fare of the passenger, what is probably the most common sense thing. 

#There are obviously, some differences. For instance the random forest doesn't give much importance to elderly or rich people, maybe 
#because it has already given more importance to the fare poor and child age columns, completely inverse to the logistic regression model. 
#At the end, these are just unimportant elections, since it doesn't matter if you give more or less importance to contrary variables, as long as, 
#you give the same whole importance to both. 
