#8

import joblib
import shap
import numpy as np

model = joblib.load('scikit_learn/Titanic_model/Joblib/Random_forest_model.plk')
X = joblib.load('/Users/Dani/Programación/scikit_learn/Titanic_model/Joblib/X_titanic_random_forest.plk')
X = X.astype(float)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


shap.summary_plot(shap_values[:, :, 1],X)
#As you can see the code looks pretty much the same for both Shap representations. Nevertheless, there are two little differences between these two.

#First of all is that shap_values with logistic regression only has 2 dimensions while in the shap_random_forest the number of dimensions goes up 
#to 3. Why's that? The cause lies on how each model completes its prediction, as while the logistic regression model just calculates the overall 
#probability for the positive class(1), surviving in this case, and if this probability or value don't go up of 0.5 it will predict the negative 
#possibility, dying. 

#Otherwise, the random forest predicts the probability for both events, so you just have all the samples for each features, 2 dimensions, and then 
#you add the probability for each class, so you end with a 3 dimensional array.  shap_values.shape == (n_samples, n_features, n_classes)  

#Then at the graph you'll notice the values goes between -0.2 and 0.2. These are not probabilities, but the log-odds. If you don't understand, don't
#worry me neither, just take note that if when we add them they add more than 0.5 our passenger survives and if not not. 

#And to finish if we analyze the graph we can see how the model gives much importance to being a male in the third class, or a female either in the 
#first or second category. Then we can see that other features such as being a baby, not having too much family members was indeed a positive feature
#for the passengers. Not being an adult was also important, and that's it there is no much more data to interpret. 

