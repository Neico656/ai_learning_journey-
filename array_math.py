import numpy as np
#Remember that arrays can describe vectors. 

#In order to make operation with arrays we can use the next easy methods

arr1= np.array[[2,4],[7,10]]

arr2= np.array[[4,6],[3,5]]

#Here we have our two bidimensional arrays, and right now we are gonna play around with them. 
#We can use the raw python simbols to make some basic operations such as add (+), substract (-), multiply (*), and divide(/).
#Numpy also offers the tools to achieve this tasks

result= np.add(arr1,arr2)
print(result)

result= np.substract(arr1, arr2)
print(result)

result= np.multiply(arr1,arr2)
print(result)

result= np.divide(arr1, arr2)
print(result)

#The dot product or scalar product is obtained thereby
result=arr1.dot(arr2) #It multiplies each number with its correspondant and then add the results alltogether