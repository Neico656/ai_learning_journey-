import numpy as np 

# For creating arrays we have different methods: 
# #First of all we can create an array with the dimensions we want by using: 

# np.(number we want to fill the array with)((shape of the array))  (Rows go first, Columns then)

arr= np.zeros((2,3), dtype=int) #By default we are gonna have floats but we can change that with dtype= data we want to use

print(arr)

#Then somethin pretty used is the np.arange function, that creates an array of the range declared 

arr= np.arange(10) #Array of length ten, that goes from the range 0 to 9--> Range 10

print(arr)

arr=np.arange(1,11,2) #We can also set an specific range and the step we want 

print(arr)

#np.linspace is gonna create an array with the number of elements you type in the third argument. The two first arguments are 
#gonna be the boundaries of your array and the distance between the different values will always be the same.That's why we are
#always gonna work with floats in this type of array, even if they are not necessary as below. 

arr= np.linspace(1 ,9 ,3)

print(arr)

#Another example of np.linspace with proper floats

arr= np.linspace(1,10, 5)
print(arr)

# The method of np.zeros only works with zero and one, so if we want to fill an array with a different we must use: 
# np.full((x,y) , n) --> Where -x is the number of rows, -y the number of columns and -n the number we are gonna full the array with

arr=np.full((3,4), 6)

print(arr)

#A matrix defines a linear transformation in which each column represent a vector

arr=np.eye(5) # The 5 stands for the number of rows and columns the identity matrix is gonna have
print(arr)

arr= np.random.random((3,6)) #It creates a random arrray of float values of the shape defined 

print(arr)