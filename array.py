import numpy as np

arr=np.array([3,2,8,10])

print(arr)

print(arr.shape) #it gives you the length of the array

# for multidimensional arrays:
arr= np.array([[1,2,3], [4,5,6]]) #An array of two dimension: X and Y

print(arr) #This returns the array in the shape of a rectangle with 2 rows and three columns

print(arr.shape) #This returns (2,3) because the array has two rows (number of arrays we have) and three because of the number
# of elements in each array. 

# Note that all arrays must have the same number of elements, if not it will return Error. 

# arr= np.array([[1,2,3,4],[5,6,7]])<--This woud return error

# Different methods: 

print(arr.size) # Number of total elements within aLL "subarrays" whithin the array

print(arr.ndim) #Number of dimensions

# For adding and deleting elements from an array, we are gonna use two methods: 
# For adding: np.append("object we want to add something to", "data we want to add")
# For deleting: np.delete("object we want to delete something from", "data we want to delete")

# Note that arrays are INMUTABLE, so when we are using these methods we are not truly changing the array but creating a copy
# with the result we want. The original array stays the same. Example: 

arr=np.array(["Helsinki", "Singapore", "Moscow", "Sidney"])

print(np.append(arr,"Bejing")) #This will only print a copy of the original array but now with the new value. "Bejing"

#If we want to truly change the array, we shall declare this new array:

arr=np.append(arr,"Bejing") #Now we have a proper array and not only a copy from the original one

#For deleting is kinda the same thing, except that with this method you must not specify the data, but the index of the data 
# you want to delete. For instance: 

arr=np.array(["Augusto", "Tiberio", "Domiciano","Adriano"])


print(np.delete(arr, 1)) #Here we are deleting the name of Tiberio from the array 

arr= np.delete(arr, 1) #With this wwe definetly change the array 

