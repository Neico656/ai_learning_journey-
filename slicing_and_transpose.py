import numpy as np

arr= np.arange(1,10)

arr=arr.reshape(3,3)

print(arr)

#Slicing diagonals in matrices: The very first number in index [0,0] has the position 0 when it comes to the diagonal.

#Slicing means cutting down some parts of the array, matrices, vector or whatever to be left with one part of them
#We execute this method trough indicating the indexes and how much we want to cut or show

#n --> Shows the exact index of the row or column we are gonna exclusively work with 
#n:-->  Means that from we are gonna pick -n and all rows or columns left all the way to the end 
#:n--> Means that we are gonna pick all the rows or columns till -n without including it
#n:z--> We are gonna select only those rows or columns going from -n to -z, including -n and leaving -z behind

#Transpose just flip transforme the rows into columns and the columns into rows, while with reshaping you give the array the exact form you want

arr=np.arange(12)
print(arr) #An array in one dimension

reshaped_arr=arr.reshape(4,3) #We have turned the one dimensional array into a two dimensional array of four rows and three columns

print(reshaped_arr)

transposed_arr=reshaped_arr.T

print(transposed_arr) #We have changed the array in a way in which the first column, is now the first row and viceversa and so forth

