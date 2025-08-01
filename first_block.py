import numpy as np
#1.	Crea una matriz 3×3 con números del 1 al 9.
#Usa np.arange() y reshape().

#📌 Objetivo: creación y forma.

arr= np.arange(1,10)

np.random.shuffle(arr)  #np.random.shuffle randomizes the order of an array

result=arr.reshape(3,3)






#	2.	Invierte las filas y columnas de una matriz.
#Haz transposición (.T) y volteo ([::-1]).
#📌 Objetivo: slicing y transposición.
#Slicing diagonals in matrices: The very first number in index [0,0] has the position 0 when it comes to the diagonal. For instance:

#We are gonna take a random matrix of 24 values 

arr= np.arange(24) #As right now it only exists we are gonna make it tridimensional to make things more exciting

np.random.shuffle(arr)

x=arr

tridimensional_arr= x.reshape(2,3,4) #Now our array is divided into two blocks of 3 rows by four columns


 #We check everything is alright

#Now we have properly built our array we are gonna slice it to obtain the value -15. OK i have just realised that the shuffled 
#function will continouisly shuffle the array changing the position of every value, and thus making my slicing unefective, so 
#we'll just create an array by ourselves

arr= np.array([[[0,7,12,20],
                [23,17,11,2],
                [19,22,13,8]],

                [[9,15,3,5],
                [21,1,18,14],
                [4,10,16,6]]]
                )

#Now we have our array to slice we are gonna slice it step by step just we get till 15. 
#First of all we are gonna slice our array into two different blocks and pick that one in which 15 is founded
#Note that the first number correspond to the "blocks", the second to the rows and the third to the columns

arr[1,0:,0:]



#after it we are gonna cut down the first column which is not necesary 

arr[1,1:,0:]

#Then we'll cut the last column also 
arr[1,0:,1:3]

#After this we'll gonna go all the way down till two numbers

arr[1,0,1:3]

#And in order to finsih

result=arr[1,0,1]
print (result)

#Transposing is really really easy to both, to understand and to realize, so, we'll just solve it in a pair of lines of code
result=arr.T

print(result)

#SO basically what is happening here is that our blocs of arrays, 2, are being turned into the numbers of columns and vice versa
#while rows in both cases remain the same


#	3.	Genera una matriz identidad de 4x4.
#Usa np.eye()
#📌 Objetivo: entender la base de las transformaciones lineales.
#An identity matrix is a matrix that when multiplied by another matrix of its same dimensions give as a result the second matrix.
#In other words, multiplying any matrix of given dimensions by an identity matrix of the same dimensions would be the same as
#multiplying a number by one. By dimensions I mean 2x2, 3x3, 4x4. 

arr=np.eye(4)
print(arr)






#	4.	Multiplica una matriz 2x3 con una 3x2 y comprueba el tamaño del resultado.
#Usa np.dot() o el operador @.
#📌 Objetivo: práctica de producto matricial.
matrix1=np.array([21,1,4,11,5,20]) #We create our matrices of the elements required. 6 in this case
matrix2=np.array([15,73,19,22,14,9])

matrix1=matrix1.reshape(2,3)
matrix2=matrix2.reshape(3,2)

final_matrix=matrix1 @ matrix2 #Here we're using the -@ operator for the multiplication of matrices 

print(final_matrix)
print(final_matrix.shape)

final_matrix= np.dot(matrix1,matrix2) 
print(final_matrix)
print(final_matrix.shape)