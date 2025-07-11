import numpy as np 
from numpy import linalg as LA

#🔹 BLOQUE 2: Álgebra lineal aplicada con NumPy

#Aquí conectas álgebra con código. Esto es lo que necesitas para entender cómo aprenden los modelos.
#	5.	Dado un vector v = [3, 4], calcula su norma (magnitud).
#Usa np.linalg.norm(v)
#📌 Objetivo: entender la longitud de vectores.

#We have been given a vector [3,4], and we have been asked to calculate its length, norm. The formula of the length of a vector is given by ||v||= sqrt(î^2+j^2)
#To completely understand this formula we have to first remember that any vector of a bidimensional plane can be created by just streching or extending others two vectors 
#and adding them. The two main vectors we use to form all the others are î (1,0) and j (0,1). 
#Knowing that we can then pass to visualize how our vector is formed. The first thing to do is to stretch or lengthen the two vectors so that they meet the longitudes of 
#the vector given. 

#Knowing that our vector moves three units into the left, we need our vector î, (1,0), to be scaled by a factor of three. Then with our vector j will need to become 
#four times bigger to meet the requisites. Having the two vectors we need is time to add them and get our wanted vector. To add the vectors you just have to pick one vector,
#and a piece of paper if you needed it, and move its tail to the tip of the other, and you will realise that the tip of the vector you have moved will end, just in the 
#same spot, in which our vector should end, that's to say [3,4]

#This may seem to have nothing to do with the problem but if you have properly drew or imagined the process you will notice that our vectors, î and j, and the third one 
#form a triangle, more specificly a rectangle triangle, in which the hipotenuse corresponds to our output vector, and as the theorem of Pitagoras sais c=sqrt(a^2+b^2)
#that's to say ||v||= sqrt(î^2+j^2).
#  So we'll do all that in just one line of code using a method of numpy that computes this formula. 
arr=np.array([3,4])


vector_length=np.linalg.norm(arr)
print(vector_length)


#	6.	Normaliza un vector (haz que tenga longitud 1).
#Divide el vector por su norma.
#📌 Objetivo: muy usado en IA para evitar que valores dominen por magnitud.
#Normalizing is the process trough which we have a vector of length -n and we shorten it till it reaches a longitude of 1, without changing its direction.
#The formule to realize this process, is really simple and easy to understand. 
#The formula goes the next way: v/||v||, where v is the vector we are given and ||v|| its magnitude, 
#I like to see a vector in this case not as a set of number, but more as an arrow pointing in one direction without a magnitude defined, 
#it wouldn't be until we multiply by a magnitude this vector that it will be defined, then we divided it by five and we obtain exactly the same 
#result we are looking for, the same vector of before, but with a longitude of 1. 

# v (just an arrow in a direction)* longitude -n/longitude -n--> v*longitude of 1. 
#Now that we understand the concept we can normalize our first vector, [3,4]
def normalizer(vector,norm): 
    
    result=[i/norm for i in vector]
    
    return result 

vector=[3,4]
vector_length=5


new_vector=normalizer(vector,vector_length)



print(f"The vector of magnitude one pointing is the same direction as [3,4] is {new_vector}")

#	7.	Crea una matriz 2x2 que escale vectores por 2 y 3 en los ejes x e y. Aplícala al vector [1,1].
#📌 Objetivo: visualizar transformaciones lineales.

matrix=np.array([[2,0],[0,3]]) #We create our matrix of 2x2 in which the values of î and j are stored. The first column belongs to the vector î 
#and the second column to the vector j. What this matrix is representing how our vectors î and j have changed as we have changed the grid. 
print(matrix[0:,0])
#So right now what we have to do, is to pick these two vectors, scale them by the proportion we have been given [1,1] and add them to get the final 
#vector. Since we must remember that any vector of a two dimensional plane can be formed by adapting the vectors î and j by the scalars of the vector
#its x and y values, and then adding them. 
#The formula then goes the next way-> x[î] + y[j]= [v]
vector=np.array([1,1])
print(vector[:1])

def linear_transformation(matrix,vector): 

    result=(vector[0]*(matrix[0:,0]))+(vector[1]*(matrix[0:,1]))
    return result 


res=linear_transformation(matrix,vector)
print(res)

#	8.	Simula una rotación 90° antihoraria de un vector [1, 0] usando una matriz de rotación.
#📌 Objetivo: aplicar rotaciones con matrices. Esto te prepara para redes neuronales convolucionales o visión por computador.
#When working with linear transformations we may encounter some cases in which we don't have the position in which our vector î and j have landed
#so we can't define a matrix and multiply it by a pair of scalars. It is in this situation in which the rotation matrix becomes useful, always that
#the basis has only ROTATED around the origin and that we have that rotation angle

#In those cases the formula for getting the resultant vector of this operation is the same as before [x2,y2]= [[a,c],[b,d]] [x1,y1], from where we only
#have x1 and y1. However, if we play a little bit with the formula we'll be able to get a formula we can work with. As we have the angles and the lenght 
#v given that we have its coordinates the first thing to do is to find a way of introducing them in the formula.

#So we'll start changing our formula by (x1,y1)--> x1 is equal to the lenght of the vector î and y1 is equal to the length of the vector j. However
#we don't have this informatio, but luckly we know that being the vector î and j the sides of the rectangle triangle formed by v, î, and j, we can 
#guess them applying the proportion between them and v determined by sen and cos of ø. 
#||v||=h
#In conclusion: (x1,y1)=(||î||,||j||)= (h*cos(ø),h*sin(ø)))

#Note that h always stay the same since we are just rotating the vector, not transforming it.
#Now that we have set clear this part we can pass to work with [x2,y2] from which we deduce, that it also must be equal to ([h*cos(ø+ß),h*in(ø+ß)]). We add the angles, because we aren't doing the rotation in just one step but in two so we need to express this numerically by the sum. 

#Doing a little bit of math we come to a final formula, that we can solve with the data we have--> [x2,y2]= [[cosß,senß],[-senß,cosß]] [x1,y1]

vector= np.array([1,0])
angle_degr=90 
angle_rad= np.radians(angle_degr)


def rotator(vector, angle):

    x2_y2= np.array([[vector[0]*np.cos(angle),vector[0]*np.sin(angle)],[vector[1]*(-np.sin(angle)),vector[1]*np.cos(angle)]])
    

    result=np.array([x2_y2[0,0]+x2_y2[1,0],x2_y2[0,1]+x2_y2[1,1]])   
    return result



print(rotator(vector,angle_rad))
#	9.	Comprueba si dos vectores son linealmente dependientes.
#Por ejemplo, si a = [2, 4] y b = [1, 2], son múltiplos → dependencia.
#Usa np.linalg.matrix_rank()
#📌 Objetivo: entender el span y la base.
#A vector is lineal dependent of another if for some number k a1/b1=a2/b2, that's to say if they are directly EQUALLY proportional in both axis. 
vector_a=np.array([2,4])
vector_b=np.array([1,2])

vectors =np.vstack([vector_a,vector_b])

def linear_dependency(vectors):

    output= np.linalg.matrix_rank(vectors)
    
    if output>1: 
        output=print(f"The vector {vector_a} and {vector_b} are linearly independent")
    else: 
        output= print(f"The vector {vector_a} and {vector_b} are linearly dependent" )
    return output

linear_dependency(vectors)
#	10.	Calcula el determinante de una matriz 3x3.
#Usa np.linalg.det()
#📌 Objetivo: saber si una matriz es invertible.
#the determinant of a matrix represents simply the proportion by which all the elements of a sheet or space
#are stretched or shortened. In a two dimensional space the determinant tell us how many times the area of
#a square has elarged and in a tridimensional space it goes for the volume. The formulas to obtain the determinant
#of a matrix are quite tricky, so just by understanding what the determinant represents we can apply the for
#mula trhough some lines of code

matrix= np.array([  [12,5,6],
                    [7,4,11],
                    [3,10,6 ]])
print(np.linalg.det(matrix))