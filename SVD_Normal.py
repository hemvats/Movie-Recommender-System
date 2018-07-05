import time
start_time = time.clock()
import random
from random import randint
import numpy as np
import pandas
from pandas import DataFrame 
from numpy import linalg as LA
from numpy import mat
from scipy.sparse import csr_matrix
from math import sqrt
from sklearn.metrics import mean_squared_error
import copy
r=1509
c=2072
matrix_original = np.zeros(shape=(r,c), dtype=np.float)    #read file in matrix

# test_row=[]
# test_col=[]

# matrix=[1,1,1,0,0,3,3,3,0,0,4,4,4,0,0,5,5,5,0,0,0,0,0,4,4,0,0,0,5,5,0,0,0,2,2]
# matrix=np.reshape(matrix,(r,c))


input_file= open("ratings.txt", "r")
lines= input_file.read()
total_ratings=0
for line in lines.split("\n"):
	  i=1
	  if len(line)!=0:
			for vals in line.split(" "):
				  if(i==1):
						row=vals
						# test_row.append(row)
						i+=1
				  elif(i==2):
						col = vals
						# test_col.append(col)
						i+=1
				  else:
						val = vals
						i+=1
	  matrix_original[int(row),int(col)] = val
	  total_ratings+=1

# Testlist = [ ( random.choice(test_row), random.choice(test_col) ) for k in range(3000) ]
# for i in range(0,len(Testlist)):
# 	matrix[int(Testlist[i][0])][int(Testlist[i][1])]=0

M_original = copy.deepcopy(matrix_original)	  
avg=[]													#Calculating avg of each user for making it zero centred matrix
for i in range(0,len(matrix_original)):
	temp_avg=0
	count=0
	for j in range(1,len(matrix_original[i])):
		if(matrix_original[i][j]>0):
			count= count+1
		temp_avg = temp_avg+matrix_original[i][j]
	if(count>0):
		avg.append(temp_avg/count)
	else:
		avg.append(temp_avg)

for i in range(0,len(matrix_original)):					#Subtracting avg of each user from its rating row if greater or equal to zero
	for j in range(1,len(matrix_original[i])):
		if matrix_original[i][j]>0:
			matrix_original[i][j]=matrix_original[i][j]-avg[i]
		else:
			matrix_original[i][j]=0						#Else leaving it as it is 


M = copy.deepcopy(matrix_original)						#Deepcopy because normal copy would  =just create a new reference
M_T=M.T 												#Calculating  Transpose of  matrix after making it zero centered
for_V=np.dot(M_T,M)										#V vector is the eigen vector of m * m_transpose
eig_values,eig_vector_V=LA.eig(for_V)					#calculating eigen vector and Values for Sigma and V

temp_index=np.iscomplex(eig_values)						#calculating all indices for which the eigen values are complex

for i in range(0,len(temp_index)):						#Removing all those values first as we dont want complex values for our subsequent calculations
	if((temp_index[len(temp_index)-i-1] == True)):		
		eig_values= np.delete(eig_values,(len(temp_index)-i-1))
		eig_vector_V=np.delete(eig_vector_V,(len(temp_index)-i-1),1)	#Deleting corresponding Eigen Vectors too from V

temp_index=[]											#Removing all the negative eigen values and their corresponding eigen vectors
for i in range(0,len(eig_values)):
	if(eig_values[i]<=1e-29):
		temp_index.append(i)
temp_index=np.sort(temp_index)[::-1]
for i in range(0,len(temp_index)):
	eig_values= np.delete(eig_values,temp_index[i])
	eig_vector_V=np.delete(eig_vector_V,temp_index[i],1)

eig_values=eig_values.real
rank=len(eig_values)	
eig_vector_V=eig_vector_V.real							#Now taking only the real part into our calculation for V matrix as their are no complex values remaining there in eigen vector

V=np.reshape(eig_vector_V,(-1,rank))					#reshaping the V matrix for calculation

U=np.dot(M,V)											#we know u[i]=(A[i].v[i])/sigma[i]

sigma=np.zeros(shape=(rank,rank),dtype=float)			#Calculating sigma vector from the modified eigen values array
for i in range(0,rank):
	sigma[i][i]=np.sqrt(eig_values[i])					##sigma is diagonal matrix with square root of eigen values in descending order
	
for i in range(0,len(U)):
    for j in range(0,len(U[i])):
        U[i][j]=U[i][j]/sigma[j][j]						#Calculating division by sigma[i] for U matrix

ans= np.dot(U,np.dot(sigma,V.T))						#Final matrix is U.sigma.(V_Transpose)

for i in range(0,len(ans)):								#We need to add the avg values back to the ans values as we try to get back the predicttion matrix
    for j in range(0,len(ans[i])):
        ans[i][j]=ans[i][j]+avg[i]


rms = sqrt(mean_squared_error(M_original, ans))			#Calculating Root Mean Square Error
print('Root Mean Square error:',rms)


diff=M_original-ans										#Calculating Spearman Rank for Original and Predicted matrix
diff_square=np.sum(np.square(diff))
n=r*c
spearman_rank=1-(6*diff_square/((n**3)-n))
print "The Spearman rank is",spearman_rank

#Calculating the K Top Precision
k=5										#Choosing  a k value
precision=0					
for i in range(0,len(ans)): 
	index_normal=[]
	index_predicted=[]
	for j in range(0,len(ans[0])):
		index_normal.append(j)
		index_predicted.append(j)
	M_original[i,:],index_normal = zip(*sorted(zip(M_original[i,:],index_normal))[::-1])	#Getting the index for asorted in descending order of rank
	ans[i,:],index_predicted=zip(*sorted(zip(ans[i,:],index_predicted))[::-1])
	index_predicted=index_predicted[0:k]
	index_normal=index_normal[0:k]
	k_top = [val for val in index_normal if val in index_predicted]
	print index_predicted
	print index_normal
	print k_top
	precision=precision+len(k_top)/k
precision=float(precision/len(ans))

print "K Top Precision where k is ",k,"is : ",precision

time_of_execution= time.clock() - start_time			#Calculating time for prediction
print "Time for prediction :",time_of_execution, "seconds"

