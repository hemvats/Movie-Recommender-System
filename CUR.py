from __future__ import division
import random
from random import randint
import pandas
import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix
import copy
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
start_time = time.clock()
num=554
p=0
r=1509
c=2072
matrix = np.zeros(shape=(r,c), dtype=np.float)    #read file in matrix
C = np.zeros(shape=(r,num), dtype=np.float)    
R = np.zeros(shape=(num,c), dtype=np.float)
input_file= open("ratings.txt", "r")
lines= input_file.read()
total_ratings=0
for line in lines.split("\n"):
	  i=1
	  if len(line)!=0:
			for vals in line.split(" "):
				  if(i==1):
						row=vals
						i+=1
				  elif(i==2):
						col = vals
						i+=1
				  else:
						val = vals
						i+=1
	  matrix[int(row),int(col)] = val
	  total_ratings+=1

M_original = copy.deepcopy(matrix)	  				#DeepCopy as it created a new variable not just references a old one 
avg=[]												#Calculating avg for each user
for i in range(0,len(matrix)):
	temp_avg=0
	count=0
	for j in range(1,len(matrix[i])):
		if(matrix[i][j]>0):
			count= count+1
		temp_avg = temp_avg+matrix[i][j]			#making the matrix zero centered
	if(count>0):
		avg.append(temp_avg/count)
	else:
		avg.append(temp_avg)

for i in range(0,len(matrix)):
	for j in range(1,len(matrix[0])):
		if matrix[i][j]>0:
			matrix[i][j]=matrix[i][j]-avg[i]
		else:
			matrix[i][j]=0
matrix=np.reshape(matrix,(r,c))

matrix_square=np.square(matrix)					#Calculating sum of squares of all the rating in the given input matrix
p=np.sum(matrix_square)
index_col = []
for i in range(0,len(matrix[0])):
	index_col.append(i)

column_square=[]
for i in range(0,len(matrix[0])):
	column_square.append(np.sum(matrix_square[:,i])/p)
index_row=[]
for i in range(0,len(matrix)):
	index_row.append(i)

row_square=[]
for i in range(0,len(matrix)):
	row_square.append(np.sum(matrix_square[i,:])/p)

#Gemerating random column index and row index based on their prpbability calculated earlier 
temp_row=np.random.choice(index_row, size=num, replace=True, p=row_square)
temp_col=np.random.choice(index_col, size=num, replace=True, p=column_square)

#Appending the columns from generated column index to C matrix
for i in range(0, num):
	  C[:,i] = matrix[:,temp_col[i]]
	  for j in range(0,len(matrix)):
	  	if(np.sqrt(column_square[i]*num)>0):
			C[j][i]=C[j][i]/(np.sqrt(column_square[temp_col[i]]*num))

#Appending the generated column index from original matrix  to C matrix
for i in range(0, num):
	R[i,:] = matrix[temp_row[i],:]
	for j in range(0,len(matrix[i])):
	  	if(np.sqrt(row_square[i]*num)>0):
	  		R[i][j]=R[i][j]/(np.sqrt((row_square[temp_row[i]]*num)))

#Calculating the intersection matrix of C and R for psuedoinverse matrix i.e U calculation 
W=np.zeros(shape=(num,num),dtype=float)
for i in range(0,num):
	for j in range (0,num):
		W[i][j]=matrix[int(temp_row[i])][int(temp_col[j])]

#Calculating the eigen values for W 
x,eigen_values,yt=LA.svd(W,full_matrices=False)
values=copy.deepcopy(eigen_values)

# sum_total=np.sum(eigen_values)
# sum_total=sum_total*0.9
# t=0
# sum=0
# for i in range(len(eigen_values)):
# 	t=i
# 	if(sum>=sum_total):
# 		break
# 	else:
# 		sum=sum+eigen_values[i]


#Taking only those starting eigen values having sum almost equal or greater than  .9*(sum of total matrix) and deleting the rest of the values from the vector
for i in range(int(num*.15),int(num)):
	eigen_values=np.delete(eigen_values,len(eigen_values)-1)
	x=np.delete(x,len(eigen_values)-1,1)
	yt=np.delete(yt,len(eigen_values)-1,0)


xt=x.T
y=yt.T

#calculating the sigma vector for U calculation 
sigma=np.zeros(shape=(len(eigen_values),len(eigen_values)),dtype=float)
for i in range(0,len(sigma)):
	sigma[i][i]=eigen_values[i]

sigma_inverse=LA.pinv(sigma)
sigma_inverse_square=np.square(sigma_inverse)

#Calculating the U vector from U=Y.(Z^2).X_T
U=np.dot(np.dot(y,sigma_inverse_square),xt)

#Calculating the final matrix
ans=np.dot(C,np.dot(U,R))

# Adding back the average subracted initially for zero-centering the matrix
for i in range(0,len(ans)):
    for j in range(0,len(ans[i])):
        ans[i][j]=ans[i][j]+avg[i]


rms = sqrt(mean_squared_error(M_original, ans))
print('Root Mean Square error:',rms)

d_square=0
count=0
for i in range(0,len(M_original)):
	for j in range(0,len(M_original[i])):
			if(M_original[i][j]>0):
				d_square=d_square+(M_original[i][j]-ans[i][j])**2
				count=count+1

spearman_rank=1-(6*d_square/((count**3)-count))
print "The Spearman Rank i:",spearman_rank

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
	precision=precision+len(k_top)/k
precision=float(precision/len(ans))

print "K Top Precision where k is ",k,"is : ",precision


print "Time for prediction :",time.clock() - start_time, "seconds"
