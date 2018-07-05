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
import time
start_time = time.clock()
r=1509
c=2072
matrix_original = np.zeros(shape=(r,c), dtype=np.float)    #read file in matrix


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
	  matrix_original[int(row),int(col)] = val
	  total_ratings+=1


M_original = copy.deepcopy(matrix_original)	 

avg=[]
for i in range(0,len(matrix_original)):					#Calculating avg of each user for making it zero centred matrix
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

for i in range(0,len(matrix_original)):
	for j in range(1,len(matrix_original[i])):
		if matrix_original[i][j]>0:
			matrix_original[i][j]=matrix_original[i][j]-avg[i]
		else:
			matrix_original[i][j]=0


M = copy.deepcopy(matrix_original)
M_T=M.T

for_V=np.dot(M_T,M)
eig_values,eig_vector_V=LA.eig(for_V)

temp_index=np.iscomplex(eig_values)
for i in range(0,len(temp_index)):
	if((temp_index[len(temp_index)-i-1] == True)):
		eig_values= np.delete(eig_values,(len(temp_index)-i-1))
		eig_vector_V=np.delete(eig_vector_V,(len(temp_index)-i-1),1)

temp_index=[]
for i in range(0,len(eig_values)):
	if(eig_values[i]<=1e-29):
		temp_index.append(i)
temp_index=np.sort(temp_index)[::-1]
for i in range(0,len(temp_index)):
	eig_values= np.delete(eig_values,temp_index[i])
	eig_vector_V=np.delete(eig_vector_V,temp_index[i],1)

eig_values=eig_values.real
rank=len(eig_values)
eig_vector_V=eig_vector_V.real
values=copy.deepcopy(eig_values)

V=np.reshape(eig_vector_V,(-1,rank))

U=np.dot(M,V)

sigma=np.zeros(shape=(rank,rank),dtype=float)
for i in range(0,rank):
	sigma[i][i]=np.sqrt(eig_values[i])
	
for i in range(0,len(U)):
    for j in range(0,len(U[i])):
        U[i][j]=U[i][j]/sigma[j][j]

'''
for i in range(int(.25*rank),rank):
	eig_values=np.delete(eig_values,len(eig_values)-1)
	U=np.delete(U,len(eig_values)-1,1)
	V = np.delete(V,len(eig_values)-1,1)

# print np.sum(eig_values)/np.sum(values)
# print np.sum(np.square(eig_values)/np.sum(np.square(values)))

'''
sum_total=.9*np.sum(np.square(eig_values))					#Retaining only those starting eigen values whose sum of sqaure is almost equal to 90% of total sum of all eigen values
t=0
sum=0
for i in range(len(eig_values)):
	t=i
	if(sum>=sum_total):
		break
	else:
		sum+=eig_values[i]

for i in range(t):
	eig_values=np.delete(eig_values,len(eig_values)-1)		#Deleting the rest of the eigen values and corresponding eigen vectors from V for calculation of U matrix
	U=np.delete(U,len(eig_values)-1,1)
	V = np.delete(V,len(eig_values)-1,1)



sigma=np.zeros(shape=(len(eig_values),len(eig_values)),dtype=float)
for i in range(0,len(eig_values)):
	sigma[i][i]=np.sqrt(eig_values[i])

ans= np.dot(U,np.dot(sigma,V.T))

for i in range(0,len(ans)):
    for j in range(0,len(ans[i])):
        ans[i][j]=ans[i][j]+avg[i]

time_of_execution=time.clock() - start_time

rms = sqrt(mean_squared_error(M_original, ans))
print('Root Mean Square error:',rms)

diff=M_original-ans
diff_square=np.sum(np.square(diff))
n=r*c
spearman_rank=1-(6*diff_square/((n**3)-n))
print "The Spearman rank is ",spearman_rank
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
	print len(k_top)
	precision=precision+len(k_top)/k
precision=float(precision/len(ans))

print "K Top Precision where k is ",k,"is : ",precision

print "Time for prediction :",time_of_execution, "seconds"
