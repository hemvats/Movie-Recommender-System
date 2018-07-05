from __future__ import division
import numpy as np
from scipy.sparse import *
from scipy import *
import scipy.sparse as sparse
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from operator import itemgetter


start_time=time.clock()
dataset=np.zeros((2072,1509))
data_for_similarity=np.zeros((2072,1509))
dictionary=[]
sorted_dict=[]
actual_rating=[]
predicted_rating=[]
actual_rating_b=[]
predicted_rating_b=[]
mu=0
mu_x=[]
mu_i=[]

"""
Readdata function reads the dataset and stores the movie ratings in a 2D matrix with columns representing users
and rows representing items(movies).
Here we implement item-item collaborative filtering
"""
def readdata():
    inp = open("ratings.txt", "r")
    data=[]
    for line in inp:
        num=line.split(" ")
        a=int(num[0])
        b=int(num[1])
        c=float(num[2])
        dataset[b,a]=c
"""
Convt_balanced funtion convert the original matrix and centers it around zero
"""
def convt_balanced(data):
    global mu_x
    global mu_i
    i=1
    non_zero = data > 0
    sum_row = data.sum(axis=1)
    num_ele = non_zero.sum(axis=1)
    avg_ele=sum_row/num_ele
    nan = np.isnan(avg_ele)
    avg_ele[nan] = 0
    mu_x=avg_ele
    mu=avg_ele.mean()
    sum_col=data.sum(axis=0)
    col_ele=non_zero.sum(axis=0)
    avg_col=sum_col/col_ele
    nan=np.isnan(avg_col)
    avg_col[nan]=0
    mu_i=avg_col
    rows=data.shape[0]
    while i < rows :
        j=1
        while j < 1509 :
            if (non_zero[i,j]):
                data[i,j]= data[i,j] - avg_ele[i]
            j=j+1
        i = i+1
    return data
"""
Convert every row into a unit vector
Each vector is for a movie because we are implementing item-item collaborative filtering
"""
def Convt_vector(data_vector):
    non_zero = data_vector != 0
    num_ele_sqr = non_zero.sum(axis=1)
    sum_sqrt=np.sqrt(np.sum(np.square(data_vector), axis=1))
    nan=np.isnan(sum_sqrt)
    sum_sqrt[nan]=0
    rows=data_vector.shape[0]
    i=1
    while i < rows:
        j=1
        while j < 1509 :
            if(non_zero[i,j]):
                data_vector[i,j]= data_vector[i,j]/sum_sqrt[i]
            j+=1
        i+=1
    return data_vector
"""
calculate precision using the top-K values
"""
def Top_K():
    print "Here we print top K value"
#     k=5										#Choosing  a k value
# precision=0					
# for i in range(0,len(ans)): 
# 	index_normal=[]
# 	index_predicted=[]
# 	for j in range(0,len(ans[0])):
# 		index_normal.append(j)
# 		index_predicted.append(j)
# 	M_original[i,:],index_normal = zip(*sorted(zip(M_original[i,:],index_normal))[::-1])	#Getting the index for asorted in descending order of rank
# 	ans[i,:],index_predicted=zip(*sorted(zip(ans[i,:],index_predicted))[::-1])
# 	index_predicted=index_predicted[0:k]
# 	index_normal=index_normal[0:k]
# 	k_top = [val for val in index_normal if val in index_predicted]
# 	precision=precision+len(k_top)/k
# precision=float(precision/len(ans))

# print "K Top Precision where k is ",k,"is : ",precision

"""
calculate the cosine similarity of two vectors
"""
def cosine_sim(a,b):
    cosine=a.dot(b)
    return cosine
"""
used to calculate b(i,x) which is used in baseline approach
"""
def Baseline(i,u):
    return mu +(mu_x[u]-mu) + (mu_i[i]-mu)


"""
create a list of vectors of items(movies) which are similar to a given movie and build a dictionary 
"""
def similar(item,user,dd,ddd):
    i=0
    while i < 2072 :
        if(ddd[i,user]!=0):
            temp=dd[i,user]
            dd[i,user]=0
            x=cosine_sim(dd[item],dd[i])
            dictionary.append((i,x))
            dd[i,user]=temp

        i+=1
    sorted_dict=[]
    sorted_dict=sorted(dictionary,key=itemgetter(1))[::-1]

    predict(item,user,np.copy(dataset),np.copy(sorted_dict))
    predict_baseline(item,user,np.copy(dataset),np.copy(sorted_dict))
"""
predict the ratings of a movie based on similarity and ratings by user to other movies
"""
def predict(item,user,data_predict,dict):
    if (data_predict[item,user]==0):
        i=0
    else :
        i=1
    cc=0
    predicted_value=0
    while i < 5 :
        a,b=dict[i]
        if(b>0):
            user = int(user)
            a = int(a)
            predicted_value=predicted_value + b* data_predict[a,user]
            cc+=1
        i+=1
    predicted_value=predicted_value/float(cc)
    if(predicted_value==0):
        predicted_value=2.5
    if(data_predict[item,user]!=0):
        actual_rating.append(data_predict[item,user])
        predicted_rating.append(predicted_value)
"""
predict movie rating using baseline approach
"""
def predict_baseline(item_b,user_b,data_b,dict_b):
    if (data_b[item_b,user_b]==0):
        i=0
    else :
        i=1
    cc=0
    predicted_value=0
    while i < 5 :
        a,b=dict_b[i]
        if(b>0):
            user_b = int(user_b)
            a = int(a)
            predicted_value=predicted_value + b* (data_b[a,user_b] - Baseline(a,user_b))
            cc+=1
        i+=1
    predicted_value=predicted_value/float(cc)
    predicted_value=Baseline(item_b,user_b) + predicted_value
    if(predicted_value==0):
        predicted_value=2.5
    if(data_b[item_b,user_b]!=0):
        actual_rating_b.append(data_b[item_b,user_b])
        predicted_rating_b.append(predicted_value)
"""
calculate Root Mean Square Error
"""
def rmse():
    rmse=np.sqrt(((np.array(actual_rating) - np.array(predicted_rating)) **2).mean())
    print "RMSE:",rmse
    rmse_b=np.sqrt(((np.array(actual_rating_b) - np.array(predicted_rating_b)) **2).mean())
    print "RMSE_Baseline:" , rmse_b
"""
Calculate Spearman Rank Co-relation
"""
def spearman():
    n=len(predicted_rating)
    sp_err=1-(6*(np.sum((np.array(actual_rating) - np.array(predicted_rating)) **2))/(n**3 - n))
    n_b=len(predicted_rating_b)
    sp_err_b=1-(6*(np.sum((np.array(actual_rating_b) - np.array(predicted_rating_b)) **2))/(n_b**3 - n_b))
    print "Spearman:",sp_err
    print "Spearman_Baseline:",sp_err_b


def main():
    readdata()
    dataset_balanced=convt_balanced(np.copy(dataset))
    dataset_vector=Convt_vector(np.copy(dataset_balanced))

    l=1
    k=1
    while l < 1000 :
        while k < 750:
            similar(l,k,np.copy(dataset_vector),np.copy(dataset))
            k+=1
        l+=1

    time_stop=time.clock()
    time_prediction=time_stop-start_time
    print "Time required to predict :",time_prediction
    Root_Mean_Square_error=rmse()
    spearman()

if __name__ == "__main__":
    main()