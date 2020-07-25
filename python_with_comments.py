import timeit
import numpy as np
import random
import math
import time



epsilon= 0.001
# generate n data each with d features
# n = number of points, d = dimensions, data-min=0, data-max=3
def generate_dataset(n, d, data_min, data_max): 
    res = [] #result
    for i in range(n):
        sub = [] #a point
        for j in range(d):
		# generate random 2 dimensions for each point with random.uniform, between min and max
            sub.append(random.uniform(data_min, data_max)) 
        res.append(sub) # append new point to the result
    return res


def print_dataset(dataset):
    print(np.asarray(dataset))
	#nicer and quicker print
	#np.asarray converts lists/tuples to np array.


# return  n*c membership function
# n = number of data points
# c = number of clusters
def generate_random_membership_function(n, c):
    membership = np.random.rand(n, c) # numpy pseudo-random matrix, n*c, with numbers between 0 to 1
	# rand() by deafult gives 0-1, but we could do also np.randon.rand(100,(3,5)) for numbers 0-100
	
	summation = [sum(point) for point in membership] #generator
	# This generator goes on each point, and sums this points "grades" for each cluster
	# and then it puts it in summation as an array of 1*n.
	# sum(point): adds the numbers in this 1*c list, to a number.
	# Example: [[0.5,0.5,0.5],[0.6,0.5,0.9]] ----> [1.5,2.0]
    normalized = [] 
    for i in range(len(membership)): #len of membership=number of points (n)
        tmp = []
        for d in membership[i]: # for c grades in membership[i] (as the number of clusters)
            tmp.append(d / summation[i]) 
			#Example: if the grades are [0.6,0.5,0.9], summation=2.0, tmp=[0.6/2,0.5/2,0.9/2]=[0.3,0.25,0.45]
			#This normalization method ensures that the sum of c grades will be 1
		normalized.append(tmp)
		#appending tmp to normalized
    return normalized #at the end, this is the normalized membership matrix

def update_cluster_centers(dataset, membership_matrix, m):
    number_of_clusters = len(membership_matrix[0]) # number of clusters is len of one row in membership_matrix
    cluster_centers = []
    for i in range(number_of_clusters): 
        u_ik = list(zip(*membership_matrix))[i] 
		# The zip() function returns a zip object, which is an iterator of tuples 
		# where the first item in each passed iterator is paired 
		# together, and then the second item in each passed iterator are 
		# paired together etc.
		# zip: returns a tuple
		# list() converts tuple to list
		# at the end, U_ik is all the n grades to c clusters in one list.
		# U_ik = sigma(x=1 to n) of Wk(x) 
		
        u_ik_m = [x ** m for x in u_ik]
		# Each value in U_ik is in power of m , the fuzzieness parameter.
		# We check also np.power but ** is faster
		# probably because our proccesseor can do power directly
		# numpy.power will be good for large M values
		
        sigma_u_ik_m = sum(u_ik_m)
	
        weighted_data = []
        for k in range(len(dataset)): # k goes 1---->n (number of points)
            weighted_vector = []
            for f in range(len(dataset[k])):  # f goes 1--->number of dimensions
                weighted_vector.append(u_ik_m[k] * dataset[k][f]) #u_ik_m * one dimension of x
            weighted_data.append(weighted_vector) # adding u_ik_m * x to the list
        sigma_data_u_ik_m = [sum(x) for x in list(zip(*weighted_data))] 
        cluster_centers.append([sigma_data_u_ik_m[d]/sigma_u_ik_m for d in range(len(sigma_data_u_ik_m))])

	return cluster_centers


def euclidean_distance(p, q):
    summation = 0
    for i in range(len(p)):
        summation += (p[i] - q[i]) ** 2
    return math.sqrt(summation)


def update_membership_matrix(dataset, clusters, m): # clusters = clustrers centers
    membership_matrix = []
    fuzzy_power = float(2 / (m-1)) # float() returns a floating point number from integer/string 
    n = len(dataset)
    c = len(clusters)
    for i in range(n): 
        denom = sum([(1/euclidean_distance(dataset[i], clusters[x])) ** fuzzy_power for x in range(c)])
        #denom = המכנה
		# ||Xi-Ck||^fuzzypower, for k from 1--->c (all clusters)
		# Finally this is sum(1/||Xi-Ck||) ^ fuzzypower,   where k from 1--->c
		membership = []
        for j in range(c):
            num = (1/euclidean_distance(dataset[i], clusters[j])) ** fuzzy_power
			#num is 1/||Xi-Cj|| ^fuzzypower
            membership.append(num/denom) #denom is the same, num changes every loop
			#num/denom = Wij that we try to minimize
        membership_matrix.append(membership)
    return membership_matrix


def get_cluster_labels(membership_matrix):
    res = []
    for membership in membership_matrix:
        max_index = membership.index(max(membership))
        res.append(max_index)
	#finding the index of maximal value in a membershiop matrix row
	#for example, row 50 in matrix is [0.1,0.1,0.8], max=0.8, maxindex=2, point 50 is in cluster 2
    return res

#m=2: Default Parameter Value
# if we call the function without this argument, it will be m=2 by default
def fcm(cluster_no, iterations, dataset, m=2):
    c = cluster_no # number of clusters
    n = len(dataset) # number of data points
	membership = generate_random_membership_function(n, c) #first we generate random membership matrix
    cluster_history=[]
	clusters = [] 
    for i in range(iterations):
        clusters = update_cluster_centers(dataset, membership, m)
        membership = update_membership_matrix(dataset, clusters, m)
		cluster_history.append(clusters)
		   if i>0: #check if delta<epsilon then return
           in_arr1 = np.array(cluster_history[i]) 
           in_arr2 = np.array(cluster_history[i-1])
           out_arr = np.subtract(in_arr1, in_arr2)
           out_arr=abs(out_arr)
           delta=np.max(out_arr)
           if delta<epsilon:
             print("the delta iteration is ", i+1 )
             return clusters, membership
    print("the iteration is ", i+1 ) 

    return clusters, membership


def run_fcm_on_a_2d_dataset():
    features = 2 # init number of dimensions for each data point
    number_of_data = 100 # init number of data points
    number_of_clusters = 3 # init number of clusters
    iterations = 50 # init max number of iterations
    data_max = 3 # max value in data set
    data_min = 0 # min value in data set
    dataset = generate_dataset(number_of_data, features, data_min, data_max)  # generate a random dataset
    cluster_centers, final_memberships = fcm(number_of_clusters, iterations, dataset)  # run FCM
    final_labels = get_cluster_labels(final_memberships)  # get labels
    print("cluster centers:")
    print_dataset(cluster_centers)

# if i rum a script myfile.py, it will start from here because this is  indentation
# level 0, and when i rum myfile's script, it sets __main__ into the __name__ variable automatically
# if someone else imports myfile, it will not go to this if.
# When i run my own program (first case) the interpreter assigns name=main
# When i import a file, named "foo", so __name__=foo and not __main__
if __name__ == '__main__': 
	start = time.time()
    run_fcm_on_a_2d_dataset()	# first function call
	end = time.time()
	print("time=" ,end-start)