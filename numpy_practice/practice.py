import numpy as np
data =np.genfromtxt('student.csv', delimiter=',',dtype = None, encoding= 'utf-8', skip_header=1)
print(data)
#dimension of array
print(data.shape)
#no of dimensions
print(data.ndim)
#no of elements
print(data.size)
#data type of elements
print(data.dtype)
#first row
print(data[0])
#second element of first row
print(data[0][1])
#slice array for 0,1,2 rows
print(data[0:3])
#dimenson of array
print(data.shape)
#data is in 1D array of tuples
#convert to 2D numeric array
data2D = np.array(data.tolist())
print(data2D)
print(data2D.shape)
#convert to 2D numeric array with specific data type
data2D = np.array(data.tolist(), dtype='float') 
print(data2D)
print(data2D.shape)
#first column
print(data2D[:, 3])
#second column first element
print(data2D[0, 1])
#slice array for 1,2,3 columns
print(data2D[:, 0:3])
#specific elements from row and col
print(data2D[[0], [1, 3]])
print(data2D[[1,4],[1,3]])
#select 0,1,3 rows and columns
print(data2D[[0,1,3], :])
print(data2D[:,[1,3]])
print(data2D.size)
print(data2D.ndim)
#sum of all elements
print(np.sum(data2D))
#mean of all elements
print(np.mean(data2D))
#median of all elements
print(np.median(data2D))
#standard deviation of all elements
print(np.std(data2D))
#filter rows based on condition
print(data2D[data2D[:, 6] > 10])
print(data2D[data2D[:, 6] > 10, 0:3])
#check missing values
print(np.isnan(data2D))
#count missing values
print(np.isnan(data2D).sum())
#here there are no missing values but if present
#handle missing values by replacing with mean of column
col_mean = np.nanmean(data2D, axis=0)
inds = np.where(np.isnan(data2D))
data2D[inds] = np.take(col_mean, inds[1])
print(data2D)
#remove rows with missing values
#row wise NaN removal
data2D = data2D[~np.isnan(data2D).any(axis=1)]
print(data2D)
#column wise NaN removal
data2D = data2D[:, ~np.isnan(data2D).any(axis=0)]
print(data2D)
#sort array based on column valuesmin ascending order
print(data2D[data2D[:, 4].argsort()])
#sort array based on column values in descending order
print(data2D[data2D[:, 4].argsort()[::-1]])