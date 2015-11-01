#########################################
# Author: Likun@stu.zzu.edu.cn
# Date: 2015-10-15
# 
# Regress with Least Squares (test Edition)
#########################################

import numpy as np
import math

data_train_A = np.genfromtxt('/home/lenovo/code/python/LSM/data/train_format.txt',delimiter = ',')
data_test = np.genfromtxt('/home/lenovo/code/python/LSM/data/test_format.txt' , delimiter = ',')

data_train_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/trainOutOne.txt')
data_test_out_one = np.genfromtxt('/home/lenovo/code/python/LSM/data/testOutOne.txt')
data_test_out_two = np.genfromtxt('/home/lenovo/code/python/LSM/data/testOutTwo.txt')

A_pinv = np.linalg.pinv(data_train_A)

#train cols one

b = np.dot(A_pinv,data_train_Y)
print('The pseudo-inverse ? matrix of Y: ')
print(b)

result_one = np.dot(data_test,b)
print('Result of one col :')
print(result_one)

print('The errors:')
print(result_one - data_test_out_one)
print('-------------------------------------------------------')
print('The RMSE(root mean squared error) of Col one is:')
print(math.sqrt(np.sum(np.square(result_one - data_test_out_one)) / result_one.size))
print('-------------------------------------------------------')

file_one_param_path = '/home/lenovo/code/python/LSM/data/result/params_one.txt'
file_one_param = open(file_one_param_path,'w')

print('Write params to file : ')
for bi in b:
	print(bi)
	file_one_param.write(str(bi) + '\n')
file_one_param.close()


# train cols two 
data_train_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/trainOutTwo.txt')

b = np.dot(A_pinv,data_train_Y)
print('\n\nThe pseudo-inverse matrix of Y: ')
print(b)

result_two = np.dot(data_test,b)
print('Result of two col :')
print(result_two)

print('The errors:')
print(result_two - data_test_out_two)
print('-------------------------------------------------------')
print('The RMSE(root mean squared error) of Col two is:')
print(math.sqrt(np.sum(np.square(result_two - data_test_out_two)) / result_two.size))
print('-------------------------------------------------------')

file_two_param_path = '/home/lenovo/code/python/LSM/data/result/params_two.txt'
file_two_param = open(file_two_param_path,'w')

print('Write params to file : ')
for bi in b:
	print(bi)
	file_two_param.write(str(bi) + '\n')
file_two_param.close()
