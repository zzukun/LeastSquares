#########################################
# Author: Likun@stu.zzu.edu.cn
# Date: 2015-10-15
# 
# Regress with Least Squares
#########################################

import numpy as np
import math

#----------------------------------------------------------------------
#Global Vars
global params

#----------------------------------------------------------------------
#train function
def train(train_x , train_y):
	global params

	pinv = np.linalg.pinv(train_x)  # cacul the pseudo-inverse matrix
	
	params = np.dot( pinv , train_y ) # figure out the params
	print('The params:')
	print(params)

#----------------------------------------------------------------------
#test function
def test(test_x , test_y , result_path):
	global params

	rs = open(result_path , 'w')

	regressValue = np.dot(test_x,params)
	print('The regress value of test data:')
	print(regressValue)

	print('-------------------------------------------------------')
	print('The errors:')
	err = regressValue - test_y
	print(err)
	print('The RMSE(root mean squared error) is:')
	MRSE  =  math.sqrt(np.sum(np.square(err)) / regressValue.size)
	print(MRSE)
	print('-------------------------------------------------------')

	#save results into file
	rs.write('---results---\n\n')
	rs.write('The params:\n')
	rs.write(str(params) + '\n')

	rs.write('\nThe regress value of test data:\n')
	rs.write(str(regressValue) + '\n')

	rs.write('\nThe errors:\n')
	rs.write(str(err) + '\n')
	rs.write('\nThe RMSE(root mean squared error) is:\n')
	rs.write(str(MRSE) + '\n')


#----------------------------------------------------------------------
#main function
def main():
	#TRAIN - 1
	train_X = np.genfromtxt('/home/lenovo/code/python/LSM/data/train_format.txt',delimiter = ',')
	train_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/trainOutOne.txt')
	
	train(train_X , train_Y)

	#TEST - 1
	test_X = np.genfromtxt('/home/lenovo/code/python/LSM/data/test_format.txt' , delimiter = ',')
	test_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/testOutOne.txt')
	result_PATH = '/home/lenovo/code/python/LSM/data/result/result_motor'

	test(test_X , test_Y , result_PATH)

	#TRAIN - 2
	train_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/trainOutTwo.txt')

	train(train_X , train_Y)

	#TEST - 2
	test_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/testOutTwo.txt')
	result_PATH = '/home/lenovo/code/python/LSM/data/result/result_total'

	test(test_X , test_Y , result_PATH)

if __name__ == '__main__':
	main()
