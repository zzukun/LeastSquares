#########################################
# Author: Likun@stu.zzu.edu.cn
# Date: 2015-10-15
# 
# Combine two matrix , and split the dataset into 'male set' and 'female set' by a column that shows the sex
#########################################

import numpy as np
import math

#result saving path
male_motor		= '/home/lenovo/code/python/LSM/data/result/result_male_motor'
male_total		= '/home/lenovo/code/python/LSM/data/result/result_male_total'
female_motor	= '/home/lenovo/code/python/LSM/data/result/result_female_motor'
female_total	= '/home/lenovo/code/python/LSM/data/result/result_female_total'

rs_mm = open(male_motor,'w') 
rs_mt = open(male_total,'w')
rs_fm = open(female_motor,'w')
rs_ft = open(female_total,'w')

rs_mm.write('---results---\n\n')
rs_mt.write('---results---\n\n')
rs_fm.write('---results---\n\n')
rs_ft.write('---results---\n\n')

#load train data
train_X = np.genfromtxt('/home/lenovo/code/python/LSM/data/train_format.txt',delimiter = ',')
train_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/trainOutOne.txt')
train_Y_2 = np.genfromtxt('/home/lenovo/code/python/LSM/data/trainOutTwo.txt')

print(train_X.shape)
print(train_Y.shape)

total = np.column_stack((train_X,train_Y))

print(total.shape)

total = np.column_stack((total, train_Y_2))

print(total.shape)

#load test data
test_X = np.genfromtxt('/home/lenovo/code/python/LSM/data/test_format.txt',delimiter = ',')
test_Y = np.genfromtxt('/home/lenovo/code/python/LSM/data/testOutOne.txt')
test_Y_2 = np.genfromtxt('/home/lenovo/code/python/LSM/data/testOutTwo.txt')

total_test = np.column_stack((test_X , test_Y))

total_test = np.column_stack((total_test , test_Y_2))

#condition = total[:,2] == 1 

#subMatrix = np.extract(condition,total)

#print(subMatrix.shape)

#condition = total[:,2] == 0
#subMatrix = np.extract(condition,total)

#print(subMatrix.shape)

#sub = total[:,2] == 1

#print(sub)


sub = []

for row in total:
	if row[2] == 0:
		sub.append(row)

#print (sub)
#zip(sub)
subMatrix_male = np.array(sub)
print(subMatrix_male.shape)

sub = []

for row in total:
	if row[2] == 1:
		sub.append(row)

subMatrix_female = np.array(sub)

print(subMatrix_female.shape)

sub = []

for row in total_test:
	if row[2] == 0:
		sub.append(row)

subMatrix_male_test = np.array(sub)
print('subMatrix_male_test.shape:')
print(subMatrix_male_test.shape)

sub = []

for row in total_test:
	if row[2] == 1:
		sub.append(row)

subMatrix_female_test = np.array(sub)


#-------------------------------------------------------------------------
#male

train_subset_x = subMatrix_male[:,:20]
train_subset_y_1 = subMatrix_male[:,20:21]
train_subset_y_2 = subMatrix_male[:,21:22]

print(train_subset_x.shape)
print(train_subset_y_1)
print(train_subset_y_2)

pinv = np.linalg.pinv(train_subset_x)

params_1 = np.dot( pinv , train_subset_y_1 )
rs_mm.write('The params:\n')
rs_mm.write(str(params_1) + '\n')

params_2 = np.dot( pinv , train_subset_y_2 )
rs_mt.write('The params:\n')
rs_mt.write(str(params_2) + '\n')

#regress
test_subset_x = subMatrix_male_test[:,:20]
test_subset_y_1 = subMatrix_male_test[:,20:21]
test_subset_y_2 = subMatrix_male_test[:,21:22]

regressValue_1 = np.dot(test_subset_x,params_1)
rs_mm.write('\nThe regress value of test data:\n')
rs_mm.write(str(regressValue_1) + '\n')

regressValue_2 = np.dot(test_subset_x,params_2)
rs_mt.write('\nThe regress value of test data:\n')
rs_mt.write(str(regressValue_2) + '\n')

delta1 = regressValue_1 - test_subset_y_1
rs_mm.write('\nThe errors:\n')
rs_mm.write(str(delta1) + '\n')

delta2 = regressValue_2 - test_subset_y_2
rs_mt.write('\nThe errors:\n')
rs_mt.write(str(delta2) + '\n')


MRSE_1  =  math.sqrt(np.sum(np.square(delta1)) / regressValue_1.size)
rs_mm.write('\nThe RMSE(root mean squared error) is:\n')
rs_mm.write(str(MRSE_1) + '\n')

MRSE_2  =  math.sqrt(np.sum(np.square(delta2)) / regressValue_2.size)
rs_mt.write('\nThe RMSE(root mean squared error) is:\n')
rs_mt.write(str(MRSE_2) + '\n')

print(MRSE_1)
print(MRSE_2)

#-------------------------------------------------------------------------
#female

train_subset_x = subMatrix_female[:,:20]
train_subset_y_1 = subMatrix_female[:,20:21]
train_subset_y_2 = subMatrix_female[:,21:22]

print(train_subset_x.shape)
print(train_subset_y_1)
print(train_subset_y_2)

pinv = np.linalg.pinv(train_subset_x)

params_1 = np.dot( pinv , train_subset_y_1 )
rs_fm.write('The params:\n')
rs_fm.write(str(params_1) + '\n')
params_2 = np.dot( pinv , train_subset_y_2 )
rs_ft.write('The params:\n')
rs_ft.write(str(params_2) + '\n')

#regress
test_subset_x = subMatrix_female_test[:,:20]
test_subset_y_1 = subMatrix_female_test[:,20:21]
test_subset_y_2 = subMatrix_female_test[:,21:22]

regressValue_1 = np.dot(test_subset_x,params_1)
rs_fm.write('\nThe regress value of test data:\n')
rs_fm.write(str(regressValue_1) + '\n')

regressValue_2 = np.dot(test_subset_x,params_2)
rs_ft.write('\nThe regress value of test data:\n')
rs_ft.write(str(regressValue_2) + '\n')

delta1 = regressValue_1 - test_subset_y_1
rs_fm.write('\nThe errors:\n')
rs_fm.write(str(delta1) + '\n')

delta2 = regressValue_2 - test_subset_y_2
rs_ft.write('\nThe errors:\n')
rs_ft.write(str(delta1) + '\n')


MRSE_1  =  math.sqrt(np.sum(np.square(delta1)) / regressValue_1.size)
rs_fm.write('\nThe RMSE(root mean squared error) is:\n')
rs_fm.write(str(MRSE_1) + '\n')

MRSE_2  =  math.sqrt(np.sum(np.square(delta2)) / regressValue_2.size)
rs_ft.write('\nThe RMSE(root mean squared error) is:\n')
rs_ft.write(str(MRSE_2) + '\n')

print(MRSE_1)
print(MRSE_2)


rs_mm.close() 
rs_mt.close() 
rs_fm.close()
rs_ft.close()
