# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:59:49 2018
@author: Nhan Tran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 

# Entropy  = h =  7.694180448822675 VS Bahram 2021 h =  7.6739952595684615

sbox_ROSIE     =  [54, 115, 104, 244, 221, 164, 20, 211, 157, 113, 246, 171, 144, 161, 26, 41, 179, 181, 52, 37, 122, 46, 127, 67, 4, 134, 58, 228, 163, 240, 8, 33, 131, 222, 170, 62, 2, 212, 133, 252, 101, 103, 34, 202, 110, 89, 165, 218, 156, 132, 85, 106, 81, 70, 32, 78, 153, 233, 65, 73, 169, 74, 237, 31, 177, 29, 123, 224, 45, 142, 214, 232, 75, 68, 242, 22, 238, 28, 213, 61, 150, 55, 193, 197, 10, 180, 12, 83, 11, 3, 76, 60, 25, 19, 90, 178, 247, 18, 15, 23, 5, 135, 191, 1, 231, 59, 100, 226, 56, 95, 201, 9, 79, 94, 249, 48, 111, 254, 198, 172, 0, 248, 200, 138, 219, 235, 146, 243, 255, 152, 16, 96, 185, 126, 174, 229, 40, 82, 203, 105, 184, 21, 204, 63, 160, 154, 6, 236, 88, 205, 155, 98, 143, 64, 207, 14, 253, 245, 49, 167, 87, 35, 208, 30, 102, 209, 24, 92, 36, 210, 158, 183, 93, 141, 148, 97, 175, 107, 39, 50, 159, 220, 182, 206, 125, 227, 149, 225, 71, 51, 99, 195, 72, 173, 53, 38, 17, 118, 91, 121, 186, 77, 120, 80, 137, 192, 47, 147, 187, 176, 199, 108, 119, 27, 129, 251, 13, 42, 130, 196, 117, 43, 217, 166, 162, 188, 7, 234, 189, 116, 124, 216, 230, 241, 84, 57, 239, 223, 194, 112, 114, 168, 145, 66, 44, 128, 109, 140, 139, 190, 215, 136, 69, 86, 151, 250]
inv_sbox_ROSIE =  [120, 103, 36, 89, 24, 100, 146, 226, 30, 111, 84, 88, 86, 216, 155, 98, 130, 196, 97, 93, 6, 141, 75, 99, 166, 92, 14, 213, 77, 65, 163, 63, 54, 31, 42, 161, 168, 19, 195, 178, 136, 15, 217, 221, 244, 68, 21, 206, 115, 158, 179, 189, 18, 194, 0, 81, 108, 235, 26, 105, 91, 79, 35, 143, 153, 58, 243, 23, 73, 252, 53, 188, 192, 59, 61, 72, 90, 201, 55, 112, 203, 52, 137, 87, 234, 50, 253, 160, 148, 45, 94, 198, 167, 172, 113, 109, 131, 175, 151, 190, 106, 40, 164, 41, 2, 139, 51, 177, 211, 246, 44, 116, 239, 9, 240, 1, 229, 220, 197, 212, 202, 199, 20, 66, 230, 184, 133, 22, 245, 214, 218, 32, 49, 38, 25, 101, 251, 204, 123, 248, 247, 173, 69, 152, 12, 242, 126, 207, 174, 186, 80, 254, 129, 56, 145, 150, 48, 8, 170, 180, 144, 13, 224, 28, 5, 46, 223, 159, 241, 60, 34, 11, 119, 193, 134, 176, 209, 64, 95, 16, 85, 17, 182, 171, 140, 132, 200, 208, 225, 228, 249, 102, 205, 82, 238, 191, 219, 83, 118, 210, 122, 110, 43, 138, 142, 149, 183, 154, 162, 165, 169, 7, 37, 78, 70, 250, 231, 222, 47, 124, 181, 4, 33, 237, 67, 187, 107, 185, 27, 135, 232, 104, 71, 57, 227, 125, 147, 62, 76, 236, 29, 233, 74, 127, 3, 157, 10, 96, 121, 114, 255, 215, 39, 156, 117, 128]

#   Bahram 2021 SBox Entropy h =  7.6739952595684615
bahram_2021_sbox = [
    0x82, 0x13, 0x9f, 0x6b, 0xd9, 0xbc, 0x76, 0xe7, 0xfa, 0xa1, 0x48, 0x55, 0x2d, 0xc4, 0x30, 0x0e,
    0x7d, 0x05, 0x7e, 0xcc, 0xa3, 0x2f, 0x59, 0x7a, 0xaf, 0x00, 0x75, 0xbf, 0x35, 0xe2, 0xfb, 0xbd,
    0xf5, 0x16, 0x1c, 0xc1, 0x74, 0xc9, 0xf8, 0xac, 0x12, 0x3a, 0x54, 0x04, 0x6f, 0xc8, 0xfe, 0xeb,
    0x28, 0xc3, 0x98, 0xff, 0x5d, 0xb8, 0x29, 0xef, 0x03, 0x43, 0x3b, 0x66, 0x6d, 0x49, 0x0f, 0x70,
    0x39, 0x56, 0x95, 0x1f, 0xcd, 0x9c, 0xab, 0x3c, 0x2c, 0x72, 0xf2, 0xfc, 0x6a, 0xe3, 0x40, 0xe1,
    0x1b, 0x09, 0xb5, 0xb2, 0x52, 0xa9, 0x1d, 0x41, 0xa4, 0xc0, 0x8d, 0x02, 0x51, 0x73, 0x21, 0x64,
    0x5f, 0x3f, 0xaa, 0x97, 0xd2, 0x25, 0x61, 0xd6, 0xe5, 0x67, 0xc2, 0xfd, 0xdf, 0xa6, 0x69, 0x44,
    0xa0, 0x0d, 0xde, 0x90, 0x79, 0x85, 0xc7, 0xe6, 0x0b, 0x8c, 0x15, 0xed, 0xd3, 0x08, 0x23, 0xdc,
    0xd7, 0xa7, 0x3d, 0xd5, 0x81, 0xe0, 0x4c, 0x32, 0x78, 0xba, 0x58, 0xf6, 0x4b, 0xbb, 0xdd, 0xbe,
    0x6c, 0x7b, 0xd1, 0x9e, 0x62, 0x7c, 0x46, 0xdb, 0x1e, 0x37, 0xd8, 0x63, 0x42, 0x96, 0x57, 0xe9,
    0xc6, 0x2b, 0xcf, 0x19, 0x91, 0x17, 0x9a, 0xd0, 0xb7, 0x27, 0x4a, 0x36, 0x4d, 0x24, 0x3e, 0x86,
    0xb1, 0x47, 0x99, 0x07, 0x65, 0x26, 0x8e, 0xc5, 0xf3, 0x84, 0xae, 0xd4, 0x83, 0xad, 0x5e, 0xca,
    0xe4, 0xa5, 0x5c, 0x31, 0xcb, 0x60, 0x80, 0x2a, 0x22, 0x18, 0x0c, 0x71, 0x2e, 0x34, 0xa8, 0x77,
    0x4e, 0x8f, 0xeb, 0x1a, 0x10, 0x9d, 0xb9, 0x38, 0x11, 0xda, 0x8a, 0x6e, 0xea, 0x53, 0xee, 0x01,
    0x93, 0x50, 0x45, 0xce, 0xf1, 0x68, 0x8b, 0x14, 0x7f, 0x4f, 0x89, 0xa2, 0x20, 0xf0, 0xb0, 0x87,
    0x0a, 0xec, 0xb6, 0xf4, 0xb4, 0x92, 0xf9, 0x5a, 0xb3, 0x33, 0xf7, 0x06, 0x5b, 0x88, 0x94, 0x9b
]


len = 256


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
#print("X = ",X)
y = dataset.iloc[:, 2].values
#print("y = ",y)

# x1 = sorted(random.sample(range(0, len), len))
# X = np.reshape(x1, (256, 1))
# #print("X = ",X)
# y = bahram_2021_sbox #
# y = sbox_ROSIE

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
# Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_linear()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_polymonial()

# Additional feature
# Making the plot line (Blue one) more smooth
def viz_polymonial_smooth():
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1) #Why do we need to reshape? (https://www.tutorialspoint.com/numpy/numpy_reshape.htm)
    # Visualizing the Polymonial Regression results
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, pol_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_polymonial_smooth()

# Predicting a new result with Linear Regression
lin_reg.predict([[5.5]])
#output should be 249500

# Predicting a new result with Polymonial Regression
pol_reg.predict(poly_reg.fit_transform([[5.5]]))
#output should be 132148.43750003