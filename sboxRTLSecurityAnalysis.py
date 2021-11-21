#*****************************************************************************
#
#                            SBOX RTL Security Analysis Code.
#                             Written  by Bilal Dastagir.
#                                Nov, 19th, 2021
#
#******************************************************************************

import time
from gettext import _error
import cv2
import numpy as np
import random 
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt, collections
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from sklearn.metrics import mean_absolute_error
import pyhomogeneity as hg
import collections
from scipy.stats import entropy
import copy
from skimage.metrics import structural_similarity as ssim
from bioinfokit import analys, visuz
from scipy import signal

#Global Variables 
BETA = [0]
ALPHA = [1]
BRAVO  = [2]
CHARLIE = [3]

ROSIEPATH = "/Users/bilaldastagir/Documents/vscode/ROSIE/"

# Proposed Dual Quad-Bit SBox  #
# SBox MSCA
sbox1FMSCA =  [15, 14, 1, 0, 3, 2, 13, 12, 4, 9, 6, 11, 8, 5, 10, 7]
sbox1BMSCA =  [3, 2, 5, 4, 8, 13, 10, 15, 12, 9, 14, 11, 7, 6, 1, 0]
sbox2FMSCA =  [8, 5, 10, 7, 4, 9, 6, 11, 3, 2, 13, 12, 15, 14, 1, 0]
sbox2BMSCA =  [15, 14, 9, 8, 4, 1, 6, 3, 0, 5, 2, 7, 11, 10, 13, 12]

# SBox RTL Entropy = 7.659934382909035
sbox1FRTL =  [15, 5, 8, 9, 10, 6, 7, 4, 3, 1, 12, 0, 11, 2, 13, 14]
sbox1BRTL =  [11, 9, 13, 8, 7, 1, 5, 6, 2, 3, 4, 12, 10, 14, 15, 0]
sbox2FRTL =  [7, 2, 14, 8, 13, 12, 11, 3, 1, 10, 9, 6, 5, 4, 15, 0]
sbox2BRTL =  [15, 8, 1, 7, 13, 12, 11, 0, 3, 10, 9, 6, 5, 4, 2, 14]
# End of Proposed Dual Quad-Bit SBox  #

#bahram_2021_Entropy = 7.6739952595684615
sbox_Bahram_2021 =  [130, 19, 159, 107, 217, 188, 118, 231, 250, 161, 240, 120, 202, 196, 48, 62, 125, 5, 126, 1, 163, 47, 89, 165, 175, 0, 117, 191, 53, 226, 251, 189, 245, 94, 200, 193, 173, 201, 248, 172, 18, 58, 199, 4, 111, 12, 254, 21, 14, 195, 152, 255, 93, 28, 41, 239, 3, 67, 41, 102, 109, 73, 174, 112, 78, 87, 149, 57, 205, 156, 171, 60, 10, 61, 242, 252, 134, 227, 208, 233, 27, 92, 181, 178, 42, 11, 65, 65, 164, 22, 247, 2, 194, 52, 33, 100, 197, 63, 170, 155, 95, 180, 59, 214, 229, 110, 76, 3, 223, 60, 219, 44, 63, 203, 222, 144, 36, 26, 6, 230, 136, 140, 23, 237, 211, 16, 18, 232, 215, 167, 0, 213, 185, 224, 76, 50, 253, 234, 218, 246, 121, 90, 221, 190, 115, 123, 209, 158, 98, 66, 157, 99, 50, 55, 216, 99, 69, 150, 147, 2, 198, 9, 207, 20, 88, 23, 154, 129, 206, 39, 98, 70, 39, 36, 62, 24, 238, 176, 83, 7, 244, 82, 142, 168, 243, 132, 137, 212, 5, 31, 143, 27, 228, 35, 92, 49, 13, 96, 160, 42, 34, 37, 12, 113, 46, 68, 168, 162, 78, 146, 235, 124, 187, 131, 103, 128, 154, 4, 138, 110, 127, 142, 114, 108, 133, 79, 29, 77, 192, 104, 119, 7, 127, 79, 137, 210, 241, 123, 176, 55, 10, 236, 74, 184, 180, 32, 139, 90, 38, 51, 8, 30, 75, 136, 46, 51]
inv_sbox_Bahram_2021 =  [130, 19, 159, 107, 217, 188, 118, 231, 250, 161, 240, 120, 202, 196, 48, 62, 125, 5, 126, 1, 163, 47, 89, 165, 175, 0, 117, 191, 53, 226, 251, 189, 245, 94, 200, 193, 173, 201, 248, 172, 18, 58, 199, 4, 111, 12, 254, 21, 14, 195, 152, 255, 93, 28, 41, 239, 3, 67, 41, 102, 109, 73, 174, 112, 78, 87, 149, 57, 205, 156, 171, 60, 10, 61, 242, 252, 134, 227, 208, 233, 27, 92, 181, 178, 42, 11, 65, 65, 164, 22, 247, 2, 194, 52, 33, 100, 197, 63, 170, 155, 95, 180, 59, 214, 229, 110, 76, 3, 223, 60, 219, 44, 63, 203, 222, 144, 36, 26, 6, 230, 136, 140, 23, 237, 211, 16, 18, 232, 215, 167, 0, 213, 185, 224, 76, 50, 253, 234, 218, 246, 121, 90, 221, 190, 115, 123, 209, 158, 98, 66, 157, 99, 50, 55, 216, 99, 69, 150, 147, 2, 198, 9, 207, 20, 88, 23, 154, 129, 206, 39, 98, 70, 39, 36, 62, 24, 238, 176, 83, 7, 244, 82, 142, 168, 243, 132, 137, 212, 5, 31, 143, 27, 228, 35, 92, 49, 13, 96, 160, 42, 34, 37, 12, 113, 46, 68, 168, 162, 78, 146, 235, 124, 187, 131, 103, 128, 154, 4, 138, 110, 127, 142, 114, 108, 133, 79, 29, 77, 192, 104, 119, 7, 127, 79, 137, 210, 241, 123, 176, 55, 10, 236, 74, 184, 180, 32, 139, 90, 38, 51, 8, 30, 75, 136, 46, 51]


s_box_aes = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

inv_s_box_aes = [
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
]

def sub_bytes_AES_Traditional(s):
    #start_time = time.time()
    so = s_box_aes[s]
    #print("sBox Execution Time --- %s seconds ---" % (time.time() - start_time))
    return so


def inv_sub_bytes_AES_Traditional(s):
    #start_time = time.time()
    so = inv_s_box_aes[s]
    #print("Inv-sBox Execution Time --- %s seconds ---" % (time.time() - start_time))
    return so

def byteSplit(integer):
    return divmod(integer, 0x10)


def byteJoin(num1, num2):
    num3 = (num1 << 4) | (num2);
    return num3


def sBoxMSCA_Forward(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteLow = sbox2FRTL[byteLow]
    newByteHigh = sbox1FRTL[byteHigh] ^ sbox2FRTL[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def sBoxMSCA_Backward(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteHigh = sbox1BRTL[byteHigh^byteLow]
    newByteLow = sbox2BRTL[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def sBoxRTL_Forward(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteLow = sbox2FRTL[byteLow]
    newByteHigh = sbox1FRTL[byteHigh] ^ sbox2FRTL[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def sBoxRTL_Backward(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteHigh = sbox1BRTL[byteHigh^byteLow]
    newByteLow = sbox2BRTL[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def sub_bytes_Bahram_2021(s):
    #start_time = time.time()
    so = sbox_Bahram_2021[s]
    #print("sBox Execution Time --- %s seconds ---" % (time.time() - start_time))
    return so


def inv_sub_bytes_Bahram_2021(s):
    #start_time = time.time()
    so = inv_sbox_Bahram_2021[s]
    #print("Inv-sBox Execution Time --- %s seconds ---" % (time.time() - start_time))
    return so

# Calculate information entropy
def getEntropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    #print(norm_counts)
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()  # log(a) b=log (c) b÷log (c) a

def get_inv_sbox(sboxFL,len):  
    # Program Started
    sboxBL = copy.deepcopy(sboxFL)
    for i in range(len):
        sboxBL[sboxFL[i]] = i
    return sboxBL
    # Program Ended  
    
def SPN_Forward(imagePlain):
    print("SPN_Forward is Started........... !!!")
    slen = 512
    key = 128
    keylistFX  = random.sample(range(0, slen), slen)
    keylistFY  = random.sample(range(0, slen), slen)
    keylistBX = get_inv_sbox(keylistFX,slen)
    keylistBY = get_inv_sbox(keylistFY,slen)
    print("\keylistFX: ",keylistFX)
    print("keylistFX Entropy -> h = ",shannon_entropy(keylistFX, 2))
    print("\keylistFY: ",keylistFY)
    print("keylistFY Entropy -> h = ",shannon_entropy(keylistFY, 2))
    print("\keylistBX: ",keylistBX)
    print("keylistBX Entropy -> h = ",shannon_entropy(keylistBX, 2))
    print("\keylistBY: ",keylistBY)
    print("keylistBY Entropy -> h = ",shannon_entropy(keylistBY, 2))
    imagePathNoise = ROSIEPATH + "noise.png"
    imageEntropyNoise = cv2.imread(imagePathNoise, cv2.IMREAD_GRAYSCALE)
    print(imageEntropyNoise)
    print("NOISE Entropy -> h = ",shannon_entropy(imageEntropyNoise, 2))
    imagePlain.show()
    imArray_In = np.array(imagePlain)
    #imArray_Out_MSCA = copy.deepcopy(imArray_In)
    imArray_Out_RTL_S = copy.deepcopy(imArray_In)#np.zeros((256, 256, 3)) #copy.deepcopy(imArray_In)
    imArray_Out_Bahrami_2021_S = copy.deepcopy(imArray_In)
    imArray_Out_RTL_D = copy.deepcopy(imArray_In)#np.zeros((256, 256, 3)) #copy.deepcopy(imArray_In)
    imArray_Out_Bahrami_2021_D = copy.deepcopy(imArray_In)
    #imArray_Out_RTL_SP = imArray_In#copy.deepcopy(imArray_In)
    # Write code Here
    for i in range(len(imArray_In)):
        for j in range(len(imArray_In[0])):
            for k in range(len(imArray_In[0][0])):  # 3):
                #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
                #if (i==0 & j==0):
                imArray_Out_Bahrami_2021_S[i][j][k] = sub_bytes_Bahram_2021(imArray_In[i][j][k])
                imArray_Out_RTL_S[keylistFX[i]][keylistFY[j]][k] = sBoxRTL_Forward(imArray_In[i][j][k])
                #else:
                    #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
                #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 
    
    for i in range(len(imArray_In)):
        for j in range(len(imArray_In[0])):
            for k in range(len(imArray_In[0][0])):  # 3):
                #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
                #if (i==0 & j==0):
                imArray_Out_Bahrami_2021_D[i][j][k] = inv_sub_bytes_Bahram_2021(imArray_Out_Bahrami_2021_S[i][j][k])
                imArray_Out_RTL_D[keylistBX[i]][keylistBY[j]][k] = sBoxRTL_Backward(imArray_Out_RTL_S[i][j][k])
                #else:
                    #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
                #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 

    
    
    #imageCipher_MSCA = Image.fromarray(imArray_Out_MSCA)
    #imageCipher_MSCA.show()
    imageCipher_Bahrami_2021_S = Image.fromarray(imArray_Out_Bahrami_2021_S)
    imageCipher_Bahrami_2021_S.show()
    imageCipherLenaPath_Bahrami_2021_S = ROSIEPATH + "imageCipherLenaBahrami_2021.png"
    imageCipher_Bahrami_2021_S.save(imageCipherLenaPath_Bahrami_2021_S)
    imageEntropyCipherLenaBahrami_2021_S = cv2.imread(imageCipherLenaPath_Bahrami_2021_S, cv2.IMREAD_GRAYSCALE)
    print("SBox Bahrami 2021 S  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaBahrami_2021_S, 2))
    
    
    
    imageCipher_RTL_S = Image.fromarray(imArray_Out_RTL_S)
    imageCipher_RTL_S.show()
    imageCipherLenaPath_RTL_S = ROSIEPATH + "imageCipherLenaRTL.png"
    imageCipher_RTL_S.save(imageCipherLenaPath_RTL_S)
    imageEntropyCipherLenaRTL_S = cv2.imread(imageCipherLenaPath_RTL_S, cv2.IMREAD_GRAYSCALE)
    print("SBox RTL S  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaRTL_S, 2))
 
    imageCipher_RTL_D = Image.fromarray(imArray_Out_RTL_D)
    imageCipher_RTL_D.show()
    imageCipherLenaPath_RTL_D = ROSIEPATH + "imageDecipherLenaRTL.png"
    imageCipher_RTL_D.save(imageCipherLenaPath_RTL_D)
    imageEntropyCipherLenaRTL_D = cv2.imread(imageCipherLenaPath_RTL_D, cv2.IMREAD_GRAYSCALE)
    print("SBox RTL D  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaRTL_D, 2))
     
    
    #imageEntropyCipherLenaRTL_SP = cv2.imread(imageCipherLenaPath_RTL_SP, cv2.IMREAD_GRAYSCALE)
    #print("SBox RTL SP Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaRTL_SP, 2))
    #h = getEntropy(imArray_Out_RTL, 2)
   # entropyImageMSCA = shannon_entropy(imageCipher_MSCA, 2)
   # print("SBox MSCA Entropy -> h = ",entropyImageMSCA)
    #entropyImageRTL =  0
   # print("SBox RTL Entropy -> h = ",entropyImageRTL)
   # entropyImageRTL = shannon_entropy(imageCipher_RTL, 2)
   # print("SBox RTL Entropy -> h = ",entropyImageRTL)
 
    
    
    
    print("SPN_Forward is Ended Successfully !!!")
    return 0   

def image_encryption():
    print("Image Encryption is Started........... !!!")
    # Write code Here
    lenaimagePath = ROSIEPATH + "lenna.png"
    imOriginal = Image.open(lenaimagePath)
    #imOriginal = cv2.imread(lenaimagePath, cv2.IMREAD_GRAYSCALE)
    #imOriginal.show()
    SPN_Forward(imOriginal)
    print("Image Encryption is Ended Successfully !!!")
    return 0

def image_decryption():
    print("Image Decryption is Started........... !!!")
    # Write code Here
    print("Image Decryption is Ended Successfully !!!")
    return 0

def run_beta():
    print("Beta Program is Started........... !!!")
    # Write code Here
    image_encryption()
    print("Beta Program is Ended Successfully !!!")
    return BETA

def run_alpha():
    print("Alpha Program is Started........... !!!")
    # Write code Here
    print("Alpha Program is Ended Successfully !!!")
    return ALPHA
    

def run_bravo():
    print("Bravo Program is Started........... !!!")
    # Write code Here
    print("Bravo Program is Ended Successfully !!!")
    return BRAVO

def run_charlie():
    print("Charlie Program is Started........... !!!")
    # Write code Here
    print("Charlie Program is Ended Successfully !!!")
    return CHARLIE
    
def switch_mode(mode):
    # Program Started
    switcher = {
        0: run_beta,
        1: run_alpha,
        2: run_bravo,
        3: run_charlie
        
        
    }
     # Get the function from switcher dictionary
    func = switcher.get(mode, lambda: "Invalid mode")
    # Execute the function
    print("Mode Selected : ",func())
    # Program Ended 

def run():
    print("......................Main Program is Started........... !!!\n")
    # write coode here
    runMode = BETA
    if (runMode == ALPHA):
        run_alpha()
    else:
        run_beta()
        
    print("\n......................Main Program is Ended Successfully !!!")

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def print_bye(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Bye, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi(' Bilal Dastagir')
    run()
    print_bye('Bilal Dastagir')