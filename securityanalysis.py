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
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
from sklearn.metrics import mean_absolute_error

from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy

import pyhomogeneity as hg
import collections
from scipy.stats import entropy
import copy
from skimage.metrics import structural_similarity as ssim
from bioinfokit import analys, visuz
from scipy import signal
import hashlib

#Global Variables 
BETA = [0]
ALPHA = [1]
BRAVO  = [2]
CHARLIE = [3]

ROSIEPATH = "/Users/bilaldastagir/Documents/vscode/ROSIE/"
imagePathLenna = ROSIEPATH + "Lenna.png"
imagePathCipher_AES = ROSIEPATH + "imageCipherAES.png"
imagePathCipher_Bahrami_2021 = ROSIEPATH + "imageCipher_Bahrami_2021.png"
imagePathCipher_DQB_RTL = ROSIEPATH + "imageCipherDQB-RTL.png"
imagePathCipher_SEB_RTL = ROSIEPATH + "imageCipherSEB-RTL.png"

def getMSE(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1]*255)

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def getMAE(imageA, imageB):
    mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float"))))
    mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
    if (mae < 0):
        return mae * -1
    else:
        return mae

def entropy_analysis():
    # Start
    #Entropy 
    imageLenna = cv2.imread(imagePathLenna, cv2.IMREAD_GRAYSCALE)
    imageCipherAES = cv2.imread(imagePathCipher_AES, cv2.IMREAD_GRAYSCALE)
    imageCipherBahrami_2021  = cv2.imread(imagePathCipher_Bahrami_2021, cv2.IMREAD_GRAYSCALE)
    imageCipherDQB_RTL = cv2.imread(imagePathCipher_DQB_RTL, cv2.IMREAD_GRAYSCALE)
    imageCipherSEB_RTL = cv2.imread(imagePathCipher_SEB_RTL, cv2.IMREAD_GRAYSCALE)
    
    print("Lena Plain Image     (Original)         Entropy -> h = ",shannon_entropy(imageLenna  , 2))
    print("AES Cipher SBox      (Traditional)      Entropy -> h = ",shannon_entropy(imageCipherAES , 2))
    print("Bahrami_2021 Cipher SBox                Entropy -> h = ",shannon_entropy(imageCipherBahrami_2021 , 2))
    print("DQB_RTL Cipher SBox  (Propose Method 1) Entropy -> h = ",shannon_entropy(imageCipherDQB_RTL , 2))
    print("SEB_RTL Cipher SBox  (Propose Method 2) Entropy -> h = ",shannon_entropy(imageCipherSEB_RTL , 2))
    print(".........................................................Entropy Remarks : Ideal is Eight or very close is Better !!!")
    # End 
 
def other_analysis():
    # Start
    #PSNR
    imgTest = cv2.imread(imagePathLenna) 
    psnrtest_test = cv2.PSNR(imgTest, imgTest)
    print("PSNR Test with Test Lena Image               = ", psnrtest_test)
    

    imgCipherAES = cv2.imread(imagePathCipher_AES) 
    psnrtest_cipherAES= cv2.PSNR(imgTest, imgCipherAES)
    print("AES Cipher SBox      (Traditional)      PSNR = ", psnrtest_cipherAES) 
    
    imgCipherBahrami2021 = cv2.imread(imagePathCipher_Bahrami_2021) 
    psnrtest_cipherBahrami2021= cv2.PSNR(imgTest, imgCipherBahrami2021)
    print("Bahrami_2021 Cipher SBox                PSNR = ", psnrtest_cipherBahrami2021) 
    
    imgCipherDQB_RTL = cv2.imread(imagePathCipher_DQB_RTL) 
    psnrtest_cipher= cv2.PSNR(imgTest, imgCipherDQB_RTL)
    print("DQB_RTL Cipher SBox  (Propose Method 1) PSNR = ", psnrtest_cipher) 
    
    imgCipherSEB_RTL  = cv2.imread(imagePathCipher_SEB_RTL ) 
    psnrtest_cipher= cv2.PSNR(imgTest, imgCipherSEB_RTL)
    print("SEB_RTL Cipher SBox  (Propose Method 2) PSNR = ", psnrtest_cipher) 
    print(".........................................................PSNR Remarks : Lower is Better !!!")
    
    # MSE
    print("MSE Original                                = ", getMSE(imgTest, imgTest))
    print("MSE Cipher AES  (Traditional)               = ", getMSE(imgTest, imgCipherAES))
    print("MSE Cipher Bahrami 2021                     = ", getMSE(imgTest, imgCipherBahrami2021))
    print("MSE DQB_RTL Cipher SBox  (Propose Method 1) = ", getMSE(imgTest, imgCipherDQB_RTL))
    print("MSE SEB_RTL Cipher SBox  (Propose Method 2) = ", getMSE(imgTest, imgCipherSEB_RTL))
    print(".........................................................PSNR Remarks : Higher is Better !!!")
    
    # MAE
    print("MAE Original                                = ", getMAE(imgTest, imgTest))
    print("MAE Cipher AES  (Traditional)               = ", getMAE(imgTest, imgCipherAES))
    print("MAE Cipher Bahrami 2021                     = ", getMAE(imgTest, imgCipherBahrami2021))
    print("MAE DQB_RTL Cipher SBox  (Propose Method 1) = ", getMAE(imgTest, imgCipherDQB_RTL))
    print("MAE SEB_RTL Cipher SBox  (Propose Method 2) = ", getMAE(imgTest, imgCipherSEB_RTL))
    print(".........................................................PSNR Remarks : Higher is Better !!!")
    
    # SSIM
    imgTestG = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    imgCipherAESG = cv2.cvtColor(imgCipherAES, cv2.COLOR_BGR2GRAY)
    imgCipherBahrami2021G = cv2.cvtColor(imgCipherBahrami2021, cv2.COLOR_BGR2GRAY)
    imgCipherDQB_RTLG = cv2.cvtColor(imgCipherDQB_RTL, cv2.COLOR_BGR2GRAY)
    imgCipherSEB_RTLG = cv2.cvtColor(imgCipherSEB_RTL, cv2.COLOR_BGR2GRAY)

    print("SSIM Original                                = ", ssim(imgTestG, imgTestG))
    print("SSIM Cipher AES  (Traditional)               = ", ssim(imgTestG, imgCipherAESG))
    print("SSIM Cipher Bahrami 2021                     = ", ssim(imgTestG, imgCipherBahrami2021G))
    print("SSIM DQB_RTL Cipher SBox  (Propose Method 1) = ", ssim(imgTestG, imgCipherDQB_RTLG))
    print("SSIM SEB_RTL Cipher SBox  (Propose Method 2) = ", ssim(imgTestG, imgCipherSEB_RTLG))
    print(".........................................................SSIM Remarks : Ideal is Zero or very close is Better !!!")

    # Correlation
    cm = np.corrcoef(imgTestG.flat, imgTestG.flat)
    r = cm[0, 1]
    print("Cor. Original                                = ",r)
    cm = np.corrcoef(imgTestG.flat, imgCipherAESG.flat)
    r = cm[0, 1]
    print("Cor. Cipher AES  (Traditional)               = ", r)
    cm = np.corrcoef(imgTestG.flat, imgCipherBahrami2021G .flat)
    r = cm[0, 1]
    print("Cor. Cipher Bahrami 2021                     = ", r)
    cm = np.corrcoef(imgTestG.flat, imgCipherDQB_RTLG.flat)
    r = cm[0, 1]
    print("Cor. DQB_RTL Cipher SBox  (Propose Method 1) = ", r)
    cm = np.corrcoef(imgTestG.flat, imgCipherSEB_RTLG.flat)
    r = cm[0, 1]
    print("Cor. SEB_RTL Cipher SBox  (Propose Method 2) = ", r)
    print(".........................................................Cor. Remarks : Ideal is Zero or very close is Better !!!")
    print("\n\n.........................................................")
    

    # End    

def run_security_analysis():
    print("Security Analysis Program is Started........... !!!")
    # Write code Here
    entropy_analysis()
    other_analysis()
    
    print("Security Analysis Program is Ended Successfully !!!")
    return BETA


def run_beta():
    print("Beta Program is Started........... !!!")
    # Write code Here
    run_security_analysis()
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