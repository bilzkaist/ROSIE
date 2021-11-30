import numpy as np
#%matplotlib notebook
from matplotlib import pyplot as plt
from skimage.io import imread,imshow,imsave
from skimage import io, color, img_as_ubyte
from PIL import Image

import copy
import random
#from Cryptodome.Cipher import AES 
from Cryptodome import Random
import cv2

from aes import AES, encrypt, decrypt

ROSIEPATH = "/Users/bilaldastagir/Documents/vscode/ROSIE/"

# SBox RTL Entropy = 7.659934382909035
sbox1FRTL =  [15, 5, 8, 9, 10, 6, 7, 4, 3, 1, 12, 0, 11, 2, 13, 14]
sbox1BRTL =  [11, 9, 13, 8, 7, 1, 5, 6, 2, 3, 4, 12, 10, 14, 15, 0]
sbox2FRTL =  [7, 2, 14, 8, 13, 12, 11, 3, 1, 10, 9, 6, 5, 4, 15, 0]
sbox2BRTL =  [15, 8, 1, 7, 13, 12, 11, 0, 3, 10, 9, 6, 5, 4, 2, 14]
# End of Proposed Dual Quad-Bit SBox  #



def byteSplit(integer):
    return divmod(integer, 0x10)


def byteJoin(num1, num2):
    num3 = (num1 << 4) | (num2);
    return num3

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

def entropy(im):
    # Compute normalized histogram -> p(g)
    p = np.array([(im==v).sum() for v in range(256)])
    p = p/p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p>0]*np.log2(p[p>0])).sum()
    
    return e


def encryptROSIE(parrayIn):
    plenX = len(parrayIn)
    plenY = len(parrayIn[0])
    pmutX  = random.sample(range(0, plenX), plenX)
    pmutY  = random.sample(range(0, plenY), plenY)
    parrayOut_C = copy.deepcopy(parrayIn)
    for i in range(plenX):
        for j in range(plenY):
            parrayOut_C[pmutX[i]][pmutY[j]] = sBoxRTL_Forward(parrayIn[i][j])
    return parrayOut_C

testentropyimagePath = ROSIEPATH + "imageCipherTest.png"
testentropyimagePathLenna = ROSIEPATH + "Lenna.png"
testentropyimagePathPlain = ROSIEPATH + "imagePlainTestAES.png"
testentropyimagePathCipher = ROSIEPATH + "imageCipherTestAES.png"
resizedtestentropyimagePath = ROSIEPATH + "resizedimageCipherTest.png"

img=cv2.imread(testentropyimagePathLenna,1)#read image
na = np.array(img)#conver it to array
x, y ,pp= img.shape[:3]#size of 3d
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blue= np.array(range(x*y), int).reshape((x, y))
enc_blue= np.array(range(x*y), int).reshape((x, y))
blue[:,:]=gray[:, :]
cv2.imwrite(testentropyimagePathPlain , blue)

blueImage = Image.open(testentropyimagePathPlain)
blueImage.show()
print("Plain Image : ",blue)
key = b'Sixteen byte key'
iv=b'0000000000000000'
#cipher = AES.new(key, AES.MODE_CFB, iv)
L2=[]
print("Entropy Plain = ",entropy(blue))
iv = b'\x02' * 16
blue1 = np.array(range(x),int)
for i in range(x):
    blue1=blue[i,:].tolist()
    blue2=bytes(blue1)
    blue3 = bytearray(blue1)
    #print("Shape = ",blue.shape)
    #print("blue1 : ",blue1)
   # print("Len : ",len(blue2))
    #msg =  cipher.encrypt(blue2)
    message = b'\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xAA\xBB\xCC\xDD\xEE\xFF'
    aes = AES(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f')
    #ciphertext = aes.encrypt_block(message)
    msg =  aes.encrypt_cbc(blue2,iv)
   
        
   # msg = encryptROSIE(blue3)
    #print("Enc(msg) RAW = ",msg)
    print("Length = ",len(msg))
    for p in msg:
        L2 += [(p)]
    #print("Enc(msg) FOR = ",L2)
    for j in range(len(enc_blue)):
        enc_blue[i][j] = msg[j]
    #enc_blue[i,:]=L2[:]
    #print("Enc(msg) FOR = ",enc_blue)
    L2=[]
print("Entropy Encrypt = ",entropy(enc_blue))
print("Encrypt Image : ",enc_blue)
enc_blue2 = encryptROSIE(blue)    
cv2.imwrite(testentropyimagePathCipher , enc_blue)


image = Image.open(testentropyimagePath)

resized_img = image.resize((512, 512))
resized_img.show()
resized_img.save(resizedtestentropyimagePath)




rgbImg = io.imread(testentropyimagePathCipher)

#grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
grayImg = io.imread(testentropyimagePathCipher)
# print("Test         Entropy -> h = ",entropy(grayImg, 2))
#print("Test (2)     Entropy -> h = ",shannon_entropy(grayImg, 2))


a = (np.ones((100,100))*255).astype('uint8')
b = (np.random.random((100,100))*256).astype('uint8')

plt.figure()
plt.subplot(1,4,1)
imshow(a)
plt.title(entropy(a))
plt.subplot(2,4,2)
imshow(b)
plt.title(entropy(b))
plt.subplot(3,4,3)
imshow(grayImg)
plt.title(entropy(grayImg))
plt.show()