import numpy as np
#%matplotlib notebook
from matplotlib import pyplot as plt
from skimage.io import imread,imshow,imsave
from skimage import io, color, img_as_ubyte
from PIL import Image


import random
from Cryptodome.Cipher import AES
from Cryptodome import Random
import cv2


ROSIEPATH = "/Users/bilaldastagir/Documents/vscode/ROSIE/"

def entropy(im):
    # Compute normalized histogram -> p(g)
    p = np.array([(im==v).sum() for v in range(256)])
    p = p/p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p>0]*np.log2(p[p>0])).sum()
    
    return e





testentropyimagePath = ROSIEPATH + "imageCipherTest.png"
testentropyimagePathPlain = ROSIEPATH + "Lenna.png"
testentropyimagePathCipher = ROSIEPATH + "imageCipherTestAES.png"
resizedtestentropyimagePath = ROSIEPATH + "resizedimageCipherTest.png"

img=cv2.imread(testentropyimagePathPlain,1)#read image
na = np.array(img)#conver it to array
x, y ,pp= img.shape[:3]#size of 3d
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blue= np.array(range(x*y), int).reshape((x, y))
enc_blue= np.array(range(x*y), int).reshape((x, y))
blue[:,:]=gray[:, :]
key = b'Sixteen byte key'
iv=b'0000000000000000'
cipher = AES.new(key, AES.MODE_CFB, iv)
L2=[]
blue1 = np.array(range(x),int)
for i in range(x):
    blue1=blue[i,:].tolist()
    blue2=bytes(blue1)
    msg =  cipher.encrypt(blue2)
    for p in msg:
        L2 += [(p)]
    enc_blue[i,:]=L2[:]
    L2=[]
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