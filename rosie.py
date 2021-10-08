#ROSIE
#*****************************************************************************
#
#                            ROSIE: SBox and Invserse SBox Code.
#                             Written  by Bilal Dastagir.
#                                Oct, 6th, 2021
#
#******************************************************************************


# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import shannon_entropy
import random 
import math
from PIL import Image 
import time
from itertools import combinations
from itertools import permutations
import statistics as st
from array import array


#Global Variables 
BETA = [0]
ALPHA = [1]
BRAVO  = [2]
CHARLIE = [3]

ROSIEPATH = "/Users/bilaldastagir/Documents/vscode/ROSIE/"

bahram_2021_Entropy = 7.6739952595684615

best_2021_Entropy = 7.694180448822675 

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

sbox_Bahram_2021 =  [130, 19, 159, 107, 217, 188, 118, 231, 250, 161, 240, 120, 202, 196, 48, 62, 125, 5, 126, 1, 163, 47, 89, 165, 175, 0, 117, 191, 53, 226, 251, 189, 245, 94, 200, 193, 173, 201, 248, 172, 18, 58, 199, 4, 111, 12, 254, 21, 14, 195, 152, 255, 93, 28, 41, 239, 3, 67, 41, 102, 109, 73, 174, 112, 78, 87, 149, 57, 205, 156, 171, 60, 10, 61, 242, 252, 134, 227, 208, 233, 27, 92, 181, 178, 42, 11, 65, 65, 164, 22, 247, 2, 194, 52, 33, 100, 197, 63, 170, 155, 95, 180, 59, 214, 229, 110, 76, 3, 223, 60, 219, 44, 63, 203, 222, 144, 36, 26, 6, 230, 136, 140, 23, 237, 211, 16, 18, 232, 215, 167, 0, 213, 185, 224, 76, 50, 253, 234, 218, 246, 121, 90, 221, 190, 115, 123, 209, 158, 98, 66, 157, 99, 50, 55, 216, 99, 69, 150, 147, 2, 198, 9, 207, 20, 88, 23, 154, 129, 206, 39, 98, 70, 39, 36, 62, 24, 238, 176, 83, 7, 244, 82, 142, 168, 243, 132, 137, 212, 5, 31, 143, 27, 228, 35, 92, 49, 13, 96, 160, 42, 34, 37, 12, 113, 46, 68, 168, 162, 78, 146, 235, 124, 187, 131, 103, 128, 154, 4, 138, 110, 127, 142, 114, 108, 133, 79, 29, 77, 192, 104, 119, 7, 127, 79, 137, 210, 241, 123, 176, 55, 10, 236, 74, 184, 180, 32, 139, 90, 38, 51, 8, 30, 75, 136, 46, 51]
inv_sbox_Bahram_2021 =  [130, 19, 159, 107, 217, 188, 118, 231, 250, 161, 240, 120, 202, 196, 48, 62, 125, 5, 126, 1, 163, 47, 89, 165, 175, 0, 117, 191, 53, 226, 251, 189, 245, 94, 200, 193, 173, 201, 248, 172, 18, 58, 199, 4, 111, 12, 254, 21, 14, 195, 152, 255, 93, 28, 41, 239, 3, 67, 41, 102, 109, 73, 174, 112, 78, 87, 149, 57, 205, 156, 171, 60, 10, 61, 242, 252, 134, 227, 208, 233, 27, 92, 181, 178, 42, 11, 65, 65, 164, 22, 247, 2, 194, 52, 33, 100, 197, 63, 170, 155, 95, 180, 59, 214, 229, 110, 76, 3, 223, 60, 219, 44, 63, 203, 222, 144, 36, 26, 6, 230, 136, 140, 23, 237, 211, 16, 18, 232, 215, 167, 0, 213, 185, 224, 76, 50, 253, 234, 218, 246, 121, 90, 221, 190, 115, 123, 209, 158, 98, 66, 157, 99, 50, 55, 216, 99, 69, 150, 147, 2, 198, 9, 207, 20, 88, 23, 154, 129, 206, 39, 98, 70, 39, 36, 62, 24, 238, 176, 83, 7, 244, 82, 142, 168, 243, 132, 137, 212, 5, 31, 143, 27, 228, 35, 92, 49, 13, 96, 160, 42, 34, 37, 12, 113, 46, 68, 168, 162, 78, 146, 235, 124, 187, 131, 103, 128, 154, 4, 138, 110, 127, 142, 114, 108, 133, 79, 29, 77, 192, 104, 119, 7, 127, 79, 137, 210, 241, 123, 176, 55, 10, 236, 74, 184, 180, 32, 139, 90, 38, 51, 8, 30, 75, 136, 46, 51]


sorted_List_256 =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255]



#Entropy = 7.659934382909035
ksbox1F =  [15, 5, 8, 9, 10, 6, 7, 4, 3, 1, 12, 0, 11, 2, 13, 14]
ksbox1B =  [11, 9, 13, 8, 7, 1, 5, 6, 2, 3, 4, 12, 10, 14, 15, 0]
ksbox2F =  [7, 2, 14, 8, 13, 12, 11, 3, 1, 10, 9, 6, 5, 4, 15, 0]
ksbox2B =  [15, 8, 1, 7, 13, 12, 11, 0, 3, 10, 9, 6, 5, 4, 2, 14]

#Entropy = 
msbox1F =  [15, 14, 0, 1, 3, 2, 12, 13, 4, 9, 7, 10, 8, 5, 11, 6]
msbox1B =  [2, 3, 5, 4, 8, 13, 15, 10, 12, 9, 11, 14, 6, 7, 1, 0]
msbox2F =  [5, 8, 6, 11, 9, 4, 10, 7, 2, 3, 13, 12, 14, 15, 1, 0]
msbox2B =  [15, 14, 8, 9, 5, 0, 2, 7, 1, 4, 6, 3, 11, 10, 12, 13]


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

def get_inv_sbox(sboxFL,len):  
    # Program Started
    sboxBL = sboxFL
    for i in range(len):
        sboxBL[sboxFL[i]] = i
    return sboxBL
    # Program Ended   

def get_inv_sbox2(sboxFL,len):  
    # Program Started
    #sboxBL = sboxFL
    sboxBL = array('i',(0 for i in range(0,255)))
    print("Starting : ",sboxBL)
    for i in range(len):
        t = sboxFL[i]
       # print("i = ",i," -> t = ", t)
        sboxBL[t] = i
        #print("i = ",i," -> ",sboxFL[i]," -> ",sboxBL[t])
        print("SBoxFL[",i,"] = ",sboxFL[i])
        print("SBoxBL[",i,"] = ",sboxBL[i])  
    
    # for k in range(len):
    #     print("SBoxFL[",k,"] = ",sboxFL[k])
    #     print("SBoxBL[",k,"] = ",sboxBL[k])  
    # print("SBoxFL = ",sboxFL)
    # print("SBoxBL = ",sboxBL)
    return sboxBL
    # Program Ended  

def byteSplit(integer):
    return divmod(integer, 0x10)

def byteJoin(num1, num2):
    #num1=0x25;
    #num2=0x71;
    num3=(num1<<4)|(num2);
    #print("%x %d",num3,num3);
    return num3


def s_box_proposed_magic(byteIn):
    #start_time = time.time()
    byteHigh, byteLow = byteSplit(byteIn)
    newByteLow = msbox2F[byteLow]
    newByteHigh = msbox1F[byteHigh]^newByteLow

    byteOut = byteJoin(newByteHigh,newByteLow)
    
    #print("\nSbox Values is ",hex(byteOut))
    #print("sBox Execution Time --- %s seconds ---" % (time.time() - start_time))

    return byteOut

def inv_s_box_proposed_magic(byteIn):
    #start_time = time.time()
    byteHigh, byteLow = byteSplit(byteIn)
    newByteHigh = msbox1B[byteHigh^byteLow]
    newByteLow = msbox2B[byteLow]
    byteOut = byteJoin(newByteHigh,newByteLow)
    #print("\nSbox Values is ",hex(byteOut))
    #print("Inv-sBox Execution Time --- %s seconds ---" % (time.time() - start_time))

    return byteOut

def s_box_proposed_rosie(byteIn):
    #start_time = time.time()
    # byteHigh, byteLow = byteSplit(byteIn)
    # newByteLow = ksbox2F[byteLow]
    # newByteHigh = ksbox1F[byteHigh]^newByteLow
    byteOut = sbox_ROSIE[byteIn]
    # byteOut = byteJoin(newByteHigh,newByteLow)
    #byteOut = s_box_proposed_magic(byteOut)
    #print("\nSbox Values is ",hex(byteOut))
    #print("sBox Execution Time --- %s seconds ---" % (time.time() - start_time))

    return byteOut

def inv_s_box_proposed_rosie(byteIn):
    #start_time = time.time()
    #byteIn = inv_s_box_proposed_magic(byteIn)
    # byteHigh, byteLow = byteSplit(byteIn)
    # newByteHigh = ksbox1B[byteHigh^byteLow]
    # newByteLow = ksbox2B[byteLow]
    byteOut = inv_sbox_ROSIE[byteIn]
    # byteOut = byteJoin(newByteHigh,newByteLow)
    #print("\nSbox Values is ",hex(byteOut))
    #print("Inv-sBox Execution Time --- %s seconds ---" % (time.time() - start_time))

    return byteOut



def prop_sbox_byte_Forward(val):
    # Program Started 
    ret =  val
    return val
    # Program Ended
    
def prop_sbox_byte_Backward(val):
    # Program Started 
    ret = val
    return ret
    # Program Ended   

def entropy(signal):
        '''
        function returns entropy of a signal
        signal must be a 1-D numpy array
        '''
        lensig=signal.size
        symset=list(set(signal))
        numsym=len(symset)
        propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
        ent=np.sum([p*np.log2(1.0/p) for p in propab])
        return ent


# Python program to generate
# odd sized magic squares
# A function to generate odd
# sized magic squares
 
 
def generateSquare(n):
 
    # 2-D array with all
    # slots set to 0
    magicSquare = [[0 for x in range(n)]
                   for y in range(n)]
 
    # initialize position of 1
    i = n / 2
    j = n - 1
 
    # Fill the magic square
    # by placing values
    num = 1
    while num <= (n * n):
        if i == -1 and j == n:  # 3rd condition
            j = n - 2
            i = 0
        else:
 
            # next number goes out of
            # right side of square
            if j == n:
                j = 0
 
            # next number goes
            # out of upper side
            if i < 0:
                i = n - 1
 
        if magicSquare[int(i)][int(j)]:  # 2nd condition
            j = j - 2
            i = i + 1
            continue
        else:
            magicSquare[int(i)][int(j)] = num
            num = num + 1
 
        j = j + 1
        i = i - 1  # 1st condition
 
    # Printing magic square
    print("Magic Square for n =", n)
    print("Sum of each row or column",
          n * (n * n + 1) / 2, "\n")
 
    for i in range(0, n):
        for j in range(0, n):
            print('%2d ' % (magicSquare[i][j]),
                  end='')
 
            # To display output
            # in matrix form
            if j == n - 1:
                print()
 
# Driver Code
 
magic_sum4x4=34
sum_range4x4=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15],[0,5,10,15],[3,6,9,12]]
 
def is_magic4x4(m):
    # Program Started
    for i in range(len(sum_range4x4)):
        if sum(m[x] for x in sum_range4x4[i])!=magic_sum4x4:
            return False
    return True
    # Program Ended 

def reorder(m,order):
    # Program Started 
    n=list(m)
    for i in range(len(order)):
        n[i]=m[order[i]]
    return n
    # Program Ended
 
def run_msca(sorted_list, size):
    # Program Started
    #comb = combinations(sorted_list, size) 
    #comb = combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255], 16) 
    #comb = combinations([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256], 16) 
   
    comb = combinations([0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15], 4) 
    comb= list(comb)
   # print(len(comb))
    comb_list=list()
    for i in range(len(comb)):
        if sum(comb[i])==30:
            comb_list.append(comb[i])
            print(comb[i])
    #print("comb = ",comb_list)
    print(len(comb_list))
    
    comb2=list(combinations(list(range(86)),4))
    possible_comb=list()
    t=0
    for i in range(len(comb2)):
        
        m= comb_list[comb2[i][0]]+comb_list[comb2[i][1]] +comb_list[comb2[i][2]]+comb_list[comb2[i][3]]
        test_list = list(set(m))
        if len(test_list)==16 :
            possible_comb.append(m)
            t +=1
    print(t)
    print(possible_comb)
    print("\n.... .... ....\n")
    index_perm=list(permutations([0,1,2,3]))
    magic4x4_list=list()
    t=0
    for g in range(len(possible_comb)):
        m=possible_comb[g]
        a=m[0:4]
        b=m[4:8]
        c=m[8:12]
        d=m[12:16]
        for p in range(len(index_perm)):
            for i in range(len(index_perm)):
                for j in range(len(index_perm)):
                    for k in range(len(index_perm)):
                        for l in range(len(index_perm)):
                            new_a=reorder(a,index_perm[i])
                            new_b=reorder(b,index_perm[j])
                            new_c=reorder(c,index_perm[k])
                            new_d=reorder(d,index_perm[l])
                            mm= [new_a,new_b,new_c,new_d]
                            new_mm=reorder(mm,index_perm[p])
                            new_m=new_mm[0]+new_mm[1]+new_mm[2]+new_mm[3]
                        
                            if is_magic4x4(new_m):
                                t+=1
                                print(t,new_m)
                                magic4x4_list.append(new_m)
                                print(new_m)
                                print("Found Pair  .... : ",new_m)
                            #else:
                                #print("Rearranging .... : ",new_m)
    print(len(magic4x4_list))
    
    # Program Ended

def calculate_entropy(colorIm,N):
    # Program Started
    greyIm=colorIm.convert('L')
    colorIm=np.array(colorIm)
    greyIm=np.array(greyIm)
    S=greyIm.shape
    E=np.array(greyIm)
    for row in range(S[0]):
            for col in range(S[1]):
                    Lx=np.max([0,col-N])
                    Ux=np.min([S[1],col+N])
                    Ly=np.max([0,row-N])
                    Uy=np.min([S[0],row+N])
                    region=greyIm[Ly:Uy,Lx:Ux].flatten()
                    E[row,col]=entropy(region)
    print("Entropy = ",E)
    return E
    # Program Ended

def runsboximage():
    # Program Started
    imagePathWBC = ROSIEPATH + "WB_Checkered.jpeg"
    imagePathNoise  = ROSIEPATH + "noise.png"
    imagePath = ROSIEPATH + "Lenna.png"
    imagePathTC = ROSIEPATH + "Lenna_Traditional_Cipher.png"
    imagePathBC = ROSIEPATH + "Lenna_Bahram_2021_Cipher.png"
    imagePathMC = ROSIEPATH + "Lenna_MagicSquare_Cipher.png"
    imagePathRC = ROSIEPATH + "Lenna_ROSIE_NOISE_Cipher.png"
    imagePathRSC = ROSIEPATH + "Lenna_ROSIE_ROSIE_Cipher.png"
    imagePathTD = ROSIEPATH + "Lenna_Traditional_Decipher.png"
    imagePathBD = ROSIEPATH + "Lenna_Bahram_2021_Decipher.png"
    imagePathMD = ROSIEPATH + "Lenna_MagicSquare_Decipher.png"
    imagePathRD = ROSIEPATH + "Lenna_ROSIE_NOISE_Decipher.png"
    imagePathRSD = ROSIEPATH + "Lenna_ROSIE_ROSIE_Decipher.png"
    print(imagePath)
    imageNoise = Image.open(imagePathNoise)
    imageWBC  = Image.open(imagePathWBC)
    image_array_wbc = np.array(imageWBC)
    image_array_noise = np.array(imageNoise)
    imageOriginal = Image.open(imagePath)
    image_array_inT = np.array(imageOriginal)
    image_array_outT = image_array_inT
    image_array_inB = np.array(imageOriginal)
    image_array_outB = image_array_inB
    image_array_inM = np.array(imageOriginal)
    image_array_outM = image_array_inM
    image_array_inR = np.array(imageOriginal)
    image_array_outR = image_array_inR
    image_array_outRS = image_array_inR
    
    for i in range(len(image_array_inT)):
        for j in range(len(image_array_inT)):
            for k in range(3):
                image_array_outT[i][j][k] = sub_bytes_AES_Traditional(image_array_inT[i][j][k])
                image_array_outB[i][j][k] = sub_bytes_Bahram_2021(image_array_inB[i][j][k])
                image_array_outM[i][j][k]=  s_box_proposed_magic(image_array_inM[i][j][k])
                image_array_outR[i][j][k]=  image_array_inR[i][j][k]^image_array_noise[i][j][k]#s_box_proposed_rosie(image_array_inR[i][j][k]^image_array_noise[i][j][k])
                image_array_outRS[i][j][k]=  image_array_inR[image_array_inT - i][image_array_inT - j][k]#^image_array_noise[i][j][k]
    imageUpdateT=Image.fromarray(image_array_outT)
    imageUpdateT.save(imagePathTC)
    imageUpdateB=Image.fromarray(image_array_outB)
    imageUpdateB.save(imagePathBC)
    imageUpdateM=Image.fromarray(image_array_outM)
    imageUpdateM.save(imagePathMC)
    imageUpdateR=Image.fromarray(image_array_outR)
    imageUpdateR.save(imagePathRC)
    imageUpdateRS=Image.fromarray(image_array_outRS)
    imageUpdateRS.save(imagePathRC)
    
    #....................
    imageTC = Image.open(imagePathTC)
    image_arrayTC = np.array(imageTC)
    image_arrayTD = image_arrayTC
    imageBC = Image.open(imagePathBC)
    image_arrayBC = np.array(imageBC)
    image_arrayBD = image_arrayBC
    
    imageMC = Image.open(imagePathMC)
    image_arrayMC = np.array(imageMC)
    image_arrayMD = image_arrayMC
    imageRC = Image.open(imagePathRC)
    image_arrayRC = np.array(imageRC)
    image_arrayRD = image_arrayRC
    
    imageEntropy = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    imageEntropyTC = cv2.imread(imagePathTC, cv2.IMREAD_GRAYSCALE)
    imageEntropyBC = cv2.imread(imagePathBC, cv2.IMREAD_GRAYSCALE)
    imageEntropyMC = cv2.imread(imagePathMC, cv2.IMREAD_GRAYSCALE)
    imageEntropyRC = cv2.imread(imagePathRC, cv2.IMREAD_GRAYSCALE)
    imageEntropyRSC = cv2.imread(imagePathRSC, cv2.IMREAD_GRAYSCALE)
    
    entropyImage = shannon_entropy(imageEntropy, 2)
    entropyImageTC = shannon_entropy(imageEntropyTC, 2)
    entropyImageBC = shannon_entropy(imageEntropyBC, 2)
    entropyImageMC = shannon_entropy(imageEntropyMC, 2)
    entropyImageRC = shannon_entropy(imageEntropyRC, 2)
    
    print("Entropy Image -> h = ", entropyImage)
    print("Entropy ImageTC -> h = ", entropyImageTC)
    print("Entropy ImageBC -> h = ", entropyImageBC)
    print("Entropy ImageMC -> h = ", entropyImageMC)
    print("Entropy ImageRC -> h = ", entropyImageRC)
    
    for i in range(len(image_array_inT)):
        for j in range(len(image_array_inT)):
            for k in range(3):
                image_arrayTD[i][j][k] = inv_sub_bytes_AES_Traditional(image_arrayTC[i][j][k])
                image_arrayBD[i][j][k] = inv_sub_bytes_Bahram_2021(image_arrayBC[i][j][k])
                image_arrayMD[i][j][k]=  inv_s_box_proposed_magic(image_arrayMC[i][j][k])
                image_arrayRD[i][j][k]=  image_arrayRC[i][j][k]^image_array_noise[i][j][k] #(inv_s_box_proposed_rosie(image_arrayRC[i][j][k]))^image_array_noise[i][j][k]
                image_array_outRS[i][j][k]=  image_array_inR[image_array_inT - i][image_array_inT - j][k]
    imageTD=Image.fromarray(image_arrayTD)
    imageTD.save(imagePathTD)
    imageBD=Image.fromarray(image_arrayBD)
    imageBD.save(imagePathBD)
    imageMD=Image.fromarray(image_arrayMD)
    imageMD.save(imagePathMD)
    imageRD=Image.fromarray(image_arrayRD)
    imageRD.save(imagePathRD)
    
    # Program Ended


def run_beta():
    print("Beta Program is Started........... !!!")
    # Write code Here
    sq_len  = 4
    slen = sq_len * sq_len
    mX = [15, 14, 0, 1, 3, 2, 12, 13, 4, 9, 7, 10, 8, 5, 11, 6]
    kX = [15, 5, 8, 9, 10, 6, 7, 4, 3, 1, 12, 0, 11, 2, 13, 14]
    rX = sbox_ROSIE
    rX_2d = np.reshape(rX, (slen, slen)) 
    bX = bahram_2021_sbox
    bX_2d = np.reshape(bX, (slen, slen)) 
    for g in range(slen):
        print("HM  [B] = ", st.harmonic_mean(bX_2d[g])," and HM [R] = ",st.harmonic_mean(rX_2d[g]))
        print("Mean [B] = ", st.mean(bX_2d[g])," and Mean [R] = ",st.mean(rX_2d[g]))
          
    kX_2d = np.reshape(kX, (sq_len, sq_len))
    print("King X : \n",kX_2d)
    mX_2d = np.reshape(mX, (sq_len, sq_len))
    print("Magic X : \n",mX_2d)
    s = sorted(random.sample(range(0, slen), slen))
    #plt.hist(mX, bins=s)#bins = 4)
    #plt.show()
    for h in range(sq_len):
            #print("SD  [M] = ",st.stdev(mX_2d[h])," and SD  [K] = ",st.stdev(kX_2d[h]))
            #print("Var [M] = ",st.variance(mX_2d[h])," and Var [K] = ",st.variance(kX_2d[h]))
            print("HM  [M] = ", st.harmonic_mean(mX_2d[h])," and HM [K] = ",st.harmonic_mean(kX_2d[h]))
            print("Mean [M] = ", st.mean(mX_2d[h])," and Mean [K] = ",st.mean(kX_2d[h]))
            #print("Mean [M] = ", st.m)
            #print("GM [M] = ", st.geometric_mean(mX_2d[h])," and GM [K] = ", st.geometric_mean[kX_2d[h]])
    #         print("Standard Deviation of Magic Square [M] is % s " % (st.stdev(mX_2d[h])))
    #         print("Variance of the Magic Square       [M] is % s" % (st.variance(mX_2d[h])))
    #         print("Standard Deviation of King Square  [K] is % s " % (st.stdev(kX_2d[h])))
    #         print("Variance of the King Square        [K] is % s" % (st.variance(kX_2d[h])))
    X = sorted(random.sample(range(0, slen), slen))
    print(X)
    for i in range(slen):
        sample = random.sample(range(0, slen), slen) 
        sample_2d = np.reshape(sample, (sq_len, sq_len))
        #print("Sample 1D: \n",sample)
        #print("Sample 2D: \n",sample_2d)
        #for j in range(sq_len):
        #    print("Standard Deviation of sample is % s " % (statistics.stdev(sample_2d[j])))
    #y = sbox_ROSIE
    #print(y)
  
    
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
    n = 16
    generateSquare(n)
    
    print("Bravo Program is Ended Successfully !!!")
    return BRAVO

def run_charlie():
    print("Charlie Program is Started........... !!!")
    # Write code Here
    runsboximage()
 

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
    runMode = CHARLIE#BETA
    if (runMode == CHARLIE):
        run_charlie()
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