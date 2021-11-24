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

# Entropy  = h =  7.694180448822675 VS Bahram 2021 h =  7.6739952595684615
sbox_ROSIE     =  [54, 115, 104, 244, 221, 164, 20, 211, 157, 113, 246, 171, 144, 161, 26, 41, 179, 181, 52, 37, 122, 46, 127, 67, 4, 134, 58, 228, 163, 240, 8, 33, 131, 222, 170, 62, 2, 212, 133, 252, 101, 103, 34, 202, 110, 89, 165, 218, 156, 132, 85, 106, 81, 70, 32, 78, 153, 233, 65, 73, 169, 74, 237, 31, 177, 29, 123, 224, 45, 142, 214, 232, 75, 68, 242, 22, 238, 28, 213, 61, 150, 55, 193, 197, 10, 180, 12, 83, 11, 3, 76, 60, 25, 19, 90, 178, 247, 18, 15, 23, 5, 135, 191, 1, 231, 59, 100, 226, 56, 95, 201, 9, 79, 94, 249, 48, 111, 254, 198, 172, 0, 248, 200, 138, 219, 235, 146, 243, 255, 152, 16, 96, 185, 126, 174, 229, 40, 82, 203, 105, 184, 21, 204, 63, 160, 154, 6, 236, 88, 205, 155, 98, 143, 64, 207, 14, 253, 245, 49, 167, 87, 35, 208, 30, 102, 209, 24, 92, 36, 210, 158, 183, 93, 141, 148, 97, 175, 107, 39, 50, 159, 220, 182, 206, 125, 227, 149, 225, 71, 51, 99, 195, 72, 173, 53, 38, 17, 118, 91, 121, 186, 77, 120, 80, 137, 192, 47, 147, 187, 176, 199, 108, 119, 27, 129, 251, 13, 42, 130, 196, 117, 43, 217, 166, 162, 188, 7, 234, 189, 116, 124, 216, 230, 241, 84, 57, 239, 223, 194, 112, 114, 168, 145, 66, 44, 128, 109, 140, 139, 190, 215, 136, 69, 86, 151, 250]
inv_sbox_ROSIE =  [120, 103, 36, 89, 24, 100, 146, 226, 30, 111, 84, 88, 86, 216, 155, 98, 130, 196, 97, 93, 6, 141, 75, 99, 166, 92, 14, 213, 77, 65, 163, 63, 54, 31, 42, 161, 168, 19, 195, 178, 136, 15, 217, 221, 244, 68, 21, 206, 115, 158, 179, 189, 18, 194, 0, 81, 108, 235, 26, 105, 91, 79, 35, 143, 153, 58, 243, 23, 73, 252, 53, 188, 192, 59, 61, 72, 90, 201, 55, 112, 203, 52, 137, 87, 234, 50, 253, 160, 148, 45, 94, 198, 167, 172, 113, 109, 131, 175, 151, 190, 106, 40, 164, 41, 2, 139, 51, 177, 211, 246, 44, 116, 239, 9, 240, 1, 229, 220, 197, 212, 202, 199, 20, 66, 230, 184, 133, 22, 245, 214, 218, 32, 49, 38, 25, 101, 251, 204, 123, 248, 247, 173, 69, 152, 12, 242, 126, 207, 174, 186, 80, 254, 129, 56, 145, 150, 48, 8, 170, 180, 144, 13, 224, 28, 5, 46, 223, 159, 241, 60, 34, 11, 119, 193, 134, 176, 209, 64, 95, 16, 85, 17, 182, 171, 140, 132, 200, 208, 225, 228, 249, 102, 205, 82, 238, 191, 219, 83, 118, 210, 122, 110, 43, 138, 142, 149, 183, 154, 162, 165, 169, 7, 37, 78, 70, 250, 231, 222, 47, 124, 181, 4, 33, 237, 67, 187, 107, 185, 27, 135, 232, 104, 71, 57, 227, 125, 147, 62, 76, 236, 29, 233, 74, 127, 3, 157, 10, 96, 121, 114, 255, 215, 39, 156, 117, 128]



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


''' 
keylistFX:  [334, 436, 293, 344, 473, 276, 34, 381, 224, 156, 486, 47, 508, 297, 70, 198, 91, 225, 145, 273, 387, 41, 493, 100, 3, 477, 470, 274, 79, 431, 267, 312, 67, 59, 175, 266, 5, 117, 228, 63, 115, 71, 17, 325, 337, 188, 410, 415, 69, 399, 452, 335, 289, 313, 343, 393, 288, 419, 358, 338, 83, 97, 75, 124, 428, 311, 55, 439, 435, 154, 38, 314, 310, 430, 465, 307, 190, 443, 252, 355, 51, 480, 382, 159, 155, 95, 25, 389, 121, 422, 1, 445, 131, 429, 365, 212, 498, 472, 262, 48, 217, 20, 56, 255, 241, 336, 340, 58, 441, 116, 446, 295, 80, 308, 349, 233, 329, 481, 324, 364, 15, 391, 475, 101, 200, 442, 215, 182, 179, 280, 57, 193, 270, 359, 456, 196, 301, 401, 342, 400, 259, 119, 331, 214, 277, 13, 44, 118, 46, 339, 207, 375, 180, 491, 197, 24, 319, 403, 417, 416, 501, 348, 346, 496, 149, 104, 281, 272, 303, 253, 134, 361, 396, 468, 434, 357, 463, 19, 230, 195, 232, 226, 178, 507, 163, 187, 23, 497, 40, 167, 354, 263, 461, 392, 141, 216, 256, 122, 471, 460, 505, 257, 285, 510, 459, 440, 406, 106, 300, 425, 211, 110, 388, 135, 234, 126, 184, 294, 450, 113, 379, 147, 32, 363, 304, 87, 49, 453, 414, 140, 107, 218, 264, 341, 164, 332, 9, 18, 109, 490, 120, 317, 43, 494, 345, 201, 94, 504, 413, 82, 511, 243, 12, 125, 326, 60, 485, 238, 231, 374, 367, 203, 209, 229, 502, 248, 457, 383, 150, 90, 380, 33, 315, 132, 305, 186, 210, 213, 181, 31, 144, 102, 194, 372, 250, 199, 265, 103, 29, 454, 244, 390, 282, 448, 409, 222, 93, 458, 68, 437, 162, 157, 451, 292, 398, 433, 170, 192, 152, 351, 10, 151, 464, 237, 449, 476, 321, 499, 427, 488, 309, 503, 278, 482, 88, 298, 370, 161, 279, 487, 11, 369, 114, 402, 189, 30, 466, 143, 142, 322, 260, 84, 111, 246, 395, 2, 368, 7, 350, 353, 404, 139, 254, 306, 62, 455, 290, 235, 371, 223, 385, 174, 495, 347, 284, 96, 258, 362, 261, 394, 418, 86, 287, 500, 65, 77, 509, 176, 165, 405, 148, 423, 242, 76, 136, 411, 153, 54, 45, 8, 296, 397, 219, 356, 239, 169, 74, 171, 377, 444, 447, 251, 27, 327, 158, 39, 177, 78, 220, 6, 478, 36, 424, 4, 492, 484, 330, 0, 172, 384, 16, 316, 112, 221, 302, 378, 99, 64, 408, 376, 108, 37, 318, 420, 66, 133, 333, 373, 138, 469, 438, 366, 205, 53, 236, 21, 92, 185, 352, 81, 479, 467, 22, 128, 191, 61, 489, 50, 462, 130, 474, 26, 386, 127, 245, 227, 89, 204, 407, 271, 320, 72, 426, 247, 73, 35, 328, 268, 432, 105, 173, 360, 52, 269, 202, 286, 299, 240, 146, 483, 275, 129, 506, 98, 208, 168, 183, 283, 206, 160, 166, 249, 123, 14, 412, 137, 421, 323, 28, 85, 42, 291]
keylistFX Entropy -> h =  9.000000000000004
keylistFY:  [195, 483, 202, 110, 503, 335, 508, 58, 295, 435, 32, 140, 361, 234, 509, 1, 449, 396, 389, 71, 422, 450, 481, 375, 70, 491, 283, 423, 253, 325, 213, 224, 78, 197, 399, 296, 219, 429, 328, 499, 193, 116, 359, 229, 176, 316, 64, 244, 215, 109, 9, 223, 346, 387, 121, 358, 478, 250, 104, 377, 292, 498, 384, 13, 332, 178, 218, 393, 293, 455, 225, 496, 145, 354, 38, 143, 208, 363, 472, 281, 434, 311, 344, 141, 133, 493, 65, 236, 383, 350, 458, 0, 368, 54, 505, 111, 192, 370, 381, 144, 366, 268, 367, 138, 123, 319, 446, 302, 155, 510, 382, 451, 427, 241, 401, 306, 251, 63, 180, 129, 378, 97, 99, 84, 454, 181, 201, 464, 279, 463, 430, 288, 262, 487, 407, 126, 11, 156, 132, 357, 373, 254, 151, 24, 154, 301, 27, 56, 386, 395, 169, 148, 438, 372, 23, 203, 175, 323, 266, 261, 101, 93, 246, 28, 107, 117, 161, 420, 80, 475, 53, 189, 469, 340, 200, 474, 247, 69, 467, 186, 280, 232, 165, 168, 441, 81, 21, 275, 428, 313, 425, 320, 398, 124, 128, 130, 351, 249, 322, 479, 267, 115, 76, 2, 19, 83, 326, 371, 230, 303, 237, 22, 164, 486, 166, 173, 159, 16, 238, 385, 349, 445, 25, 52, 391, 259, 40, 57, 442, 336, 317, 191, 495, 120, 44, 48, 284, 347, 50, 33, 59, 461, 185, 113, 211, 494, 190, 468, 460, 221, 36, 245, 298, 432, 287, 17, 406, 452, 305, 260, 187, 177, 360, 100, 94, 312, 233, 217, 194, 272, 511, 369, 29, 410, 31, 334, 256, 158, 473, 338, 26, 433, 20, 417, 376, 85, 459, 388, 457, 447, 131, 289, 497, 409, 286, 300, 92, 108, 476, 162, 171, 252, 364, 62, 278, 488, 5, 324, 112, 329, 321, 146, 55, 424, 291, 337, 102, 51, 209, 206, 270, 277, 174, 404, 490, 471, 282, 243, 235, 119, 228, 400, 365, 98, 380, 294, 136, 403, 348, 88, 35, 492, 506, 489, 465, 248, 304, 413, 10, 184, 39, 297, 342, 67, 60, 134, 204, 502, 172, 355, 477, 47, 30, 231, 160, 210, 227, 327, 153, 484, 470, 255, 8, 139, 276, 68, 163, 14, 315, 263, 198, 392, 41, 448, 352, 290, 500, 242, 34, 183, 379, 419, 74, 408, 91, 66, 89, 285, 345, 147, 42, 106, 214, 207, 182, 333, 444, 205, 239, 167, 118, 196, 226, 485, 269, 103, 4, 466, 271, 504, 188, 456, 95, 257, 453, 307, 135, 339, 61, 18, 341, 418, 436, 273, 127, 43, 414, 310, 482, 212, 72, 222, 49, 157, 87, 149, 480, 86, 240, 462, 397, 73, 440, 390, 501, 142, 82, 137, 75, 426, 331, 318, 443, 216, 362, 439, 152, 90, 79, 150, 353, 37, 45, 416, 274, 264, 258, 437, 96, 46, 220, 309, 199, 330, 308, 421, 265, 356, 7, 122, 170, 125, 314, 114, 415, 507, 402, 77, 6, 15, 431, 412, 374, 12, 179, 343, 105, 405, 299, 411, 394, 3]
keylistFY Entropy -> h =  9.000000000000004
keylistBX:  [417, 90, 345, 24, 413, 36, 409, 347, 389, 236, 310, 330, 252, 145, 503, 120, 420, 42, 237, 177, 101, 445, 452, 186, 155, 86, 461, 402, 508, 288, 335, 279, 222, 271, 6, 475, 411, 431, 70, 405, 188, 21, 510, 242, 146, 388, 148, 11, 99, 226, 457, 80, 482, 443, 387, 66, 102, 130, 107, 33, 255, 455, 354, 39, 427, 374, 434, 32, 298, 48, 14, 41, 471, 474, 396, 62, 383, 375, 407, 28, 112, 449, 249, 60, 341, 509, 371, 225, 324, 466, 269, 16, 446, 296, 246, 85, 365, 61, 493, 426, 23, 123, 281, 287, 165, 479, 207, 230, 430, 238, 211, 342, 422, 219, 332, 40, 109, 37, 147, 141, 240, 88, 197, 502, 63, 253, 215, 463, 453, 491, 459, 92, 273, 435, 170, 213, 384, 505, 438, 351, 229, 194, 338, 337, 280, 18, 488, 221, 380, 164, 268, 311, 308, 386, 69, 84, 9, 301, 404, 83, 499, 327, 300, 184, 234, 378, 500, 189, 495, 395, 306, 397, 418, 480, 361, 34, 377, 406, 182, 128, 152, 278, 127, 496, 216, 447, 275, 185, 45, 334, 76, 454, 307, 131, 282, 179, 135, 154, 15, 285, 124, 245, 484, 261, 467, 442, 498, 150, 494, 262, 276, 210, 95, 277, 143, 126, 195, 100, 231, 392, 408, 423, 295, 359, 8, 17, 181, 465, 38, 263, 178, 258, 180, 115, 214, 357, 444, 313, 257, 394, 487, 104, 382, 251, 290, 464, 343, 473, 265, 501, 284, 401, 78, 169, 352, 103, 196, 201, 366, 140, 340, 368, 98, 191, 232, 286, 35, 30, 477, 483, 132, 469, 167, 19, 27, 490, 5, 144, 322, 328, 129, 166, 292, 497, 364, 202, 485, 372, 56, 52, 356, 511, 303, 2, 217, 111, 390, 13, 325, 486, 208, 136, 424, 168, 224, 274, 353, 75, 113, 320, 72, 65, 31, 53, 71, 272, 421, 241, 432, 156, 470, 316, 339, 507, 118, 43, 254, 403, 476, 116, 416, 142, 235, 436, 0, 51, 105, 44, 59, 149, 106, 233, 138, 54, 3, 244, 162, 363, 161, 114, 348, 309, 448, 349, 190, 79, 393, 175, 58, 133, 481, 171, 367, 223, 119, 94, 441, 260, 346, 331, 326, 358, 283, 437, 259, 151, 429, 398, 425, 220, 270, 7, 82, 267, 419, 360, 462, 20, 212, 87, 291, 121, 193, 55, 369, 344, 172, 391, 304, 49, 139, 137, 333, 157, 350, 379, 206, 468, 428, 294, 46, 385, 504, 248, 228, 47, 159, 158, 370, 57, 433, 506, 89, 381, 412, 209, 472, 318, 64, 93, 73, 29, 478, 305, 174, 68, 1, 299, 440, 67, 205, 108, 125, 77, 399, 91, 110, 400, 293, 314, 218, 302, 50, 227, 289, 355, 134, 266, 297, 204, 199, 192, 458, 176, 312, 74, 336, 451, 173, 439, 26, 198, 97, 4, 460, 122, 315, 25, 410, 450, 81, 117, 323, 489, 415, 256, 10, 329, 319, 456, 239, 153, 414, 22, 243, 362, 163, 187, 96, 317, 373, 160, 264, 321, 247, 200, 492, 183, 12, 376, 203, 250]
keylistBX Entropy -> h =  9.000000000000004
keylistBY:  [91, 15, 203, 511, 416, 306, 498, 488, 372, 50, 348, 136, 503, 63, 377, 499, 217, 255, 429, 204, 282, 186, 211, 154, 143, 222, 280, 146, 163, 272, 362, 274, 10, 239, 388, 340, 250, 471, 74, 350, 226, 382, 400, 435, 234, 472, 479, 361, 235, 442, 238, 317, 223, 170, 93, 312, 147, 227, 7, 240, 354, 428, 303, 117, 46, 86, 395, 353, 375, 177, 24, 19, 440, 451, 392, 458, 202, 497, 32, 468, 168, 185, 456, 205, 123, 285, 447, 444, 339, 396, 467, 394, 296, 161, 264, 422, 478, 121, 333, 122, 263, 160, 316, 415, 58, 506, 401, 164, 297, 49, 3, 95, 308, 243, 493, 201, 41, 165, 410, 329, 233, 54, 489, 104, 193, 491, 135, 434, 194, 119, 195, 290, 138, 84, 355, 426, 336, 457, 103, 373, 11, 83, 455, 75, 99, 72, 311, 399, 151, 445, 469, 142, 466, 368, 144, 108, 137, 443, 277, 216, 364, 166, 299, 376, 212, 182, 214, 409, 183, 150, 490, 300, 358, 215, 322, 156, 44, 261, 65, 504, 118, 125, 404, 389, 349, 242, 179, 260, 420, 171, 246, 231, 96, 40, 268, 0, 411, 33, 380, 482, 174, 126, 2, 155, 356, 407, 319, 403, 76, 318, 365, 244, 439, 30, 402, 48, 463, 267, 66, 36, 480, 249, 441, 51, 31, 70, 412, 366, 330, 43, 208, 363, 181, 266, 13, 328, 87, 210, 218, 408, 448, 113, 387, 327, 47, 251, 162, 176, 345, 197, 57, 116, 301, 28, 141, 371, 276, 423, 476, 225, 259, 159, 132, 379, 475, 486, 158, 200, 101, 414, 320, 418, 269, 433, 474, 187, 374, 321, 304, 128, 180, 79, 326, 26, 236, 397, 294, 254, 131, 291, 385, 314, 60, 68, 335, 8, 35, 351, 252, 508, 295, 145, 107, 209, 346, 258, 115, 425, 484, 481, 437, 81, 265, 189, 492, 378, 45, 230, 461, 105, 191, 310, 198, 157, 307, 29, 206, 367, 38, 309, 483, 460, 64, 405, 275, 5, 229, 315, 279, 427, 173, 430, 352, 505, 82, 398, 52, 237, 338, 220, 89, 196, 384, 470, 73, 359, 487, 139, 55, 42, 262, 12, 464, 77, 302, 332, 100, 102, 92, 271, 97, 207, 153, 140, 502, 23, 284, 59, 120, 390, 334, 98, 110, 88, 62, 219, 148, 53, 287, 18, 453, 224, 381, 67, 510, 149, 17, 450, 192, 34, 331, 114, 496, 337, 323, 507, 256, 134, 393, 293, 273, 509, 501, 347, 436, 494, 473, 283, 431, 391, 167, 485, 20, 27, 313, 190, 459, 112, 188, 37, 130, 500, 253, 281, 80, 9, 432, 477, 152, 465, 452, 184, 228, 462, 406, 221, 106, 289, 383, 16, 21, 111, 257, 424, 124, 69, 421, 288, 90, 286, 248, 241, 449, 129, 127, 344, 417, 178, 247, 172, 370, 325, 78, 278, 175, 169, 298, 360, 56, 199, 446, 22, 438, 1, 369, 413, 213, 133, 305, 343, 324, 25, 341, 85, 245, 232, 71, 292, 61, 39, 386, 454, 357, 4, 419, 94, 342, 495, 6, 14, 109, 270]
keylistBY Entropy -> h =  9.000000000000004 
'''

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

def sub_bytes_ROSIE(s):
    #start_time = time.time()
    so = sbox_ROSIE[s]
    #print("sBox Execution Time --- %s seconds ---" % (time.time() - start_time))
    return so


def inv_sub_bytes_ROSIE(s):
    #start_time = time.time()
    so = inv_sbox_ROSIE[s]
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
    
def get_dec_key(enc_key,len):  
    # Program Started
    dec_key = copy.deepcopy(enc_key)
    for i in range(len):
        dec_key[enc_key[i]] = i
    return dec_key
    # Program Ended  

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

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
    
def SPN_Forward(testimagePath):
    print("SPN_Forward is Started........... !!!")
    imagePlain = Image.open(testimagePath)
    imagePlain.show()
    imArray_In = np.array(imagePlain)
    slen = 512
    key = 128
    xlen = len(imArray_In)
    ylen = len(imArray_In[0])
    keylistFX  = random.sample(range(0, xlen), xlen)
    keylistFY  = random.sample(range(0, ylen), ylen)
    keylistBX = get_dec_key(keylistFX,xlen)
    keylistBY = get_dec_key(keylistFY,ylen)
    print("keylistFX: ",keylistFX)
    print("keylistFX Entropy -> h = ",shannon_entropy(keylistFX, 2))
    print("keylistFY: ",keylistFY)
    print("keylistFY Entropy -> h = ",shannon_entropy(keylistFY, 2))
    print("keylistBX: ",keylistBX)
    print("keylistBX Entropy -> h = ",shannon_entropy(keylistBX, 2))
    print("keylistBY: ",keylistBY)
    print("keylistBY Entropy -> h = ",shannon_entropy(keylistBY, 2))
    imagePathNoise = ROSIEPATH + "noise.png"
    imageEntropyNoise = cv2.imread(imagePathNoise, cv2.IMREAD_GRAYSCALE)
    print(imageEntropyNoise)
    print("NOISE Entropy -> h = ",shannon_entropy(imageEntropyNoise, 2))
    
    #imArray_Out_MSCA = copy.deepcopy(imArray_In)
    imArray_Out_AES_C = copy.deepcopy(imArray_In)
    imArray_Out_RTL_C = copy.deepcopy(imArray_In)#np.zeros((256, 256, 3)) #copy.deepcopy(imArray_In)
    imArray_Out_Bahrami_2021_C = copy.deepcopy(imArray_In)
    imArray_Out_ROSIE_C = copy.deepcopy(imArray_In)
    imArray_Out_AES_D = copy.deepcopy(imArray_In)
    imArray_Out_RTL_D = copy.deepcopy(imArray_In)#np.zeros((256, 256, 3)) #copy.deepcopy(imArray_In)
    imArray_Out_Bahrami_2021_D = copy.deepcopy(imArray_In)
    imArray_Out_ROSIE_D = copy.deepcopy(imArray_In)
    #imArray_Out_RTL_SP = imArray_In#copy.deepcopy(imArray_In)
    # Write code Here
    ilow = int(len(imArray_In)/4)
    ihigh = int(len(imArray_In)-ilow)
    jlow = int(len(imArray_In[0])/4)
    jhigh = int(len(imArray_In[0])-jlow)
    print("ilow = ",ilow,", ihigh = ",ihigh,", jlow = ",jlow," and jhigh = ",jhigh)
    start_time = time.time()
    for i in range(len(imArray_In)):
        for j in range(len(imArray_In[0])):
            for k in range(len(imArray_In[0][0])):  # 3):
                #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
                #if (i>ilow & i<ihigh & j>jlow & j<jhigh):
                # imArray_Out_AES_C[keylistFX[i]][keylistFY[j]][k] = sub_bytes_AES_Traditional(imArray_In[i][j][k])
                # imArray_Out_ROSIE_C[keylistFX[i]][keylistFY[j]][k] = sub_bytes_ROSIE(imArray_In[i][j][k])
                # imArray_Out_Bahrami_2021_C[keylistFX[i]][keylistFY[j]][k] = sub_bytes_Bahram_2021(imArray_In[i][j][k])
                # imArray_Out_RTL_C[keylistFX[i]][keylistFY[j]][k] = sBoxRTL_Forward(imArray_In[i][j][k])
              
                imArray_Out_AES_C[i][j][k] = sub_bytes_AES_Traditional(imArray_In[i][j][k])
                imArray_Out_ROSIE_C[i][j][k] = sub_bytes_ROSIE(imArray_In[i][j][k])
                imArray_Out_Bahrami_2021_C[i][j][k]= sub_bytes_Bahram_2021(imArray_In[i][j][k])
                imArray_Out_RTL_C[i][j][k] = sBoxRTL_Forward(imArray_In[i][j][k])
              
                #imArray_Out_RTL_S[i][j][k] = sBoxRTL_Forward(imArray_In[i][j][k])
                #else:
                    #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
                #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 
    stop_time = time.time()
    enc_time = stop_time-start_time
    print("Encryption Time : ",enc_time," seconds")
    start_time = time.time()
    for i in range(len(imArray_In)):
        for j in range(len(imArray_In[0])):
            for k in range(len(imArray_In[0][0])):  # 3):
                #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
                #if (i==0 & j==0):
                #if (i>ilow & i<ihigh & j>jlow & j<jhigh):
                # imArray_Out_AES_D[keylistBX[i]][keylistBY[j]][k] = inv_sub_bytes_AES_Traditional(imArray_Out_AES_C[i][j][k])
                # imArray_Out_ROSIE_D[keylistBX[i]][keylistBY[j]][k] = inv_sub_bytes_ROSIE(imArray_Out_ROSIE_C[i][j][k])
                # imArray_Out_Bahrami_2021_D[keylistBX[i]][keylistBY[j]][k] = inv_sub_bytes_Bahram_2021(imArray_Out_Bahrami_2021_C[i][j][k])
                # imArray_Out_RTL_D[keylistBX[i]][keylistBY[j]][k] = sBoxRTL_Backward(imArray_Out_RTL_C[i][j][k])
                
                imArray_Out_AES_D[i][j][k]= inv_sub_bytes_AES_Traditional(imArray_Out_AES_C[i][j][k])
                imArray_Out_ROSIE_D[i][j][k]= inv_sub_bytes_ROSIE(imArray_Out_ROSIE_C[i][j][k])
                imArray_Out_Bahrami_2021_D[i][j][k]= inv_sub_bytes_Bahram_2021(imArray_Out_Bahrami_2021_C[i][j][k])
                imArray_Out_RTL_D[i][j][k] = sBoxRTL_Backward(imArray_Out_RTL_C[i][j][k])
               
                
                #imArray_Out_RTL_D[i][j][k] = sBoxRTL_Backward(imArray_Out_RTL_S[i][j][k])
                #else:
                    #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
                #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 
    stop_time = time.time()
    dec_time = stop_time-start_time
    print("Decryption Time : ",dec_time," seconds")
    # print("Dimension = ",imArray_In.shape)
    # testimage = Image.fromarray(imArray_In)
    # testimage.show()
    # for i in range(len(imArray_In)):
    #     for j in range(len(imArray_In[0])):
    #         #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
    #         if (i>ilow & i<ihigh & j>jlow & j<jhigh):
    #             imArray_Out_Bahrami_2021_S[i][j] = sub_bytes_Bahram_2021(imArray_In[i][j])
    #         imArray_Out_RTL_S[keylistFX[i]][keylistFY[j]]= sBoxRTL_Forward(imArray_In[i][j])
    #         #else:
    #             #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
    #         #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 

    # for i in range(len(imArray_In)):
    #     for j in range(len(imArray_In[0])):
    #         #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
    #         #if (i==0 & j==0):
    #         if (i>ilow & i<ihigh & j>jlow & j<jhigh):
    #             imArray_Out_Bahrami_2021_D[i][j] = inv_sub_bytes_Bahram_2021(imArray_Out_Bahrami_2021_S[i][j])
    #         imArray_Out_RTL_D[keylistBX[i]][keylistBY[j]] = sBoxRTL_Backward(imArray_Out_RTL_S[i][j])
    #         #else:
    #             #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
    #         #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 

    
    
    #imageCipher_MSCA = Image.fromarray(imArray_Out_MSCA)
    #imageCipher_MSCA.show()
    imageCipher_ROSIE_C = Image.fromarray(imArray_Out_ROSIE_C)
    imageCipher_ROSIE_C.show()
    imageCipherLenaPath_ROSIE_C = ROSIEPATH + "imageCipherLenaROSIE.png"
    imageCipher_ROSIE_C.save(imageCipherLenaPath_ROSIE_C)
    imageEntropyCipherLenaROSIE_C = cv2.imread(imageCipherLenaPath_ROSIE_C, cv2.IMREAD_GRAYSCALE)
    print("SBox ROSIE C  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaROSIE_C, 2))
    
    
    
    imageCipher_Bahrami_2021_C = Image.fromarray(imArray_Out_Bahrami_2021_C)
    imageCipher_Bahrami_2021_C.show()
    imageCipherLenaPath_Bahrami_2021_C = ROSIEPATH + "imageCipherLenaBahrami_2021.png"
    imageCipher_Bahrami_2021_C.save(imageCipherLenaPath_Bahrami_2021_C)
    imageEntropyCipherLenaBahrami_2021_C = cv2.imread(imageCipherLenaPath_Bahrami_2021_C, cv2.IMREAD_GRAYSCALE)
    print("SBox Bahrami 2021 C  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaBahrami_2021_C, 2))
    
    imageCipher_AES_C = Image.fromarray(imArray_Out_AES_C)
    imageCipher_AES_C.convert('L')
    imageCipher_AES_C.show()
    #imageCipherLenaPath_RTL_S = ROSIEPATH + "imageCipherLenaRTL.png"
    imageCipherLenaPath_AES_C = ROSIEPATH + "imageCipherAES.png"
    imageCipher_AES_C.save(imageCipherLenaPath_AES_C)
    imageEntropyCipherLenaAES_C  = cv2.imread(imageCipherLenaPath_AES_C, cv2.IMREAD_GRAYSCALE)
    print("SBox AES C  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaAES_C, 2))
    
    
    imageCipher_RTL_C = Image.fromarray(imArray_Out_RTL_C)
    imageCipher_RTL_C.convert('L')
    imageCipher_RTL_C.show()
    #imageCipherLenaPath_RTL_S = ROSIEPATH + "imageCipherLenaRTL.png"
    imageCipherTestPath = ROSIEPATH + "imageCipherTest.png"
    imageCipher_RTL_C.save(imageCipherTestPath)
    imageEntropyCipherTest = cv2.imread(imageCipherTestPath, cv2.IMREAD_GRAYSCALE)
    print("SBox RTL C  Entropy -> h = ",shannon_entropy(imageEntropyCipherTest, 2))
    
    imageDecipher_ROSIE_D = Image.fromarray(imArray_Out_ROSIE_D)
    imageDecipher_ROSIE_D.show()
    imageDecipherLenaPath_ROSIE_D = ROSIEPATH + "imageDecipherLenaROSIE.png"
    imageDecipher_ROSIE_D.save(imageDecipherLenaPath_ROSIE_D)
    imageEntropyDecipherLenaROSIE_D = cv2.imread(imageDecipherLenaPath_ROSIE_D, cv2.IMREAD_GRAYSCALE)
    print("SBox ROSIE D  Entropy -> h = ",shannon_entropy(imageEntropyDecipherLenaROSIE_D, 2))
    
    imageDecipher_Bahrami_2021_D = Image.fromarray(imArray_Out_Bahrami_2021_D)
    imageDecipher_Bahrami_2021_D.show()
    imageDecipherLenaPath_Bahrami_2021_D = ROSIEPATH + "imageDecipherLenaBahrami_2021.png"
    imageDecipher_Bahrami_2021_D.save(imageDecipherLenaPath_Bahrami_2021_D)
    imageEntropyDecipherLenaBahrami_2021_D = cv2.imread(imageDecipherLenaPath_Bahrami_2021_D, cv2.IMREAD_GRAYSCALE)
    print("SBox Bahrami 2021 D  Entropy -> h = ",shannon_entropy(imageEntropyDecipherLenaBahrami_2021_D, 2))
    
    
 
    imageDecipher_RTL_D = Image.fromarray(imArray_Out_RTL_D)
    imageDecipher_RTL_D.show()
    #imageCipherLenaPath_RTL_D = ROSIEPATH + "imageDecipherLenaRTL.png"
    imageDecipherTestPath = ROSIEPATH + "imageDeipherTest.png"
    imageDecipher_RTL_D.save(imageDecipherTestPath)
    imageEntropyDecipherTest= cv2.imread(imageDecipherTestPath, cv2.IMREAD_GRAYSCALE)
    print("SBox RTL D  Entropy -> h = ",shannon_entropy(imageEntropyDecipherTest, 2))
    
    imageDecipher_AES_D = Image.fromarray(imArray_Out_AES_D)
    imageDecipher_AES_D.convert('L')
    imageDecipher_AES_D.show()
    #imageCipherLenaPath_RTL_S = ROSIEPATH + "imageCipherLenaRTL.png"
    imageDecipherLenaPath_AES_D = ROSIEPATH + "imageDecipherAES.png"
    imageDecipher_AES_D.save(imageDecipherLenaPath_AES_D)
    imageEntropyDecipherLenaAES_D  = cv2.imread(imageDecipherLenaPath_AES_D, cv2.IMREAD_GRAYSCALE)
    print("SBox AES D  Entropy -> h = ",shannon_entropy(imageEntropyDecipherLenaAES_D, 2))
    
    
    
    # Entropy
    print("Original      Entropy -> h = ",shannon_entropy(imagePlain, 2))
    print("AES           Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaAES_C, 2))
    print("RTL           Entropy -> h = ",shannon_entropy(imageEntropyCipherTest, 2))
    print("Bahrami 2021  Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaBahrami_2021_C, 2))
    print("ROSIE         Entropy -> h = ",shannon_entropy(imageEntropyCipherLenaROSIE_C, 2))
    
    
    # PSNR 
    imgTest = cv2.imread(testimagePath) 
    psnrtest_test = cv2.PSNR(imgTest, imgTest)
    print("PSNR Test with Test Image                     = ", psnrtest_test)
    imgCipher = cv2.imread(imageCipherTestPath) 
    psnrtest_cipher= cv2.PSNR(imgTest, imgCipher)
    print("PSNR Test with Cipher                         = ", psnrtest_cipher) 
    
    imgCipherAES = cv2.imread(imageCipherLenaPath_AES_C) 
    psnrtest_cipherAES= cv2.PSNR(imgTest, imgCipherAES)
    print("PSNR Test with Cipher AES                      = ", psnrtest_cipherAES) 
    
    imgCipherBahrami2021 = cv2.imread(imageCipherLenaPath_Bahrami_2021_C) 
    psnrtest_cipherBahrami2021= cv2.PSNR(imgTest, imgCipherBahrami2021)
    print("PSNR Test with Cipher Bahrami2021             = ", psnrtest_cipherBahrami2021) 
    
    imgCipherROSIE = cv2.imread(imageCipherLenaPath_ROSIE_C) 
    psnrtest_cipherROSIE= cv2.PSNR(imgTest, imgCipherROSIE)
    print("PSNR Test with Cipher ROSIE                   = ", psnrtest_cipherROSIE) 
    
    # MSE
    print("MSE Original                          = ", getMSE(imgTest, imgTest))
    print("MSE Cipher AES                        = ", getMSE(imgTest, imgCipherAES))
    print("MSE Cipher RTL                        = ", getMSE(imgTest, imgCipher))
    print("MSE Cipher Bahrami 2021               = ", getMSE(imgTest, imgCipherBahrami2021))
    print("MSE Cipher ROSIE                      = ", getMSE(imgTest, imgCipherROSIE))
    
    # MAE
    print("MAE Original                          = ", getMAE(imgTest, imgTest))
    print("MAE Cipher AES                        = ", getMAE(imgTest, imgCipherAES))
    print("MAE Cipher RTL                        = ", getMAE(imgTest, imgCipher))
    print("MAE Cipher Bahrami 2021               = ", getMAE(imgTest, imgCipherBahrami2021))
    print("MAE Cipher ROSIE                      = ", getMAE(imgTest, imgCipherROSIE))
    
    # SSIM
    imgTestG = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    imgCipherAESG = cv2.cvtColor(imgCipherAES, cv2.COLOR_BGR2GRAY)
    imgCipherG = cv2.cvtColor(imgCipher, cv2.COLOR_BGR2GRAY)
    imgCipherBahrami2021G = cv2.cvtColor(imgCipherBahrami2021, cv2.COLOR_BGR2GRAY)
    imgCipherROSIEG = cv2.cvtColor(imgCipherROSIE, cv2.COLOR_BGR2GRAY)

    print("SSIM Original                         = ", ssim(imgTestG, imgTestG))
    print("SSIM AES                              = ", ssim(imgTestG, imgCipherAESG))
    print("SSIM RTL                              = ", ssim(imgTestG, imgCipherG))
    print("SSIM Bahrami 2021                     = ", ssim(imgTestG, imgCipherBahrami2021G))
    print("SSIM ROSIE                            = ", ssim(imgTestG, imgCipherROSIEG))

     # Correlation
    cm = np.corrcoef(imgTestG.flat, imgTestG.flat)
    r = cm[0, 1]
    print("Cor. Original                         = ",r)
    cm = np.corrcoef(imgTestG.flat, imgCipherAESG.flat)
    r = cm[0, 1]
    print("Cor. AES                              = ", r)
    cm = np.corrcoef(imgTestG.flat, imgCipherG.flat)
    r = cm[0, 1]
    print("Cor. RTL                              = ", r)
    cm = np.corrcoef(imgTestG.flat, imgCipherBahrami2021G.flat)
    r = cm[0, 1]
    print("Cor. Bahrami                          = ", r)
    cm = np.corrcoef(imgTestG.flat, imgCipherROSIEG.flat)
    r = cm[0, 1]
    print("Cor. ROISE                            = ", r)
    
    
    #Histogram
    im = io.imread(testimagePath)
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

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

def calculate():
    # Start
    img0 = cv2.imread('zelda.png')
    # End

def image_encryption():
    print("Image Encryption is Started........... !!!")
    # Write code Here
    #lenaimagePath = ROSIEPATH + "lenna.png"
    testimagePath = ROSIEPATH + "lenna.png"
    #imOriginal = Image.open(lenaimagePath)
    #imOriginal = Image.open(testimagePath)
    #imOriginal = cv2.imread(lenaimagePath, cv2.IMREAD_GRAYSCALE)
    #imOriginal.show()
    SPN_Forward(testimagePath)
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