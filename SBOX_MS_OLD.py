import time
from gettext import _error
import cv2
import numpy as np
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


s_box = (
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
)

inv_s_box = (
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
)

# Bahram 2020 Generated sBox256F & sBox256B : 0a - > e8
sboxF256_Bahram_2020a =  [20, 1, 169, 194, 253, 131, 127, 106, 215, 62, 232, 85, 150, 64, 43, 188, 235, 40, 166, 153, 72, 224, 96, 74, 125, 255, 50, 206, 252, 189, 236, 184, 175, 214, 121, 151, 9, 78, 161, 48, 58, 132, 110, 19, 46, 117, 59, 95, 65, 225, 8, 0, 140, 111, 122, 179, 178, 124, 191, 204, 88, 211, 135, 229, 201, 24, 6, 52, 162, 228, 145, 30, 173, 147, 14, 77, 249, 134, 49, 240, 216, 29, 97, 192, 244, 231, 66, 10, 233, 172, 108, 137, 185, 168, 163, 186, 250, 83, 170, 112, 105, 79, 109, 133, 28, 247, 218, 118, 103, 159, 180, 221, 5, 241, 92, 209, 7, 181, 13, 81, 107, 87, 155, 197, 82, 38, 213, 42, 190, 164, 37, 200, 89, 203, 4, 93, 60, 25, 91, 226, 165, 34, 17, 245, 99, 113, 36, 146, 32, 208, 171, 139, 15, 138, 198, 27, 71, 75, 35, 41, 54, 12, 212, 174, 234, 237, 126, 31, 23, 57, 158, 177, 63, 116, 104, 248, 141, 144, 51, 94, 129, 2, 33, 123, 53, 142, 70, 243, 47, 11, 193, 3, 39, 210, 183, 16, 56, 86, 98, 55, 238, 149, 202, 222, 219, 199, 220, 68, 114, 26, 18, 227, 76, 223, 84, 242, 119, 176, 205, 21, 100, 207, 136, 196, 156, 130, 230, 239, 67, 143, 246, 160, 157, 73, 128, 115, 152, 120, 182, 195, 80, 254, 44, 148, 90, 167, 154, 45, 251, 102, 217, 101, 22, 187, 61, 69]
sboxB256_Bahram_2020a =  [51, 1, 181, 191, 134, 112, 66, 116, 50, 36, 87, 189, 161, 118, 74, 152, 195, 142, 210, 43, 0, 219, 252, 168, 65, 137, 209, 155, 104, 81, 71, 167, 148, 182, 141, 158, 146, 130, 125, 192, 17, 159, 127, 14, 242, 247, 44, 188, 39, 78, 26, 178, 67, 184, 160, 199, 196, 169, 40, 46, 136, 254, 9, 172, 13, 48, 86, 228, 207, 255, 186, 156, 20, 233, 23, 157, 212, 75, 37, 101, 240, 119, 124, 97, 214, 11, 197, 121, 60, 132, 244, 138, 114, 135, 179, 47, 22, 82, 198, 144, 220, 251, 249, 108, 174, 100, 7, 120, 90, 102, 42, 53, 99, 145, 208, 235, 173, 45, 107, 216, 237, 34, 54, 183, 57, 24, 166, 6, 234, 180, 225, 5, 41, 103, 77, 62, 222, 91, 153, 151, 52, 176, 185, 229, 177, 70, 147, 73, 243, 201, 12, 35, 236, 19, 246, 122, 224, 232, 170, 109, 231, 38, 68, 94, 129, 140, 18, 245, 93, 2, 98, 150, 89, 72, 163, 32, 217, 171, 56, 55, 110, 117, 238, 194, 31, 92, 95, 253, 15, 29, 128, 58, 83, 190, 3, 239, 223, 123, 154, 205, 131, 64, 202, 133, 59, 218, 27, 221, 149, 115, 193, 61, 162, 126, 33, 8, 80, 250, 106, 204, 206, 111, 203, 213, 21, 49, 139, 211, 69, 63, 226, 85, 10, 88, 164, 16, 30, 165, 200, 227, 79, 113, 215, 187, 84, 143, 230, 105, 175, 76, 96, 248, 28, 4, 241, 25]

# Bahram 2020 Generated sBox256F & sBox256B : 10 - > e8
sboxF256_Bahram_2020b =  [20, 1, 169, 194, 253, 131, 127, 106, 215, 62, 235, 85, 150, 64, 43, 188, 232, 40, 166, 153, 72, 224, 96, 74, 125, 255, 50, 206, 252, 189, 236, 184, 175, 214, 121, 151, 9, 78, 161, 48, 58, 132, 110, 19, 46, 117, 59, 95, 65, 225, 8, 0, 140, 111, 122, 179, 178, 124, 191, 204, 88, 211, 135, 229, 201, 24, 6, 52, 162, 228, 145, 30, 173, 147, 14, 77, 249, 134, 49, 240, 216, 29, 97, 192, 244, 231, 66, 10, 233, 172, 108, 137, 185, 168, 163, 186, 250, 83, 170, 112, 105, 79, 109, 133, 28, 247, 218, 118, 103, 159, 180, 221, 5, 241, 92, 209, 7, 181, 13, 81, 107, 87, 155, 197, 82, 38, 213, 42, 190, 164, 37, 200, 89, 203, 4, 93, 60, 25, 91, 226, 165, 34, 17, 245, 99, 113, 36, 146, 32, 208, 171, 139, 15, 138, 198, 27, 71, 75, 35, 41, 54, 12, 212, 174, 234, 237, 126, 31, 23, 57, 158, 177, 63, 116, 104, 248, 141, 144, 51, 94, 129, 2, 33, 123, 53, 142, 70, 243, 47, 11, 193, 3, 39, 210, 183, 16, 56, 86, 98, 55, 238, 149, 202, 222, 219, 199, 220, 68, 114, 26, 18, 227, 76, 223, 84, 242, 119, 176, 205, 21, 100, 207, 136, 196, 156, 130, 230, 239, 67, 143, 246, 160, 157, 73, 128, 115, 152, 120, 182, 195, 80, 254, 44, 148, 90, 167, 154, 45, 251, 102, 217, 101, 22, 187, 61, 69]
sboxB256_Bahram_2020b =  [51, 1, 181, 191, 134, 112, 66, 116, 50, 36, 87, 189, 161, 118, 74, 152, 195, 142, 210, 43, 0, 219, 252, 168, 65, 137, 209, 155, 104, 81, 71, 167, 148, 182, 141, 158, 146, 130, 125, 192, 17, 159, 127, 14, 242, 247, 44, 188, 39, 78, 26, 178, 67, 184, 160, 199, 196, 169, 40, 46, 136, 254, 9, 172, 13, 48, 86, 228, 207, 255, 186, 156, 20, 233, 23, 157, 212, 75, 37, 101, 240, 119, 124, 97, 214, 11, 197, 121, 60, 132, 244, 138, 114, 135, 179, 47, 22, 82, 198, 144, 220, 251, 249, 108, 174, 100, 7, 120, 90, 102, 42, 53, 99, 145, 208, 235, 173, 45, 107, 216, 237, 34, 54, 183, 57, 24, 166, 6, 234, 180, 225, 5, 41, 103, 77, 62, 222, 91, 153, 151, 52, 176, 185, 229, 177, 70, 147, 73, 243, 201, 12, 35, 236, 19, 246, 122, 224, 232, 170, 109, 231, 38, 68, 94, 129, 140, 18, 245, 93, 2, 98, 150, 89, 72, 163, 32, 217, 171, 56, 55, 110, 117, 238, 194, 31, 92, 95, 253, 15, 29, 128, 58, 83, 190, 3, 239, 223, 123, 154, 205, 131, 64, 202, 133, 59, 218, 27, 221, 149, 115, 193, 61, 162, 126, 33, 8, 80, 250, 106, 204, 206, 111, 203, 213, 21, 49, 139, 211, 69, 63, 226, 85, 16, 88, 164, 10, 30, 165, 200, 227, 79, 113, 215, 187, 84, 143, 230, 105, 175, 76, 96, 248, 28, 4, 241, 25]

#https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0241890
# N Siddiqui Generated sBox256F & sBox256B : SBox 1
sboxF256_Nasir_2020_s1 =  [0, 244, 165, 89, 22, 147, 2, 105, 71, 186, 163, 184, 48, 30, 39, 49, 164, 94, 211, 143, 219, 187, 7, 100, 72, 46, 181, 231, 132, 252, 174, 154, 216, 88, 200, 113, 151, 65, 199, 9, 224, 102, 215, 67, 19, 99, 189, 220, 96, 138, 226, 177, 82, 179, 13, 59, 108, 81, 230, 63, 32, 50, 190, 8, 104, 35, 24, 79, 75, 131, 153, 145, 37, 26, 155, 92, 85, 222, 16, 120, 23, 196, 52, 5, 31, 208, 250, 180, 202, 87, 28, 198, 51, 139, 109, 106, 248, 95, 27, 29, 93, 170, 133, 58, 166, 78, 124, 176, 221, 157, 210, 90, 218, 251, 235, 175, 207, 34, 66, 117, 17, 209, 135, 107, 21, 12, 127, 253, 84, 161, 213, 54, 125, 168, 150, 61, 204, 228, 4, 146, 247, 98, 152, 173, 242, 191, 121, 233, 212, 232, 169, 80, 229, 238, 217, 40, 255, 126, 225, 171, 223, 68, 140, 56, 10, 25, 42, 160, 188, 15, 97, 129, 115, 118, 128, 144, 194, 70, 77, 185, 116, 243, 74, 112, 130, 159, 91, 36, 33, 178, 110, 38, 246, 41, 114, 192, 162, 172, 14, 55, 214, 249, 62, 76, 148, 11, 193, 203, 241, 134, 86, 44, 60, 206, 141, 239, 240, 20, 237, 57, 183, 156, 236, 47, 53, 18, 1, 142, 245, 83, 103, 69, 158, 101, 167, 122, 45, 73, 136, 43, 201, 3, 195, 64, 123, 197, 119, 6, 234, 227, 137, 111, 182, 149, 205, 254]
sboxB256_Nasir_2020_s1 =  [0, 226, 6, 241, 138, 83, 247, 22, 63, 39, 164, 205, 125, 54, 198, 169, 78, 120, 225, 44, 217, 124, 4, 80, 66, 165, 73, 98, 90, 99, 13, 84, 60, 188, 117, 65, 187, 72, 191, 14, 155, 193, 166, 239, 211, 236, 25, 223, 12, 15, 61, 92, 82, 224, 131, 199, 163, 219, 103, 55, 212, 135, 202, 59, 243, 37, 118, 43, 161, 231, 177, 8, 24, 237, 182, 68, 203, 178, 105, 67, 151, 57, 52, 229, 128, 76, 210, 89, 33, 3, 111, 186, 75, 100, 17, 97, 48, 170, 141, 45, 23, 233, 41, 230, 64, 7, 95, 123, 56, 94, 190, 251, 183, 35, 194, 172, 180, 119, 173, 246, 79, 146, 235, 244, 106, 132, 157, 126, 174, 171, 184, 69, 28, 102, 209, 122, 238, 250, 49, 93, 162, 214, 227, 19, 175, 71, 139, 5, 204, 253, 134, 36, 142, 70, 31, 74, 221, 109, 232, 185, 167, 129, 196, 10, 16, 2, 104, 234, 133, 150, 101, 159, 197, 143, 30, 115, 107, 51, 189, 53, 87, 26, 252, 220, 11, 179, 9, 21, 168, 46, 62, 145, 195, 206, 176, 242, 81, 245, 91, 38, 34, 240, 88, 207, 136, 254, 213, 116, 85, 121, 110, 18, 148, 130, 200, 42, 32, 154, 112, 20, 47, 108, 77, 160, 40, 158, 50, 249, 137, 152, 58, 27, 149, 147, 248, 114, 222, 218, 153, 215, 216, 208, 144, 181, 1, 228, 192, 140, 96, 201, 86, 113, 29, 127, 255, 156]

#https://www.mdpi.com/1099-4300/21/3/245
# N Siddiqui Generated sBox256F & sBox256B :
sboxF256_Amjad_2019 =  [120, 250, 193, 180, 88, 223, 185, 112, 210, 242, 233, 241, 91, 95, 53, 174, 132, 115, 125, 220, 74, 135, 190, 80, 72, 104, 43, 8, 239, 38, 194, 186, 183, 153, 31, 160, 116, 157, 114, 165, 48, 13, 52, 221, 244, 63, 24, 119, 46, 171, 169, 158, 9, 177, 42, 123, 140, 122, 111, 216, 245, 98, 70, 197, 203, 235, 168, 187, 12, 26, 137, 138, 101, 60, 225, 100, 113, 28, 195, 146, 29, 199, 189, 86, 214, 102, 200, 39, 178, 191, 227, 44, 27, 15, 246, 141, 144, 134, 255, 19, 22, 204, 18, 139, 82, 35, 156, 57, 209, 181, 79, 93, 188, 231, 206, 97, 77, 128, 143, 155, 167, 59, 208, 175, 253, 3, 73, 218, 62, 61, 47, 159, 78, 68, 136, 126, 58, 36, 152, 252, 249, 45, 67, 229, 54, 56, 99, 6, 94, 198, 145, 226, 173, 247, 34, 11, 85, 87, 248, 118, 192, 213, 133, 212, 237, 21, 92, 20, 215, 121, 219, 49, 109, 50, 238, 64, 0, 176, 66, 1, 76, 254, 150, 222, 106, 129, 205, 40, 196, 127, 230, 179, 154, 69, 30, 163, 33, 10, 4, 55, 2, 105, 7, 117, 71, 65, 81, 251, 148, 170, 182, 217, 232, 236, 151, 124, 224, 17, 131, 41, 166, 161, 96, 184, 107, 83, 162, 37, 130, 172, 228, 75, 25, 103, 240, 147, 108, 207, 211, 234, 32, 110, 51, 23, 16, 201, 202, 164, 14, 84, 149, 243, 142, 5, 90, 89]
sboxB256_Amjad_2019 =  [176, 179, 200, 125, 198, 253, 147, 202, 27, 52, 197, 155, 68, 41, 248, 93, 244, 217, 102, 99, 167, 165, 100, 243, 46, 232, 69, 92, 77, 80, 194, 34, 240, 196, 154, 105, 137, 227, 29, 87, 187, 219, 54, 26, 91, 141, 48, 130, 40, 171, 173, 242, 42, 14, 144, 199, 145, 107, 136, 121, 73, 129, 128, 45, 175, 205, 178, 142, 133, 193, 62, 204, 24, 126, 20, 231, 180, 116, 132, 110, 23, 206, 104, 225, 249, 156, 83, 157, 4, 255, 254, 12, 166, 111, 148, 13, 222, 115, 61, 146, 75, 72, 85, 233, 25, 201, 184, 224, 236, 172, 241, 58, 7, 76, 38, 17, 36, 203, 159, 47, 0, 169, 57, 55, 215, 18, 135, 189, 117, 185, 228, 218, 16, 162, 97, 21, 134, 70, 71, 103, 56, 95, 252, 118, 96, 150, 79, 235, 208, 250, 182, 214, 138, 33, 192, 119, 106, 37, 51, 131, 35, 221, 226, 195, 247, 39, 220, 120, 66, 50, 209, 49, 229, 152, 15, 123, 177, 53, 88, 191, 3, 109, 210, 32, 223, 6, 31, 67, 112, 82, 22, 89, 160, 2, 30, 78, 188, 63, 149, 81, 86, 245, 246, 64, 101, 186, 114, 237, 122, 108, 8, 238, 163, 161, 84, 168, 59, 211, 127, 170, 19, 43, 183, 5, 216, 74, 151, 90, 230, 143, 190, 113, 212, 10, 239, 65, 213, 164, 174, 28, 234, 11, 9, 251, 44, 60, 94, 153, 158, 140, 1, 207, 139, 124, 181, 98]


# Proposed Dual Quad-Bit SBox  #
sbox1F =  [15, 14, 1, 0, 3, 2, 13, 12, 4, 9, 6, 11, 8, 5, 10, 7]
sbox1B =  [3, 2, 5, 4, 8, 13, 10, 15, 12, 9, 14, 11, 7, 6, 1, 0]
sbox2F =  [8, 5, 10, 7, 4, 9, 6, 11, 3, 2, 13, 12, 15, 14, 1, 0]
sbox2B =  [15, 14, 9, 8, 4, 1, 6, 3, 0, 5, 2, 7, 11, 10, 13, 12]
# End of Proposed Dual Quad-Bit SBox  #


def byteSplit(integer):
    return divmod(integer, 0x10)


def byteJoin(num1, num2):
    num3 = (num1 << 4) | (num2);
    return num3

def s_box_forward_proposed(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteLow = sbox2F[byteLow]
    newByteHigh = sbox1F[byteHigh] ^ sbox2F[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def s_box_backword_proposed(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteHigh = sbox1B[byteHigh^byteLow]
    newByteLow = sbox2B[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def s_box_forward_proposed2(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteLow = sbox2F[byteLow]
    newByteHigh = sbox1F[byteHigh] ^ sbox2F[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def s_box_backword_proposed2(byteIn):
    byteHigh, byteLow = byteSplit(byteIn)
    newByteHigh = sbox1B[byteHigh^byteLow]
    newByteLow = sbox2B[byteLow]
    byteOut = byteJoin(newByteHigh, newByteLow)
    return byteOut

def s_boxF256_Bahram_2020a(byteIn):
    return sboxF256_Bahram_2020a[byteIn]

def s_boxB256_Bahram_2020a(byteIn):
    return sboxB256_Bahram_2020a[byteIn]

def s_boxF256_Bahram_2020b(byteIn):
    return sboxF256_Bahram_2020b[byteIn]

def s_boxB256_Bahram_2020b(byteIn):
    return sboxB256_Bahram_2020b[byteIn]

def s_boxF256_Nasir_2020_s1(byteIn):
    return sboxF256_Nasir_2020_s1[byteIn]

def s_boxB256_Nasir_2020_s1(byteIn):
    return sboxB256_Nasir_2020_s1[byteIn]

def s_boxF256_Amjad_2019(byteIn):
    return sboxF256_Amjad_2019[byteIn]

def s_boxB256_Amjad_2019(byteIn):
    return sboxB256_Amjad_2019[byteIn]

def sub_bytes_AES(s):
    so = s_box[s]
    return so


def inv_sub_bytes_AES(s):
    so = inv_s_box[s]
    return so

# Calculate information entropy
def getEntropy(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    #print(norm_counts)
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()  # log(a) b=log (c) b÷log (c) a


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





def estimate_shannon_entropy(dna_sequence):
    bases = collections.Counter([tmp_base for tmp_base in dna_sequence])
    # define distribution
    dist = [x / sum(bases.values()) for x in bases.values()]

    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)

    return entropy_value


def information_entropy(data):
    # Create a counter for all the 256 different possible values
    possible_vals = dict(((chr(x), 0) for x in range(0, 256)))

    # Increment the counter if the byte has the same value as one of the keys
    for byte in data:
        possible_vals[chr(byte)] += 1

    data_len = len(data)
    entropy = 0.0

    # Compute the entropy of the data
    for count in possible_vals:
        if possible_vals[count] == 0:
            continue

        p = float(possible_vals[count] / data_len)
        entropy -= p * np.math.log(p, 2)

    return entropy

def HWA():
    #            00 11  2233445566778899aabbccddeeff
    plainText = [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 236, 255]
    CipherText = "8ea2b7ca516745bfeafc49904b496089"

    print("Est Shannon Entropy h = ", estimate_shannon_entropy(plainText))
    print("Information Entropy h = ", information_entropy(plainText))

def main2():
    #runEDSBox()
    imOriginal = Image.open(r"zelda.png")
    imOriginal.show()

    image_array_inT = np.array(imOriginal)
    imageOriginal = image_array_inT
    image_array_outT = image_array_inT
    image_array_inP = np.array(imOriginal)
    image_array_outP = image_array_inP
    for i in range(len(image_array_inP)):
        for j in range(len(image_array_inP[0])):
            for k in range(len(image_array_inP[0][0])):#3):
                image_array_outP[i][j][k] = s_box_forward_proposed(image_array_inP[i][j][k])
                image_array_outT[i][j][k] = sub_bytes_AES(image_array_inT[i][j][k])
    imageP = Image.fromarray(image_array_outP)
    imageP.save('zelda_Cipher_Proposed.png')
    imageT = Image.fromarray(image_array_outT)
    imageT.save('zelda_Cipher_Trad_AES.png')

def runAnalysis():
    print("Security Analysis Started !!!")


    imOriginal = Image.open(r"zelda.png")
    imOriginal.show()
    imArray_Originals = np.array(imOriginal)


    imArray_Plain_Trad_AES = copy.deepcopy(imArray_Originals)
    imArray_Cipher_Trad_AES = imArray_Plain_Trad_AES
    #print(len(imArray_Cipher_Trad_AES))

    imArray_Plain_Amjad_2019 = copy.deepcopy(imArray_Originals)
    imArray_Cipher_Amjad_2019 = imArray_Plain_Amjad_2019

    imArray_Plain_Nasir_2020 = copy.deepcopy(imArray_Originals)
    imArray_Cipher_Nasir_2020 = imArray_Plain_Nasir_2020

    imArray_Plain_Bahram_2020a = copy.deepcopy(imArray_Originals)
    imArray_Cipher_Bahram_2020a = imArray_Plain_Bahram_2020a

    imBahram_2020b = Image.open(r"zelda.png")
    imArray_Bahram_2020b = np.array(imBahram_2020b)
    imArray_Plain_Bahram_2020b = copy.deepcopy(imArray_Originals)
    imArray_Cipher_Bahram_2020b = imArray_Plain_Bahram_2020b

    imProposed = Image.open(r"zelda.png")
    imArray_Proposed = np.array(imProposed)
    imArray_Plain_Proposed = copy.deepcopy(imArray_Proposed)
    imArray_Cipher_Proposed = imArray_Plain_Proposed


    for i in range(len(imArray_Plain_Proposed)):
        for j in range(len(imArray_Plain_Proposed[0])):
            for k in range(len(imArray_Plain_Proposed[0][0])):  # 3):
                imArray_Cipher_Trad_AES[i][j][k] = sub_bytes_AES(imArray_Plain_Trad_AES[i][j][k])
                imArray_Cipher_Amjad_2019[i][j][k] = s_boxF256_Amjad_2019(imArray_Plain_Amjad_2019[i][j][k])
                imArray_Cipher_Nasir_2020[i][j][k] = s_boxF256_Nasir_2020_s1(imArray_Plain_Nasir_2020[i][j][k])
                imArray_Cipher_Bahram_2020a[i][j][k] = s_boxF256_Bahram_2020a(imArray_Plain_Bahram_2020a[i][j][k])
                imArray_Cipher_Bahram_2020b[i][j][k] = s_boxF256_Bahram_2020b(imArray_Plain_Bahram_2020b[i][j][k])
                imArray_Cipher_Proposed[i][j][k] = s_box_forward_proposed(imArray_Plain_Proposed[i][j][k])

    im_Cipher_Trad_AES = Image.fromarray(imArray_Cipher_Trad_AES)
    im_Cipher_Trad_AES.save('zelda_Cipher_Trad_AES.png')

    im_Cipher_Amjad_2019 = Image.fromarray(imArray_Cipher_Amjad_2019)
    im_Cipher_Amjad_2019.save('zelda_Cipher_Amjad_2019.png')

    im_Cipher_Nasir_2020 = Image.fromarray(imArray_Cipher_Nasir_2020)
    im_Cipher_Nasir_2020.save('zelda_Cipher_Nasir_2020.png')

    im_Cipher_Bahram_2020a = Image.fromarray(imArray_Cipher_Bahram_2020a)
    im_Cipher_Bahram_2020a.save('zelda_Cipher_Bahram_2020a.png')

    im_Cipher_Bahram_2020b = Image.fromarray(imArray_Cipher_Bahram_2020b)
    im_Cipher_Bahram_2020b.save('zelda_Cipher_Bahram_2020b.png')
    
    im_Cipher_Proposed = Image.fromarray(imArray_Cipher_Proposed)
    im_Cipher_Proposed.save('zelda_Cipher_Proposed.png')
    im_Cipher_Proposed.save('zelda_Cipher_Proposed_SBox4.png')

    # ....
    colorIm = Image.open('zelda.png')
    #colorIm.show()
    greyIm = colorIm.convert('L')
    #greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)
    #print('Lenne Original Shannon -> h=', shannon_entropy(greyIm, 2))
    print('zelda_Original Entropy ->            h=', h)



    colorIm = Image.open('zelda_Cipher_Trad_AES.png')
    # colorIm.show()
    greyIm = colorIm.convert('L')
    # greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)
    print('zelda_Cipher_Trad_AES Entropy ->     h=', h)

    colorIm = Image.open('zelda_Cipher_Amjad_2019.png')
    # colorIm.show()
    greyIm = colorIm.convert('L')
    # greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)

    print('zelda_Cipher_Amjad_2019 Entropy ->   h=', h)

    colorIm = Image.open('zelda_Cipher_Nasir_2020.png')
    # colorIm.show()
    greyIm = colorIm.convert('L')
    # greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)
    print('zelda_Cipher_Nasir_2020 Entropy ->   h=', h)

    colorIm = Image.open('zelda_Cipher_Bahram_2020a.png')
    # colorIm.show()
    greyIm = colorIm.convert('L')
    # greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)
    print('zelda_Cipher_Bahram_2020a Entropy -> h=', h)

    colorIm = Image.open('zelda_Cipher_Bahram_2020b.png')
    # colorIm.show()
    greyIm = colorIm.convert('L')
    # greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)
    print('zelda_Cipher_Bahram_2020b Entropy -> h=', h)

    colorIm = Image.open('zelda_Cipher_Proposed_SBox4.png')
    # colorIm.show()
    greyIm = colorIm.convert('L')
    # greyIm.show()
    colorIm = np.array(colorIm)
    greyIm = np.array(greyIm)
    h = getEntropy(greyIm, 2)

    print('zelda_Cipher_Proposed Entropy ->     h=', h)

    img0 = cv2.imread('zelda.png')
    img1 = cv2.imread('zelda_Cipher_Trad_AES.png')
    img2 = cv2.imread('zelda_Cipher_Amjad_2019.png')
    img3 = cv2.imread('zelda_Cipher_Nasir_2020.png')
    img4 = cv2.imread('zelda_Cipher_Bahram_2020a.png')
    img5 = cv2.imread('zelda_Cipher_Bahram_2020b.png')
    img6 = cv2.imread('zelda_Cipher_Proposed_SBox4.png')

    psnr00 = cv2.PSNR(img0, img0)
    print("PSNR zelda Image                     = ", psnr00)

    psnr01 = cv2.PSNR(img0, img1)
    print("PSNR zelda_Cipher_Trad_AES Image     = ", psnr01)

    psnr02 = cv2.PSNR(img0, img2)
    print("PSNR zelda_Cipher_Amjad_2019 Image   = ", psnr02)

    psnr03 = cv2.PSNR(img0, img3)
    print("PSNR zelda_Cipher_Nasir_2020 Image   = ", psnr03)

    psnr04 = cv2.PSNR(img0, img4)
    print("PSNR zelda_Cipher_Bahram_2020a Image = ", psnr04)

    psnr05 = cv2.PSNR(img0, img5)
    print("PSNR zelda_Cipher_Bahram_2020b Image = ", psnr05)

    psnr06 = cv2.PSNR(img0, img6)
    print("PSNR zelda_Cipher_Proposed Image     = ", psnr06)



    # MSE
    print("MSE Original                          = ", getMSE(img0, img0))
    print("MSE zelda_Cipher_Trad_AES Image      = ", getMSE(img0, img1))
    print("MSE zelda_Cipher_Amjad_2019 Image    = ", getMSE(img0, img2))
    print("MSE zelda_Cipher_Nasir_2020 Image    = ", getMSE(img0, img3))
    print("MSE zelda_Cipher_Bahram_2020a Image  = ", getMSE(img0, img4))
    print("MSE zelda_Cipher_Bahram_2020b Image  = ", getMSE(img0, img5))
    print("MSE zelda_Cipher_Proposed Image      = ", getMSE(img0, img6))

    # MAE
    print("MAE Original                          = ", getMAE(img0, img0))
    print("MAE zelda_Cipher_Trad_AES Image      = ", getMAE(img0, img1))
    print("MAE zelda_Cipher_Amjad_2019 Image    = ", getMAE(img0, img2))
    print("MAE zelda_Cipher_Nasir_2020 Image    = ", getMAE(img0, img3))
    print("MAE zelda_Cipher_Bahram_2020a Image  = ", getMAE(img0, img4))
    print("MAE zelda_Cipher_Bahram_2020b Image  = ", getMAE(img0, img5))
    print("MAE zelda_Cipher_Proposed Image      = ", getMAE(img0, img6))



    # SSIM
    img0G = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1G = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2G = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3G = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4G = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    img5G = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    img6G = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
    print("SSIM Original                         = ", ssim(img0G, img0G))
    print("SSIM zelda_Cipher_Trad_AES Image     = ", ssim(img0G, img1G))
    print("SSIM zelda_Cipher_Amjad_2019 Image   = ", ssim(img0G, img2G))
    print("SSIM zelda_Cipher_Nasir_2020 Image   = ", ssim(img0G, img3G))
    print("SSIM zelda_Cipher_Bahram_2020a Image = ", ssim(img0G, img4G))
    print("SSIM zelda_Cipher_Bahram_2020b Image = ", ssim(img0G, img5G))
    print("SSIM zelda_Cipher_Proposed Image     = ", ssim(img0G, img6G))

    # Correlation
    cm = np.corrcoef(img0G.flat, img0G.flat)
    r = cm[0, 1]
    print("Cor. Original                         = ",r)
    cm = np.corrcoef(img0G.flat, img1G.flat)
    r = cm[0, 1]
    print("Cor. zelda_Cipher_Trad_AES Image     = ", r)
    cm = np.corrcoef(img0G.flat, img2G.flat)
    r = cm[0, 1]
    print("Cor. zelda_Cipher_Amjad_2019 Image   = ", r)
    cm = np.corrcoef(img0G.flat, img3G.flat)
    r = cm[0, 1]
    print("Cor. zelda_Cipher_Nasir_2020 Image   = ", r)
    cm = np.corrcoef(img0G.flat, img4G.flat)
    r = cm[0, 1]
    print("Cor. zelda_Cipher_Bahram_2020a Image = ", r)
    cm = np.corrcoef(img0G.flat, img5G.flat)
    r = cm[0, 1]
    print("Cor. zelda_Cipher_Bahram_2020b Image = ", r)
    cm = np.corrcoef(img0G.flat, img6G.flat)
    r = cm[0, 1]
    print("Cor. zelda_Cipher_Proposed Image     = ", r)
    #print("Correlation Original = ", signal.correlate(img0G, img0G))
    #print("Correlation Original = ", signal.correlate2d(img0G, img0G))




    # Homogeneity
    #print(imArray_Originals.ravel())
    #result = hg.pettitt_test(imArray_Originals)
    #print("Homogeneity = ",result)
    # ....


    imArray_Decipher_Trad_AES = imArray_Plain_Trad_AES

    imArray_Decipher_Amjad_2019 = imArray_Plain_Amjad_2019

    imArray_Decipher_Nasir_2020 = imArray_Plain_Nasir_2020

    imArray_Decipher_Bahram_2020a = imArray_Plain_Bahram_2020a

    imArray_Decipher_Bahram_2020b = imArray_Plain_Bahram_2020b

    imArray_Decipher_Proposed = imArray_Plain_Proposed

    for i in range(len(imArray_Originals)):
        for j in range(len(imArray_Originals[0])):
            for k in range(len(imArray_Originals[0][0])):  # 3):
                imArray_Decipher_Trad_AES[i][j][k] = inv_sub_bytes_AES(imArray_Cipher_Trad_AES[i][j][k])
                imArray_Decipher_Amjad_2019[i][j][k] = s_boxB256_Amjad_2019(imArray_Cipher_Amjad_2019[i][j][k])
                imArray_Decipher_Nasir_2020[i][j][k] = s_boxB256_Nasir_2020_s1(imArray_Cipher_Nasir_2020[i][j][k])
                imArray_Decipher_Bahram_2020a[i][j][k] = s_boxB256_Bahram_2020a(imArray_Cipher_Bahram_2020a[i][j][k])
                imArray_Decipher_Bahram_2020b[i][j][k] = s_boxB256_Bahram_2020b(imArray_Cipher_Bahram_2020b[i][j][k])
                imArray_Decipher_Proposed[i][j][k] = s_box_backword_proposed(imArray_Cipher_Proposed[i][j][k])

    im_Cipher_Trad_AES = Image.fromarray(imArray_Cipher_Trad_AES)
    im_Cipher_Trad_AES.save('zelda_Decipher_Trad_AES.png')

    im_Cipher_Amjad_2019 = Image.fromarray(imArray_Cipher_Amjad_2019)
    im_Cipher_Amjad_2019.save('zelda_Decipher_Amjad_2019.png')

    im_Cipher_Nasir_2020 = Image.fromarray(imArray_Cipher_Nasir_2020)
    im_Cipher_Nasir_2020.save('zelda_Decipher_Nasir_2020.png')

    im_Cipher_Bahram_2020a = Image.fromarray(imArray_Cipher_Bahram_2020a)
    im_Cipher_Bahram_2020a.save('zelda_Decipher_Bahram_2020a.png')

    im_Cipher_Bahram_2020b = Image.fromarray(imArray_Cipher_Bahram_2020b)
    im_Cipher_Bahram_2020b.save('zelda_Decipher_Bahram_2020b.png')

    im_Cipher_Proposed = Image.fromarray(imArray_Cipher_Proposed)
    im_Cipher_Proposed.save('zelda_Decipher_Proposed.png')
    im_Cipher_Proposed.save('zelda_Decipher_Proposed_SBox4.png')

    #Histogram
    im = io.imread('zelda.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

    im = io.imread('zelda_Cipher_Trad_AES.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

    im = io.imread('zelda_Cipher_Amjad_2019.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

    im = io.imread('zelda_Cipher_Nasir_2020.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

    im = io.imread('zelda_Cipher_Bahram_2020a.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

    im = io.imread('zelda_Cipher_Bahram_2020b.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()

    im = io.imread('zelda_Cipher_Proposed_SBox4.png')
    plt.hist(im.ravel(), bins=8)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.show()



    print("End of  Security Analysis Started !!!")

def main():
    print("Main Program Started !!!")
    # code here
    #main2()
    runAnalysis()
    #HWA()
    print("Main Program run successfully!!!")

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Starting, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    main()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('SBox 4.0: Program !!!')