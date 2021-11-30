#!/usr/bin/env python3
"""
This is an exercise in secure symmetric-key encryption, implemented in pure
Python (no external libraries needed).
Original AES-128 implementation by Bo Zhu (http://about.bozhu.me) at 
https://github.com/bozhu/AES-Python . PKCS#7 padding, CBC mode, PKBDF2, HMAC,
byte array and string support added by me at https://github.com/boppreh/aes. 
Other block modes contributed by @righthandabacus.
Although this is an exercise, the `encrypt` and `decrypt` functions should
provide reasonable security to encrypted messages.
"""


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

# ROSIE Entropy  = h =  7.694180448822675 VS Bahram 2021 h =  7.6739952595684615
s_box_ROSIE    =  [54, 115, 104, 244, 221, 164, 20, 211, 157, 113, 246, 171, 144, 161, 26, 41, 179, 181, 52, 37, 122, 46, 127, 67, 4, 134, 58, 228, 163, 240, 8, 33, 131, 222, 170, 62, 2, 212, 133, 252, 101, 103, 34, 202, 110, 89, 165, 218, 156, 132, 85, 106, 81, 70, 32, 78, 153, 233, 65, 73, 169, 74, 237, 31, 177, 29, 123, 224, 45, 142, 214, 232, 75, 68, 242, 22, 238, 28, 213, 61, 150, 55, 193, 197, 10, 180, 12, 83, 11, 3, 76, 60, 25, 19, 90, 178, 247, 18, 15, 23, 5, 135, 191, 1, 231, 59, 100, 226, 56, 95, 201, 9, 79, 94, 249, 48, 111, 254, 198, 172, 0, 248, 200, 138, 219, 235, 146, 243, 255, 152, 16, 96, 185, 126, 174, 229, 40, 82, 203, 105, 184, 21, 204, 63, 160, 154, 6, 236, 88, 205, 155, 98, 143, 64, 207, 14, 253, 245, 49, 167, 87, 35, 208, 30, 102, 209, 24, 92, 36, 210, 158, 183, 93, 141, 148, 97, 175, 107, 39, 50, 159, 220, 182, 206, 125, 227, 149, 225, 71, 51, 99, 195, 72, 173, 53, 38, 17, 118, 91, 121, 186, 77, 120, 80, 137, 192, 47, 147, 187, 176, 199, 108, 119, 27, 129, 251, 13, 42, 130, 196, 117, 43, 217, 166, 162, 188, 7, 234, 189, 116, 124, 216, 230, 241, 84, 57, 239, 223, 194, 112, 114, 168, 145, 66, 44, 128, 109, 140, 139, 190, 215, 136, 69, 86, 151, 250]
inv_s_box_ROSIE =  [120, 103, 36, 89, 24, 100, 146, 226, 30, 111, 84, 88, 86, 216, 155, 98, 130, 196, 97, 93, 6, 141, 75, 99, 166, 92, 14, 213, 77, 65, 163, 63, 54, 31, 42, 161, 168, 19, 195, 178, 136, 15, 217, 221, 244, 68, 21, 206, 115, 158, 179, 189, 18, 194, 0, 81, 108, 235, 26, 105, 91, 79, 35, 143, 153, 58, 243, 23, 73, 252, 53, 188, 192, 59, 61, 72, 90, 201, 55, 112, 203, 52, 137, 87, 234, 50, 253, 160, 148, 45, 94, 198, 167, 172, 113, 109, 131, 175, 151, 190, 106, 40, 164, 41, 2, 139, 51, 177, 211, 246, 44, 116, 239, 9, 240, 1, 229, 220, 197, 212, 202, 199, 20, 66, 230, 184, 133, 22, 245, 214, 218, 32, 49, 38, 25, 101, 251, 204, 123, 248, 247, 173, 69, 152, 12, 242, 126, 207, 174, 186, 80, 254, 129, 56, 145, 150, 48, 8, 170, 180, 144, 13, 224, 28, 5, 46, 223, 159, 241, 60, 34, 11, 119, 193, 134, 176, 209, 64, 95, 16, 85, 17, 182, 171, 140, 132, 200, 208, 225, 228, 249, 102, 205, 82, 238, 191, 219, 83, 118, 210, 122, 110, 43, 138, 142, 149, 183, 154, 162, 165, 169, 7, 37, 78, 70, 250, 231, 222, 47, 124, 181, 4, 33, 237, 67, 187, 107, 185, 27, 135, 232, 104, 71, 57, 227, 125, 147, 62, 76, 236, 29, 233, 74, 127, 3, 157, 10, 96, 121, 114, 255, 215, 39, 156, 117, 128]


# Proposed RTL_Light 
s_box_SEB_RTL =  [55, 200, 157, 98, 251, 4, 81, 174, 191, 64, 21, 234, 115, 140, 217, 38, 199, 56, 109, 146, 11, 244, 161, 94, 79, 176, 229, 26, 131, 124, 41, 214, 151, 104, 61, 194, 91, 164, 241, 14, 31, 224, 181, 74, 211, 44, 121, 134, 103, 152, 205, 50, 171, 84, 1, 254, 239, 16, 69, 186, 35, 220, 137, 118, 247, 8, 93, 162, 59, 196, 145, 110, 127, 128, 213, 42, 179, 76, 25, 230, 7, 248, 173, 82, 203, 52, 97, 158, 143, 112, 37, 218, 67, 188, 233, 22, 87, 168, 253, 2, 155, 100, 49, 206, 223, 32, 117, 138, 19, 236, 185, 70, 167, 88, 13, 242, 107, 148, 193, 62, 47, 208, 133, 122, 227, 28, 73, 182, 183, 72, 29, 226, 123, 132, 209, 46, 63, 192, 149, 106, 243, 12, 89, 166, 71, 184, 237, 18, 139, 116, 33, 222, 207, 48, 101, 154, 3, 252, 169, 86, 23, 232, 189, 66, 219, 36, 113, 142, 159, 96, 53, 202, 83, 172, 249, 6, 231, 24, 77, 178, 43, 212, 129, 126, 111, 144, 197, 58, 163, 92, 9, 246, 119, 136, 221, 34, 187, 68, 17, 238, 255, 0, 85, 170, 51, 204, 153, 102, 135, 120, 45, 210, 75, 180, 225, 30, 15, 240, 165, 90, 195, 60, 105, 150, 215, 40, 125, 130, 27, 228, 177, 78, 95, 160, 245, 10, 147, 108, 57, 198, 39, 216, 141, 114, 235, 20, 65, 190, 175, 80, 5, 250, 99, 156, 201, 54]
inv_s_box_SEB_RTL  =  [201, 54, 99, 156, 5, 250, 175, 80, 65, 190, 235, 20, 141, 114, 39, 216, 57, 198, 147, 108, 245, 10, 95, 160, 177, 78, 27, 228, 125, 130, 215, 40, 105, 150, 195, 60, 165, 90, 15, 240, 225, 30, 75, 180, 45, 210, 135, 120, 153, 102, 51, 204, 85, 170, 255, 0, 17, 238, 187, 68, 221, 34, 119, 136, 9, 246, 163, 92, 197, 58, 111, 144, 129, 126, 43, 212, 77, 178, 231, 24, 249, 6, 83, 172, 53, 202, 159, 96, 113, 142, 219, 36, 189, 66, 23, 232, 169, 86, 3, 252, 101, 154, 207, 48, 33, 222, 139, 116, 237, 18, 71, 184, 89, 166, 243, 12, 149, 106, 63, 192, 209, 46, 123, 132, 29, 226, 183, 72, 73, 182, 227, 28, 133, 122, 47, 208, 193, 62, 107, 148, 13, 242, 167, 88, 185, 70, 19, 236, 117, 138, 223, 32, 49, 206, 155, 100, 253, 2, 87, 168, 233, 22, 67, 188, 37, 218, 143, 112, 97, 158, 203, 52, 173, 82, 7, 248, 25, 230, 179, 76, 213, 42, 127, 128, 145, 110, 59, 196, 93, 162, 247, 8, 137, 118, 35, 220, 69, 186, 239, 16, 1, 254, 171, 84, 205, 50, 103, 152, 121, 134, 211, 44, 181, 74, 31, 224, 241, 14, 91, 164, 61, 194, 151, 104, 41, 214, 131, 124, 229, 26, 79, 176, 161, 94, 11, 244, 109, 146, 199, 56, 217, 38, 115, 140, 21, 234, 191, 64, 81, 174, 251, 4, 157, 98, 55, 200]




#bahram_2021_Entropy = 7.6739952595684615
s_box_Bahrami_2021 =  [130, 19, 159, 107, 217, 188, 118, 231, 250, 161, 240, 120, 202, 196, 48, 62, 125, 5, 126, 1, 163, 47, 89, 165, 175, 0, 117, 191, 53, 226, 251, 189, 245, 94, 200, 193, 173, 201, 248, 172, 18, 58, 199, 4, 111, 12, 254, 21, 14, 195, 152, 255, 93, 28, 41, 239, 3, 67, 41, 102, 109, 73, 174, 112, 78, 87, 149, 57, 205, 156, 171, 60, 10, 61, 242, 252, 134, 227, 208, 233, 27, 92, 181, 178, 42, 11, 65, 65, 164, 22, 247, 2, 194, 52, 33, 100, 197, 63, 170, 155, 95, 180, 59, 214, 229, 110, 76, 3, 223, 60, 219, 44, 63, 203, 222, 144, 36, 26, 6, 230, 136, 140, 23, 237, 211, 16, 18, 232, 215, 167, 0, 213, 185, 224, 76, 50, 253, 234, 218, 246, 121, 90, 221, 190, 115, 123, 209, 158, 98, 66, 157, 99, 50, 55, 216, 99, 69, 150, 147, 2, 198, 9, 207, 20, 88, 23, 154, 129, 206, 39, 98, 70, 39, 36, 62, 24, 238, 176, 83, 7, 244, 82, 142, 168, 243, 132, 137, 212, 5, 31, 143, 27, 228, 35, 92, 49, 13, 96, 160, 42, 34, 37, 12, 113, 46, 68, 168, 162, 78, 146, 235, 124, 187, 131, 103, 128, 154, 4, 138, 110, 127, 142, 114, 108, 133, 79, 29, 77, 192, 104, 119, 7, 127, 79, 137, 210, 241, 123, 176, 55, 10, 236, 74, 184, 180, 32, 139, 90, 38, 51, 8, 30, 75, 136, 46, 51]
inv_s_box_Bahrami_2021 =  [130, 19, 159, 107, 217, 188, 118, 231, 250, 161, 240, 120, 202, 196, 48, 62, 125, 5, 126, 1, 163, 47, 89, 165, 175, 0, 117, 191, 53, 226, 251, 189, 245, 94, 200, 193, 173, 201, 248, 172, 18, 58, 199, 4, 111, 12, 254, 21, 14, 195, 152, 255, 93, 28, 41, 239, 3, 67, 41, 102, 109, 73, 174, 112, 78, 87, 149, 57, 205, 156, 171, 60, 10, 61, 242, 252, 134, 227, 208, 233, 27, 92, 181, 178, 42, 11, 65, 65, 164, 22, 247, 2, 194, 52, 33, 100, 197, 63, 170, 155, 95, 180, 59, 214, 229, 110, 76, 3, 223, 60, 219, 44, 63, 203, 222, 144, 36, 26, 6, 230, 136, 140, 23, 237, 211, 16, 18, 232, 215, 167, 0, 213, 185, 224, 76, 50, 253, 234, 218, 246, 121, 90, 221, 190, 115, 123, 209, 158, 98, 66, 157, 99, 50, 55, 216, 99, 69, 150, 147, 2, 198, 9, 207, 20, 88, 23, 154, 129, 206, 39, 98, 70, 39, 36, 62, 24, 238, 176, 83, 7, 244, 82, 142, 168, 243, 132, 137, 212, 5, 31, 143, 27, 228, 35, 92, 49, 13, 96, 160, 42, 34, 37, 12, 113, 46, 68, 168, 162, 78, 146, 235, 124, 187, 131, 103, 128, 154, 4, 138, 110, 127, 142, 114, 108, 133, 79, 29, 77, 192, 104, 119, 7, 127, 79, 137, 210, 241, 123, 176, 55, 10, 236, 74, 184, 180, 32, 139, 90, 38, 51, 8, 30, 75, 136, 46, 51]


s_box_DQB_RTL =  [135, 210, 30, 120, 45, 60, 75, 195, 225, 90, 105, 150, 165, 180, 15, 240, 39, 114, 190, 216, 141, 156, 235, 99, 65, 250, 201, 54, 5, 20, 175, 80, 247, 162, 110, 8, 93, 76, 59, 179, 145, 42, 25, 230, 213, 196, 127, 128, 231, 178, 126, 24, 77, 92, 43, 163, 129, 58, 9, 246, 197, 212, 111, 144, 215, 130, 78, 40, 125, 108, 27, 147, 177, 10, 57, 198, 245, 228, 95, 160, 23, 66, 142, 232, 189, 172, 219, 83, 113, 202, 249, 6, 53, 36, 159, 96, 7, 82, 158, 248, 173, 188, 203, 67, 97, 218, 233, 22, 37, 52, 143, 112, 55, 98, 174, 200, 157, 140, 251, 115, 81, 234, 217, 38, 21, 4, 191, 64, 71, 18, 222, 184, 237, 252, 139, 3, 33, 154, 169, 86, 101, 116, 207, 48, 103, 50, 254, 152, 205, 220, 171, 35, 1, 186, 137, 118, 69, 84, 239, 16, 183, 226, 46, 72, 29, 12, 123, 243, 209, 106, 89, 166, 149, 132, 63, 192, 119, 34, 238, 136, 221, 204, 187, 51, 17, 170, 153, 102, 85, 68, 255, 0, 199, 146, 94, 56, 109, 124, 11, 131, 161, 26, 41, 214, 229, 244, 79, 176, 87, 2, 206, 168, 253, 236, 155, 19, 49, 138, 185, 70, 117, 100, 223, 32, 167, 242, 62, 88, 13, 28, 107, 227, 193, 122, 73, 182, 133, 148, 47, 208, 151, 194, 14, 104, 61, 44, 91, 211, 241, 74, 121, 134, 181, 164, 31, 224]
inv_s_box_DQB_RTL =  [191, 152, 209, 135, 125, 28, 91, 96, 35, 58, 73, 198, 165, 228, 242, 14, 159, 184, 129, 215, 29, 124, 107, 80, 51, 42, 201, 70, 229, 164, 2, 254, 223, 136, 177, 151, 93, 108, 123, 16, 67, 202, 41, 54, 245, 4, 162, 238, 143, 216, 145, 183, 109, 92, 27, 112, 195, 74, 57, 38, 5, 244, 226, 174, 127, 24, 81, 103, 189, 156, 219, 128, 163, 234, 249, 6, 37, 52, 66, 206, 31, 120, 97, 87, 157, 188, 139, 208, 227, 170, 9, 246, 53, 36, 194, 78, 95, 104, 113, 23, 221, 140, 187, 144, 243, 10, 169, 230, 69, 196, 34, 62, 111, 88, 17, 119, 141, 220, 155, 176, 3, 250, 233, 166, 197, 68, 50, 46, 47, 56, 65, 199, 173, 236, 251, 0, 179, 154, 217, 134, 117, 20, 82, 110, 63, 40, 193, 71, 237, 172, 11, 240, 147, 186, 137, 214, 21, 116, 98, 94, 79, 200, 33, 55, 253, 12, 171, 224, 211, 138, 185, 150, 85, 100, 114, 30, 207, 72, 49, 39, 13, 252, 235, 160, 131, 218, 153, 182, 101, 84, 18, 126, 175, 232, 241, 7, 45, 60, 75, 192, 115, 26, 89, 102, 181, 148, 210, 142, 239, 168, 1, 247, 61, 44, 203, 64, 19, 122, 105, 86, 149, 180, 130, 222, 255, 8, 161, 231, 77, 204, 43, 48, 83, 106, 121, 22, 213, 132, 178, 158, 15, 248, 225, 167, 205, 76, 59, 32, 99, 90, 25, 118, 133, 212, 146, 190]


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


def sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]


def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inv_s_box[s[i][j]]


def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]


def inv_shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]

def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]


# learned from http://cs.ucsb.edu/~koc/cs178/projects/JT/aes.c
xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    # see Sec 4.1.2 in The Design of Rijndael
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])


def inv_mix_columns(s):
    # see Sec 4.1.3 in The Design of Rijndael
    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u
        s[i][1] ^= v
        s[i][2] ^= u
        s[i][3] ^= v

    mix_columns(s)


r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def bytes2matrix(text):
    """ Converts a 16-byte array into a 4x4 matrix.  """
    return [list(text[i:i+4]) for i in range(0, len(text), 4)]

def matrix2bytes(matrix):
    """ Converts a 4x4 matrix into a 16-byte array.  """
    return bytes(sum(matrix, []))

def xor_bytes(a, b):
    """ Returns a new byte array with the elements xor'ed. """
    return bytes(i^j for i, j in zip(a, b))

def inc_bytes(a):
    """ Returns a new byte array with the value increment by 1 """
    out = list(a)
    for i in reversed(range(len(out))):
        if out[i] == 0xFF:
            out[i] = 0
        else:
            out[i] += 1
            break
    return bytes(out)

def pad(plaintext):
    """
    Pads the given plaintext with PKCS#7 padding to a multiple of 16 bytes.
    Note that if the plaintext size is a multiple of 16,
    a whole block will be added.
    """
    padding_len = 16 - (len(plaintext) % 16)
    padding = bytes([padding_len] * padding_len)
    return plaintext + padding

def unpad(plaintext):
    """
    Removes a PKCS#7 padding, returning the unpadded text and ensuring the
    padding was correct.
    """
    padding_len = plaintext[-1]
    assert padding_len > 0
    message, padding = plaintext[:-padding_len], plaintext[-padding_len:]
    assert all(p == padding_len for p in padding)
    return message

def split_blocks(message, block_size=16, require_padding=True):
        assert len(message) % block_size == 0 or not require_padding
        return [message[i:i+16] for i in range(0, len(message), block_size)]


class AES:
    """
    Class for AES-128 encryption with CBC mode and PKCS#7.
    This is a raw implementation of AES, without key stretching or IV
    management. Unless you need that, please use `encrypt` and `decrypt`.
    """
    rounds_by_key_size = {16: 10, 24: 12, 32: 14}
    def __init__(self, master_key):
        """
        Initializes the object with a given key.
        """
        assert len(master_key) in AES.rounds_by_key_size
        self.n_rounds = AES.rounds_by_key_size[len(master_key)]
        self._key_matrices = self._expand_key(master_key)

    def _expand_key(self, master_key):
        """
        Expands and returns a list of key matrices for the given master_key.
        """
        # Initialize round keys with raw key material.
        key_columns = bytes2matrix(master_key)
        iteration_size = len(master_key) // 4

        # Each iteration has exactly as many columns as the key material.
        columns_per_iteration = len(key_columns)
        i = 1
        while len(key_columns) < (self.n_rounds + 1) * 4:
            # Copy previous word.
            word = list(key_columns[-1])

            # Perform schedule_core once every "row".
            if len(key_columns) % iteration_size == 0:
                # Circular shift.
                word.append(word.pop(0))
                # Map to S-BOX.
                word = [s_box[b] for b in word]
                # XOR with first byte of R-CON, since the others bytes of R-CON are 0.
                word[0] ^= r_con[i]
                i += 1
            elif len(master_key) == 32 and len(key_columns) % iteration_size == 4:
                # Run word through S-box in the fourth iteration when using a
                # 256-bit key.
                word = [s_box[b] for b in word]

            # XOR with equivalent word from previous iteration.
            word = xor_bytes(word, key_columns[-iteration_size])
            key_columns.append(word)

        # Group key words in 4x4 byte matrices.
        return [key_columns[4*i : 4*(i+1)] for i in range(len(key_columns) // 4)]

    def encrypt_block(self, plaintext):
        """
        Encrypts a single block of 16 byte long plaintext.
        """
        assert len(plaintext) == 16

        plain_state = bytes2matrix(plaintext)

        add_round_key(plain_state, self._key_matrices[0])

        for i in range(1, self.n_rounds):
            sub_bytes(plain_state)
            shift_rows(plain_state)
            mix_columns(plain_state)
            add_round_key(plain_state, self._key_matrices[i])

        sub_bytes(plain_state)
        shift_rows(plain_state)
        add_round_key(plain_state, self._key_matrices[-1])

        return matrix2bytes(plain_state)

    def decrypt_block(self, ciphertext):
        """
        Decrypts a single block of 16 byte long ciphertext.
        """
        assert len(ciphertext) == 16

        cipher_state = bytes2matrix(ciphertext)

        add_round_key(cipher_state, self._key_matrices[-1])
        inv_shift_rows(cipher_state)
        inv_sub_bytes(cipher_state)

        for i in range(self.n_rounds - 1, 0, -1):
            add_round_key(cipher_state, self._key_matrices[i])
            inv_mix_columns(cipher_state)
            inv_shift_rows(cipher_state)
            inv_sub_bytes(cipher_state)

        add_round_key(cipher_state, self._key_matrices[0])

        return matrix2bytes(cipher_state)

    def encrypt_cbc(self, plaintext, iv):
        """
        Encrypts `plaintext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 16

        plaintext = pad(plaintext)

        blocks = []
        previous = iv
        for plaintext_block in split_blocks(plaintext):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            block = self.encrypt_block(xor_bytes(plaintext_block, previous))
            blocks.append(block)
            previous = block

        return b''.join(blocks)

    def decrypt_cbc(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using CBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        previous = iv
        for ciphertext_block in split_blocks(ciphertext):
            # CBC mode decrypt: previous XOR decrypt(ciphertext)
            blocks.append(xor_bytes(previous, self.decrypt_block(ciphertext_block)))
            previous = ciphertext_block

        return unpad(b''.join(blocks))

    def encrypt_pcbc(self, plaintext, iv):
        """
        Encrypts `plaintext` using PCBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 16

        plaintext = pad(plaintext)

        blocks = []
        prev_ciphertext = iv
        prev_plaintext = bytes(16)
        for plaintext_block in split_blocks(plaintext):
            # PCBC mode encrypt: encrypt(plaintext_block XOR (prev_ciphertext XOR prev_plaintext))
            ciphertext_block = self.encrypt_block(xor_bytes(plaintext_block, xor_bytes(prev_ciphertext, prev_plaintext)))
            blocks.append(ciphertext_block)
            prev_ciphertext = ciphertext_block
            prev_plaintext = plaintext_block

        return b''.join(blocks)

    def decrypt_pcbc(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using PCBC mode and PKCS#7 padding, with the given
        initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        prev_ciphertext = iv
        prev_plaintext = bytes(16)
        for ciphertext_block in split_blocks(ciphertext):
            # PCBC mode decrypt: (prev_plaintext XOR prev_ciphertext) XOR decrypt(ciphertext_block)
            plaintext_block = xor_bytes(xor_bytes(prev_ciphertext, prev_plaintext), self.decrypt_block(ciphertext_block))
            blocks.append(plaintext_block)
            prev_ciphertext = ciphertext_block
            prev_plaintext = plaintext_block

        return unpad(b''.join(blocks))

    def encrypt_cfb(self, plaintext, iv):
        """
        Encrypts `plaintext` with the given initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        prev_ciphertext = iv
        for plaintext_block in split_blocks(plaintext, require_padding=False):
            # CFB mode encrypt: plaintext_block XOR encrypt(prev_ciphertext)
            ciphertext_block = xor_bytes(plaintext_block, self.encrypt_block(prev_ciphertext))
            blocks.append(ciphertext_block)
            prev_ciphertext = ciphertext_block

        return b''.join(blocks)

    def decrypt_cfb(self, ciphertext, iv):
        """
        Decrypts `ciphertext` with the given initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        prev_ciphertext = iv
        for ciphertext_block in split_blocks(ciphertext, require_padding=False):
            # CFB mode decrypt: ciphertext XOR decrypt(prev_ciphertext)
            plaintext_block = xor_bytes(ciphertext_block, self.encrypt_block(prev_ciphertext))
            blocks.append(plaintext_block)
            prev_ciphertext = ciphertext_block

        return b''.join(blocks)

    def encrypt_ofb(self, plaintext, iv):
        """
        Encrypts `plaintext` using OFB mode initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        previous = iv
        for plaintext_block in split_blocks(plaintext, require_padding=False):
            # OFB mode encrypt: plaintext_block XOR encrypt(previous)
            block = self.encrypt_block(previous)
            ciphertext_block = xor_bytes(plaintext_block, block)
            blocks.append(ciphertext_block)
            previous = block

        return b''.join(blocks)

    def decrypt_ofb(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using OFB mode initialization vector (iv).
        """
        assert len(iv) == 16

        blocks = []
        previous = iv
        for ciphertext_block in split_blocks(ciphertext, require_padding=False):
            # OFB mode decrypt: ciphertext XOR encrypt(previous)
            block = self.encrypt_block(previous)
            plaintext_block = xor_bytes(ciphertext_block, block)
            blocks.append(plaintext_block)
            previous = block

        return b''.join(blocks)

    def encrypt_ctr(self, plaintext, iv):
        """
        Encrypts `plaintext` using CTR mode with the given nounce/IV.
        """
        assert len(iv) == 16

        blocks = []
        nonce = iv
        for plaintext_block in split_blocks(plaintext, require_padding=False):
            # CTR mode encrypt: plaintext_block XOR encrypt(nonce)
            block = xor_bytes(plaintext_block, self.encrypt_block(nonce))
            blocks.append(block)
            nonce = inc_bytes(nonce)

        return b''.join(blocks)

    def decrypt_ctr(self, ciphertext, iv):
        """
        Decrypts `ciphertext` using CTR mode with the given nounce/IV.
        """
        assert len(iv) == 16

        blocks = []
        nonce = iv
        for ciphertext_block in split_blocks(ciphertext, require_padding=False):
            # CTR mode decrypt: ciphertext XOR encrypt(nonce)
            block = xor_bytes(ciphertext_block, self.encrypt_block(nonce))
            blocks.append(block)
            nonce = inc_bytes(nonce)

        return b''.join(blocks)


import os
from hashlib import pbkdf2_hmac
from hmac import new as new_hmac, compare_digest

AES_KEY_SIZE = 16
HMAC_KEY_SIZE = 16
IV_SIZE = 16

SALT_SIZE = 16
HMAC_SIZE = 32

def get_key_iv(password, salt, workload=100000):
    """
    Stretches the password and extracts an AES key, an HMAC key and an AES
    initialization vector.
    """
    stretched = pbkdf2_hmac('sha256', password, salt, workload, AES_KEY_SIZE + IV_SIZE + HMAC_KEY_SIZE)
    aes_key, stretched = stretched[:AES_KEY_SIZE], stretched[AES_KEY_SIZE:]
    hmac_key, stretched = stretched[:HMAC_KEY_SIZE], stretched[HMAC_KEY_SIZE:]
    iv = stretched[:IV_SIZE]
    return aes_key, hmac_key, iv


def encrypt(key, plaintext, workload=100000):
    """
    Encrypts `plaintext` with `key` using AES-128, an HMAC to verify integrity,
    and PBKDF2 to stretch the given key.
    The exact algorithm is specified in the module docstring.
    """
    if isinstance(key, str):
        key = key.encode('utf-8')
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')

    salt = os.urandom(SALT_SIZE)
    key, hmac_key, iv = get_key_iv(key, salt, workload)
    ciphertext = AES(key).encrypt_cbc(plaintext, iv)
    hmac = new_hmac(hmac_key, salt + ciphertext, 'sha256').digest()
    assert len(hmac) == HMAC_SIZE

    return hmac + salt + ciphertext


def decrypt(key, ciphertext, workload=100000):
    """
    Decrypts `ciphertext` with `key` using AES-128, an HMAC to verify integrity,
    and PBKDF2 to stretch the given key.
    The exact algorithm is specified in the module docstring.
    """

    assert len(ciphertext) % 16 == 0, "Ciphertext must be made of full 16-byte blocks."

    assert len(ciphertext) >= 32, """
    Ciphertext must be at least 32 bytes long (16 byte salt + 16 byte block). To
    encrypt or decrypt single blocks use `AES(key).decrypt_block(ciphertext)`.
    """

    if isinstance(key, str):
        key = key.encode('utf-8')

    hmac, ciphertext = ciphertext[:HMAC_SIZE], ciphertext[HMAC_SIZE:]
    salt, ciphertext = ciphertext[:SALT_SIZE], ciphertext[SALT_SIZE:]
    key, hmac_key, iv = get_key_iv(key, salt, workload)

    expected_hmac = new_hmac(hmac_key, salt + ciphertext, 'sha256').digest()
    assert compare_digest(hmac, expected_hmac), 'Ciphertext corrupted or tampered.'

    return AES(key).decrypt_cbc(ciphertext, iv)


def benchmark():
    key = b'P' * 16
    message = b'M' * 16
    aes = AES(key)
    for i in range(30000):
        aes.encrypt_block(message)

__all__ = [encrypt, decrypt, AES]

if __name__ == '__main__':
    import sys
    write = lambda b: sys.stdout.buffer.write(b)
    read = lambda: sys.stdin.buffer.read()

    if len(sys.argv) < 2:
        print('Usage: ./aes.py encrypt "key" "message"')
        print('Running tests...')
        from tests import *
        run()
    elif len(sys.argv) == 2 and sys.argv[1] == 'benchmark':
        benchmark()
        exit()
    elif len(sys.argv) == 3:
        text = read()
    elif len(sys.argv) > 3:
        text = ' '.join(sys.argv[2:])

    if 'encrypt'.startswith(sys.argv[1]):
        write(encrypt(sys.argv[2], text))
    elif 'decrypt'.startswith(sys.argv[1]):
        write(decrypt(sys.argv[2], text))
    else:
        print('Expected command "encrypt" or "decrypt" in first argument.')

    # encrypt('my secret key', b'0' * 1000000) # 1 MB encrypted in 20 seconds.