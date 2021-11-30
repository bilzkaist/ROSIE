import numpy as np
import random 

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

s_box_RTL =  [135, 210, 30, 120, 45, 60, 75, 195, 225, 90, 105, 150, 165, 180, 15, 240, 39, 114, 190, 216, 141, 156, 235, 99, 65, 250, 201, 54, 5, 20, 175, 80, 247, 162, 110, 8, 93, 76, 59, 179, 145, 42, 25, 230, 213, 196, 127, 128, 231, 178, 126, 24, 77, 92, 43, 163, 129, 58, 9, 246, 197, 212, 111, 144, 215, 130, 78, 40, 125, 108, 27, 147, 177, 10, 57, 198, 245, 228, 95, 160, 23, 66, 142, 232, 189, 172, 219, 83, 113, 202, 249, 6, 53, 36, 159, 96, 7, 82, 158, 248, 173, 188, 203, 67, 97, 218, 233, 22, 37, 52, 143, 112, 55, 98, 174, 200, 157, 140, 251, 115, 81, 234, 217, 38, 21, 4, 191, 64, 71, 18, 222, 184, 237, 252, 139, 3, 33, 154, 169, 86, 101, 116, 207, 48, 103, 50, 254, 152, 205, 220, 171, 35, 1, 186, 137, 118, 69, 84, 239, 16, 183, 226, 46, 72, 29, 12, 123, 243, 209, 106, 89, 166, 149, 132, 63, 192, 119, 34, 238, 136, 221, 204, 187, 51, 17, 170, 153, 102, 85, 68, 255, 0, 199, 146, 94, 56, 109, 124, 11, 131, 161, 26, 41, 214, 229, 244, 79, 176, 87, 2, 206, 168, 253, 236, 155, 19, 49, 138, 185, 70, 117, 100, 223, 32, 167, 242, 62, 88, 13, 28, 107, 227, 193, 122, 73, 182, 133, 148, 47, 208, 151, 194, 14, 104, 61, 44, 91, 211, 241, 74, 121, 134, 181, 164, 31, 224]
inv_s_box_RTL  =  [191, 152, 209, 135, 125, 28, 91, 96, 35, 58, 73, 198, 165, 228, 242, 14, 159, 184, 129, 215, 29, 124, 107, 80, 51, 42, 201, 70, 229, 164, 2, 254, 223, 136, 177, 151, 93, 108, 123, 16, 67, 202, 41, 54, 245, 4, 162, 238, 143, 216, 145, 183, 109, 92, 27, 112, 195, 74, 57, 38, 5, 244, 226, 174, 127, 24, 81, 103, 189, 156, 219, 128, 163, 234, 249, 6, 37, 52, 66, 206, 31, 120, 97, 87, 157, 188, 139, 208, 227, 170, 9, 246, 53, 36, 194, 78, 95, 104, 113, 23, 221, 140, 187, 144, 243, 10, 169, 230, 69, 196, 34, 62, 111, 88, 17, 119, 141, 220, 155, 176, 3, 250, 233, 166, 197, 68, 50, 46, 47, 56, 65, 199, 173, 236, 251, 0, 179, 154, 217, 134, 117, 20, 82, 110, 63, 40, 193, 71, 237, 172, 11, 240, 147, 186, 137, 214, 21, 116, 98, 94, 79, 200, 33, 55, 253, 12, 171, 224, 211, 138, 185, 150, 85, 100, 114, 30, 207, 72, 49, 39, 13, 252, 235, 160, 131, 218, 153, 182, 101, 84, 18, 126, 175, 232, 241, 7, 45, 60, 75, 192, 115, 26, 89, 102, 181, 148, 210, 142, 239, 168, 1, 247, 61, 44, 203, 64, 19, 122, 105, 86, 149, 180, 130, 222, 255, 8, 161, 231, 77, 204, 43, 48, 83, 106, 121, 22, 213, 132, 178, 158, 15, 248, 225, 167, 205, 76, 59, 32, 99, 90, 25, 118, 133, 212, 146, 190]



s_box =  [55, 200, 157, 98, 251, 4, 81, 174, 191, 64, 21, 234, 115, 140, 217, 38, 199, 56, 109, 146, 11, 244, 161, 94, 79, 176, 229, 26, 131, 124, 41, 214, 151, 104, 61, 194, 91, 164, 241, 14, 31, 224, 181, 74, 211, 44, 121, 134, 103, 152, 205, 50, 171, 84, 1, 254, 239, 16, 69, 186, 35, 220, 137, 118, 247, 8, 93, 162, 59, 196, 145, 110, 127, 128, 213, 42, 179, 76, 25, 230, 7, 248, 173, 82, 203, 52, 97, 158, 143, 112, 37, 218, 67, 188, 233, 22, 87, 168, 253, 2, 155, 100, 49, 206, 223, 32, 117, 138, 19, 236, 185, 70, 167, 88, 13, 242, 107, 148, 193, 62, 47, 208, 133, 122, 227, 28, 73, 182, 183, 72, 29, 226, 123, 132, 209, 46, 63, 192, 149, 106, 243, 12, 89, 166, 71, 184, 237, 18, 139, 116, 33, 222, 207, 48, 101, 154, 3, 252, 169, 86, 23, 232, 189, 66, 219, 36, 113, 142, 159, 96, 53, 202, 83, 172, 249, 6, 231, 24, 77, 178, 43, 212, 129, 126, 111, 144, 197, 58, 163, 92, 9, 246, 119, 136, 221, 34, 187, 68, 17, 238, 255, 0, 85, 170, 51, 204, 153, 102, 135, 120, 45, 210, 75, 180, 225, 30, 15, 240, 165, 90, 195, 60, 105, 150, 215, 40, 125, 130, 27, 228, 177, 78, 95, 160, 245, 10, 147, 108, 57, 198, 39, 216, 141, 114, 235, 20, 65, 190, 175, 80, 5, 250, 99, 156, 201, 54]
inv_s_box  =  [201, 54, 99, 156, 5, 250, 175, 80, 65, 190, 235, 20, 141, 114, 39, 216, 57, 198, 147, 108, 245, 10, 95, 160, 177, 78, 27, 228, 125, 130, 215, 40, 105, 150, 195, 60, 165, 90, 15, 240, 225, 30, 75, 180, 45, 210, 135, 120, 153, 102, 51, 204, 85, 170, 255, 0, 17, 238, 187, 68, 221, 34, 119, 136, 9, 246, 163, 92, 197, 58, 111, 144, 129, 126, 43, 212, 77, 178, 231, 24, 249, 6, 83, 172, 53, 202, 159, 96, 113, 142, 219, 36, 189, 66, 23, 232, 169, 86, 3, 252, 101, 154, 207, 48, 33, 222, 139, 116, 237, 18, 71, 184, 89, 166, 243, 12, 149, 106, 63, 192, 209, 46, 123, 132, 29, 226, 183, 72, 73, 182, 227, 28, 133, 122, 47, 208, 193, 62, 107, 148, 13, 242, 167, 88, 185, 70, 19, 236, 117, 138, 223, 32, 49, 206, 155, 100, 253, 2, 87, 168, 233, 22, 67, 188, 37, 218, 143, 112, 97, 158, 203, 52, 173, 82, 7, 248, 25, 230, 179, 76, 213, 42, 127, 128, 145, 110, 59, 196, 93, 162, 247, 8, 137, 118, 35, 220, 69, 186, 239, 16, 1, 254, 171, 84, 205, 50, 103, 152, 121, 134, 211, 44, 181, 74, 31, 224, 241, 14, 91, 164, 61, 194, 151, 104, 41, 214, 131, 124, 229, 26, 79, 176, 161, 94, 11, 244, 109, 146, 199, 56, 217, 38, 115, 140, 21, 234, 191, 64, 81, 174, 251, 4, 157, 98, 55, 200]


def run():
    print("......................Main Program is Started........... !!!\n")
    # write coode here
    slen = 256
    print("s_box = ",s_box)
    print("inv_s_box  = ",inv_s_box)
    print("\n\n\n\n.........\n\n\n")
    sbox  = random.sample(range(0, slen), slen)
    inv_sbox = random.sample(range(0, slen), slen)
    for i in range(256):
        sbox[i] = sBoxRTL_Forward(i)
        inv_sbox[i] = sBoxRTL_Backward(i)
    print("s_box = ",sbox)
    print("inv_s_box  = ",inv_sbox)
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