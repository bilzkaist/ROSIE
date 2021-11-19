#*****************************************************************************
#
#                            Dual Quad Bit SBox Code.
#                             Written  by Bilal Dastagir.
#                                Oct, 13th, 2021
#
#******************************************************************************



#Global Variables 
BETA = [0]
ALPHA = [1]
BRAVO  = [2]
CHARLIE = [3]

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

def binary2int(binary): 
    # Program Started 
    int_val, i, n = 0, 0, 0
    while(binary != 0): 
        a = binary % 10
        int_val = int_val + a * pow(2, i) 
        binary = binary//10
        i += 1
    print(int_val) 
    # Program Ended



# y3 = (a & b & ~d) | (a & ~b & d) | ( ~a & b && c) | ( ~a & ~b & ~c)


def y3(a,b,c,d):
    # Program Started 
    return 0#(a & b & ~d) | (a & ~b & d) | ( ~a & b && c) | ( ~a & ~b & ~c)
    # Program Ended

def getBit(byteIn, pos):
    # Program Started 
    #return (bin(byteIn >> pos & 0b1))
    return byteIn >> pos & 1
    # Program Ended 

def subByte(byteIn):
    # Program Started
    byte4H, byte4L = byteSplit(byteIn)
    binary4H = bin(byte4H)
    print(binary4H)
    binary4L = bin(byte4L)
    print(binary4L)
    q = 10
    pos = 3
    print ("q = ",q,"(",bin(q),")")
    q = q<<1 
    print ("q = ",q,"(",bin(q),")")
    print("q bit at ",pos," = ",getBit(q,pos))
    i = 0 
    for i in range(16):
        print("y3 (",bin(i),") = ",y3(getBit(i,3),getBit(i,2),getBit(i,1),getBit(i,0)))
    byteOut = byteIn  
    return byteOut
    # Program Ended 

def sboxDQ():
    # Program Started 
    sbox1F = [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
    sbox1B = [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
    sbox2F = [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
    sbox2B = [00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00, 00]
    byteIn = 100;
    byteOut = subByte(byteIn)
    print("Byte In =",byteIn," -> sbox -> ",byteOut)
    return sbox1F,sbox1B, sbox2F, sbox2B
    # Program Ended 

def run_beta():
    print("Beta Program is Started........... !!!")
    # Write code Here
    sbox1F,sbox1B, sbox2F, sbox2B = sboxDQ()
    print("Dual Quad SBox Pair : ")
    print("sbox1F = ",sbox1F)
    print("sbox1B = ",sbox1B)
    print("sbox2F = ",sbox2F)
    print("sbox2B = ",sbox2B)
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