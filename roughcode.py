     
#    start_time = time.time()
"""     for i in range(len(imArray_In)):
        for j in range(len(imArray_In[0])):
            for k in range(len(imArray_In[0][0])):  """ # 3):
                #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
                #if (i>ilow & i<ihigh & j>jlow & j<jhigh):
                # imArray_Out_AES_C[keylistFX[i]][keylistFY[j]][k] = sub_bytes_AES_Traditional(imArray_In[i][j][k])
                # imArray_Out_ROSIE_C[keylistFX[i]][keylistFY[j]][k] = sub_bytes_ROSIE(imArray_In[i][j][k])
                # imArray_Out_Bahrami_2021_C[keylistFX[i]][keylistFY[j]][k] = sub_bytes_Bahram_2021(imArray_In[i][j][k])
                # imArray_Out_RTL_C[keylistFX[i]][keylistFY[j]][k] = sBoxRTL_Forward(imArray_In[i][j][k])
              
                # imArray_Out_AES_C[i][j][k] = sub_bytes_AES_Traditional(imArray_In[i][j][k])
                # imArray_Out_ROSIE_C[i][j][k] = sub_bytes_ROSIE(imArray_In[i][j][k])
                # imArray_Out_Bahrami_2021_C[i][j][k]= sub_bytes_Bahram_2021(imArray_In[i][j][k])
                # imArray_Out_RTL_C[i][j][k] = sBoxRTL_Forward(imArray_In[i][j][k])
              
                #imArray_Out_RTL_S[i][j][k] = sBoxRTL_Forward(imArray_In[i][j][k])
                #else:
                    #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
                #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 
    #stop_time = time.time()
   # enc_time = stop_time-start_time
    #print("Encryption Time : ",enc_time," seconds")
    #start_time = time.time()
"""     for i in range(len(imArray_In)):
        for j in range(len(imArray_In[0])):
            for k in range(len(imArray_In[0][0])):  """ # 3):
                #imArray_Out_MSCA[i][j][k] = sBoxMSCA_Forward(imArray_In[i][j][k])
                #if (i==0 & j==0):
                #if (i>ilow & i<ihigh & j>jlow & j<jhigh):
                # imArray_Out_AES_D[keylistBX[i]][keylistBY[j]][k] = inv_sub_bytes_AES_Traditional(imArray_Out_AES_C[i][j][k])
                # imArray_Out_ROSIE_D[keylistBX[i]][keylistBY[j]][k] = inv_sub_bytes_ROSIE(imArray_Out_ROSIE_C[i][j][k])
                # imArray_Out_Bahrami_2021_D[keylistBX[i]][keylistBY[j]][k] = inv_sub_bytes_Bahram_2021(imArray_Out_Bahrami_2021_C[i][j][k])
                # imArray_Out_RTL_D[keylistBX[i]][keylistBY[j]][k] = sBoxRTL_Backward(imArray_Out_RTL_C[i][j][k])
                
                # imArray_Out_AES_D[i][j][k]= inv_sub_bytes_AES_Traditional(imArray_Out_AES_C[i][j][k])
                # imArray_Out_ROSIE_D[i][j][k]= inv_sub_bytes_ROSIE(imArray_Out_ROSIE_C[i][j][k])
                # imArray_Out_Bahrami_2021_D[i][j][k]= inv_sub_bytes_Bahram_2021(imArray_Out_Bahrami_2021_C[i][j][k])
                # imArray_Out_RTL_D[i][j][k] = sBoxRTL_Backward(imArray_Out_RTL_C[i][j][k])
               
                
                #imArray_Out_RTL_D[i][j][k] = sBoxRTL_Backward(imArray_Out_RTL_S[i][j][k])
                #else:
                    #imArray_Out_RTL_S[i][j][k].append(imArray_Out_RTL_S[i][j][k]) 
                #imArray_Out_RTL_SP[sBoxRTL_Forward(i)][sBoxRTL_Forward(j)][k] = sBoxRTL_Forward(imArray_In[i][j][k]) 
    #stop_time = time.time()
    #dec_time = stop_time-start_time
    #print("Decryption Time : ",dec_time," seconds")
    # print("Dimension = ",imArray_In.shape)
    # testimage = Image.fromarray(imArray_In)
    # testimage.show()
    