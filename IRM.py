import numpy as np 
from PIL import Image
import PIL
import matplotlib.pyplot as plt 
import os  
import colorsys 
from imshowpair import imshowpair

'''
FILE HEADER
- Signature        1 octet
- Total File size  4 octets
- data offset      2 octets

IMAGE HEADER
- Height           4 octets
- Width            4 octets
- Nbr des plans    4 bits
- Compression type 4 bits
- 

DATA : 
DATA HEADER :  (depends on compression type)
- maxbit           2 octets
- Len Huffman_dict 2 octets
CODE  :  (Huffman : dictionnaire + code) 
'''

def taux(irm_path,imagepath):
    ''' Parameters : irm_path : String  ( le chemin où se trouve le fichier irm )
                     imagepath : String ( le chemin où se trouve l'image à compresser ) 
        retour :    Poucentage ( Taux de compression )
    ''' 
    im=PIL.Image.open(imagepath)
    
    if(im.mode=='L'):n=1
    else:n=3
    
    im=np.array(im)
    irmsize=os.path.getsize(irm_path)
    
    return round(100-((irmsize/(im.shape[0]*im.shape[1]*n)))*100,2) 

def MSE(irm_path,imagepath):
    ''' Parameters : irm_path : String  ( le chemin où se trouve le fichier irm )
                     imagepath : String ( le chemin où se trouve l'image à compresser ) 
        retour :     Float ( l'Mse )
    ''' 
    im1=PIL.Image.open(imagepath)
    
    if(im1.mode=='L'):n=1
    else:n=3
        
    im1=np.array(im1)
    
    im2=irmDECO(irm_path)
    
    return round(np.mean(np.power(im1-im2,2))/n,2)

def entropie(imagepath):
    im=PIL.Image.open(imagepath)
    
    if(im.mode=='L'):n=1
    else:n=3
    im=np.array(im)
    occ1=occ_img(im) 
    occ=[j/(im.shape[0]*im.shape[1]*n) for i , j in occ1]
    
    I=np.array(occ)
    
    ent=np.sum(I*np.log2(I+1e-10))
    return round(-ent,2)  

def long_moy(imagepath):
    im1=PIL.Image.open(imagepath)
    
    if(im1.mode=='L'):n=1
    else:n=3
        
    im=np.array(im1)
    
    occ1=occ_img(im) 
    arbre=huffman_arbre(occ1)
    c="";dictc={} 
    for i in arbre:
        decom(i,c,dictc)
        c="" 
    long=0
    occ=[(i,j/(im.shape[0]*im.shape[1]*n)) for i , j in occ1]
    for i in range(len(occ)): 
        long+=occ[i][1]*len(dictc[occ1[i][0]]) 
    return long

def efficacite(imagepath): 
    im=PIL.Image.open(imagepath)
    if(im.mode=='L'):n=1
    else:n=3
    im=np.array(im)
    
    return round(entropie(imagepath)/long_moy(imagepath),3)*100

def rendement(imagepath):
    
    im=PIL.Image.open(imagepath)
    if(im.mode=='L'):n=1
    else:n=3
    im=np.array(im)
    occ1=occ_img(im) 
    arbre=huffman_arbre(occ1)
    c="";dictc={} 
    for i in arbre:
        decom(i,c,dictc)
        c="" 
        
    return round((entropie(imagepath)/(long_moy(imagepath)*np.log2(len(dictc)))),3)

def redondance(imagepath):
    return 1-rendement(imagepath) 

def irmCode(image_name,ctype):
    ''' Parameters : image_name : String ( le chemin où se trouve l'image à compresser )
                     ctype : Int (Compression type){0 (lossless) , 1 (quant), 2 (hsl) , 3 (quant+hsl)}
        return :     Ecrire la sequence compressé dans un fichier .irm
    ''' 
    image=Image.open(image_name)
    image=np.array(image)
    shape=image.shape
    head=format(232,"08b")     #signature irm
    data_offset=40+16+32+32+4+4+16+16 
    head+=format(data_offset,"016b")
    file=""
    height=shape[0]    
    width=shape[1]      
    
    if(len(shape)==3):         #if len shape = 3  then kayn 3 plans or 4  
        nb_plan=shape[2]  
    elif(len(shape)==2): nb_plan=1     #if not then its gray

    #add width , height , nb plans and compression type 
    head+=format(height,"032b")+format(width,"032b")+format(nb_plan,"04b")+format(ctype,"04b")
    
    
    if ctype==0:    #huffman
        huffman_code=Huffman_codage(image)
        code=huffman_code[0]
        huff_entet=huffman_code[1]
    elif ctype==1:
        huffman_code=Huffman_codage(quant(image,8))
        code=huffman_code[0]
        huff_entet=huffman_code[1]
    elif ctype==2:
        huffman_code=Huffman_codage(rgb2hls(image))
        code=huffman_code[0]
        huff_entet=huffman_code[1]
    elif ctype==3:
        huffman_code=Huffman_codage(quant(np.array(rgb2hls(image)*255,int),8))
        code=huffman_code[0]
        huff_entet=huffman_code[1]
    
    head+=format(huff_entet[3],"016b")+format(huff_entet[4],"016b")
    
    Total_file_size=160+len(code)
    
    head=head[:8]+format(Total_file_size,"032b")+head[8:]
    
    if(ctype==0) :  
        new_path = os.path.splitext(image_name)[0]+"_lossless.irm" 
    else: new_path = os.path.splitext(image_name)[0]+"_lossy.irm" 
    
    
    #head = [signature(1),Total_file_size(4),data_offset(2),height(4),width(4),nb_plan+ctype(1),]
    file=head+code
    
    return write_bin(file,new_path)
    
def irmDECO(filename):
    ''' Parameters : filename : String ( le chemin où se trouve le fichier irm )

        return :     Array : image irm decompressé
    '''    
    filebin=read_bin(filename)
    
    signature=int(filebin[:8],2)
    filesize=int(filebin[8:40],2)      #Total File size
    data_offset=int(filebin[40:56],2)  #Data offset  
    height=int(filebin[56:88],2)       #Height
    width=int(filebin[88:120],2)       #Width
    plans=int(filebin[120:124],2)      #nbr plans
    ctype=int(filebin[124:128],2)      #Compression Type
    max_bit=int(filebin[128:144],2)
    dict_length=int(filebin[144:160],2)
    
    entet=[height,width,plans,max_bit,dict_length]
    
    data=filebin[data_offset:]         #data header + compressed data 
    
    if ctype==0 :
        imdeco=Huffman_decodage(entet,data)
    elif ctype==1:                      #huffman direct on image lossless
        imdeco=Huffman_decodage(entet,data)
    elif ctype==2:
        imdeco=hls2rgb(Huffman_decodage(entet,data)/255)
    elif ctype==3:
        imdeco=hls2rgb(Huffman_decodage(entet,data)/255)
    return imdeco

def rgb2hls(image):
    ''' Parameters : image : array ( image rgb )

        return :     Array hls
    '''   
    hls_array = np.zeros((image.shape[0], image.shape[1], 3),dtype=float)
    for i in range(0,image.shape[0]) :
          for j in range(0,image.shape[1]):
            rgb = image[i,j,:]
            hls = colorsys.rgb_to_hls(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hls_array[i,j, 0] = hls[0] 
            hls_array[i,j, 1] = hls[1] 
            hls_array[i,j, 2] = hls[2] 
    return hls_array

def hls2rgb(hls_array):
    ''' Parameters : hls_array : array ( image hls )

        return :     Array rgb
    ''' 
    new_image = np.zeros((hls_array.shape[0], hls_array.shape[1],3),int)
    for i in range(0,new_image.shape[0]):
        for j in range(0,new_image.shape[1]):
            rgb = colorsys.hls_to_rgb(hls_array[i,j, 0],
                                      hls_array[i,j, 1],
                                      hls_array[i,j, 2])

            rgb = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

            new_image[i,j,:]=rgb

    return new_image

def occ_img(img):
    unique, counts = np.unique(img, return_counts=True)
    return list(zip(unique, counts))
def huffman_arbre(t):
    while(len(t)!=1):
        t=sorted(t,key=lambda x:x[1])
        new=[( ('0', t[0][0]),('1', t[1][0]) ) , t[0][1]+t[1][1]]
        t=t[2:]
        t.append(new)
    return t[0][0]
def decom(tup,c,l):
    c+=tup[0]
    if type(tup[1]) is tuple:
        for i in tup[1]:
            decom(i,c,l)
    else:
        l[tup[1]]=c
    
   
    
def Huffman_codage(img,quant=1):
    ''' Parameters : img : array

        return :     Tuple : ( sequence binaire ,[img width,img height,nbr de plans,max longueurs des codes, longueur de dictionnare]) 
    '''
    image=img.copy()
    if np.max(image)<=1: #Si l'image est normalisé
        image*=255
    nl,nc=image.shape[0],image.shape[1]
    
    image=np.array(image,int)
    arbre=huffman_arbre(occ_img(image))#création de l'arbre
    
    #ces deux variables pour prendre les signification de chaque élement
    c="";l={}
    for i in arbre:
        decom(i,c,l)
        c=""
    
    #cette étape pour prendre la longueur des codes pour lui coder dans le dictionaire
    len_code=[len(l[i]) for i in l.keys()]
    max_len_code=max(len_code)
    
    bit = "0"+str(len(format(255//quant,"0b")))+"b"
    
    #Concatenate le dectionnaire
    dictionnary=""
    for i in l.keys():
        s=format(i,bit)
        dictionnary+=s+format(len(l[i]),"0"+str(max_len_code)+"b")+l[i]
    
    #la longueur de dictionnaire dans l'en-tête
    entete=""
    dict_length=format(len(dictionnary),"0b")
    entete=format(len(dict_length),"08b")+dict_length #the length of the dictionnary
    
    #concatenation de la longueur et largeur dans l'en-tête
    nl,nc=format(nl,"0b"),format(nc,"0b")
    entete=format(max_len_code,"08b")+format(len(nl),"08b")+nl+format(len(nc),"08b")+nc+entete
    
    #la données de l'image
    donne=""
    nl,nc=image.shape[0],image.shape[1]
    
    if len(image.shape)==3:
        pl=3
        count=0
        while(count<3):
            for i in range(nl):
                for j in range(nc):
                    donne+=l[image[i,j,count]]
            count+=1
    else:
        pl=1
        for i in range(nl):
            for j in range(nc):
                donne+=l[image[i,j]]
    
    return dictionnary+donne,[nl,nc,pl,max_len_code,int(dict_length,2)]

def Huffman_decodage(entete,code,quant=1): #entete = [img width,img height,nbr de plans,max longueurs des codes, longueur de dictionnare]
    ''' Parameters : entete : list ([img width,img height,nbr de plans,max longueurs des codes, longueur de dictionnare])
                      code : String ( sequence binaire )  
        return :     Array : image decodée 
    '''
    
    nl,nc,plans,len_max,length=entete
    

    image=np.zeros((nl,nc),int)
    imageFinale=np.zeros((nl,nc,3),int)
    
    dic=code[:length]
    chaine=code[length:]
    
    bit=len(format(255//quant,"0b"))
    dic1={} #here what gonna concatenate the dectionary
    while dic!="":
        car=int(dic[:bit],2)
        dic=dic[bit:]
        lon=int(dic[:len_max],2)
        dic=dic[len_max:]
        dic1[dic[:lon]]=car
        dic=dic[lon:]
    c="";k="" 
    i,j=0,0
    imgIndex=0
    for x in range(len(chaine)): 
        c+=chaine[x]
        if c in dic1.keys():
            image[i,j]=dic1[c]
            c="";j+=1
            if j==nc:
                i+=1;j=0
                if i==nl:
                    if plans==3:
                        i,j=0,0
                        imageFinale[:,:,imgIndex]=image
                        imgIndex+=1
                    else:
                        return image 
                
    return imageFinale

def quant(img,n):
    return (img//n)*n

def write_bin(seq,nom): #on passe en argument notre suite binaire et le nom de fichier ou on souhaite ecrire
    ''' Parameters : seq : String ( suite binaire )
                     nom : String ( le nom de fichier )  
        return :     Ecrit la sequence dans le fichier 
    '''
    f=open(nom,"wb")
    for i in range(0,len(seq),8):#je parcours par un pas de 8 pour faire le pacquetage en octet
        bt=int(str(seq[i:i+8]),2) #chaque suite de 8 est convertie en entier corespondant
        byy=bt.to_bytes(1,byteorder="big")#c'est ici où on realise le pacquetage
        f.write(byy) 
    f.close()
    
def read_bin(file):# on passe le nom de fichier a lire , et le pas de parcours
    ''' Parameters : file : String ( le nom de fichier )  
        return :     String : ( sequence binaire  )
    '''
    f=open(file,"rb")
    r="" #la sequence binaire qu'on va reconstruire
    ll=f.read()
    totale=ll[1:5]
    reste=""
    for i in totale:
        reste+=format(i,"08b")
    len_tot=int(reste,2)%8
    if not len_tot:
        for i in range(len(ll)): 
            r+=format(ll[i],"08b")
    else:
        for i in range(len(ll)-1): 
            r+=format(ll[i],"08b")
        r+=format(ll[-1],"b").zfill(len_tot)
    return r 

def showtwo(im1,im2): 
    plt.figure(figsize =(100,50))
    imshowpair(im1, im2,cmap='gray') 