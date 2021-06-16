import numpy as np
import cv2
import math
import scipy.ndimage
from scipy.special import iv  # Modified bessel function
from bimvee.importRpgDvsRos import importRpgDvsRos
import oriens_utils as oriens

MAXLEVEL = 10
NUMORI = 8
R0 = 2
WIDTHVM = 13
HEIGHTVM = 13
W = 1
ITERATIONS = 10
ALPHA = 2

def makePyramid(img):

    '''

        It takes a base image and create an image pyramid

    '''
    depth = MAXLEVEL
    pyr = []
    pyr.append(img)

    for level in range(1,depth):

        scale = 1/pow(math.sqrt(2),(level-1))
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)

        pyr.append(cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC))

    return pyr

def makeVonMises(R0,theta0,dim1,dim2):

    '''

        Make VonMises

    '''

    eps = np.finfo(float).eps

    # Make grid
    [X,Y] = np.meshgrid(dim1,dim2)

    # Convert to polar coordinates
    R = np.sqrt(np.power(X,2)+np.power(Y,2))
    theta = np.arctan2(Y,X)

    # Make masks
    msk1 = np.exp(R0*np.cos(theta-(theta0)))/iv(0,R-R0)
    msk1[R==0] = 0
    msk1 = msk1/np.amax(np.amax(msk1))

    msk2 = np.exp(R0*np.cos(theta-(theta0+math.pi)))/iv(0,R-R0)
    msk2[R==0] = 0
    msk2 = msk2/np.amax(np.amax(msk2))

    # Added to avoid overlapping RFs
    msk1_final = msk1*((msk1 - msk2) > eps)
    msk2_final = msk2*((msk2 - msk1) > eps)

    return (msk1_final,msk2_final)

'''def vfcolor(x,y):

    eps = np.finfo(float).eps

    xf = x
    yf = y

    nrow,ncol = np.shape(xf)
    h = 8*(np.arctan2(yf,xf)/(2*math.pi))%8
    h2 = (np.ceil(h)%8)+1
    h1 = (h2-2%8)+1
    hr = np.ceil(h)-h
    s = np.sqrt(np.power(xf,2)+np.power(yf,2))

    s = s/(np.amax(s[:])+eps)

    c = np.reshape(np.matrix('1 0 0; 1 0.5 0; 1 1 0; 0 1 0; 0 0.6 0; 0 1 1; 0 0 1; 1 0 1'),(8,1,3))
    img1 = np.zeros((nrow,ncol,3))
    img1[:] = c[h1,0,:]
    img2 = np.zeros((nrow,ncol,3))
    img2[:] = c[h2,0,:]
    rhr = np.tile(hr,(1,1,3))
    img = rhr*img1+(1-rhr)*img2
    rs = np.tile(s,(1,1,3))
    img = rs*img+(1-rs)

    return img'''

def mergeLevel(pyr):

    '''

        Merges pyramid orientations at each level

    '''

    newPyr = []
    
    for level in range(MAXLEVEL):

        temp = np.zeros((np.shape(pyr[level])[0],np.shape(pyr[level])[1]))

        for ori in range(NUMORI):

            temp += pyr[level][:,:,ori]

        newPyr.append(temp)

    return newPyr

def sumPyr(pyr1,pyr2,type):

    '''

        Sums the levels of two input pyramids

    '''

    pyr = []

    if(type == 'data'):

        for level in range(MAXLEVEL):
            
            temp = pyr1[level]+pyr2[level]
            temp[temp<0] = 0

            pyr.append(temp) 

    elif(type == 'orientation'):

        for level in range(MAXLEVEL):

            temp = np.empty((np.shape(pyr1[level])[0],np.shape(pyr1[level])[1],NUMORI))

            for ori in range(NUMORI):

                temp[:,:,ori] = pyr1[level][:,:,ori]+pyr2[level][:,:,ori]

            pyr.append(temp)

    return pyr

def subPyr(pyr1,pyr2):

    '''

        Subtracts the levels of two input pyramids

    '''

    newPyr1 = []
    newPyr2 = []

    for level in range(MAXLEVEL):

        temp = pyr1[level]-pyr2[level]

        newPyr1.append(pyr1[level]*(temp>0))
        newPyr2.append(pyr2[level]*(temp>0))

    return newPyr1,newPyr2

def vonMisesPyramid(pyr,invmsk1,invmsk2):

    '''

        Convolves input pyramid with von Mises distribution

    '''

    pyr1 = []
    pyr2 = []

    for level in range(MAXLEVEL):

        temp1 = np.empty((np.shape(pyr[level])[0],np.shape(pyr[level])[1],NUMORI))
        temp2 = np.empty((np.shape(pyr[level])[0],np.shape(pyr[level])[1],NUMORI))

        for ori in range(NUMORI):

            temp1[:,:,ori] = scipy.ndimage.correlate(pyr[level], invmsk1[ori], mode='constant')
            temp2[:,:,ori] = scipy.ndimage.correlate(pyr[level], invmsk2[ori], mode='constant')

        pyr1.append(temp1)
        pyr2.append(temp2)

    return (pyr1,pyr2)

def vonMisesSum(pyr,invmsk1,invmsk2):

    '''

        Convolves the center surround maps with the Von Mises masks and summates
        over different pyramid levels

    '''

    vmPyr1,vmPyr2 = vonMisesPyramid(pyr,invmsk1,invmsk2)

    map1 = []
    map2 = []

    for minL in range(MAXLEVEL):

        temp1 = np.zeros((np.shape(pyr[minL])[0],np.shape(pyr[minL])[1],NUMORI))
        temp2 = np.zeros((np.shape(pyr[minL])[0],np.shape(pyr[minL])[1],NUMORI))

        for level in range(minL,MAXLEVEL):

            for ori in range(NUMORI):

                temp1[:,:,ori] +=  np.power((1/2),(level-minL))*cv2.resize(vmPyr1[level][:,:,ori],np.shape(vmPyr1[minL][:,:,ori]),interpolation=cv2.INTER_CUBIC).transpose()
                temp2[:,:,ori] +=  np.power((1/2),(level-minL))*cv2.resize(vmPyr2[level][:,:,ori],np.shape(vmPyr2[minL][:,:,ori]),interpolation=cv2.INTER_CUBIC).transpose()

        map1.append(temp1)
        map2.append(temp2)

    return (map1,map2)

def makeBorderOwnership(edgeMapPyr,oriensMatrixPyr):

    '''

        Creates border ownership cells by using the
        edge and orientation pyramid as input
            (FEEDFORWARD step)

    '''

    bl_1_pyr = []
    bl_2_pyr = []
    bd_1_pyr = []
    bd_2_pyr = []

    s_pyr = []

    for level in range(MAXLEVEL):

        bl_1 = np.empty((np.shape(edgeMapPyr[level])[0],np.shape(edgeMapPyr[level])[1],NUMORI))
        bl_2 = np.empty((np.shape(edgeMapPyr[level])[0],np.shape(edgeMapPyr[level])[1],NUMORI))

        bd_1 = np.empty((np.shape(edgeMapPyr[level])[0],np.shape(edgeMapPyr[level])[1],NUMORI))
        bd_2 = np.empty((np.shape(edgeMapPyr[level])[0],np.shape(edgeMapPyr[level])[1],NUMORI))

        s = np.empty((np.shape(edgeMapPyr[level])[0],np.shape(edgeMapPyr[level])[1],NUMORI+8))

        for ori in range(NUMORI):

            ori_index = np.round(oriensMatrixPyr[level]/(2*math.pi)*16)+4%16

            s[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori))
            s[:,:,ori+7] = np.multiply(edgeMapPyr[level],(ori_index==ori+7))

            # Light object on dark background
            bl_1[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori))
            bl_2[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori+7))

            # Dark object on light background
            bd_1[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori+7))
            bd_2[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori))

        bl_1_pyr.append(bl_1)
        bl_2_pyr.append(bl_2)
        bd_1_pyr.append(bd_1)
        bd_2_pyr.append(bd_2)

        s_pyr.append(s)

    return (bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,s_pyr)

def groupingFeedforward(bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2):

    '''

        Calculates grouping activity from border ownership activity

    '''

    gl_1_pyr = []
    gl_2_pyr = []
    gd_1_pyr = []
    gd_2_pyr = []

    for level in range(MAXLEVEL):

        gl_1 = np.empty((np.shape(bl_1_pyr[level])[0],np.shape(bl_1_pyr[level])[1],NUMORI))
        gl_2 = np.empty((np.shape(bl_2_pyr[level])[0],np.shape(bl_2_pyr[level])[1],NUMORI))

        gd_1 = np.empty((np.shape(bd_1_pyr[level])[0],np.shape(bd_1_pyr[level])[1],NUMORI))
        gd_2 = np.empty((np.shape(bd_2_pyr[level])[0],np.shape(bd_2_pyr[level])[1],NUMORI))

        for ori in range(NUMORI):

            if (np.allclose(bl_1_pyr[level][:,:,ori],bd_2_pyr[level][:,:,ori])):
                gl_1[:,:,ori] = scipy.ndimage.correlate(bl_1_pyr[level][:,:,ori], invmsk1[ori], mode='constant')
                gl_2[:,:,ori] = scipy.ndimage.correlate(bl_2_pyr[level][:,:,ori], invmsk2[ori], mode='constant')
                gd_1[:,:,ori] = scipy.ndimage.correlate(bd_1_pyr[level][:,:,ori], invmsk1[ori], mode='constant')
                gd_2[:,:,ori] = scipy.ndimage.correlate(bd_2_pyr[level][:,:,ori], invmsk2[ori], mode='constant')
            else:
                gl_1[:,:,ori] = scipy.ndimage.correlate(bl_1_pyr[level][:,:,ori]-(W*bd_2_pyr[level][:,:,ori]), invmsk1[ori], mode='constant')
                gl_2[:,:,ori] = scipy.ndimage.correlate(bl_2_pyr[level][:,:,ori]-(W*bd_1_pyr[level][:,:,ori]), invmsk2[ori], mode='constant')
                gd_1[:,:,ori] = scipy.ndimage.correlate(bd_1_pyr[level][:,:,ori]-(W*bl_2_pyr[level][:,:,ori]), invmsk1[ori], mode='constant')
                gd_2[:,:,ori] = scipy.ndimage.correlate(bd_2_pyr[level][:,:,ori]-(W*bl_1_pyr[level][:,:,ori]), invmsk2[ori], mode='constant')

        gl_1_pyr.append(gl_1)
        gl_2_pyr.append(gl_2)
        gd_1_pyr.append(gd_1)
        gd_2_pyr.append(gd_2)

    return (gl_1_pyr,gl_2_pyr,gd_1_pyr,gd_2_pyr)

def groupingFeedback(gl_pyr,gd_pyr,bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2,s_pyr):

    '''

        Computes feedback from grouping pyramids to border-ownership cells

    '''

    # Light on dark
    vmL1,vmL2 = vonMisesSum(gl_pyr,invmsk1,invmsk2)

    # Dark on light
    vmD1,vmD2 = vonMisesSum(gd_pyr,invmsk1,invmsk2)

    bl_1_pyr_new = []
    bl_2_pyr_new = []
    bd_1_pyr_new = []
    bd_2_pyr_new = []

    for level in range(MAXLEVEL):

        bl_1 = np.empty((np.shape(bl_1_pyr[level])[0],np.shape(bl_1_pyr[level])[1],NUMORI))
        bl_2 = np.empty((np.shape(bl_2_pyr[level])[0],np.shape(bl_2_pyr[level])[1],NUMORI))

        bd_1 = np.empty((np.shape(bd_1_pyr[level])[0],np.shape(bd_1_pyr[level])[1],NUMORI))
        bd_2 = np.empty((np.shape(bd_2_pyr[level])[0],np.shape(bd_2_pyr[level])[1],NUMORI))

        for ori in range(NUMORI):

            temp = s_pyr[level][:,:,ori] * (1+ALPHA*(1/(1+np.exp(-((1/1)*(vmL1[level][:,:,ori]-vmD2[level][:,:,ori]))))-0.5))
            temp[np.logical_or(bl_1_pyr[level][:,:,ori]<0,np.isnan(bl_1_pyr[level][:,:,ori]))]
            bl_1[:,:,ori] = temp

            temp = s_pyr[level][:,:,ori+7] * (1+ALPHA*(1/(1+np.exp(-((1/1)*(vmL2[level][:,:,ori]-vmD1[level][:,:,ori]))))-0.5))
            temp[np.logical_or(bl_2_pyr[level][:,:,ori]<0,np.isnan(bl_2_pyr[level][:,:,ori]))]
            bl_2[:,:,ori] = temp

            temp = s_pyr[level][:,:,ori+7] * (1+ALPHA*(1/(1+np.exp(-((1/1)*(vmD1[level][:,:,ori]-vmL2[level][:,:,ori]))))-0.5))
            temp[np.logical_or(bd_1_pyr[level][:,:,ori]<0,np.isnan(bd_1_pyr[level][:,:,ori]))]
            bd_1[:,:,ori] = temp

            temp= s_pyr[level][:,:,ori] * (1+ALPHA*(1/(1+np.exp(-((1/1)*(vmD2[level][:,:,ori]-vmL1[level][:,:,ori]))))-0.5))
            temp[np.logical_or(bd_2_pyr[level][:,:,ori]<0,np.isnan(bd_2_pyr[level][:,:,ori]))]
            bd_2[:,:,ori] = temp

        bl_1_pyr_new.append(bl_1)
        bl_2_pyr_new.append(bl_2)
        bd_1_pyr_new.append(bd_1)
        bd_2_pyr_new.append(bd_2)

    return (bl_1_pyr_new,bl_2_pyr_new,bd_1_pyr_new,bd_2_pyr_new)

def makeGrouping(bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2,s_pyr):

    '''

        Perform feedforward and feedback loop (iterative algorithm)

    '''

    gl_1_pyr,gl_2_pyr,gd_1_pyr,gd_2_pyr = groupingFeedforward(bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2)

    # Light on dark
    gl_1 = mergeLevel(gl_1_pyr)
    gl_2 = mergeLevel(gl_2_pyr)

    gl_pyr = sumPyr(gl_1,gl_2,'data')

    # Dark on light
    gd_1 = mergeLevel(gd_1_pyr)
    gd_2 = mergeLevel(gd_2_pyr)

    gd_pyr = sumPyr(gd_1,gd_2,'data')

    # Subtract these two g_maps from each other (inhibition between different polarity g-cells)
    gl_pyr,gd_pyr = subPyr(gl_pyr,gd_pyr)

    bl_1_pyr_new,bl_2_pyr_new,bd_1_pyr_new,bd_2_pyr_new = groupingFeedback(gl_pyr,gd_pyr,bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2,s_pyr)

    return(gl_pyr,gd_pyr,bl_1_pyr_new,bl_2_pyr_new,bd_1_pyr_new,bd_2_pyr_new)

if __name__ == '__main__':

    ori = np.array([0,22.5,45,67.5]) # 8 orientations
    oris = np.deg2rad(np.concatenate((ori,ori+90),0))

    # Load the events fram from the bag
    filePathOrName = './out015.bag'

    template = {
        'events': {
                'dvs': '/cam0/events'
                }
    }

    container = importRpgDvsRos(filePathOrName=filePathOrName, template=template)

    events = container["data"]["events"]["dvs"]
    x = events["x"]
    y = events["y"]
    pol = events["pol"]
    dimX = events["dimX"]
    dimY = events["dimY"]

    image = np.zeros((dimY,dimX))

    image[y,x] = pol

    # Create the multivariate DoG
    mu = np.array([0,0])
    sigma1 = np.array([[0.8,0],[0,0.4]])
    sigma2 = np.array([[0.8,0],[0,0.3]])

    orienslist = list(np.arange(0, 337.5+1,22.5))

    G = oriens.mvDog(mu,mu,sigma1,sigma2)
    response = oriens.getOriensResp(image,G,orienslist)

    # Compute the orientation matrix with the edge maps at different orientations
    corfresponse, oriensMatrix = oriens.calc_viewimage(response,list(range(1,len(orienslist)+1)), np.multiply(orienslist,math.pi/180))

    # Create images pyramids
    edgeMapPyr = makePyramid(image)
    oriensMatrixPyr = makePyramid(oriensMatrix)


    # Create VonMises
    dim1 = np.arange(-3*R0,3*R0)
    dim2 = dim1

    invmsk1 = []
    invmsk2 = []

    for ori in range(NUMORI):

        temp1,temp2 = makeVonMises(R0,oris[ori]+math.pi/2,dim1,dim2)

        invmsk1.append(temp1)
        invmsk2.append(temp2)

    # Generate border ownership(feedforward step)
    bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,s_pyr = makeBorderOwnership(edgeMapPyr,oriensMatrixPyr)

    # Iterative algorithm
    for i in range(ITERATIONS):

        print('Iteration ' +str(i)+ '\n')

        gl_pyr,gd_pyr,bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr = makeGrouping(bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2,s_pyr)

    # Combine border-ownership maps
    bTotal1 = sumPyr(bl_1_pyr,bd_1_pyr,'orientation')
    bTotal2 = sumPyr(bl_2_pyr,bd_2_pyr,'orientation')

    # Combine grouping maps
    gTotal = sumPyr(gl_pyr,gd_pyr,'data')

    # Combine the maps
    groupData = np.zeros((np.shape(image)[0],np.shape(image)[1]))

    b1_pyr = []
    b2_pyr = []

    for level in range(MAXLEVEL):

        groupData += cv2.resize(gTotal[level], np.shape(image)).transpose()

        temp1 = np.empty((np.shape(bl_1_pyr[level])[0],np.shape(bl_1_pyr[level])[1],NUMORI))
        temp2 = np.empty((np.shape(bl_1_pyr[level])[0],np.shape(bl_1_pyr[level])[1],NUMORI))

        for ori in range(NUMORI):

            temp1[:,:,ori] = bTotal1[level][:,:,ori]
            temp2[:,:,ori] = bTotal2[level][:,:,ori]

        b1_pyr.append(temp1)
        b2_pyr.append(temp2)

    # Visualize border ownership at the highest resolution
    X = np.zeros((np.shape(b1_pyr[0])[0],np.shape(b1_pyr[0])[1]))
    Y = np.zeros((np.shape(b1_pyr[0])[0],np.shape(b1_pyr[0])[1]))

    for ori in range(NUMORI):

        # BOS points towards outside of circle
        X += np.cos(oris[ori]-math.pi/2)*(b2_pyr[0][:,:,ori]-b1_pyr[0][:,:,ori])
        Y += np.sin(oris[ori]-math.pi/2)*(b2_pyr[0][:,:,ori]-b1_pyr[0][:,:,ori])    

    # Figure-ground segmentation
    occ_map = np.arctan2(Y,X)

    #occ_map = vfcolor(X,Y)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", occ_map)
    cv2.waitKey(0)

    # Grouping
    #group_map = cv2.normalize(groupData)
    #group_map = group_map*(group_map>0.1)