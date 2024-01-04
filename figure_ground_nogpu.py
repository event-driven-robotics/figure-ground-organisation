import numpy as np
import cv2
import math
import scipy.ndimage
from scipy.special import iv  # Modified bessel function
from bimvee.importRpgDvsRos import importRpgDvsRos
from bimvee.importIitYarp import importIitYarp
from bimvee.importProph import importProph
import oriens_utils as oriens
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import imageio as iio

MAXLEVEL = 10  #pyramid level  (originally 10)
NUMORI = 8 #orientation numbers
R0 = 8 #VM size, radius
WIDTHVM = 13 #VM witdh
HEIGHTVM = 13 #VM height (originally 13)
W = 1 #weight of inhibition BO
ITERATIONS = 10 #iterations number (originally 10)
ALPHA = 2 #BO feedback formulation, paramater

def makePyramid(img):

    '''

        It takes a base image and create an image pyramid

    '''
    depth = MAXLEVEL
    pyr = []
    pyr.append(img)

    for level in range(2,depth+1):

        scale = 1/pow(math.sqrt(2),(level-1))
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width+1, height+1)

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
    msk1 = np.exp(R0*np.cos(theta-theta0))/iv(0,R-R0)
    msk1[R==0] = 0
    msk1 = msk1/np.amax(np.amax(msk1))

    msk2 = np.exp(R0*np.cos(theta-(theta0+math.pi)))/iv(0,R-R0)
    msk2[R==0] = 0
    msk2 = msk2/np.amax(np.amax(msk2))

    # Added to avoid overlapping RFs
    msk1_final = msk1*((msk1 - msk2) > eps)
    msk2_final = msk2*((msk2 - msk1) > eps)

    return (msk1_final,msk2_final)


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
        newPyr2.append(pyr2[level]*(temp<0))

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

    for minL in range(1,MAXLEVEL+1):

        temp1 = np.zeros((np.shape(pyr[minL-1])[0],np.shape(pyr[minL-1])[1],NUMORI))
        temp2 = np.zeros((np.shape(pyr[minL-1])[0],np.shape(pyr[minL-1])[1],NUMORI))

        for level in range(minL,MAXLEVEL+1):

            for ori in range(NUMORI):

                if level == minL:

                    temp1[:,:,ori] = np.zeros((np.shape(vmPyr1[minL-1][:,:,ori])))
                    temp2[:,:,ori] = np.zeros((np.shape(vmPyr2[minL-1][:,:,ori])))

                temp1[:,:,ori] += np.power((1/2),(level-minL))*cv2.resize(vmPyr1[level-1][:,:,ori],(np.shape(vmPyr1[minL-1][:,:,ori])[1],np.shape(vmPyr1[minL-1][:,:,ori])[0]),interpolation=cv2.INTER_CUBIC)
                temp2[:,:,ori] += np.power((1/2),(level-minL))*cv2.resize(vmPyr2[level-1][:,:,ori],(np.shape(vmPyr2[minL-1][:,:,ori])[1],np.shape(vmPyr2[minL-1][:,:,ori])[0]),interpolation=cv2.INTER_CUBIC)

        map1.append(temp1)
        map2.append(temp2)

    return (map1,map2)

def makeBorderOwnership(edgeMapPyr,oriensMatrixPyr,orienslist):

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

        s = np.empty((np.shape(edgeMapPyr[level])[0],np.shape(edgeMapPyr[level])[1],2*NUMORI))

        ori_index = (np.round(oriensMatrixPyr[level]/(2*math.pi)*16)+4)%16

        for ori in range(NUMORI):

            s[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori))
            s[:,:,ori+NUMORI] = np.multiply(edgeMapPyr[level],(ori_index==ori+(NUMORI)))

            # Light object on dark background
            bl_1[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori))
            bl_2[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori+(NUMORI)))

            # Dark object on light background
            bd_1[:,:,ori] = np.multiply(edgeMapPyr[level],(ori_index==ori+(NUMORI)))
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
                gl_1[:,:,ori] = scipy.ndimage.correlate(bl_1_pyr[level][:,:,ori], invmsk2[ori], mode='constant')
                gl_2[:,:,ori] = scipy.ndimage.correlate(bl_2_pyr[level][:,:,ori], invmsk1[ori], mode='constant')
                gd_1[:,:,ori] = scipy.ndimage.correlate(bd_1_pyr[level][:,:,ori], invmsk2[ori], mode='constant')
                gd_2[:,:,ori] = scipy.ndimage.correlate(bd_2_pyr[level][:,:,ori], invmsk1[ori], mode='constant')
            else:
                gl_1[:,:,ori] = scipy.ndimage.correlate(bl_1_pyr[level][:,:,ori]-(W*bd_2_pyr[level][:,:,ori]), invmsk2[ori], mode='constant')
                gl_2[:,:,ori] = scipy.ndimage.correlate(bl_2_pyr[level][:,:,ori]-(W*bd_1_pyr[level][:,:,ori]), invmsk1[ori], mode='constant')
                gd_1[:,:,ori] = scipy.ndimage.correlate(bd_1_pyr[level][:,:,ori]-(W*bl_2_pyr[level][:,:,ori]), invmsk2[ori], mode='constant')
                gd_2[:,:,ori] = scipy.ndimage.correlate(bd_2_pyr[level][:,:,ori]-(W*bl_1_pyr[level][:,:,ori]), invmsk1[ori], mode='constant')

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

            temp = s_pyr[level][:,:,ori] * (1+ALPHA*((1/(1+np.exp(-(vmL1[level][:,:,ori]-vmD2[level][:,:,ori]),dtype=np.float128)))-0.5))
            temp[np.logical_or(temp<0,np.isnan(temp))] = 0
            bl_1[:,:,ori] = temp

            temp = s_pyr[level][:,:,ori+NUMORI] * (1+ALPHA*((1/(1+np.exp(-(vmL2[level][:,:,ori]-vmD1[level][:,:,ori]),dtype=np.float128)))-0.5))
            temp[np.logical_or(temp<0,np.isnan(temp))] = 0
            bl_2[:,:,ori] = temp

            temp = s_pyr[level][:,:,ori+NUMORI] * (1+ALPHA*((1/(1+np.exp(-(vmD1[level][:,:,ori]-vmL2[level][:,:,ori]),dtype=np.float128)))-0.5))
            temp[np.logical_or(temp<0,np.isnan(temp))] = 0
            bd_1[:,:,ori] = temp

            temp= s_pyr[level][:,:,ori] * (1+ALPHA*((1/(1+np.exp(-(vmD2[level][:,:,ori]-vmL1[level][:,:,ori]),dtype=np.float128)))-0.5))
            temp[np.logical_or(temp<0,np.isnan(temp))] = 0
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

    # Read image name from command line
    '''path = str(sys.argv[1]).split("/")
    image_name = path[2]
    # SUFFIX = "_"+image_name'''

    RealWorldData = True
    DataLogFLAG = False
    RosBagFLAG = False
    RawDataFLAG = False
    SaveFilesFLAG = True


    image_name = 'test'
    SUFFIX = "_" + image_name

    ori = np.array([0,22.5,45,67.5]) # 8 orientations
    oris = np.deg2rad(np.concatenate((ori,ori+90),0))

    filpFLAG = 0;
    # name_list=['heart', 'footprint','cat']
    # name_list=['cilinder_cup_bottle']# 'tv', 'square_sasso', 'cilinder_cup_bottle', 'key_mouse_flip','calib_circles']

    # filePathOrName = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/FBEDFGBerkley/shaked_imgs/'
    # name_list=os.listdir(filePathOrName)

    # name_list = ['12003', '12074', '22090', '24063', '28075', '35008', '35058', '35070', '41004', '105053', '112082', '113016', '156079','159091','368016', '43070', '113016', '156079']

    name_list = ['realworld']

    for name in name_list:
        # Read events

        # save_path = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/resultspattericubreal/results_r08/'+name+'/'
        # os.mkdir('/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/resultspattericubreal/results_r08/'+name)

        # save_path = '/home/giuliadangelo/figure-ground-organisation/Berkleyresults/data/'+name+'/'
        # os.mkdir(save_path)

        save_path = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/real-world-data/results/'


        if DataLogFLAG:
            if name.__eq__('cilinder_cup_bottle') | name.__eq__('key_mouse_flip'):
                filePathOrName = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/icub-real/' +name+ '/data/'
                codecbit='24bit'
                camera_events = 'left'
            elif name.__eq__('calib_circles'):
                filePathOrName = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/icub-real/' +name+ '/data/'
                codecbit = '20bit'
                camera_events = 'right'
            else:
                filePathOrName = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/patterns/'+name+'/data/'
                codecbit = '24bit'
                camera_events = 'left'

            events = importIitYarp(filePathOrName=filePathOrName, codec = codecbit)
            #events = importIitYarp(filePathOrName='/Tesi/Datasets/icub_datasets/square/data')



            xs = events['data'][camera_events]['dvs']['x']
            ys = events['data'][camera_events]['dvs']['y']
            ts = events['data'][camera_events]['dvs']['ts']
            pol = events['data'][camera_events]['dvs']['pol']

            # Create events frame
            tw_seconds = 0.18 #how many seconds you want to accumulate events (0.15) new research (0.1)
            period = tw_seconds
            time2accumulate = 4.42 #where to start the accumulation (decided with mustard) (1.50)new research(3.90)
            time_istant = np.where(ts >= time2accumulate)[0][0]
            start_time = ts[time_istant]
            start_time_tw = ts[time_istant]
            i = time_istant
            frame_list = []
            frame = np.full((241, 305), 0.5)
            # frame = np.full((341, 481), 0.5)


        # Load the events fram from the bag
        if RosBagFLAG:
            filePathOrName = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/FBEDFGBerkley/shaked_imgs/'+name+'/out010.bag'

            template = {
                'events': {
                        'dvs': '/cam0/events'
                        }
            }

            container = importRpgDvsRos(filePathOrName=filePathOrName, template=template, log='info')

            events = container["data"]["events"]["dvs"]
            xs = events["x"]
            ys = events["y"]
            pol = events["pol"]
            ts = events["ts"]

            frame_list = []
            frame = np.full((events["dimY"],events["dimX"]), 0.5)
            frame[ys, xs] = pol
            frame_list.append(frame.copy())

            # Create events frame
            tw_seconds = 0.01  # how many seconds you want to accumulate events
            time2accumulate = 0.00 # where to start the accumulation (decided with mustard)
            time_istant = np.where(ts >= time2accumulate)[0][0]
            start_time = ts[time_istant]
            start_time_tw = ts[time_istant]
            i = time_istant
            frame_list = []

            img_path = '/home/giuliadangelo/workspace/data/DATASETs/figure-ground-segmentation/paper/FBEDFGBerkley/shaked_imgs/' + name + '/0.jpg'
            im = cv2.imread(img_path)
            [rows,cols,ch]=im.shape
            frame = np.full((rows, cols), 0.5)

        if RawDataFLAG:
            events = importProph(filePathOrName='/home/giuliadangelo/figure-ground-organisation/clutter_dynamic_cappelloDATASET/dataset/clutter.raw')

            camera_events = 'ch0'

            xs = events['data'][camera_events]['dvs']['x']
            ys = events['data'][camera_events]['dvs']['y']
            ts = events['data'][camera_events]['dvs']['ts']
            pol = events['data'][camera_events]['dvs']['pol']

            # Create events frame
            tw_seconds = 0.15  # how many seconds you want to accumulate events
            period = tw_seconds
            time2accumulate = 0  # where to start the accumulation (decided with mustard)
            time_istant = np.where(ts >= time2accumulate)[0][0]
            start_time = ts[time_istant]
            start_time_tw = ts[time_istant]
            i = time_istant
            frame_list = []

            frame = np.full((480, 640), 0.5)

        if RealWorldData:
            filePathOrName = '/home/giuliadangelo/figure-ground-organisation/real-world-data/realworld/data/'
            codecbit = '24bit'
            camera_events = 'left'
            # events = importIitYarp(filePathOrName=filePathOrName, codec=codecbit)

            with open('realworlddata.pkl', 'rb') as f:
                events = pickle.load(f)

            xs = events['data'][camera_events]['dvs']['x']
            ys = events['data'][camera_events]['dvs']['y']
            ts = events['data'][camera_events]['dvs']['ts']
            pol = events['data'][camera_events]['dvs']['pol']

            # Create events frame
            tw_seconds = 0.10 #how many seconds you want to accumulate events (0.15) new research (0.1)
            period = tw_seconds
            max_time = max(ts) #### HERE
            print(max_time)
            curr_ts = 0
            frame_list = []
            # frame = np.full((241, 305), 0.5)
            frame = np.full((480, 640), 0.5)

        #events frame accumulating event for a specific period, only for data.log or ros.bag if you don't want a frame with all the events of the dataset
        while True:
            current_time = ts[curr_ts]
            frame[ys[curr_ts], xs[curr_ts]] = pol[curr_ts]

            if current_time >= tw_seconds:
                frame_list.append(frame.copy())
                plt.imshow(frame)
                plt.draw()
                plt.pause(0.2)
                print(current_time)
                frame = np.full((480, 640), 0.5)
                tw_seconds = tw_seconds + period
            if current_time >= max_time:
                break
            curr_ts +=1

        orienslist = list(np.arange(0, 337.5 + 1, 22.5))

        # Create the multivariate DoG
        mu = np.array([0, 0])
        sigma1 = np.array([[0.8, 0], [0, 0.4]])
        sigma2 = np.array([[0.8, 0], [0, 0.3]])
        G = oriens.mvDog(mu, mu, sigma1, sigma2)

        i=0
        image_sum = 0
        response_list = []
        all_respone = []

        for i in range(0, len(frame_list)):
            print(str(len(frame_list)))
            saving_path = save_path+str(i)+'/'
            os.mkdir(saving_path)
            image = frame_list[i]

            # Split events frame into pos e neg
            img_pos = np.zeros(np.shape(image))
            img_neg = np.zeros(np.shape(image))
            img_hndl = np.zeros(np.shape(image))

            if filpFLAG:
                image = cv2.flip(image, -1)
            img_pos[image > 0.7] = 1
            img_neg[image < 0.3] = 1

            #g ahead and make results
            if SaveFilesFLAG:
                np.savetxt(saving_path+'img_pos' + SUFFIX + '.csv', img_pos)
                np.savetxt(saving_path+'img_neg' + SUFFIX + '.csv', img_neg)

            image[image == 0] = 1
            image[image == 0.5] = 0

            image_sum += image

            edgeMapPyr_pos = makePyramid(img_pos)
            edgeMapPyr_neg = makePyramid(img_neg)

            # Compute edge maps at different orientations
            for j in range(MAXLEVEL):

                response_pos = oriens.getOriensResp(edgeMapPyr_pos[j], G, orienslist[:NUMORI], 0.3)  # 0.15
                response_neg = oriens.getOriensResp(edgeMapPyr_neg[j], G, orienslist[NUMORI:], 0.3)  # 0.15

                if (SaveFilesFLAG and (j==0)):# and (i==0)):
                    for t in range(np.shape(response_pos)[2]):
                        np.savetxt(saving_path+'resp'+str(orienslist[t]) + SUFFIX + '.csv', response_pos[:,:,t])

                    for t in range(np.shape(response_neg)[2]):
                        np.savetxt(saving_path+'resp'+str(orienslist[NUMORI+t]) + SUFFIX + '.csv', response_neg[:,:,t])

                response = np.concatenate((response_pos, response_neg), axis=2)

                response_list.append(response)

            all_respone.append(response_list)
            response_list = []

            #it was a teab back
            response_pyr = []
            response = []
            # Sum the edge map at the same orientation together
            for i in range(MAXLEVEL):
                for k in range(len(orienslist)):
                    sum_ori = 0
                    for j in range(len(all_respone)):

                        sum_ori += all_respone[j][i][:,:,k]
                        sum_ori[sum_ori<0.2] = 0

                    response.append(sum_ori)

                response_pyr.append(response)
                response = []

            # Compute orientations matrix
            oriensMatrixPyr = []
            for i in range(MAXLEVEL):
                # Compute the orientation matrix with the edge maps at different orientations
                corfresponse, oriensMatrix = oriens.calc_viewimage(response_pyr[i],list(range(1,len(orienslist)+1)), np.multiply(orienslist,math.pi/180))

                oriensMatrixPyr.append(oriensMatrix)

            if SaveFilesFLAG:
                np.savetxt(saving_path+'oriens'+SUFFIX+'.csv',oriensMatrixPyr[0])
                np.savetxt(saving_path+'frame'+SUFFIX+'.csv', image_sum)

            edgeMapPyr = makePyramid(image_sum)

            # Create VonMises
            dim1 = np.arange(-3*R0,3*R0+1)
            dim2 = np.flip(dim1,0)

            invmsk1 = []
            invmsk2 = []

            for ori in range(NUMORI):

                temp1,temp2 = makeVonMises(R0,oris[ori]+math.pi/2,dim1,dim2)

                invmsk1.append(temp1)
                invmsk2.append(temp2)

            # Generate border ownership(feedforward step)
            bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,s_pyr = makeBorderOwnership(edgeMapPyr,oriensMatrixPyr,np.multiply(orienslist,math.pi/180))

            # Iterative algorithm
            for i in range(ITERATIONS):

                print('Iteration ' +str(i+1)+ '\n')

                gl_pyr,gd_pyr,bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr = makeGrouping(bl_1_pyr,bl_2_pyr,bd_1_pyr,bd_2_pyr,invmsk1,invmsk2,s_pyr)

            # Combine border-ownership maps
            bTotal1 = sumPyr(bl_1_pyr,bd_1_pyr,'orientation')
            bTotal2 = sumPyr(bl_2_pyr,bd_2_pyr,'orientation')

            # Visualize border ownership at the highest resolution
            X = np.zeros((np.shape(bTotal1[0])[0],np.shape(bTotal1[0])[1]))
            Y = np.zeros((np.shape(bTotal1[0])[0],np.shape(bTotal1[0])[1]))

            for ori in range(NUMORI):

                # BOS points towards outside of circle
                X += (math.cos(oris[ori]-(math.pi/2))*(bTotal2[0][:,:,ori]-bTotal1[0][:,:,ori]))
                Y += (math.sin(oris[ori]-(math.pi/2))*(bTotal2[0][:,:,ori]-bTotal1[0][:,:,ori]))

            # Combine grouping maps
            gTotal = sumPyr(gl_pyr, gd_pyr, 'data')

            # Combine the maps
            groupData = np.zeros((np.shape(image)[0],np.shape(image)[1]))
            for level in range(MAXLEVEL):
                groupData += cv2.resize(gTotal[level], (np.shape(image)[1],np.shape(image)[0]))

            if SaveFilesFLAG:
                np.savetxt(saving_path+'X'+SUFFIX+'.csv',X)
                np.savetxt(saving_path+'Y'+SUFFIX+'.csv',Y)
                np.savetxt(saving_path+'grouping' + SUFFIX + '.csv', groupData)


                #how to draw a window
                # cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                # cv2.imshow("Image", occ_map)
                # cv2.waitKey(0)
