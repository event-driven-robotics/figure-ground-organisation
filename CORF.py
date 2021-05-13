import cv2
import numpy as np
import math
from scipy.ndimage.measurements import label
import numpy.matlib
import skimage.morphology
import scipy
import time

def makePyramid(img):
    depth = 10
    scale = 1/pow(math.sqrt(2),(2-1))
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    image = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    return image

def rescaleImage(image, mn, mx):

    if (image.size == 0) == False:
        maximum = np.amax(image[:])
        minimum = np.min(image[:])
        if maximum - minimum == 0:
            image = np.ones(np.shape(image)) * mx
        else:
            image = (((image-minimum)/(maximum-minimum) * (mx - mn)) + mn)

    return image

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

def getDoG(img,sigma, onoff, sigmaRatio, config, width):

    # Create Difference of Gaussian Kernel
    sz = (np.shape(img)[0]+ 2*width,np.shape(img)[1]+ 2*width)

    # Create Gaussians
    g1 = matlab_style_gauss2D(sz,sigma)
    g2 = matlab_style_gauss2D(sz,sigma*sigmaRatio)

    # LGN cell
    if onoff == 1:
        # Center-OFF RF
        G = g2 - g1
    else:
        # Center-ON RF
        G = g1 - g2

    # Intesity distribution in the input image
    resultimg = np.pad(img,(width,width),'symmetric')
   
    # Compute LGN response as linear spatial summation
    # FFTSHIFT in order to recover the phase since it is important for reconstructing the correct spatial structure of the image
    output = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(G,(sz[0],sz[1]))*np.fft.fft2(resultimg))) # Applying the inverse transform => f*g

    return output.real

def pol2cart(th,r):

    x = np.multiply(r,np.cos(th))
    y = np.multiply(r,np.sin(th))
    return(x, y)

def cart2pol(x, y):

    th = np.arctan2(y, x)
    r = np.sqrt(np.power(abs(x),2) + np.power(abs(y),2))
    return(th, r)
  
def determineProperties(input,rho,fp,t2):
    
    # Responses of model LGN cells
    gb = input.max(axis=2)
    #np.savetxt('gb.csv',gb)

    t = float('inf')
    for i in range(0,np.shape(input)[2]): 
        mxlist = np.amax(np.amax(input[:,:,i]))
        if mxlist < t:
            t = mxlist
    t = np.amin(mxlist)

    x = list(range(1, 360*2+1))
    
    if rho > 0:
        # Building the circle
        a = np.round(fp[0] + np.multiply(rho,np.cos(np.multiply(x,math.pi/180))))-1
        b = np.round(fp[1]+np.multiply(rho,np.sin(np.multiply(x,math.pi/180))))-1

        # Get responses of model LGN cells along the circle
        tempo = []
        for i in range(0,len(a)):
            tempo.append(a[i].astype(int)+np.shape(input)[0]*b[i].astype(int))
        gb = gb.ravel('F') 
        y = gb[tempo]    
        y = np.roll(y.conj().transpose(),270).conj().transpose() #FOR CLEAR VISUALIZATION

        # Check if all elements are equal
        if len(np.unique(y)) == 1:
            onoff = []
            rho = []
            phi = []              
            return (onoff,rho,phi)

        y[y < 0.01] = 0  
        y = np.round(y*1000)/1000

        # Find local maxima inside the responses of model LGN cells along the circle
        nikos = skimage.morphology.local_maxima(y) #Locations of regional maxima
        BW, ncomponents = label(nikos) #Label matrix of contiguous regions
        npeaks = BW[:].max(0)
        peaks = np.zeros((1,npeaks))

        # Compute the value of the peaks as mean of
        # the positions of the regional maxima
        for i in range(0,npeaks):
            peaks[0,i] = np.mean(np.nonzero(BW == i+1))
       
        f = np.logical_and(peaks >= 180,peaks < 540) #TAKE HALF OF THE CIRCLE

        # Positions (in angle °) of the sub-units on the circle
        phivalues = peaks[f]

        x, y = pol2cart((phivalues-270)*math.pi/180,rho)

        # Positions (in angle rad) of the sub-units on the circle
        phi = phivalues*math.pi/180

        # Get polarity of the sub-inits knowing that
        # position 0 is LGN response center-ON
        # position 1 is LGN response center-OFF
        # The polarity is the position (0 or 1) of the maximum response
        onoff = np.zeros((1,len(phi)))
        for i in range(0,np.size(phi)):
            idx = np.argmax(input[round(fp[1]+x[i]),round(fp[0]+y[i]),:])
            onoff[0,i] = idx
        
    else:
        centreResponses = np.reshape(input[round(fp[1]),round(fp[0]),:],1,2)
        mx = np.amax(centreResponses,0)
        if np.amax(mx) >= t:
            onoff = np.argwhere(mx > t2*np.amax(mx)) - 1
            phi = np.zeros((1,len(onoff)))
        else:
            onoff = []
            phi = []
    
    rho = np.tile(rho,(1,len(phi)))
    phi = np.expand_dims(phi,0)
    return (onoff,rho,phi)

def getHashKey(vector):

    primelist = [2,3,5,7,11,13]
    n = len(vector)

    # based on prime factorization
    hashkey = np.prod(np.power(primelist[0:n],vector))
    return hashkey

def modifyModel(model,*args):

    '''
        Compute the new CORF model cell that is selective
        for vertical edges with opposite contrast
    '''

    nargs = len(args)
    names = []
    values = []

    for i in range(0,nargs-1,2):
        names.append(args[i])

    for i in range(1,nargs,2):
        values.append(args[i])

    for i in range(0,len(names)):
        if names[i] == "invertpolarity":
            # invert polarity
            if values[i]== 1:
                model[0,:] = 1-model[0,:]
        elif names[i] == "thetaoffset":
            # add angular offset
            model[3,:] = (model[3,:]+values[i])%(2*math.pi)
        elif names[i] == "overlappixels":
            rho = model[2,:]
            phi = model[3,:]
            x, y = pol2cart(phi,rho)
            negx = x < 0    
            x[negx] = x[negx] - values[i] #x = x-beta beta/2?
            x[~negx] = x[~negx] + values[i] #x = x+beta beta/2?
            phi, model[2,:] = cart2pol(x,y)
            model[3,:] = (phi+2*math.pi)%(2*math.pi) #NON CHIARO    

    return model

def imshift(im,shiftRows,shiftCols):

    A = np.zeros(np.shape(im))
    shiftRows = int(shiftRows)
    shiftCols = int(shiftCols)

    if shiftRows >= 0 and shiftCols >= 0:
        A[shiftRows:,shiftCols:] = im[0:int(np.shape(im)[0]-shiftRows),0:int(np.shape(im)[1]-shiftCols)]
    elif shiftRows >= 0 and shiftCols < 0:
        A[shiftRows:,0:int(np.shape(A)[1]+shiftCols)] = im[0:int(np.shape(im)[0]-shiftRows),-shiftCols:]
    elif shiftRows < 0 and shiftCols >= 0:
        A[0:int(np.shape(A)[0]+shiftRows),shiftCols:] = im[-shiftRows:,0:int(np.shape(im)[1]-shiftCols)]
    else:
        A[0:int(np.shape(A)[0]+shiftRows),0:int(np.shape(A)[1]+shiftCols)] = im[-shiftRows:,-shiftCols:]

    return A

def calc_viewimage(matrices, dispcomb, theta):

    '''
        Calculates the maximum-superposition of all the matrices stored
        in MATRICES (according to the L-infinity norm). It uses only the matrices for
        which the index is entered in DISPCOMB, e.g. if DISPCOMB contains the values
        1,2,4: only the first, second and fourth matrix contained in MATRICES are used
        for the superposition. This method also calculates the orientationmatrix (ORIENSMATRIX) 
        which stores the maximum orientation response of each point in the resulting matrix (RESULT)
    '''

    # initialize values
    oriensMatrix = 0
    tmpMaxConv = float('-inf')
    result = float('-inf')
    cnt1 = 0

    if (np.shape(dispcomb)[0] == 1):
        result = matrices[:,:,dispcomb[0]]
    else:

        # calculate the superposition (L-infinity norm)
        while (cnt1 < np.shape(dispcomb)[0]):
            #calculate the maximum orientation-response in each point (based on the absolute values)
            oriensMatrixtmp1 = np.multiply((abs(matrices[:,:,dispcomb[cnt1]-1]) > tmpMaxConv),theta[dispcomb[cnt1]-1])
            oriensMatrixtmp2 = np.multiply((abs(matrices[:,:,dispcomb[cnt1]-1]) <= tmpMaxConv),oriensMatrix)
            oriensMatrix = oriensMatrixtmp1 + oriensMatrixtmp2
            tmpMaxConv = np.maximum(abs(matrices[:,:,dispcomb[cnt1]-1]), tmpMaxConv)
        
            # calculate the superposition
            result = np.maximum(result,abs(matrices[:,:,dispcomb[cnt1]-1]))
            cnt1 = cnt1 + 1
    
    return(result, oriensMatrix)

class Corf():

    def __init__(self,img,sigma, beta, inhibitionFactor, highthresh=-1):
        self.img = img
        self.sigma = sigma
        self.beta = beta
        self.inhibitionFactor = inhibitionFactor
        self.highthresh = highthresh
        self.blurringType = "Sum"
        self.simpleCell_excitation = np.array([])

    def getsimpleCellExcitation(self):
        return self.simpleCell_excitation

    def configureSimpleCell(self,sigmaRatio,t2):

        '''
            Starting from LGN cells in order to
            create sub-units of CORF cell
        '''

        # Create a synthetic stimulus of a vertical edge
        stimulus = np.zeros((200,200))
        stimulus[:,0:100] = 1

        self.eta = 1
        self.radius = math.ceil(self.sigma*2.5)*2+1

        # Linear spatial summation
        DoG_pos = getDoG(stimulus, self.sigma, 0, sigmaRatio, 1, 0)
        DoG_neg = -DoG_pos

        # Half wave rectification(only takes positive elements)
        DoG_pos = np.multiply(DoG_pos,(DoG_pos > 0))
        DoG_neg = np.multiply(DoG_neg,(DoG_neg > 0))

        # LGN cell response(ON+OFF)
        DoG = np.stack((DoG_pos,DoG_neg),2)

        # Choose rho list according to given sigma value
        # Sigma: scale parameter of the LGN cells
        # Rho: radius of concetric circles
        if self.sigma >= 1 and self.sigma <= 2:
            self.rho = (14.38,6.9796,3.0310,1.4135)
        elif self.sigma > 2 and self.sigma < 2.4:
            self.rho = (19.18,9.369,4.5128,2.1325)
        elif self.sigma >= 2.4 and self.sigma <= 3.5:
            self.rho = (24.62,12.6488,6.1992,3.0515)
        elif self.sigma >= 4 and self.sigma <= 5:
            self.rho = (34.43,18.08,9.2467,4.7877,3.3021)

        self.alpha = 0.9

        # Create configuration set

        # Image of size 100x100px
        fp = [100,100]       
        for r in range(0,len(self.rho)):
            # Onoff: polarity of sub-unit
            # Rho: radius of concetric circles
            # Phi: polar angle of RF center of sub-unit with respect to the center of the CORF cell 
            onoff,rho,phi = determineProperties(DoG,self.rho[r],fp,t2)        
            if (onoff.size == 0) == False:
                xs = np.concatenate((onoff,np.matlib.repmat(self.sigma,1,np.shape(onoff)[1]),rho,phi),0)
                self.simpleCell_excitation = np.concatenate((self.simpleCell_excitation,xs),1) if self.simpleCell_excitation.size else xs
                
        self.simpleCell_excitation[3,:] = self.simpleCell_excitation[3,:]%(2*math.pi) #NON CHIARO

        self.sigma0 = 2
        self.sigmaRatio = sigmaRatio
        self.radius = round(np.amax(self.rho) + (2 + self.alpha*np.amax(self.rho))/2) #NON CHIARO

    def computeBlurredResponse(self):

        '''
            Starting from LGN cells in order to
            compute sub-units' responses of CORF cell
        '''

        # Get sigma values of sub-units from the exication S cell
        sigmaList = self.simpleCell_excitation[1,:]

        # Get polarity,sigma and rho values of sub-units from the exication S cell
        paramsList = np.transpose(self.simpleCell_excitation[(0,1,2),:])
        #if isfield(model.simpleCell,'inhibition')

        # Merge sigma values of sub-units from the exication and inhibition S cell
        sigmaList =  np.expand_dims(sigmaList,0)

        # Merge polarity,sigma and rho values of sub-units from the exication and inhibition S cell
        sigmaList = np.concatenate((sigmaList,np.expand_dims(self.simpleCell_inhibition[1,:],0)),1)   
        paramsList = np.concatenate((paramsList,np.transpose(self.simpleCell_inhibition[(0,1,2),:])),0)
        
        sigmaList = np.unique(sigmaList)
        paramsList = np.unique(np.around(paramsList*1000)/1000,axis=0)

        nsigmaList = len(sigmaList)
        nparamsList = np.shape(paramsList)[0]

        LGNHashTable = {}

        for s in range(0,nsigmaList):
            delta = 0
            key_0 = getHashKey(np.array([delta,sigmaList[s]]))

            # Linear spatial summation
            LGNHashTable[key_0] = getDoG(self.img,sigmaList[s],0,self.sigmaRatio,delta,self.radius)

            delta = 1
            key_1 = getHashKey(np.array([delta,sigmaList[s]]))
            LGNHashTable[key_1] = -LGNHashTable[key_0]
            
            # LGN cell response(ON+OFF)
            LGNHashTable[key_0] = np.multiply(LGNHashTable[key_0],(LGNHashTable[key_0] > 0))
            LGNHashTable[key_1] = np.multiply(LGNHashTable[key_1],(LGNHashTable[key_1] > 0))

        # Sigma_prime of the CORF model response
        weightsigma = np.amax(paramsList[:,2])/ 3

        SimpleCellHashTable = {}

        for p in range(0,nparamsList):
            # Polarity
            delta = paramsList[p,0]

            sigma = paramsList[p,1]
            rho = paramsList[p,2]
           
            LGNHashKey = getHashKey(np.array([delta,sigma]))

            # LGN response
            LGN = LGNHashTable[LGNHashKey]   
            
            # 3*((d_0+alpha*rho)/6)
            r = round((self.sigma0 + self.alpha*rho)/2)
            my_key = getHashKey(np.array([delta,sigma,rho]))
            if r > 0: 
                if self.blurringType == "Sum":
                    smoothfilter = matlab_style_gauss2D((2*r+1,2*r+1),r/3) # Size of the kernel = 6*sigma, where sigma=(d_0+alpha*rho)/6 => sigma_prime

                    # Response of the sub-unit
                    SimpleCellHashTable[my_key] = scipy.signal.convolve2d(LGN,smoothfilter,"same")
                           
                #elif self.blurringType == "Max":
                    #SimpleCellValueList[p] = maxgaussianfilter(LGN,r/3,0,0,[1 1],size(LGN));        
            else:
                SimpleCellHashTable[my_key] = LGN

            # Part of CORF model response (only elevation to w_i)
            SimpleCellHashTable[my_key] = np.power(SimpleCellHashTable[my_key],math.exp(-rho**2/(2*weightsigma*weightsigma)))

            #SimpleCellValueList{p} = SimpleCellValueList{p} .^ exp(-rho^2/(2*weightsigma*weightsigma))
        return SimpleCellHashTable,weightsigma

    def getResponse(self,SimpleCellHashTable,simpleCell,weightsigma):

        '''
            Starting from sub-units' responses in order to
            compute CORF model response
        '''

        ntuples = np.shape(simpleCell)[1]
        w = np.zeros((1,ntuples))
        response = 1

        for i in range(0,ntuples):
            delta = simpleCell[0,i]
            sigma = simpleCell[1,i]
            rho   = round(simpleCell[2,i]*1000)/1000
            phi   = simpleCell[3,i]
            
            [col, row] = pol2cart(phi,rho)

            # Part of CORF model response (only elevation to w_i)
            blurredResponse = SimpleCellHashTable[getHashKey(np.array([delta,sigma,rho]))]
            
            shiftedResponse = imshift(blurredResponse,np.fix(row),-np.fix(col)) # Probably minus row???
                                                                
            w[0,i] = math.exp(-rho**2/(2*weightsigma*weightsigma))

            # Production part of the CORF model response       
            response = np.multiply(response,shiftedResponse)

        # Response CORF model cell
        response = np.power(response,(1/np.sum(w)))
        response = response[self.radius:np.shape(response)[0]-self.radius,self.radius:np.shape(response)[1]-self.radius] #NON CHIARO
        
        return response
    
    def getSimpleCellResponse(self,orientlist):

        self.SimpleCellHashTable, self.weightsigma = self.computeBlurredResponse()

        response = np.zeros((np.shape(self.img)[0],np.shape(self.img)[1],len(orientlist)))
        
        for i in range(0,len(orienslist)): 
            
            # In order to obtain the new push-pull inhibition CORF model with different preferred orientations
            simpleCell_excitation = modifyModel(self.simpleCell_excitation.copy(),'thetaoffset',orientlist[i]*math.pi/180)
            simpleCell_inhibition = modifyModel(self.simpleCell_inhibition.copy(),'thetaoffset',orientlist[i]*math.pi/180)

            # Compute the excitatory response of the simple cell
            excitationresponse = self.getResponse(self.SimpleCellHashTable,simpleCell_excitation,self.weightsigma)

            #if isfield(model.simpleCell,'inhibition')
            # Compute the antiphase inhibitory response of the simple cell
            inhibitionresponse = self.getResponse(self.SimpleCellHashTable,simpleCell_inhibition,self.weightsigma)          

            # Compute the net response
            rotresponse = excitationresponse - self.inhibitionFactor*inhibitionresponse
            
            rotresponse = np.multiply(rotresponse,(rotresponse > 0))
            #else
                # If no inhibition, then we only take the excitatory response
                #rotresponse = excitationresponse;
            response[:,:,i] = rotresponse

        return response

if __name__ == '__main__':

    start = time.time()

    img = cv2.imread('42049.jpg')
    img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    
    # Make colors
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    intesity = (r+g+b)/3
    
    #img = makePyramid(intesity)
    img = intesity

    img  = rescaleImage(np.double(img),0,1)
    
    corf = Corf(img,2.2,4,1.8) 
    corf.configureSimpleCell(0.5,0.5)
  
    
    corf.simpleCell_inhibition = modifyModel(corf.getsimpleCellExcitation().copy(),"invertpolarity",1,"overlappixels",corf.beta)

    # 16 equidistant (22.5°) orientation preferences 
    orienslist = list(np.arange(0, 337.5+1,22.5)) 

    output = corf.getSimpleCellResponse(orienslist)
    
    corfresponse, oriensMatrix = calc_viewimage(output,list(range(1,len(orienslist)+1)), np.multiply(orienslist,math.pi/180))
    
    end = time.time()
    print ("Time elapsed:", end - start)
    #np.savetxt('oriensMatrix.csv',oriensMatrix)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", oriensMatrix)
    cv2.waitKey(0)
    
