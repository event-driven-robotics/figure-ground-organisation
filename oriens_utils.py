import numpy as np
from scipy.stats import multivariate_normal
import scipy.signal
import imutils

def mvnpdf(mu,sigma):

    x1 = list(np.arange(-3, 3+0.2,0.2))
    x2 = list(np.arange(-3, 3+0.2,0.2))

    X1,X2 = np.meshgrid(x1,x2)

    X = np.stack((X1.ravel('F'),X2.ravel('F')),1)

    y = multivariate_normal.pdf(X, mu, sigma)
    y = np.reshape(y,(len(x2),len(x1)))

    return y

def mvDog(mu1,mu2,sigma1,sigma2):

    y1 = mvnpdf(mu1,sigma1)
    y2 = mvnpdf(mu2,sigma2)

    G = y2-y1

    return G

def getOriensResp(img,G,orienslist):

    result = scipy.signal.convolve2d(img,G,"same")

    response = np.zeros((np.shape(result)[0],np.shape(result)[1],len(orienslist)))

    response[:,:,0] = result

    for i in range(1,len(orienslist)):
        Grot = imutils.rotate(G, angle=orienslist[i])
        temp = scipy.signal.convolve2d(img,Grot,"same")
        temp[temp < 0.08] = 0
        response[:,:,i] = temp

    return response


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