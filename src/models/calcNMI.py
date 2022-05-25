#######################
## Mutual Information##
#######################
import numpy as np
import torch

def calcMI(Z1, Z2):
    '''
    Function to calculate mutual information between two assignment matrices
    Input:
        Z1  D1 x N  Assignment matrix
        Z2  D2 x N  Assignment matrix
    output:
        MI Mutual information between Z1 and Z2
    Inspired from source code provided by Morten Mørup
    '''
    #if Z1.shape[1] != Z2.shape[1]:
    #    print('OOPS! Number of columns need to be equal in the two matrices! (Try transposing one of them)')
    #    return np.nan
    
    P=Z1@Z2.T
    PXY=P/torch.sum(torch.sum(P))
    PXPY=torch.sum(PXY,1).unsqueeze(1) * torch.sum(PXY,0)
    ind=torch.where(PXY>0)
    MI=sum(PXY[ind]*torch.log(PXY[ind]/PXPY[ind]))
    return MI

def calcNMI(Z1, Z2):
    '''
    WIP 
    Function to calculate mutual information between two assignment matrices
    Input:
        Z1  D1 x N  Assignment matrix
        Z2  D2 x N  Assignment matrix
    output:
        NMI Normalizes Mutual information between Z1 and Z2
    Inspired from source code provided by Morten Mørup
    ''' 
    return 2 * calcMI(Z1, Z2) / (calcMI(Z1, Z1) + calcMI(Z2, Z2))
torch.random.manual_seed(0)
Z1 = torch.rand((10, 5))
Z2 = torch.rand((10, 10))
Z1 = Z1.T #Need number of columns to be equal, not rows <3
Z2 = Z2.T
print(calcNMI(Z1, Z1))
print(calcMI(Z1,Z2)) #Gives 0.026 as Mortens matlab solution!
