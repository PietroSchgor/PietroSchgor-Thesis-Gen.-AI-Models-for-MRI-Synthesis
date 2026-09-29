import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as FUN
import numpy as np
import math
import itertools
import random
import lpips
import csv
import time
import os

def ssim_structure(img1, img2):
    """
    Compute the structural similarity index (SSIM) focusing only on the structure component.
    
    Parameters:
    - img1: First image (grayscale).
    - img2: Second image (grayscale).
    
    Returns:
    - Structural similarity index focusing only on structure.
    """
    # Ensure the input images are in float format
    #img1 = img1.astype(np.float64)
    #img2 = img2.astype(np.float64)
    
    # Define constants for numerical stability
    #C3 = 1e-15
    C3 = (0.03 * (2**8 - 1))**2 / 2

    # Compute the mean of the images
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Compute the standard deviation of the images
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    
    # Compute the covariance between the images
    covariance = np.mean((img1 - mu1) * (img2 - mu2))
    
    # Compute the structural similarity index focusing only on the structure component
    ssim_structure_component = (covariance + C3) / (sigma1 * sigma2 + C3)
    
    return ssim_structure_component

def normalize_image(image):
    """ Normalize image to the range [0, 1]. """
    image = image.astype(np.float64)
    return 255*(image - np.min(image)) / (np.max(image) - np.min(image))

def calculate_3d_ssim(image1, image2):
    assert image1.shape == image2.shape, "Input images must have the same dimensions"
    
    # Normalize images to [0, 1] range
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    
    # Dimensions of the 3D images
    H, W, D = image1.shape
    
    # Calculate SSIM for each slice along the third axis (depth)
    ssim_values = []
    for i in range(D):
        slice1 = image1[:, :, i]
        slice2 = image2[:, :, i]
        slice_ssim = ssim_structure(slice1, slice2)  # data_range is 1 since images are normalized
        ssim_values.append(slice_ssim)
        
    for i in range(W):
        slice1 = image1[:, i, :]
        slice2 = image2[:, i, :]
        slice_ssim = ssim_structure(slice1, slice2)  # data_range is 1 since images are normalized
        ssim_values.append(slice_ssim)
    
    for i in range(H):
        slice1 = image1[i, :, :]
        slice2 = image2[i, :, :]
        slice_ssim = ssim_structure(slice1, slice2) # data_range is 1 since images are normalized
        ssim_values.append(slice_ssim)
    
    # Average the SSIM values for all slices
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    
    
    return mean_ssim, ssim_values, std_ssim

def calculate_ssim_for_image_lists(list1, list2):
    # Ensure the two lists have the same length
    assert len(list1) == len(list2), "The two lists must have the same number of images"
    
    # Initialize a list to store the SSIM indices for each pair of images
    ssim_results = []
    
    # Loop through each pair of images from list1 and list2
    for img1, img2 in zip(list1, list2):
        mean_ssim, ssim_values, std_ssim = calculate_3d_ssim(img1, img2)
        
        ssim_results.append(mean_ssim)
    
    return ssim_results

def calculate_ssim_for_single_list(image_list, max_pairs=None, seed=None):
    """
    Compute full 3D SSIM for randomly selected pairs from a single list of 3D images.

    Parameters:
        image_list (list of np.ndarray): List of 3D images (H, W, D).
        max_pairs (int, optional): Maximum number of pairs to evaluate.
                                   If None, use all possible pairs.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        mean_val (float): Mean SSIM across selected pairs.
        std_val (float): Std.dev SSIM across selected pairs.
        values (list of float): Individual SSIM values for each pair.
        pairs (list of tuple): Indices of image pairs that were evaluated.
    """
    n = len(image_list)
    assert n >= 2, "Need at least two images to compute SSIM."

    # All possible unique pairs
    all_pairs = list(itertools.combinations(range(n), 2))

    # Subsample pairs if requested
    if max_pairs is not None and max_pairs < len(all_pairs):
        if seed is not None:
            random.seed(seed)
        pairs = random.sample(all_pairs, max_pairs)
    else:
        pairs = all_pairs

    values = []
    for i, j in pairs:
        mean_ssim, _, _ = calculate_3d_ssim(image_list[i], image_list[j])
        values.append(mean_ssim)

    mean_val = float(np.mean(values)) if values else float("nan")
    std_val = float(np.std(values)) if values else float("nan")

    return mean_val, std_val, values, pairs



def ssim_full(img1, img2):
    """
    Compute the full SSIM (luminance, contrast, structure) between two 2D slices.
    """
    C1 = (0.01 * (2**8 - 1))**2
    C2 = (0.03 * (2**8 - 1))**2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return numerator / denominator


def normalize_image(image):
    """ Normalize image to [0, 255] range (float). """
    image = image.astype(np.float64)
    return 255 * (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-12)


def calculate_3d_full_ssim(image1, image2):
    """
    Compute mean and std of full SSIM across all 2D slices along H, W, and D.
    """
    assert image1.shape == image2.shape, "Input images must have the same dimensions"
    
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    
    H, W, D = image1.shape
    ssim_values = []

    # Slice along depth
    for i in range(D):
        ssim_values.append(ssim_full(image1[:, :, i], image2[:, :, i]))
    # Slice along width
    for i in range(W):
        ssim_values.append(ssim_full(image1[:, i, :], image2[:, i, :]))
    # Slice along height
    for i in range(H):
        ssim_values.append(ssim_full(image1[i, :, :], image2[i, :, :]))
    
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    
    return mean_ssim, ssim_values, std_ssim


def calculate_full_ssim_for_image_lists(list1, list2):
    """
    Compute full 3D SSIM for two lists of 3D images (same length).
    Returns a list of mean SSIM values (one per pair).
    """
    assert len(list1) == len(list2), "Both lists must have the same number of images"
    
    ssim_results = []
    for img1, img2 in zip(list1, list2):
        mean_ssim, _, _ = calculate_3d_full_ssim(img1, img2)
        ssim_results.append(mean_ssim)
    
    return ssim_results


def calculate_full_ssim_for_single_list(image_list, max_pairs=None, seed=None):
    """
    Compute full 3D SSIM for randomly selected pairs from a single list of 3D images.

    Parameters:
        image_list (list of np.ndarray): List of 3D images (H, W, D).
        max_pairs (int, optional): Maximum number of pairs to evaluate.
                                   If None, use all possible pairs.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        mean_val (float): Mean SSIM across selected pairs.
        std_val (float): Std.dev SSIM across selected pairs.
        values (list of float): Individual SSIM values for each pair.
        pairs (list of tuple): Indices of image pairs that were evaluated.
    """
    n = len(image_list)
    assert n >= 2, "Need at least two images to compute SSIM."

    # All possible unique pairs
    all_pairs = list(itertools.combinations(range(n), 2))

    # Subsample pairs if requested
    if max_pairs is not None and max_pairs < len(all_pairs):
        if seed is not None:
            random.seed(seed)
        pairs = random.sample(all_pairs, max_pairs)
    else:
        pairs = all_pairs

    values = []
    for i, j in pairs:
        mean_ssim, _, _ = calculate_3d_full_ssim(image_list[i], image_list[j])
        values.append(mean_ssim)

    mean_val = float(np.mean(values)) if values else float("nan")
    std_val = float(np.std(values)) if values else float("nan")

    return mean_val, std_val, values, pairs



def ssim_luminance(img1, img2):
    """
    Compute the luminance component of SSIM between two 2D slices.
    """
    C1 = (0.01 * (2**8 - 1))**2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    l = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    return l


def calculate_3d_luminance_ssim(image1, image2):
    """
    Compute mean and std of luminance SSIM across all 2D slices along H, W, and D.
    """
    assert image1.shape == image2.shape, "Input images must have the same dimensions"

    # Normalize images to [0,255]
    def normalize_image(img):
        img = img.astype(np.float64)
        return 255 * (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-12)

    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    H, W, D = image1.shape
    ssim_values = []

    # Slice along depth
    for i in range(D):
        ssim_values.append(ssim_luminance(image1[:, :, i], image2[:, :, i]))
    # Slice along width
    for i in range(W):
        ssim_values.append(ssim_luminance(image1[:, i, :], image2[:, i, :]))
    # Slice along height
    for i in range(H):
        ssim_values.append(ssim_luminance(image1[i, :, :], image2[i, :, :]))

    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)

    return mean_ssim, ssim_values, std_ssim


def calculate_luminance_ssim_for_single_list(image_list, max_pairs=None, seed=None):
    """
    Compute luminance-only 3D SSIM for randomly selected pairs from a single list of 3D images.

    Parameters:
        image_list (list of np.ndarray): List of 3D images (H, W, D).
        max_pairs (int, optional): Maximum number of pairs to evaluate.
                                   If None, use all possible pairs.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        mean_val (float): Mean luminance SSIM across selected pairs.
        std_val (float): Std.dev luminance SSIM across selected pairs.
        values (list of float): Individual luminance SSIM values for each pair.
        pairs (list of tuple): Indices of image pairs that were evaluated.
    """
    n = len(image_list)
    assert n >= 2, "Need at least two images to compute SSIM."

    all_pairs = list(itertools.combinations(range(n), 2))

    # Subsample if requested
    if max_pairs is not None and max_pairs < len(all_pairs):
        if seed is not None:
            random.seed(seed)
        pairs = random.sample(all_pairs, max_pairs)
    else:
        pairs = all_pairs

    values = []
    for i, j in pairs:
        mean_ssim, _, _ = calculate_3d_luminance_ssim(image_list[i], image_list[j])
        values.append(mean_ssim)

    mean_val = float(np.mean(values)) if values else float("nan")
    std_val = float(np.std(values)) if values else float("nan")

    return mean_val, std_val, values, pairs



'''
This code is a direct pytorch implementation of the original FSIM code provided by
Lin ZHANG, Lei Zhang, Xuanqin Mou and David Zhang in Matlab. For the original version
please see: 

https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm

'''


class FSIM_base(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.cuda_computation = False
        self.nscale = 4 # Number of wavelet scales
        self.norient = 4 # Number of filter orientations
        self.k = 2.0 # No of standard deviations of the noise
                     # energy beyond the mean at which we set the
                     # noise threshold point. 
                     # below which phase congruency values get
                     # penalized.

        self.epsilon = .0001 # Used to prevent division by zero
        self.pi = math.pi
        
        minWaveLength = 6  # Wavelength of smallest scale filter
        mult = 2  # Scaling factor between successive filters
        sigmaOnf = 0.55 # Ratio of the standard deviation of the
                        # Gaussian describing the log Gabor filter's
                        # transfer function in the frequency domain
                        # to the filter center frequency.    
        dThetaOnSigma = 1.2 # Ratio of angular interval between filter orientations    
                            # and the standard deviation of the angular Gaussian
                            # function used to construct filters in the
                            # freq. plane.
        
        self.thetaSigma = self.pi/self.norient/dThetaOnSigma # Calculate the standard deviation of the
                                                             # angular Gaussian function used to
                                                             # construct filters in the freq. plane.
        

        self.fo = (1.0/(minWaveLength*pt.pow(mult,(pt.arange(0,self.nscale,dtype=pt.float64))))).unsqueeze(0) # Centre frequency of filter
        self.den = 2*(math.log(sigmaOnf))**2
        self.dx = -pt.tensor([[[[3, 0, -3], [10, 0,-10], [3,0,-3]]]])/16.0
        self.dy = -pt.tensor([[[[3, 10, 3], [0, 0, 0],   [-3 ,-10, -3]]]])/16.0
        self.T1 = 0.85
        self.T2 = 160
        self.T3 = 200;
        self.T4 = 200;
        self.lambdac = 0.03

    def set_arrays_to_cuda(self):
        self.cuda_computation = True
        self.fo = self.fo.cuda()
        self.dx = self.dx.cuda()
        self.dy = self.dy.cuda()
    
    def forward_gradloss(self,imgr,imgd):
        I1,Q1,Y1 = self.process_image_channels(imgr)
        I2,Q2,Y2 = self.process_image_channels(imgd)

        
        #PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)
        
        gradientSimMatrix = self.calculate_gradient_sim(gradientMap1,gradientMap2)
        #gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        gradloss = pt.sum(pt.sum(pt.sum(gradientSimMatrix,1),1))
        return gradloss
    
    def calculate_fsim(self,gradientSimMatrix,PCSimMatrix,PCm):
        SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
        FSIM = pt.sum(pt.sum(SimMatrix,1),1) / pt.sum(pt.sum(PCm,1),1)
        return FSIM

    def calculate_fsimc(self, I1,Q1,I2,Q2,gradientSimMatrix,PCSimMatrix,PCm):

        ISimMatrix = (2*I1*I2 + self.T3) / (pt.pow(I1,2) + pt.pow(I2,2) + self.T3)
        QSimMatrix = (2*Q1*Q2 + self.T4) / (pt.pow(Q1,2) + pt.pow(Q2,2) + self.T4)
        SimMatrixC = gradientSimMatrix*PCSimMatrix*(pt.pow(pt.abs(ISimMatrix*QSimMatrix),self.lambdac))*PCm
        FSIMc = pt.sum(pt.sum(SimMatrixC,1),1)/pt.sum(pt.sum(PCm,1),1)

        return FSIMc
    
    def lowpassfilter(self, rows, cols):
        cutoff = .45
        n = 15
        x, y = self.create_meshgrid(cols,rows)
        radius = pt.sqrt(pt.pow(x,2) + pt.pow(y,2)).unsqueeze(0)       
        f = self.ifftshift2d( 1 / (1.0 + pt.pow(pt.div(radius,cutoff),2*n)) ) 
        return f
    
    def calculate_gradient_sim(self,gradientMap1,gradientMap2):

        gradientSimMatrix = (2*gradientMap1*gradientMap2 + self.T2) /(pt.pow(gradientMap1,2) + pt.pow(gradientMap2,2) + self.T2)
        return gradientSimMatrix

    def calculate_gradient_map(self,Y):
        IxY = FUN.conv2d(Y,self.dx, padding=1)
        IyY = FUN.conv2d(Y,self.dy, padding=1)
        gradientMap1 = pt.sqrt(pt.pow(IxY,2) + pt.pow(IyY,2))
        return gradientMap1
    
    def calculate_phase_score(self,PC1,PC2):
        PCSimMatrix = (2 * PC1 * PC2 + self.T1) / (pt.pow(PC1,2) + pt.pow(PC2,2) + self.T1)
        PCm = pt.where(PC1>PC2, PC1,PC2)
        return PCSimMatrix,PCm    
    
    def roll_1(self,x, n):  
        return pt.cat((x[:,-n:,:,:,:], x[:,:-n,:,:,:]), dim=1)        
        
    def ifftshift(self,tens,var_axis):
        len11 = int(tens.size()[var_axis]/2)
        len12 = tens.size()[var_axis]-len11
        return pt.cat((tens.narrow(var_axis,len11,len12),tens.narrow(var_axis,0,len11)),axis=var_axis)

    def ifftshift2d(self,tens):
        return self.ifftshift(self.ifftshift(tens,1),2)

    #def create_meshgrid(self,cols,rows):
    #    '''
    #    Set up X and Y matrices with ranges normalised to +/- 0.5
    #    The following code adjusts things appropriately for odd and even values
    #    of rows and columns.
    #    '''
#
    #    if cols%2:
    #        xrange = pt.arange(start = -(cols-1)/2, end = (cols-1)/2+1, step = 1, requires_grad=False)/(cols-1)
    #    else:
    #        xrange = pt.arange(-(cols)/2, (cols)/2, step = 1, requires_grad=False)/(cols)
#
    #    if rows%2:
    #        yrange = pt.arange(-(rows-1)/2, (rows-1)/2+1, step = 1, requires_grad=False)/(rows-1)
    #    else:
    #        yrange = pt.arange(-(rows)/2, (rows)/2, step = 1, requires_grad=False)/(rows)
#
    #    x, y = pt.meshgrid([xrange, yrange])
    #    
    #    if self.cuda_computation:
    #        x, y = x.cuda(), y.cuda()
    #        
    #    return x.T, y.T
    
    def create_meshgrid(self, cols, rows):
        '''
        Set up X and Y matrices with ranges normalised to +/- 0.5
        The following code adjusts things appropriately for odd and even values
        of rows and columns.
        '''
    
        if cols%2:
            xrange = pt.arange(start = -(cols-1)/2, end = (cols-1)/2+1, step = 1, requires_grad=False)/(cols-1)
        else:
            xrange = pt.arange(-(cols)/2, (cols)/2, step = 1, requires_grad=False)/(cols)
    
        if rows%2:
            yrange = pt.arange(-(rows-1)/2, (rows-1)/2+1, step = 1, requires_grad=False)/(rows-1)
        else:
            yrange = pt.arange(-(rows)/2, (rows)/2, step = 1, requires_grad=False)/(rows)
    
        # OLD: x, y = pt.meshgrid([xrange, yrange])
        # NEW: Add indexing='ij' to maintain original behavior
        x, y = pt.meshgrid(xrange, yrange, indexing='ij')
        
        if self.cuda_computation:
            x, y = x.cuda(), y.cuda()
            
        return x.T, y.T

    def process_image_channels(self,img):


        batch, rows, cols = img.shape[0],img.shape[2],img.shape[3]

        minDimension = min(rows,cols)    

        Ycoef = pt.tensor([[0.299,0.587,0.114]])
        Icoef = pt.tensor([[0.596,-0.274,-0.322]])
        Qcoef = pt.tensor([[0.211,-0.523,0.312]])
        
        if self.cuda_computation:
            Ycoef, Icoef, Qcoef = Ycoef.cuda(), Icoef.cuda(), Qcoef.cuda()

        Yfilt=pt.cat(batch*[pt.cat(rows*cols*[Ycoef.unsqueeze(2)],dim=2).view(1,3,rows,cols)],0)
        Ifilt=pt.cat(batch*[pt.cat(rows*cols*[Icoef.unsqueeze(2)],dim=2).view(1,3,rows,cols)],0)
        Qfilt=pt.cat(batch*[pt.cat(rows*cols*[Qcoef.unsqueeze(2)],dim=2).view(1,3,rows,cols)],0)
        
        # If images have three chanels
        if img.size()[1]==3:
            Y = pt.sum(Yfilt*img,1).unsqueeze(1)
            I = pt.sum(Ifilt*img,1).unsqueeze(1)
            Q = pt.sum(Qfilt*img,1).unsqueeze(1)
        else:
            Y = pt.mean(img,1).unsqueeze(1)
            I = pt.ones(Y.size(),dtype=pt.float64)
            Q = pt.ones(Y.size(),dtype=pt.float64)

        F = max(1,round(minDimension / 256))

        aveKernel = nn.AvgPool2d(kernel_size = F, stride = F, padding =0)# max(0, math.floor(F/2)))
        if self.cuda_computation:
            aveKernel = aveKernel.cuda()
            
        # Make sure that the dimension of the returned image is the same as the input
        I = aveKernel(I)
        Q = aveKernel(Q)
        Y = aveKernel(Y)
        return I,Q,Y

        
    def phasecong2(self, img):
        batch, rows, cols = img.shape[0], img.shape[2], img.shape[3]
    
        # OLD: imagefft = pt.rfft(img, signal_ndim=2, onesided=False)
        # NEW: Use torch.fft.fft2 for 2D FFT
        imagefft = pt.fft.fft2(img.squeeze(1))  # Remove channel dim, apply FFT
        # Stack real and imaginary parts to match old format
        imagefft = pt.stack([imagefft.real, imagefft.imag], dim=-1).unsqueeze(1)
    
        x, y = self.create_meshgrid(cols, rows)
    
        radius = pt.cat(batch*[pt.sqrt(pt.pow(x,2) + pt.pow(y,2)).unsqueeze(0)],0)
        theta = pt.cat(batch*[pt.atan2(-y,x).unsqueeze(0)],0)
    
        radius = self.ifftshift2d(radius)
        theta = self.ifftshift2d(theta)
    
        radius[:,0,0] = 1 
        
        sintheta = pt.sin(theta)
        costheta = pt.cos(theta)
    
        lp = self.lowpassfilter(rows,cols)
        lp = pt.cat(batch*[lp.unsqueeze(0)],0)
     
        term1 = pt.cat(rows*cols*[self.fo.unsqueeze(2)],dim=2).view(-1,self.nscale,rows,cols)
        term1 = pt.cat(batch*[term1.unsqueeze(0)],0).view(-1,self.nscale,rows,cols)
    
        term2 = pt.log(pt.cat(self.nscale*[radius.unsqueeze(1)],1)/term1)
        logGabor = pt.exp(-pt.pow(term2,2)/self.den)
        logGabor = logGabor*lp
        logGabor[:,:,0,0] = 0
    
        angl = pt.arange(0,self.norient,dtype=pt.float64)/self.norient*self.pi
    
        if self.cuda_computation:
            angl = angl.cuda()
        ds_t1 = pt.cat(self.norient*[sintheta.unsqueeze(1)],1)*pt.cos(angl).view(-1,self.norient,1,1)
        ds_t2 = pt.cat(self.norient*[costheta.unsqueeze(1)],1)*pt.sin(angl).view(-1,self.norient,1,1)
        dc_t1 = pt.cat(self.norient*[costheta.unsqueeze(1)],1)*pt.cos(angl).view(-1,self.norient,1,1)
        dc_t2 = pt.cat(self.norient*[sintheta.unsqueeze(1)],1)*pt.sin(angl).view(-1,self.norient,1,1)
        ds = ds_t1-ds_t2
        dc = dc_t1+dc_t2
        dtheta = pt.abs(pt.atan2(ds,dc))
        spread = pt.exp(-pt.pow(dtheta,2)/(2*self.thetaSigma**2))
    
        logGabor_rep = pt.repeat_interleave(logGabor,self.norient,1).view(-1,self.nscale,self.norient,rows,cols)
    
        spread_rep = pt.cat(self.nscale*[spread]).view(-1,self.nscale,self.norient,rows,cols)
        filter_log_spread = logGabor_rep*spread_rep
        
        # Convert filter to complex for FFT operations
        filter_complex = pt.complex(filter_log_spread, pt.zeros_like(filter_log_spread))
        
        # OLD: Used pt.ifft with manual zero padding
        # NEW: Use torch.fft.ifft2
        ifftFilterArray = pt.fft.ifft2(filter_complex).real * math.sqrt(rows*cols)
    
        # Convert imagefft back to complex format for convolution
        imagefft_complex = pt.complex(imagefft[..., 0], imagefft[..., 1])
        imagefft_repeat = pt.cat(self.nscale*self.norient*[imagefft_complex],dim=1).view(-1,self.nscale,self.norient,rows,cols)
        
        # Convolve in frequency domain
        EO = pt.fft.ifft2(filter_complex * imagefft_repeat)
    
        E = EO.real
        O = EO.imag
        
        An = pt.sqrt(pt.pow(E,2)+pt.pow(O,2))
        sumAn_ThisOrient = pt.sum(An,1)
        sumE_ThisOrient = pt.sum(E,1)
        sumO_ThisOrient = pt.sum(O,1)
    
        XEnergy = pt.sqrt(pt.pow(sumE_ThisOrient,2) + pt.pow(sumO_ThisOrient,2)) + self.epsilon
        MeanE = sumE_ThisOrient / XEnergy
        MeanO = sumO_ThisOrient / XEnergy
        
        MeanO = pt.cat(self.nscale*[MeanO.unsqueeze(1)],1)
        MeanE = pt.cat(self.nscale*[MeanE.unsqueeze(1)],1)
    
        Energy = pt.sum( E*MeanE+O*MeanO - pt.abs(E*MeanO-O*MeanE),1)
        abs_EO  = pt.sqrt(pt.pow(E,2) + pt.pow(O,2))
    
        medianE2n = pt.pow(abs_EO.select(1,0),2).view(-1,self.norient,rows*cols).median(2).values
    
        EM_n = pt.sum(pt.sum(pt.pow(filter_log_spread.select(1,0),2),3),2)
        noisePower = -(medianE2n/math.log(0.5))/EM_n
        
        EstSumAn2 = pt.sum(pt.pow(ifftFilterArray,2),1)
    
        sumEstSumAn2 = pt.sum(pt.sum(EstSumAn2,2),2)
        roll_t1 = ifftFilterArray*self.roll_1(ifftFilterArray,1)
        roll_t2 = ifftFilterArray*self.roll_1(ifftFilterArray,2)
        roll_t3 = ifftFilterArray*self.roll_1(ifftFilterArray,3)
        rolling_mult = roll_t1+roll_t2+roll_t3
        EstSumAiAj = pt.sum(rolling_mult,1)/2
        sumEstSumAiAj = pt.sum(pt.sum(EstSumAiAj,2),2)
    
        EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2+4*noisePower*sumEstSumAiAj
        tau = pt.sqrt(EstNoiseEnergy2/2)
        EstNoiseEnergy = tau*math.sqrt(self.pi/2)
        EstNoiseEnergySigma = pt.sqrt( (2-self.pi/2)*pt.pow(tau,2))
    
        T = (EstNoiseEnergy + self.k*EstNoiseEnergySigma)/1.7
        
        T_exp = pt.cat(rows*cols*[T.unsqueeze(2)],dim=2).view(-1,self.norient,rows,cols)
        AnAll = pt.sum(sumAn_ThisOrient,1)
        array_of_zeros_energy = pt.zeros(Energy.size(),dtype=pt.float64)
        if self.cuda_computation:
            array_of_zeros_energy = array_of_zeros_energy.cuda()
            
        EnergyAll = pt.sum(pt.where((Energy - T_exp)<0.0, array_of_zeros_energy,Energy - T_exp ),1)
        ResultPC = EnergyAll/AnAll
        
        return ResultPC
    
class FSIM(FSIM_base):
    '''
    Note, the input is expected to be from 0 to 255
    '''

    def __init__(self):
        super().__init__()
        
    def forward(self,imgr,imgd):
        if imgr.is_cuda:
            self.set_arrays_to_cuda()
            
        I1,Q1,Y1 = self.process_image_channels(imgr)
        I2,Q2,Y2 = self.process_image_channels(imgd)
        PC1 = self.phasecong2(Y1)
        PC2 = self.phasecong2(Y2)
        
        PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)
        
        gradientSimMatrix = self.calculate_gradient_sim(gradientMap1,gradientMap2)
        gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        FSIM = self.calculate_fsim(gradientSimMatrix,PCSimMatrix,PCm)

        return FSIM.mean()

class FSIMc(FSIM_base, nn.Module):
    '''
    Note, the input is expected to be from 0 to 255
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self,imgr,imgd):
        if imgr.is_cuda:
            self.set_arrays_to_cuda()
            
            
        I1,Q1,Y1 = self.process_image_channels(imgr)
        I2,Q2,Y2 = self.process_image_channels(imgd)
        PC1 = self.phasecong2(Y1)
        PC2 = self.phasecong2(Y2)
        
        PCSimMatrix,PCm = self.calculate_phase_score(PC1,PC2)
        gradientMap1 = self.calculate_gradient_map(Y1)
        gradientMap2 = self.calculate_gradient_map(Y2)
        
        gradientSimMatrix = self.calculate_gradient_sim(gradientMap1,gradientMap2)
        gradientSimMatrix= gradientSimMatrix.view(PCSimMatrix.size())
        FSIMc = self.calculate_fsimc(I1.squeeze(),Q1.squeeze(),I2.squeeze(),Q2.squeeze(),gradientSimMatrix,PCSimMatrix,PCm)

        return FSIMc.mean()



import time
import os
import csv
import numpy as np
import torch
import lpips  
import torchio as tio
from tqdm import tqdm
import warnings

# (Assicurati che la classe FSIM() e le funzioni calculate_3d_psnr_volume, 
# calculate_3d_full_ssim e calculate_3d_ncc_volume siano state importate)

@torch._dynamo.disable
def valuta_modello_quantitativo_completo(G, dataset, device, num_campioni=10, num_slices_percettive=10, 
                                         direction='A2B', patch_size=(128, 128, 128), 
                                         patch_overlap=(64, 64, 64), batch_size=4, save_path=None):
    """
    Calcola MAE, PSNR, SSIM, NCC, LPIPS e FSIM (Media e Deviazione Standard).
    Versione PATCH-BASED: usa GridSampler e GridAggregator per evitare OutOfMemory 
    e crash legati alle dimensioni fisiche dei tensori full-volume.
    I risultati possono essere esportati in formato CSV tramite il parametro 'save_path'.
    """
    G.eval()

    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    
    # =========
    # INIZIALIZZAZIONE MODELLI PERCETTIVI
    # =========
    print("Caricamento modelli percettivi (LPIPS e FSIM) in VRAM...")
    
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_model.eval()
    
    fsim_model = FSIM().to(device)
    fsim_model.eval()
    
    mae_list, psnr_list, ssim_list, ncc_list = [], [], [], []
    lpips_list, fsim_list = [], []
    
    num_campioni = min(num_campioni, len(dataset))
    print(f"\nAvvio valutazione completa Patch-Based su {num_campioni} volumi (Direzione: {direction})...")
    start_time = time.time()
    
    with torch.inference_mode():
        for i in tqdm(range(num_campioni), desc="Progresso Valutazione", unit="vol"):
            # Estraiamo il subject originale
            subject = dataset.dataset[i] if hasattr(dataset, 'dataset') else dataset[i]
            
            # =========
            # INFERENZA PATCH-BASED (GridSampler + Aggregator)
            # =========
            grid_sampler = tio.inference.GridSampler(
                subject,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='average')

            for patches_batch in patch_loader:
                locations = patches_batch[tio.LOCATION]
                if direction == 'A2B':
                    real_in_patch = patches_batch['A'][tio.DATA].to(device)
                else:
                    real_in_patch = patches_batch['B'][tio.DATA].to(device)
                
                # --- CORREZIONE ASSI IN INFERENZA ---
                real_in_dhw = real_in_patch.permute(0, 1, 4, 3, 2)
                fake_patch_dhw = G(real_in_dhw)
                fake_patch_whd = fake_patch_dhw.permute(0, 1, 4, 3, 2)
                
                aggregator.add_batch(fake_patch_whd, locations)
                
            fake_out_whd = aggregator.get_output_tensor().unsqueeze(0).to(device)
            if direction == 'A2B':
                real_out_whd = subject['B'][tio.DATA].unsqueeze(0).to(device)
            else:
                real_out_whd = subject['A'][tio.DATA].unsqueeze(0).to(device)
            
            # --- CORREZIONE: Riportiamo a [D, H, W] per far funzionare bene le metriche LPIPS/SSIM ---
            fake_out = fake_out_whd.permute(0, 1, 4, 3, 2)
            real_out = real_out_whd.permute(0, 1, 4, 3, 2)
            
            # =========
            # 1. METRICHE VOLUMETRICHE GLOBALI (Array Numpy 3D)
            # =========
            real_out_np = real_out.squeeze().cpu().numpy()
            fake_out_np = fake_out.squeeze().cpu().numpy()
            
            mae_list.append(np.mean(np.abs(real_out_np - fake_out_np)))
            
            p_val = calculate_3d_psnr_volume(real_out_np, fake_out_np)
            psnr_list.append(p_val)
            
            s_mean, _, _ = calculate_3d_full_ssim(real_out_np, fake_out_np)
            ssim_list.append(s_mean)
            
            ncc_mean, _, _ = calculate_3d_ncc_volume(fake_out_np, real_out_np)
            ncc_list.append(ncc_mean)
            
            # =========
            # 2. METRICHE PERCETTIVE 2D BATCH (Slice assiali centrali)
            # =========
            D_dim = real_out.shape[2]
            start_slice = (D_dim // 2) - (num_slices_percettive // 2)
            end_slice = start_slice + num_slices_percettive
            
            real_slices = real_out[0, 0, start_slice:end_slice, :, :].unsqueeze(1)
            fake_slices = fake_out[0, 0, start_slice:end_slice, :, :].unsqueeze(1)
            
            # --- Calcolo LPIPS ---
            real_rgb = real_slices.repeat(1, 3, 1, 1) 
            fake_rgb = fake_slices.repeat(1, 3, 1, 1)
            
            lpips_score = lpips_model(fake_rgb, real_rgb).mean().item()
            lpips_list.append(lpips_score)
            
            # --- Calcolo FSIM ---
            real_255 = ((real_slices + 1.0) / 2.0) * 255.0
            fake_255 = ((fake_slices + 1.0) / 2.0) * 255.0
            
            fsim_score = fsim_model(fake_255, real_255).item()
            fsim_list.append(fsim_score)

    # =========
    # 3. CALCOLO MEDIE E DEVIAZIONI STANDARD
    # =========
    m_mae, m_psnr, m_ssim = np.mean(mae_list), np.mean(psnr_list), np.mean(ssim_list)
    m_ncc, m_lpips, m_fsim = np.mean(ncc_list), np.mean(lpips_list), np.mean(fsim_list)
    
    s_mae, s_psnr, s_ssim = np.std(mae_list), np.std(psnr_list), np.std(ssim_list)
    s_ncc, s_lpips, s_fsim = np.std(ncc_list), np.std(lpips_list), np.std(fsim_list)

    # =========
    # 4. SALVATAGGIO IN CSV (Se richiesto)
    # =========
    if save_path is not None:
        # Crea la cartella se non esiste
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, mode='w', newline='', encoding='utf-8') as file_csv:
            writer = csv.writer(file_csv)
            
            # Header per i singoli volumi
            writer.writerow(['Volume_Idx', 'MAE', 'PSNR', 'SSIM', 'NCC', 'LPIPS', 'FSIM'])
            
            # Dati per ciascun volume
            for i in range(num_campioni):
                writer.writerow([
                    i, 
                    f"{mae_list[i]:.6f}", 
                    f"{psnr_list[i]:.6f}", 
                    f"{ssim_list[i]:.6f}", 
                    f"{ncc_list[i]:.6f}", 
                    f"{lpips_list[i]:.6f}", 
                    f"{fsim_list[i]:.6f}"
                ])
            
            # Spazio di separazione
            writer.writerow([])
            
            # Header per le statistiche globali
            writer.writerow(['Statistica', 'MAE', 'PSNR', 'SSIM', 'NCC', 'LPIPS', 'FSIM'])
            
            # Riga Media
            writer.writerow([
                'Media', 
                f"{m_mae:.6f}", 
                f"{m_psnr:.6f}", 
                f"{m_ssim:.6f}", 
                f"{m_ncc:.6f}", 
                f"{m_lpips:.6f}", 
                f"{m_fsim:.6f}"
            ])
            
            # Riga Deviazione Standard
            writer.writerow([
                'Dev_Std', 
                f"{s_mae:.6f}", 
                f"{s_psnr:.6f}", 
                f"{s_ssim:.6f}", 
                f"{s_ncc:.6f}", 
                f"{s_lpips:.6f}", 
                f"{s_fsim:.6f}"
            ])
        print(f"\n[INFO] Risultati esportati con successo in: {save_path}")

    # =========
    # 5. STAMPA STATISTICHE FINALI
    # =========
    tempo_totale = time.time() - start_time
    minuti, secondi = int(tempo_totale // 60), int(tempo_totale % 60)

    print(f"\n=== RISULTATI MEDI CLINICI ({direction}) ===")
    print(f"Tempo di calcolo: {minuti}m {secondi}s")
    print("-" * 65)
    print(f"MAE Medio:    {m_mae:.4f} ± {s_mae:.4f}  (↓ Più basso è meglio)")
    print(f"PSNR Medio:   {m_psnr:.2f} ± {s_psnr:.2f} dB (↑ Più alto è meglio)")
    print(f"SSIM Medio:   {m_ssim:.4f} ± {s_ssim:.4f}  (↑ Vicino a 1 - Similarità Strutturale)")
    print(f"NCC Medio:    {m_ncc:.4f} ± {s_ncc:.4f}  (↑ Vicino a 1 - Invariante alla luminosità)")
    print("-" * 65)
    print(f"LPIPS Medio:  {m_lpips:.4f} ± {s_lpips:.4f}  (↓ Vicino a 0 - Distanza Percettiva)")
    print(f"FSIM Medio:   {m_fsim:.4f} ± {s_fsim:.4f}  (↑ Vicino a 1 - Nitidezza e Phase Congruency)")
    
    return (m_mae, m_psnr, m_ssim, m_ncc, m_lpips, m_fsim,
            s_mae, s_psnr, s_ssim, s_ncc, s_lpips, s_fsim)



import matplotlib.pyplot as plt
import numpy as np

def plot_risultati_finali_barre(risultati_tuple, save_path='/kaggle/working/risultati_finali_test.png'):
    """
    Genera un Bar Chart elegante per le 6 metriche cliniche calcolate sul Test Set.
    Mostra la Media come altezza della barra e la Deviazione Standard come 'error bar'.
    """
    # 1. Spacchettiamo la tupla
    medie = risultati_tuple[:6]  # I primi 6 elementi sono le medie
    stds = risultati_tuple[6:]   # Gli ultimi 6 elementi sono le std
    
    # 2. Definiamo i metadati per i grafici
    titoli = [
        'MAE (↓ Più basso è meglio)',
        'PSNR [dB] (↑ Più alto è meglio)',
        'SSIM (↑ Più vicino a 1)',
        'NCC (↑ Più vicino a 1)',
        'LPIPS (↓ Più vicino a 0)',
        'FSIM (↑ Più vicino a 1)'
    ]
    colori = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    
    # Creiamo una griglia 2x3 per i subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Risultati Finali sul Dataset di Test', fontsize=24, fontweight='bold', y=0.98)
    
    for i in range(6):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        
        # Disegniamo la singola barra con l'errore (yerr)
        bar = ax.bar(['Test Set'], [medie[i]], yerr=[stds[i]], 
                     color=colori[i], alpha=0.85, edgecolor='black', linewidth=1.5, 
                     capsize=12, ecolor='black', error_kw={'elinewidth': 2})
        
        # --- CORREZIONE QUI ---
        # Definiamo solo il puro specificatore di formato
        formato = ".2f" if i == 1 else ".4f" 
        testo_valore = f"{medie[i]:{formato}}\n(±{stds[i]:{formato}})"
        # ----------------------
        
        # Scriviamo il valore esatto sopra la barra (e sopra la barra d'errore)
        ax.text(0, medie[i] + stds[i] + (medie[i]*0.05), testo_valore, 
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='#333333')
        
        # Estetica del subplot
        ax.set_title(titoli[i], fontsize=15, pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True) # Mette la griglia dietro la barra
        
        # Aggiustiamo il limite Y per fare spazio al testo in alto
        ax.set_ylim(0, (medie[i] + stds[i]) * 1.25)
        
        # Rimuoviamo i bordi brutti
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Rendiamo l'etichetta dell'asse X un po' più grande
        ax.tick_params(axis='x', labelsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvataggio
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Bar chart salvato con successo in: {save_path}")
        
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

def plot_confronto_modelli(lista_risultati, 
                           nomi_modelli=None, 
                           save_path='/kaggle/working/confronto_modelli.png'):
    """
    Genera una griglia di Bar Chart per confrontare 6 metriche cliniche tra un numero N arbitrario di modelli.
    'lista_risultati' deve essere una lista di tuple, dove ogni tupla contiene 12 valori (6 medie + 6 std).
    """
    num_modelli = len(lista_risultati)
    
    # Setup dinamico dei nomi dei modelli se non forniti o errati
    if nomi_modelli is None or len(nomi_modelli) != num_modelli:
        nomi_modelli = [f'Model {i+1}' for i in range(num_modelli)]
        
    # Estraiamo medie e std per ogni modello dinamicamente
    medie_modelli = [ris[:6] for ris in lista_risultati]
    stds_modelli = [ris[6:] for ris in lista_risultati]
    
    # Metadati per i grafici (Tradotti in Inglese)
    titoli = [
        ('MAE', '↓ Lower is better (Mean Absolute Error)'),
        ('PSNR [dB]', '↑ Higher is better (Signal-to-Noise Ratio)'),
        ('SSIM', '↑ Closer to 1 is better (Structural Similarity)'),
        ('NCC', '↑ Closer to 1 is better (Normalized Cross-Correlation)'),
        ('LPIPS', '↓ Closer to 0 is better (Perceptual Distance)'),
        ('FSIM', '↑ Closer to 1 is better (Feature Similarity)')
    ]
    
    # Generazione dinamica dei colori (supporta comodamente fino a 10/20 modelli)
    cmap = plt.get_cmap('tab10')
    colori_modelli = [cmap(i % 10) for i in range(num_modelli)]
    
    # Creiamo la griglia 2x3
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle(f'Final Metrics Comparison across {num_modelli} Models', fontsize=28, fontweight='bold', y=0.98)
    
    # Adattiamo la grandezza del testo se ci sono troppi modelli per evitare sovrapposizioni
    dimensione_font_valori = 13 if num_modelli <= 3 else 10
    
    for i in range(6):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        
        # Estraiamo i valori per la singola metrica 'i' da tutti i modelli N
        medie_metrica = [medie_modelli[m][i] for m in range(num_modelli)]
        stds_metrica = [stds_modelli[m][i] for m in range(num_modelli)]
        
        # Disegniamo le N barre nel subplot
        bars = ax.bar(nomi_modelli, medie_metrica, yerr=stds_metrica, 
                      color=colori_modelli, alpha=0.85, edgecolor='black', linewidth=1.5, 
                      capsize=10, ecolor='black', error_kw={'elinewidth': 2})
        
        # Specificatore di formato
        formato = ".2f" if i == 1 else ".4f" 
        
        # Aggiungiamo i testi sopra ogni singola barra
        for j, bar in enumerate(bars):
            valore = medie_metrica[j]
            errore = stds_metrica[j]
            testo_valore = f"{valore:{formato}}\n(±{errore:{formato}})"
            
            # Calcoliamo la posizione Y del testo per non sovrapporsi all'error bar
            altezza_testo = valore + errore + (max(medie_metrica) * 0.05)
            
            ax.text(bar.get_x() + bar.get_width() / 2, altezza_testo, testo_valore, 
                    ha='center', va='bottom', fontsize=dimensione_font_valori, 
                    fontweight='bold', color='#333333')
        
        # ========
        # ESTETICA TITOLO E ASSE Y
        # ========
        nome_metrica, sottotitolo = titoli[i]
        
        # Riquadro (Badge)
        ax.set_title(nome_metrica, fontsize=22, fontweight='bold', pad=40,
                     bbox=dict(facecolor='#f0f4f8', edgecolor='#cbd5e1', boxstyle='round,pad=0.4', linewidth=1.5))
        
        # Sottotitolo
        ax.text(0.5, 1.01, sottotitolo, transform=ax.transAxes, ha='center', va='bottom', 
                fontsize=13, fontstyle='italic', color='#555555')
        
        # Label sull'asse Y
        ax.set_ylabel(nome_metrica, fontsize=14, fontweight='bold', color='#444444')
        
        # Griglia e assi
        ax.grid(axis='y', linestyle='-', alpha=0.2, color='black')
        ax.set_axisbelow(True)
        
        # Aggiustiamo dinamicamente il limite Y
        max_y = max([m + s for m, s in zip(medie_metrica, stds_metrica)])
        ax.set_ylim(0, max_y * 1.35) 
        
        # Rimozione bordi
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Riduciamo leggermente i label X se i modelli sono tanti
        ax.tick_params(axis='x', labelsize=max(9, 16 - num_modelli), pad=10)

    # Regoliamo gli spazi tra i subplot
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.subplots_adjust(hspace=0.40)
    
    # Salvataggio
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Comparative bar chart successfully saved at: {save_path}")
        
    plt.show()