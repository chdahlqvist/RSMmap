"""
Set of functions used by the PyRSM class to compute detection maps, optimize the parameters
of the RSM algorithm and PSF-subtraction techniques via the auto-RSM and auto-S/N frameworks,
generate contrast curves, and characterise detected astrophysical signals.
"""
__author__ = 'Carl-Henrik Dahlqvist'

from scipy.interpolate import Rbf
import pandas as pd
import numpy.linalg as la
from vip_hci.var import get_annulus_segments, frame_center,prepare_matrix,cube_filter_highpass
import numpy as np
from vip_hci.preproc import cube_derotate, cube_collapse, check_pa_vector,check_scal_vector
from vip_hci.preproc.derotation import _find_indices_adi
from vip_hci.preproc.rescaling import _find_indices_sdi
import scipy as sp
from multiprocessing import cpu_count
from vip_hci.conf.utils_conf import pool_map, iterable
from vip_hci.pca.svd import get_eigenvectors
from vip_hci.llsg.llsg import _patch_rlrps
from vip_hci.preproc import cube_rescaling_wavelengths as scwave
from sklearn.decomposition import NMF
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize

def check_delta_sep(scale_list,delta_sep,minradius,fwhm,c):
    wl = np.asarray(scale_list)
    wl_ref = wl[len(wl)//2]
    sep_lft = (wl_ref - wl) / wl_ref * ((minradius + fwhm * delta_sep) / fwhm)
    sep_rgt = (wl - wl_ref) / wl_ref * ((minradius - fwhm * delta_sep) / fwhm)
    map_lft = sep_lft >= delta_sep
    map_rgt = sep_rgt >= delta_sep
    indices = np.nonzero(map_lft | map_rgt)[0]

    if indices.size == 0:
        raise RuntimeError(("No frames left after radial motion threshold for cube {}. Try "
                           "decreasing the value of `delta_sep`").format(c))   
                                        
def rot_scale(step,cube,cube_scaled,angle_list,scale_list, imlib, interpolation):
    
    """
    Function used to rescale the frames when relying on ADI+SDI before the computation the reference PSF
    (step='ini') and rescale and derotate the frames to generate the cube of residuals used by the RSM 
    algorithm (step='fin').
        
        Parameters
        ----------

        step: str
            'ini' before the reference PSF computation and 'fin' after PSF subtraction.
        cube: numpy ndarray, 3d or 4d
            Original cube
        cube_scaled: numpy ndarray, 3d
            Cube of residuals to be rescaled and derotated (None for the step='ini')
        angle_list : numpy ndarray, 1d
            Parallactic angles for each frame of the ADI sequences. 
        scale_list: numpy ndarray, 1d, optional
            Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
            scaling factors are the central channel wavelength divided by the
            shortest wavelength in the cube (more thorough approaches can be used
            to get the scaling factors). This scaling factors are used to re-scale
            the spectral channels and align the speckles. Default is None
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    """
    
    if cube.ndim == 4:
        
        z, n, y_in, x_in = cube.shape
        scale_list = check_scal_vector(scale_list)
        
        if step=='ini':
        # rescaled cube, aligning speckles for SDI
            for i in range(n):
                if i==0:
                    fin_cube = scwave(cube[:, i, :, :], scale_list,
                                      imlib=imlib, interpolation=interpolation)[0]
                    fin_pa=np.repeat(angle_list[i],z)
                    fin_scale=scale_list
                else: 
                    
                    fin_cube = np.append(fin_cube,scwave(cube[:, i, :, :], scale_list,
                                      imlib=imlib, interpolation=interpolation)[0],axis=0)
                    fin_pa=np.append(fin_pa,np.repeat(angle_list[i],z),axis=0)
                    fin_scale=np.append(fin_scale,scale_list,axis=0)
                    
            return fin_cube,fin_pa,fin_scale
                

        elif step=='fin':  
            
                cube_fin=np.zeros((n,y_in, x_in))
                                
                cube_rescaled = scwave(cube_scaled, scale_list, 
                                full_output=True, inverse=True,
                                y_in=y_in, x_in=x_in, imlib=imlib,
                                interpolation=interpolation)[0]
                
                cube_derotated=cube_derotate(cube_rescaled,angle_list, interpolation=interpolation,imlib=imlib)

                 
                for i in range(n):
                    
                    cube_fin[i]=np.mean(cube_derotated[(i*z):((i+1)*z),:,:],axis=0)
                                    
                return cube_fin
                
    if cube.ndim == 3:
        
        if step=='ini':
     
            return cube,angle_list,None
        
        elif step=='fin':    

            cube_derotated=cube_derotate(cube_scaled,angle_list, interpolation=interpolation,imlib=imlib)
            
            return cube_derotated

def _define_annuli(angle_list, ann, n_annuli, fwhm, radius_int, annulus_width,
                   delta_rot, n_segments, verbose, strict=False):
    """ Function that defines the annuli geometry using the input parameters.
    Returns the parallactic angle threshold, the inner radius and the annulus
    center for each annulus. Function taken from python package VIP_HCI (Gomez et al. 2017).
    """
    if ann == n_annuli - 1:
        inner_radius = radius_int + (ann * annulus_width - 1)
    else:
        inner_radius = radius_int + ann * annulus_width
    ann_center = inner_radius + (annulus_width / 2)
    pa_threshold = np.rad2deg(2 * np.arctan(delta_rot * fwhm / (2 * ann_center)))
    mid_range = np.abs(np.amax(angle_list) - np.amin(angle_list)) / 2
    if pa_threshold >= mid_range - mid_range * 0.1:
        pa_threshold = float(mid_range - mid_range * 0.1)

    return pa_threshold, inner_radius, ann_center

    
    
def remove_outliers(time_s, range_sel, k=5, t0=3):
    """
    Hampel Filter to remove potential outliers in the set of selected parameters 
    for the annular mode of the auto-RSM framework
    """
    vals=pd.DataFrame(data=time_s[range_sel])
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=threshold[outlier_idx]
    return(vals.to_numpy().reshape(-1))
       
def interpolation(time_s,range_sel): 
    
    """
    Interpolation algorithm for the RSM parameters 
    for the annular mode of the auto-RSM framework
    """
    
    time_series=time_s.copy()
    time_series[range_sel]=remove_outliers(time_series,range_sel)
    fit = Rbf(range_sel,time_s[range_sel])
    inter_point = np.linspace(range_sel[0],range_sel[-1]+1, num=(range_sel[-1]-range_sel[0]+1), endpoint=True)
    return fit(inter_point)

def poly_fit(time_s,range_sel,poly_n):
    
    """
    Smoothing procedure for the computation of the final radial thresholds
    which are subtracted from the final RSM detection map in the final step
    of the auto-RSM framework
    """
    
    time_series=time_s.copy()
    time_series[range_sel]=remove_outliers(time_series,range_sel)
    fit_p=np.poly1d(np.polyfit(range_sel,time_series[range_sel], poly_n))
    time_series=fit_p(range(len(time_series)))
    return time_series

def get_time_series(mcube,ann_center):
    
        """
        Function defining and ordering (anti-clockwise) the pixels composing
        an annulus at a radial distance of ann_center for an ADI sequence mcube
        """
        if mcube.ndim == 4:
            indices = get_annulus_segments(mcube[0,0,:,:], ann_center,1,4,90)
        else:
            indices = get_annulus_segments(mcube[0], ann_center,1,4,90)

        tempind=np.vstack((indices[0][0],indices[0][1]))
        ind = np.lexsort((tempind[0], tempind[1]))

        indicesy=tempind[0,ind[::-1]]
        indicesx=tempind[1,ind[::-1]] 

        tempind=np.vstack((indices[1][0],indices[1][1]))
        ind = np.lexsort((-tempind[0], tempind[1]))

        indicesy=np.hstack((indicesy,tempind[0,ind[::-1]]))
        indicesx=np.hstack((indicesx,tempind[1,ind[::-1]]))

        tempind=np.vstack((indices[2][0],indices[2][1]))
        ind = np.lexsort((tempind[0], tempind[1]))

        indicesy=np.hstack((indicesy,tempind[0,ind]))
        indicesx=np.hstack((indicesx,tempind[1,ind])) 

        tempind=np.vstack((indices[3][0],indices[3][1]))
        ind = np.lexsort((-tempind[0], tempind[1]))

        indicesy=np.hstack((indicesy,tempind[0,ind]))
        indicesx=np.hstack((indicesx,tempind[1,ind]))
            
        return indicesy,indicesx
 
def perturb(frame,model_matrix,numbasis,evals_matrix, evecs_matrix, KL_basis_matrix,sci_mean_sub_matrix,refs_mean_sub_matrix, angle_list, fwhm, pa_threshold, ann_center):
    

    """
    Function allowing the estimation of the PSF forward model when relying on KLIP
    for the computation of the speckle field. The code is based on the PyKLIP library
     considering only the ADI case with a singlle number of principal components considered.
    For more details about the code, consider the PyKLIP library or the originall articles
    (Pueyo, L. 2016, ApJ, 824, 117 or
     Ruffio, J.-B., Macintosh, B., Wang, J. J., & Pueyo, L. 2017, ApJ, 842)
    """
    
    #Selection of the reference library based on the given parralactic angle threshold

    if pa_threshold != 0:
        indices_left = _find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False)

        models_ref = model_matrix[indices_left]

    else:
        models_ref = model_matrix


    #Computation of the self-subtraction and over-subtraction for the current frame
    
    model_sci = model_matrix[frame]  
    KL_basis=KL_basis_matrix[frame]
    sci_mean_sub=sci_mean_sub_matrix[frame]
    refs_mean_sub=refs_mean_sub_matrix[frame]
    evals=evals_matrix[frame]
    evecs=evecs_matrix[frame]

    max_basis = KL_basis.shape[0]
    N_pix = KL_basis.shape[1]

    models_mean_sub = models_ref - np.nanmean(models_ref, axis=1)[:,None] 
    models_mean_sub[np.where(np.isnan(models_mean_sub))] = 0
    
    model_sci_mean_sub = model_sci- np.nanmean(model_sci)
    model_sci_mean_sub[np.where(np.isnan(model_sci_mean_sub))] = 0
    model_sci_mean_sub_rows = np.reshape(model_sci_mean_sub,(1,N_pix))
    sci_mean_sub_rows = np.reshape(sci_mean_sub,(1,N_pix))
    
    delta_KL = np.zeros([max_basis, N_pix])

    models_mean_sub_X_refs_mean_sub_T = models_mean_sub.dot(refs_mean_sub.transpose())

    for k in range(max_basis):
        Zk = np.reshape(KL_basis[k,:],(1,KL_basis[k,:].size))
        Vk = (evecs[:,k])[:,None]


        diagVk_X_models_mean_sub_X_refs_mean_sub_T = (Vk.T).dot(models_mean_sub_X_refs_mean_sub_T)
        models_mean_sub_X_refs_mean_sub_T_X_Vk = models_mean_sub_X_refs_mean_sub_T.dot(Vk)
        DeltaZk = -(1/(2*np.sqrt(evals[k])))*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vk) + ((Vk.T).dot(models_mean_sub_X_refs_mean_sub_T_X_Vk))).dot(Zk)+(Vk.T).dot(models_mean_sub)


        for j in range(k):
            Zj = KL_basis[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + ((Vj.T).dot(models_mean_sub_X_refs_mean_sub_T_X_Vk))).dot(Zj)
        for j in range(k+1, max_basis):
            Zj = KL_basis[j, :][None,:]
            Vj = evecs[:, j][:,None]
            DeltaZk += np.sqrt(evals[j])/(evals[k]-evals[j])*(diagVk_X_models_mean_sub_X_refs_mean_sub_T.dot(Vj) + ((Vj.T).dot(models_mean_sub_X_refs_mean_sub_T_X_Vk))).dot(Zj)

        delta_KL[k] = DeltaZk/np.sqrt(evals[k])
        
    oversubtraction_inner_products = np.dot(model_sci_mean_sub_rows, KL_basis.T)  
    
    selfsubtraction_1_inner_products = np.dot(sci_mean_sub_rows, delta_KL.T)
    selfsubtraction_2_inner_products = np.dot(sci_mean_sub_rows, KL_basis.T)

    oversubtraction_inner_products[max_basis::] = 0
    klipped_oversub = np.dot(oversubtraction_inner_products, KL_basis)
    
    selfsubtraction_1_inner_products[0,max_basis::] = 0
    selfsubtraction_2_inner_products[0,max_basis::] = 0
    klipped_selfsub = np.dot(selfsubtraction_1_inner_products, KL_basis) + \
                          np.dot(selfsubtraction_2_inner_products, delta_KL)

    return model_sci[None,:] - klipped_oversub - klipped_selfsub   
        



def KLIP(cube, angle_list, nann=None, local=False,radius_int=None,radius_out=None, fwhm=4, asize=2, n_segments=1,delta_rot=1, ncomp=1,min_frames_lib=2, max_frames_lib=200,imlib='opencv',nframes=None, interpolation='lanczos4', collapse='median',full_output=False, verbose=1):

    """
    Function allowing the estimation of the cube of residuals after
    the subtraction of the speckle field modeled via the KLIP framework 
    """
    
    array = cube
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')

    n, y, _ = array.shape
    
    angle_list = check_pa_vector(angle_list)
    
    if asize is None:
        annulus_width = int(np.ceil(2 * fwhm))
    elif isinstance(asize, int):
        annulus_width = asize
        
    # Annulus parametrization 
    

    if local==True:
            if nann> 2*annulus_width:
                n_annuli = 5
                radius_int=(nann//annulus_width-2)*annulus_width 
            else:
                n_annuli = 4 
                radius_int=(nann//annulus_width-1)*annulus_width
    else:
        if radius_int%asize>int(asize/2):
            radius_int=(radius_int//asize)*asize
        elif radius_int>=asize:
            radius_int=(radius_int//asize-1)*asize
        else:
            radius_int=0
            
        if radius_out is not None:
            
            if radius_out%asize>int(asize/2):
                radius_out=(radius_out//asize+2)*asize
            else:
                radius_out=(radius_out//asize+1)*asize
            
            n_annuli = int((radius_out - radius_int) / asize)
            
        elif radius_out==None:
            n_annuli = int((y / 2 - radius_int) / asize)
        
        elif radius_out>int(y / 2): 
            
            n_annuli = int((y / 2 - radius_int) / asize)
        
            
    # Definition of the number of segment for the diifferent annuli

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  
        n_segments.append(3)  
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = '# annuli = {}, Ann width = {}, FWHM = {:.3f}'
        print(msg.format(n_annuli, asize, fwhm))
        print('PCA per annulus (or annular sectors):')


    # Definition of the annuli and the corresmponding parralactic angle threshold 
    
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):
        if isinstance(ncomp, list) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msge = 'If ncomp is a list, it must match the number of annuli'
                raise TypeError(msge)
        else:
            ncompann = ncomp

        
        inner_radius = radius_int + ann * annulus_width
        n_segments_ann = n_segments[ann]


        if verbose:
            print('{} : in_rad={}, n_segm={}'.format(ann+1, inner_radius,
                                                     n_segments_ann))


        theta_init = 90
        res_ann_par = _define_annuli(angle_list, ann, int((y / 2 - radius_int) / asize), fwhm,radius_int, annulus_width, delta_rot,n_segments_ann, verbose)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius, annulus_width,n_segments_ann,theta_init)
        
        # Computation of the speckle field for the different frames and estimation of the cube of residuals
        
        for j in range(n_segments_ann):

            for k in range(array.shape[0]):
                
                res =KLIP_patch(k,array[:, indices[j][0], indices[j][1]], ncompann, angle_list, fwhm, pa_thr, ann_center,nframes=nframes)
                cube_out[k,indices[j][0], indices[j][1]] = res[3]


    # Cube is derotated according to the parallactic angle and collapsed
    
    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame
    
def KLIP_patch(frame, matrix, numbasis, angle_list, fwhm, pa_threshold, ann_center,nframes=None):

    """          
    Function allowing the computation via KLIP of the speckle field for a 
    given sub-region of the original ADI sequence. Code inspired by the PyKLIP librabry
    """
    
    max_frames_lib=200
    
    if pa_threshold != 0:
        if ann_center > fwhm*20:
            indices_left = _find_indices_adi(angle_list,frame,pa_threshold, truncate=True,max_frames=max_frames_lib)
        else:
            indices_left = _find_indices_adi(angle_list, frame,pa_threshold, truncate=False,nframes=nframes)

        refs = matrix[indices_left]
        
    else:
        refs = matrix

    sci = matrix[frame]
    sci_mean_sub = sci - np.nanmean(sci)
    #sci_mean_sub[np.where(np.isnan(sci_mean_sub))] = 0
    refs_mean_sub = refs- np.nanmean(refs, axis=1)[:, None]
    #refs_mean_sub[np.where(np.isnan(refs_mean_sub))] = 0

    # Covariance matrix definition
    covar_psfs = np.cov(refs_mean_sub)
    covar_psfs *= (np.size(sci)-1)

    tot_basis = covar_psfs.shape[0]

    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    max_basis = np.max(numbasis) + 1

    #Computation of the eigenvectors/values of the covariance matrix
    evals, evecs = la.eigh(covar_psfs)
    evals = np.copy(evals[int(tot_basis-max_basis):int(tot_basis)])
    evecs = np.copy(evecs[:,int(tot_basis-max_basis):int(tot_basis)])
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1])

    # Computation of the principal components
    
    KL_basis = np.dot(refs_mean_sub.T,evecs)
    KL_basis = KL_basis * (1. / np.sqrt(evals))[None,:]
    KL_basis = KL_basis.T 

    N_pix = np.size(sci_mean_sub)
    sci_rows = np.reshape(sci_mean_sub, (1,N_pix))
    
    inner_products = np.dot(sci_rows, KL_basis.T)
    inner_products[0,int(max_basis)::]=0

    #Projection of the science image on the selected prinicpal component
    #to generate the speckle field model

    klip_reconstruction = np.dot(inner_products, KL_basis)

    # Subtraction of the speckle field model from the riginal science image
    #to obtain the residual frame
    
    sub_img_rows = sci_rows - klip_reconstruction 

    return evals,evecs,KL_basis,np.reshape(sub_img_rows, (N_pix)),refs_mean_sub,sci_mean_sub


def LOCI_FM(cube, psf, ann_center, angle_list,scale_list, asize,fwhm, Tol,delta_rot,delta_sep):


    """
    Computation of the optimal factors weigthing the linear combination of reference
    frames used to obtain the modeled speckle field for each frame and allowing the 
    determination of the forward modeled PSF. Estimation of the cube 
    of residuals based on the modeled speckle field.
    """


    cube_res = np.zeros_like(cube)
    ceny, cenx = frame_center(cube[0])
    radius_int=ann_center-int(1.5*asize)
    if radius_int<=0:
        radius_int=1
            
    for ann in range(3):
        n_segments_ann = 1
        inner_radius_ann = radius_int + ann*asize
        pa_threshold = _define_annuli(angle_list, ann, 3, asize,
                                      radius_int, asize, delta_rot,
                                      n_segments_ann, verbose=False)[0]
        
        indices = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)
        ind_opt = get_annulus_segments(cube[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann,
                                       optim_scale_fact=2)
        
        ayxyx = [inner_radius_ann,pa_threshold, indices[0][0], indices[0][1],
                   ind_opt[0][0], ind_opt[0][1]]
                
        matrix_res, ind_ref, coef, yy, xx = _leastsq_patch(ayxyx,
                         angle_list,scale_list,fwhm,cube,ann_center,'manhattan', 100,delta_sep,
                         'lstsq', Tol,formod=True,psf=psf)
        
        if ann==1:
            ind_ref_list=ind_ref
            coef_list=coef
        
        cube_res[:, yy, xx] = matrix_res
    
    return cube_res, ind_ref_list,coef_list



def annular_NMF(cube, angle_list, nann=None, local=False, fwhm=4, asize=2, n_segments=1, ncomp=20,imlib='opencv', interpolation='lanczos4', collapse='median',max_iter=100,
        random_state=None,full_output=False, verbose=False):
   
    """
    Function allowing the estimation of the cube of residuals after
    the subtraction of the speckle field modeled via the NMF framework.
    This codes is an adaptation of the VIP NMF function to the case of annular
    computation of the modeled speckle fields
    (only full-frame estimation in Gonzalez et al. AJ, 154:7,2017)
    """
    
    array = cube
    if array.ndim != 3:
        raise TypeError('Input array is not a cube or 3d array')
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError('Input vector or parallactic angles has wrong length')

    n, y, _ = array.shape
    
    angle_list = check_pa_vector(angle_list)
    
    if asize is None:
        annulus_width = int(np.ceil(2 * fwhm))
    elif isinstance(asize, int):
        annulus_width = asize
        
    # Annulus parametrization 
    
    radius_int=fwhm
    if local==True:
            if nann> 2*annulus_width:
                n_annuli = 5
                radius_int=(nann//annulus_width-2)*annulus_width 
            else:
                n_annuli = 4 
                radius_int=(nann//annulus_width-1)*annulus_width
    else:
            n_annuli = int((y / 2 - radius_int) / asize)
            

    # Definition of the annuli and the corresponding parralactic angle threshold 
    
    cube_out = np.zeros_like(array)
    for ann in range(n_annuli):


        inner_radius = radius_int + ann * annulus_width


        if verbose:
            print('{} : in_rad={}'.format(ann+1, inner_radius))

        theta_init = 90
        indices = get_annulus_segments(array[0], inner_radius, annulus_width,n_segments,theta_init)
        
        # Computation of the speckle field for the different frames and estimation of the cube of residuals
        
        for j in range(n_segments):

            
            cube_out[:,indices[j][0], indices[j][1]] =NMF_patch(array[:, indices[j][0], indices[j][1]], ncomp, max_iter,random_state,verbose)


    # Cube is derotated according to the parallactic angle and collapsed
    
    cube_der = cube_derotate(cube_out, angle_list, imlib=imlib,interpolation=interpolation)
    frame = cube_collapse(cube_der, mode=collapse)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame
    


def nmf_adisdi(cube, angle_list,scale_list=None, cube_ref=None, ncomp=1, scaling=None, max_iter=100,
        random_state=None, mask_center_px=None, imlib='opencv',
        interpolation='lanczos4', collapse='median', full_output=False,
        verbose=True, **kwargs):
    """ Non Negative Matrix Factorization for ADI or ADI+SDI sequences.This function embeds the 
    scikit-learn NMF algorithm solved through coordinate descent method.     
    """
    
    array,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            


    n, y, x = array.shape
    
    matrix_ref = prepare_matrix(array, scaling, mask_center_px, mode='fullfr',
                            verbose=verbose)
    matrix_ref += np.abs(matrix_ref.min())
    if cube_ref is not None:
        matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                    mode='fullfr', verbose=verbose)
        matrix_ref += np.abs(matrix_ref.min())
 
      
    mod = NMF(n_components=ncomp, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=100,tol=1e-3)  

    W = mod.fit_transform(matrix_ref)
    H = mod.components_

    
    reconstructed = np.dot(W, H)
    residuals = matrix_ref - reconstructed
               
    array_out = np.zeros_like(array)
    for i in range(n):
        array_out[i] = residuals[i].reshape(y,x)
            
    cube_der=rot_scale('fin',cube,array_out,angle_list_t,scale_list_t, imlib, interpolation)
    frame_fin = cube_collapse(cube_der, mode=collapse)
    
    return cube_der,frame_fin



def NMF_patch(matrix, ncomp, max_iter,random_state):

    """
    Function allowing the computation via NMF of the speckle field for a 
    given sub-region of the original ADI sequence. The code is a partial reproduction of
    the VIP function NMF_patch (Gonzalez et al. AJ, 154:7,2017)
    """

    refs = matrix+ np.abs(matrix.min())
    

    
    mod = NMF(n_components=ncomp, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=100,tol=1e-3)  

    W = mod.fit_transform(refs)
    H = mod.components_

    
    reconstructed = np.dot(W, H)

    residuals = refs - reconstructed

    return residuals



def annular_pca_adisdi(cube, angle_list,scale_list=None, radius_int=0,radius_out=None, fwhm=4, asize=2, n_segments=1,
                 delta_rot=1,delta_sep=0.1, ncomp=1, svd_mode='lapack', nproc=None,
                 min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None,
                 imlib='opencv', interpolation='lanczos4', collapse='median',
                 full_output=False, verbose=False, cube_ref=None, weights=None):
    """ PCA exploiting angular and spectral variability (ADI or ADI+SDI fashion).
    """

    array,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            
    n, y, _ = array.shape

    angle_list_t = check_pa_vector(angle_list_t)
    
    if radius_int%asize>int(asize/2):
        radius_int=(radius_int//asize)*asize
    elif radius_int>=asize:
        radius_int=(radius_int//asize-1)*asize
    else:
        radius_int=0
        
    if radius_out is not None:
        
        if radius_out%asize>int(asize/2):
            radius_out=(radius_out//asize+2)*asize
        else:
            radius_out=(radius_out//asize+1)*asize
        
        n_annuli = int((radius_out - radius_int) / asize)
        
    elif radius_out==None:
        n_annuli = int((y / 2 - radius_int) / asize)
    
    elif radius_out>int(y / 2): 
        
        n_annuli = int((y / 2 - radius_int) / asize)
 
    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot = [delta_rot] * n_annuli

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = 'N annuli = {}, FWHM = {:.3f}'
        print(msg.format(n_annuli, fwhm))
        print('PCA per annulus (or annular sectors):')

    if nproc is None:   # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    
    for ann in range(n_annuli):
        if isinstance(ncomp, tuple) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                raise TypeError('If `ncomp` is a tuple, it must match the '
                                'number of annuli')
        else:
            ncompann = ncomp

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(angle_list_t, ann, n_annuli, fwhm,
                                     radius_int, asize, delta_rot[ann],
                                     n_segments_ann, verbose)
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(array[0], inner_radius, asize,
                                       n_segments_ann)
        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
            else:
                matrix_segm_ref = None

            res = pool_map(nproc, do_pca_patch, matrix_segm, iterable(range(n)),
                           angle_list_t,scale_list_t, fwhm, pa_thr,delta_sep, ann_center, svd_mode,
                           ncompann, min_frames_lib, max_frames_lib, tol,
                           matrix_segm_ref)
            res = np.array(res)
            residuals = np.array(res[:, 0])

            for fr in range(n):
                cube_out[fr][yy, xx] = residuals[fr]

    # Cube is derotated according to the parallactic angle and collapsed
    cube_der=rot_scale('fin',cube,cube_out,angle_list_t,scale_list_t, imlib, interpolation)
    
    frame = cube_collapse(cube_der, mode=collapse)

    return cube_der, frame
   
def do_pca_patch(matrix, frame, angle_list,scale_list, fwhm, pa_threshold, delta_sep, ann_center,
                 svd_mode, ncomp, min_frames_lib, max_frames_lib, tol,
                 matrix_ref):
    
    """ 
    Function  doing the SVD/PCA for each frame patch. The code is a partial reproduction of
    the VIP function do_pca_patch (Gonzalez et al. AJ, 154:7,2017)  
    """


    if scale_list is not None:
    
        indices_left = np.intersect1d(_find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False),_find_indices_sdi(scale_list, ann_center, frame,
                                             fwhm, delta_sep))
    else:
        indices_left = _find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False)
    

    data_ref = matrix[indices_left]
    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        data_ref = np.vstack((matrix_ref, data_ref))

    curr_frame = matrix[frame]  # current frame
    V = get_eigenvectors(ncomp, data_ref, svd_mode, noise_error=tol)
    transformed = np.dot(curr_frame, V.T)
    reconstructed = np.dot(transformed.T, V)
    residuals = curr_frame - reconstructed
    return residuals, V.shape[0], data_ref.shape[0]


def do_pca_patch_range(matrix, frame, angle_list,scale_list, fwhm, pa_threshold,delta_sep, ann_center,
                 svd_mode, ncomp_range, min_frames_lib, max_frames_lib, tol,
                 matrix_ref):
    """ 
    Function  doing the SVD/PCA for each frame patch for a range of principal
    component ncomp_range. The code is a partial reproduction of
    the VIP function do_pca_patch (Gonzalez et al. AJ, 154:7,2017)  
    """

    if scale_list is not None:
    
        indices_left = np.intersect1d(_find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False),_find_indices_sdi(scale_list, ann_center, frame,
                                             fwhm, delta_sep))
    else:
        indices_left = _find_indices_adi(angle_list, frame,
                                             pa_threshold, truncate=False)

    data_ref = matrix[indices_left]
    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        data_ref = np.vstack((matrix_ref, data_ref))

    curr_frame = matrix[frame]  # current frame
    V = get_eigenvectors(ncomp_range[len(ncomp_range)-1], data_ref, svd_mode, noise_error=tol)
    residuals=[]
    for i in ncomp_range:
        V_trunc=V[ncomp_range[0]:i,:]
        transformed = np.dot(curr_frame, V_trunc.T)
        reconstructed = np.dot(transformed.T, V_trunc)
        residuals.append(curr_frame - reconstructed)
        
    return residuals, V.shape[0], data_ref.shape[0]

         
def loci_adisdi(cube, angle_list,scale_list=None, fwhm=4, metric='manhattan',
                 dist_threshold=50, delta_rot=0.5,delta_sep=0.1, radius_int=0,radius_out=None, asize=4,
                 n_segments=1, nproc=1, solver='lstsq', tol=1e-3,
                 optim_scale_fact=1, imlib='opencv', interpolation='lanczos4',
                 collapse='median', nann=None,local=False, verbose=True, full_output=False):
    """ Least-squares model PSF subtraction for ADI or ADI+SDI. This code is an adaptation of the VIP
    xloci function to provide, if required, the residuals after speckle field subtraction
    for a given annulus.
    """
    
    cube_rot_scale,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            
            
    y = cube_rot_scale.shape[1]
    if not asize < y // 2:
        raise ValueError("asize is too large")

    angle_list = check_pa_vector(angle_list)
    if local==True:
            n_annuli = 3 
            radius_int=nann-asize
    else:
        if radius_int%asize>int(asize/2):
            radius_int=(radius_int//asize)*asize
        elif radius_int>=asize:
            radius_int=(radius_int//asize-1)*asize
        else:
            radius_int=0
            
        if radius_out is not None:
            
            if radius_out%asize>int(asize/2):
                radius_out=(radius_out//asize+2)*asize
            else:
                radius_out=(radius_out//asize+1)*asize
            
            n_annuli = int((radius_out - radius_int) / asize)
        
        elif radius_out==None:
            n_annuli = int((y / 2 - radius_int) / asize)
        
        elif radius_out>int(y / 2): 
            
            n_annuli = int((y / 2 - radius_int) / asize)
        
            
    if verbose:
        print("Building {} annuli:".format(n_annuli))

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif isinstance(delta_rot, (int, float)):
        delta_rot = [delta_rot] * n_annuli

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    annulus_width = asize
    if isinstance(n_segments, int):
        n_segments = [n_segments]*n_annuli
    elif n_segments == 'auto':
        n_segments = list()
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    # annulus-wise least-squares combination and subtraction
    cube_res = np.zeros_like(cube_rot_scale)

    ayxyx = []  # contains per-segment data

    for ann in range(n_annuli):
        n_segments_ann = n_segments[ann]
        inner_radius_ann = radius_int + ann*annulus_width

        # angles
        pa_threshold = _define_annuli(angle_list, ann, n_annuli, fwhm,
                                      radius_int, asize, delta_rot[ann],
                                      n_segments_ann, verbose)[0]

        # indices
        indices = get_annulus_segments(cube_rot_scale[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann)
        ind_opt = get_annulus_segments(cube_rot_scale[0], inner_radius=inner_radius_ann,
                                       width=asize, nsegm=n_segments_ann,
                                       optim_scale_fact=optim_scale_fact)

        # store segment data for multiprocessing
        ayxyx += [(inner_radius_ann+asize//2,pa_threshold, indices[nseg][0], indices[nseg][1],
                   ind_opt[nseg][0], ind_opt[nseg][1]) for nseg in
                  range(n_segments_ann)]



    msg = 'Patch-wise least-square combination and subtraction:'
    # reverse order of processing, as outer segments take longer
    res_patch = pool_map(nproc, _leastsq_patch, iterable(ayxyx[::-1]),
                         angle_list_t,scale_list_t,fwhm,cube_rot_scale, None, metric, dist_threshold,delta_sep,
                         solver, tol, verbose=verbose, msg=msg,
                         progressbar_single=True)

    for patch in res_patch:
        matrix_res, yy, xx = patch
        cube_res[:, yy, xx] = matrix_res
        
    cube_der=rot_scale('fin',cube,cube_res,angle_list_t,scale_list_t, imlib, interpolation)
    frame_der_median = cube_collapse(cube_der, collapse)

    if verbose:
        print('Done processing annuli')

    return cube_der, frame_der_median


def _leastsq_patch(ayxyx, angle_list,scale_list,fwhm,cube, nann,metric, dist_threshold,delta_sep,
                   solver, tol,formod=False,psf=None):

    """
    Function allowing th estimation of the optimal factors for the modeled speckle field
    estimation via the LOCI framework. The code has been developped based on the VIP 
    python function _leastsq_patch, but return additionnaly the set of coefficients used for
    the speckle field computation.
    """
    
    ann_center,pa_threshold, yy, xx, yy_opti, xx_opti = ayxyx
    
    ind_ref_list=[]
    coef_list=[]
    
    yy_opt=[]
    xx_opt=[]
        
    for j in range(0,len(yy_opti)):
        if not any(x in np.where(yy==yy_opti[j])[0] for x in np.where(xx==xx_opti[j])[0]):
            xx_opt.append(xx_opti[j])
            yy_opt.append(yy_opti[j])
    

    values = cube[:, yy, xx]  
    matrix_res = np.zeros((values.shape[0], yy.shape[0]))
    values_opt = cube[:, yy_opti, xx_opti]
    n_frames = cube.shape[0]


    for i in range(n_frames):
        
        if scale_list is not None:
    
            ind_fr_i = np.intersect1d(_find_indices_adi(angle_list, i,
                                                 pa_threshold, truncate=False),_find_indices_sdi(scale_list, ann_center, i,
                                                 fwhm, delta_sep))
        else:
            ind_fr_i = _find_indices_adi(angle_list, i,
                                                 pa_threshold, truncate=False)
        if len(ind_fr_i) > 0:
            A = values_opt[ind_fr_i]
            b = values_opt[i]
            if solver == 'lstsq':
                coef = np.linalg.lstsq(A.T, b, rcond=tol)[0]     # SVD method
            elif solver == 'nnls':
                coef = sp.optimize.nnls(A.T, b)[0]
            elif solver == 'lsq':   
                coef = sp.optimize.lsq_linear(A.T, b, bounds=(0, 1),
                                              method='trf',
                                              lsq_solver='lsmr')['x']
            else:
                raise ValueError("`solver` not recognized")
        else:
            msg = "No frames left in the reference set. Try increasing "
            msg += "`dist_threshold` or decreasing `delta_rot`."
            raise RuntimeError(msg)


        if formod==True:
            ind_ref_list.append(ind_fr_i)
            coef_list.append(coef)       
            
        recon = np.dot(coef, values[ind_fr_i])
        matrix_res[i] = values[i] - recon
    
    if formod==True:
        return matrix_res,ind_ref_list,coef_list, yy, xx,
    else:
        return matrix_res, yy,xx
    
      
def llsg_adisdi(cube, angle_list,scale_list, fwhm, rank=10, thresh=1, max_iter=10,
         low_rank_ref=False, low_rank_mode='svd', auto_rank_mode='noise',
         residuals_tol=1e-1, cevr=0.9, thresh_mode='soft', nproc=1,
         asize=None, n_segments=4, azimuth_overlap=None, radius_int=None,radius_out=None,
         random_seed=None, imlib='opencv', interpolation='lanczos4',
         high_pass=None, collapse='median', full_output=True, verbose=True,
         debug=False):
    
    """ Local low rank plus Gaussian PSF subtraction for ADI or ADI+SDI. This 
    code is an adaptation of the VIP llsg function.
    """
    
    cube_rot_scale,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
    
    list_l, list_s, list_g, f_l, frame_fin, f_g = llsg(cube_rot_scale, angle_list_t, fwhm, rank=rank,asize=asize,radius_int=radius_int,radius_out=radius_out,thresh=1,n_segments=n_segments, max_iter=40, random_seed=10, nproc=nproc,full_output=True,verbose=False)
    res_s=np.array(list_s)
    residuals_cube_=cube_derotate(res_s[0],-angle_list_t)
    cube_der=rot_scale('fin',cube,residuals_cube_,angle_list_t,scale_list_t, imlib, interpolation)
    frame_fin=cube_collapse(cube_der, collapse)
    return cube_der,frame_fin




def llsg(cube, angle_list, fwhm, rank=10, thresh=1, max_iter=10,
         low_rank_ref=False, low_rank_mode='svd', auto_rank_mode='noise',
         residuals_tol=1e-1, cevr=0.9, thresh_mode='soft', nproc=1,
         asize=None, n_segments=4, azimuth_overlap=None, radius_int=None, radius_out=None,
         random_seed=None, imlib='opencv', interpolation='lanczos4',
         high_pass=None, collapse='median', full_output=False, verbose=True,
         debug=False):
    """ Local Low-rank plus Sparse plus Gaussian-noise decomposition (LLSG) as
    described in Gomez Gonzalez et al. 2016. This first version of our algorithm
    aims at decomposing ADI cubes into three terms L+S+G (low-rank, sparse and
    Gaussian noise). Separating the noise from the S component (where the moving
    planet should stay) allow us to increase the SNR of potential planets.
    The three tunable parameters are the *rank* or expected rank of the L
    component, the ``thresh`` or threshold for encouraging sparsity in the S
    component and ``max_iter`` which sets the number of iterations. The rest of
    parameters can be tuned at the users own risk (do it if you know what you're
    doing).
    Parameters
    ----------
    cube : numpy ndarray, 3d
        Input ADI cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    fwhm : float
        Known size of the FHWM in pixels to be used.
    rank : int, optional
        Expected rank of the L component.
    thresh : float, optional
        Factor that scales the thresholding step in the algorithm.
    max_iter : int, optional
        Sets the number of iterations.
    low_rank_ref :
        If True the first estimation of the L component is obtained from the
        remaining segments in the same annulus.
    low_rank_mode : {'svd', 'brp'}, optional
        Sets the method of solving the L update.
    auto_rank_mode : {'noise', 'cevr'}, str optional
        If ``rank`` is None, then ``auto_rank_mode`` sets the way that the
        ``rank`` is determined: the noise minimization or the cumulative
        explained variance ratio (when 'svd' is used).
    residuals_tol : float, optional
        The value of the noise decay to be used when ``rank`` is None and
        ``auto_rank_mode`` is set to ``noise``.
    cevr : float, optional
        Float value in the range [0,1] for selecting the cumulative explained
        variance ratio to choose the rank automatically (if ``rank`` is None).
    thresh_mode : {'soft', 'hard'}, optional
        Sets the type of thresholding.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to cpu_count()/2. By default the algorithm works
        in single-process mode.
    asize : int or None, optional
        If ``asize`` is None then each annulus will have a width of ``2*asize``.
        If an integer then it is the width in pixels of each annulus.
    n_segments : int or list of ints, optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli.
    azimuth_overlap : int or None, optional
        Sets the amount of azimuthal averaging.
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular area is discarded.
    random_seed : int or None, optional
        Controls the seed for the Pseudo Random Number generator.
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    high_pass : odd int or None, optional
        If set to an odd integer <=7, a high-pass filter is applied to the
        frames. The ``vip_hci.var.frame_filter_highpass`` is applied twice,
        first with the mode ``median-subt`` and a large window, and then with
        ``laplacian-conv`` and a kernel size equal to ``high_pass``. 5 is an
        optimal value when ``fwhm`` is ~4.
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: bool, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    debug : bool, optional
        Whether to output some intermediate information.
    Returns
    -------
    frame_s : numpy ndarray, 2d
        Final frame (from the S component) after rotation and median-combination.
    If ``full_output`` is True, the following intermediate arrays are returned:
    list_l_array_der, list_s_array_der, list_g_array_der, frame_l, frame_s,
    frame_g
    """
    if cube.ndim != 3:
        raise TypeError("Input array is not a cube (3d array)")
    if not cube.shape[0] == angle_list.shape[0]:
        msg = "Angle list vector has wrong length. It must equal the number"
        msg += " frames in the cube"
        raise TypeError(msg)

    if low_rank_mode == 'brp':
        if rank is None:
            msg = "Auto rank only works with SVD low_rank_mode."
            msg += " Set a value for the rank parameter"
            raise ValueError(msg)
        if low_rank_ref:
            msg = "Low_rank_ref only works with SVD low_rank_mode"
            raise ValueError(msg)

    if high_pass is not None:
        cube_init = cube_filter_highpass(cube, 'median-subt', median_size=19,
                                         verbose=False)
        cube_init = cube_filter_highpass(cube_init, 'laplacian-conv',
                                         kernel_size=high_pass, verbose=False)
    else:
        cube_init = cube

    n, y, x = cube.shape

    if azimuth_overlap == 0:
        azimuth_overlap = None

    if radius_int is None:
        radius_int = 0

    if nproc is None:
        nproc = cpu_count() // 2        # Hyper-threading doubles the # of cores

    # Same number of pixels per annulus
    if asize is None:
        annulus_width = int(np.ceil(2 * fwhm))  # as in the paper
    elif isinstance(asize, int):
        annulus_width = asize
        
    if radius_int%asize>int(asize/2):
        radius_int=(radius_int//asize)*asize
    elif radius_int>=asize:
        radius_int=(radius_int//asize-1)*asize
    else:
        radius_int=0
        
    if radius_out is not None:
        
        if radius_out%asize>int(asize/2):
            radius_out=(radius_out//asize+2)*asize
        else:
            radius_out=(radius_out//asize+1)*asize
        
        if radius_out>int(y / 2):
            n_annuli = int((y / 2 - radius_int) / asize)
        else:
            n_annuli = int((radius_out - radius_int) / asize)
        
    else :
        
        n_annuli = int((y / 2 - radius_int) / asize)

        

    if n_segments is None:
        n_segments = [4 for _ in range(n_annuli)]   # as in the paper
    elif isinstance(n_segments, int):
        n_segments = [n_segments]*n_annuli
    elif n_segments == 'auto':
        n_segments = []
        n_segments.append(2)    # for first annulus
        n_segments.append(3)    # for second annulus
        ld = 2 * np.tan(360/4/2) * annulus_width
        for i in range(2, n_annuli):    # rest of annuli
            radius = i * annulus_width
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360/ang)))

    if verbose:
        print('Annuli = {}'.format(n_annuli))

    # Azimuthal averaging of residuals
    if azimuth_overlap is None:
        azimuth_overlap = 360   # no overlapping, single config of segments
    n_rots = int(360 / azimuth_overlap)

    matrix_s = np.zeros((n_rots, n, y, x))
    if full_output:
        matrix_l = np.zeros((n_rots, n, y, x))
        matrix_g = np.zeros((n_rots, n, y, x))

    # Looping the he annuli
    if verbose:
        print('Processing annulus: ')
    for ann in range(n_annuli):
        inner_radius = radius_int + ann * annulus_width
        n_segments_ann = n_segments[ann]
        if verbose:
            print('{} : in_rad={}, n_segm={}'.format(ann+1, inner_radius,
                                                     n_segments_ann))

        
        for i in range(n_rots):
            theta_init = i * azimuth_overlap
            indices = get_annulus_segments(cube[0], inner_radius,
                                           annulus_width, n_segments_ann,
                                           theta_init)

            patches = pool_map(nproc, _decompose_patch, indices,
                               iterable(range(n_segments_ann)),cube_init, n_segments_ann,
                               rank, low_rank_ref, low_rank_mode, thresh,
                               thresh_mode, max_iter, auto_rank_mode, cevr,
                               residuals_tol, random_seed, debug, full_output)

            for j in range(n_segments_ann):
                yy = indices[j][0]
                xx = indices[j][1]

                if full_output:
                    matrix_l[i, :, yy, xx] = patches[j][0]
                    matrix_s[i, :, yy, xx] = patches[j][1]
                    matrix_g[i, :, yy, xx] = patches[j][2]
                else:
                    matrix_s[i, :, yy, xx] = patches[j]

    if full_output:
        list_s_array_der = [cube_derotate(matrix_s[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_s = [cube_collapse(list_s_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_s = cube_collapse(np.array(list_frame_s), mode=collapse)

        list_l_array_der = [cube_derotate(matrix_l[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_l = [cube_collapse(list_l_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_l = cube_collapse(np.array(list_frame_l), mode=collapse)

        list_g_array_der = [cube_derotate(matrix_g[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_g = [cube_collapse(list_g_array_der[k], mode=collapse)
                        for k in range(n_rots)]
        frame_g = cube_collapse(np.array(list_frame_g), mode=collapse)

    else:
        list_s_array_der = [cube_derotate(matrix_s[k], angle_list, imlib=imlib,
                                          interpolation=interpolation)
                            for k in range(n_rots)]
        list_frame_s = [cube_collapse(list_s_array_der[k], mode=collapse)
                        for k in range(n_rots)]

        frame_s = cube_collapse(np.array(list_frame_s), mode=collapse)


    if full_output:
        return(list_l_array_der, list_s_array_der, list_g_array_der,
               frame_l, frame_s, frame_g)
    else:
        return frame_s
    

def _decompose_patch(indices, i_patch, cube_init, n_segments_ann, rank, low_rank_ref,
                     low_rank_mode, thresh, thresh_mode, max_iter,
                     auto_rank_mode, cevr, residuals_tol, random_seed,
                     debug=False, full_output=False):
    """ Patch decomposition.
    """
    j = i_patch
    yy = indices[j][0]
    xx = indices[j][1]
    data_segm = cube_init[:, yy, xx]

    if low_rank_ref:
        ref_segments = list(range(n_segments_ann))
        ref_segments.pop(j)
        for m, n in enumerate(ref_segments):
            if m == 0:
                yy_ref = indices[n][0]
                xx_ref = indices[n][1]
            else:
                yy_ref = np.hstack((yy_ref, indices[n][0]))
                xx_ref = np.hstack((xx_ref, indices[n][1]))
        data_ref = cube_init[:, yy_ref, xx_ref]
    else:
        data_ref = data_segm

    patch = _patch_rlrps(data_segm, data_ref, rank, low_rank_ref,
                         low_rank_mode, thresh, thresh_mode,
                         max_iter, auto_rank_mode, cevr,
                         residuals_tol, random_seed, debug=debug,
                         full_output=full_output)
    return patch


def Hessian(x, f, step=0.05):
    
    def matrix_indices(n):
        for i in range(n):
            for j in range(i, n):
                yield i, j
    
    n = len(x)

    h = np.empty(n)
    h.fill(step)

    ee = np.diag(h)

    f0 = f(x)
    g = np.zeros(n)
    #print(x,f0)
    
    new_opti =True
    while new_opti ==True:
        new_opti =False
        for i in range(n):
            g[i] = f(x+ee[i, :])
            #print(x + ee[i, :],g[i])
            if g[i]<f0:
                f0=g[i]
                x+= ee[i, :]
                new_opti ==True 
                break

    hess = np.outer(h, h)
    new_opti =True
    while new_opti ==True:
        new_opti =False
        for i, j in matrix_indices(n):
            f_esti=f(x + ee[i, :] + ee[j, :])
            if f_esti>=f0:
            
                hess[i, j] = ( f_esti -
                          g[i] - g[j] + f0)/hess[i, j]
                hess[j, i] = hess[i, j]
            else:
                f0=f_esti
                x+= ee[i, :] + ee[j, :]
                new_opti =True
                break
            
        return x,hess
    

def bayesian_optimization(loss_function,bounds, param_type,args=None,n_random_esti=30, n_iters=20,random_search=True,multi_search=False,n_restarts=200,ncore=1):

    """

    Parameters
    ----------
    loss_function: function,
            function to optimize
    bounds: numpy ndarray 2d,
            boundary conditions for the set of parameters 
    param_type: list,
            define if the parameters are float ('float') or integer ('int')
    n_random_esti: int,
            number of loss function estimations used to initialize the Gaussian Process
    n_iters: int,
            the number of iterations of the Bayesian optimization algorithm
    random_search: bool,
            define if a random search is used to define the next point to sample based on
            the maximization of the expected improvement. If False, a L-BFGS-B minimisation
            is perform to define the next point of the parameter space to sample
    n_restarts: int,
            Number of points of the parameter space for which the expected improvement
            is computed if random_search=True and the number of tested starting points if
            random_search=False.
    n_core: int,
            number of cores used to perform the loss function estimations during the Gaussian 
            process initialization.
    """
    
    def expected_improvement(x, gauss_proc, eval_loss, space_dim):
    
        x_p = x.reshape(-1, space_dim)
    
        mu, sigma = gauss_proc.predict(x_p, return_std=True)
    
        opti_loss = np.max(eval_loss)
    
        # In case sigma equals zero
        with np.errstate(divide='ignore'):
            Z = (mu - opti_loss) / sigma
            expected_improvement = (mu - opti_loss) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] == 0.0
    
        return -1* expected_improvement
    
    
    
    def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss,
                                   bounds=(0, 10),param_type='int', n_restarts=10):
    
        n_params = bounds.shape[0]
        
        params=[]
          
        for i in range(len(param_type)):
            
            if param_type[i]=='int':
                params.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (n_restarts)))
            else:
                params.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (n_restarts)))
    
        ei_temp=[]
        param_temp=[]
        param_sel=[]
        for starting_point in np.array(params).T:
            
            res = minimize(fun=acquisition_func,
                           x0=starting_point.reshape(1, -1),
                           bounds=bounds,
                           method='L-BFGS-B',
                           args=(gaussian_process, evaluated_loss, n_params))
    
            
            ei_temp.append(-res.fun[0])
            param_temp.append(res.x)
            param_sel.append(res.x)
            for j in range(len(param_sel[-1])):
                param_sel[-1][j]=round(param_sel[-1][j],ndigits=np.where(np.log10(param_sel[-1][j])>0,3,int(abs(np.log10(param_sel[-1][j])))+3)) 
       
        param_sel,index_sel=np.unique(np.array(param_sel),return_index=True,axis=0)
        param_temp=np.take(np.array(param_temp),index_sel,axis=0)
        ei_temp=np.take(np.array(ei_temp),index_sel)
     
        return ei_temp,param_temp
    
    np.random.seed(10)
    space_dim = bounds.shape[0]
    
    params_m=[]
    
    for i in range(len(param_type)):
        if param_type[i]=='int':
            params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (n_random_esti)))
        else:
            params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (n_random_esti)))
            
            
    res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T),*args)
    
    x_ini=[]
    y_ini=[]
    
    for res_temp in res_param:
        x_ini.append(res_temp[0])
        y_ini.append(res_temp[1])
        
    #kernel = gp.kernels.RationalQuadratic(length_scale=1.0, alpha=0.5, length_scale_bounds=(0.5,5),alpha_bounds=(0.6, 5))    
    #kernel =gp.kernels.Matern(length_scale=1, nu=10, length_scale_bounds=(0.5,5))
    kernel = gp.kernels.RBF(1.0, length_scale_bounds=(0.5,5))
    
    model = gp.GaussianProcessRegressor( kernel,
                                       alpha=5e-6,                               
                                    n_restarts_optimizer=0,
                                    normalize_y=False)


    for n in range(n_iters):
    
        model.fit(np.array(x_ini),np.array(y_ini))
    
        if random_search:
            x_sel=[]
            for i in range(len(param_type)):
                if param_type[i]=='int':
                    x_sel.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (n_restarts)))
                else:
                    x_sel.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (n_restarts)))
                    
            x_sel=np.array(x_sel).T
            ei = -1 * expected_improvement(x_sel, model, y_ini, space_dim=space_dim) 
            
        else:         
            ei,x_sel = sample_next_hyperparameter(expected_improvement, model, y_ini, bounds=bounds,param_type=param_type, n_restarts=n_restarts)
        
    
        if multi_search:
            ind = np.argpartition(ei, -multi_search)[-multi_search:]
            params_m = np.take(x_sel,ind,axis=0)
        else:               
            params_m = [x_sel[np.argmax(ei),:]]
            
        
        # Duplicates break the GP
        for j in range(params_m.shape[0]):
             
            if any((np.abs(params_m[j] - x_ini)/x_ini).sum(axis=1) < 1e-3):
                          
                for i in range(len(param_type)):                    
                    if param_type[i]=='int':
                        params_m[j][i]=np.random.random_integers(bounds[i, 0], bounds[i, 1], 1)[0]
                    else:
                        params_m[j][i]=np.random.uniform(bounds[i, 0], bounds[i, 1], 1)[0]

        res_param = pool_map(ncore, loss_function, iterable(params_m),*args)
                
        for res_temp in res_param:
            x_ini.append(res_temp[0])
            y_ini.append(res_temp[1])
        
    return x_ini[np.argmax(y_ini)],np.max(y_ini) 
