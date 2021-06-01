"""
Set of functions used by the PyRSM class to compute detection maps and optimize the parameters
of the RSM algorithm and PSF-subtraction techniques via the auto-RSM and auto-S/N frameworks
"""
__author__ = 'Carl-Henrik Dahlqvist'

from scipy.interpolate import Rbf
import pandas as pd
import numpy.linalg as la
from vip_hci.var import get_annulus_segments, frame_center,prepare_matrix
from vip_hci.preproc.derotation import _define_annuli
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
import vip_hci as vip
from sklearn.decomposition import NMF as NMF_sklearn
 
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
        



def KLIP(cube, angle_list, nann=None, local=False, fwhm=4, asize=2, n_segments=1,delta_rot=1, ncomp=1,min_frames_lib=2, max_frames_lib=200,imlib='opencv',nframes=None, interpolation='lanczos4', collapse='median',full_output=False, verbose=1):

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


def nmf_adisdi(cube, angle_list,scale_list=None, cube_ref=None, ncomp=1, scaling=None, max_iter=100,
        random_state=None, mask_center_px=None, imlib='opencv',
        interpolation='lanczos4', collapse='median', full_output=False,
        verbose=True, **kwargs):
    """ Non Negative Matrix Factorization for ADI or ADI+SDI sequences.This function embeds the 
    scikit-learn NMF algorithm solved through coordinate descent method.     
    """
    
    array,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            


    n, y, x = array.shape
    
    matrix = prepare_matrix(array, scaling, mask_center_px, mode='fullfr',
                            verbose=verbose)
    matrix += np.abs(matrix.min())
    if cube_ref is not None:
        matrix_ref = prepare_matrix(cube_ref, scaling, mask_center_px,
                                    mode='fullfr', verbose=verbose)
        matrix_ref += np.abs(matrix_ref.min())
           
    mod = NMF_sklearn(n_components=ncomp, alpha=0, solver='cd', init='nndsvd', 
              max_iter=max_iter, random_state=random_state, **kwargs) 
    
    # H [ncomp, n_pixels]: Non-negative components of the data
    if cube_ref is not None:
        H = mod.fit(matrix_ref).components_
    else:
        H = mod.fit(matrix).components_          
    
    # W: coefficients [n_frames, ncomp]
    W = mod.transform(matrix)
        
    reconstructed = np.dot(W, H)
    residuals = matrix - reconstructed
               
    array_out = np.zeros_like(array)
    for i in range(n):
        array_out[i] = residuals[i].reshape(y,x)
            
    cube_der=rot_scale('fin',cube,array_out,angle_list_t,scale_list_t, imlib, interpolation)
    frame_fin = cube_collapse(cube_der, mode=collapse)
    
    return cube_der,frame_fin


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
    

def NMF_patch(matrix, ncomp, max_iter,random_state,sklearn=False):

    """
    Function allowing the computation via NMF of the speckle field for a 
    given sub-region of the original ADI sequence. The code is a partial reproduction of
    the VIP function NMF_patch (Gonzalez et al. AJ, 154:7,2017)
    """

    refs = matrix+ np.abs(matrix.min())
    
    if sklearn==True:
        
        mod = NMF_sklearn(n_components=ncomp, alpha=0, solver='cd', init='nndsvd', 
                  max_iter=max_iter, random_state=random_state) 
        
        # H [ncomp, n_pixels]: Non-negative components of the data
        
        H = mod.fit(refs).components_          
    
        W = mod.transform(refs)
            
        reconstructed = np.dot(W, H)
    
    else:
        
        mod = NMF(X=refs, n_components=ncomp)
        
        mod.SolveNMF(maxiters=max_iter, tol=0.001)
    
    
        H=mod.H
        W=mod.W
        reconstructed = np.dot(W, H)
        
    residuals = refs - reconstructed

    return residuals

def NMF_patch_range(matrix, ncomp_range, max_iter,random_state,verbose):

    """
    Function allowing the computation via NMF of the speckle field for a range of principal 
    components ncomp_range and a given sub-region of the original ADI sequence. The code is a
    partial reproduction of the VIP function NMF_patch (Gonzalez et al. AJ, 154:7,2017)
    """


    refs = matrix+ np.abs(matrix.min())

    mod = NMF(X=refs, n_components=ncomp_range[len(ncomp_range)-1])
    
    mod.SolveNMF(maxiters=max_iter, tol=0.001)

    if verbose:  
        print('Done NMF with sklearn.NMF.')

    residuals=[]
    for i in ncomp_range:
        H=mod.H[ncomp_range[0]:i,:]
        W=mod.W[:,ncomp_range[0]:i]
        reconstructed = np.dot(W, H)
        residuals.append(refs - reconstructed)

    return residuals

def annular_pca_adisdi(cube, angle_list,scale_list=None, radius_int=0, fwhm=4, asize=2, n_segments=1,
                 delta_rot=1,delta_sep=0.1, ncomp=1, svd_mode='lapack', nproc=None,
                 min_frames_lib=2, max_frames_lib=200, tol=1e-1, scaling=None,
                 imlib='opencv', interpolation='lanczos4', collapse='median',
                 full_output=False, verbose=False, cube_ref=None, weights=None):
    """ PCA exploiting angular and spectral variability (ADI or ADI+SDI fashion).
    """

    array,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
            
    n, y, _ = array.shape

    angle_list_t = check_pa_vector(angle_list_t)
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
                 dist_threshold=50, delta_rot=0.5,delta_sep=0.1, radius_int=0, asize=4,
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
            n_annuli= int((y / 2 - radius_int) / asize)
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
         asize=None, n_segments=4, azimuth_overlap=None, radius_int=None,
         random_seed=None, imlib='opencv', interpolation='lanczos4',
         high_pass=None, collapse='median', full_output=True, verbose=True,
         debug=False):
    
    """ Local low rank plus Gaussian PSF subtraction for ADI or ADI+SDI. This 
    code is an adaptation of the VIP llsg function.
    """
    
    cube_rot_scale,angle_list_t,scale_list_t=rot_scale('ini',cube,None,angle_list,scale_list,imlib, interpolation)
    
    list_l, list_s, list_g, f_l, frame_fin, f_g = vip.llsg.llsg(cube_rot_scale, angle_list_t, fwhm, rank=rank,asize=asize, thresh=1,n_segments=n_segments, max_iter=40, random_seed=10, nproc=nproc,full_output=True,verbose=False)
    res_s=np.array(list_s)
    residuals_cube_=cube_derotate(res_s[0],-angle_list_t)
    cube_der=rot_scale('fin',cube,residuals_cube_,angle_list_t,scale_list_t, imlib, interpolation)
    frame_fin=cube_collapse(cube_der, collapse)
    return cube_der,frame_fin
    

def _decompose_patch(indices, i_patch,cube_init, n_segments_ann, rank, low_rank_ref,
                     low_rank_mode, thresh, thresh_mode, max_iter,
                     auto_rank_mode, cevr, residuals_tol, random_seed,
                     debug=False, full_output=False):
    

    """ Patch decomposition from the LLSG VIP function.
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

_largenumber = 1E100
_smallnumber = 1E-5

class NMF:
    """
    Nonnegative Matrix Factorization - Build a set of nonnegative basis components given 
    a dataset with Heteroscedastic uncertainties and missing data with a vectorized update rule.
    Algorithm:
      -- Iterative multiplicative update rule
    Input: 
      -- X: m x n matrix, the dataset
    Optional Input/Output: 
      -- n_components: desired size of the basis set, default 5
      -- V: m x n matrix, the weight, (usually) the inverse variance
      -- M: m x n binary matrix, the mask, False means missing/undesired data
      -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
      -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
    Comments:
      -- Between W and H, which one is the basis set and which one is the coefficient 
         depends on how you interpret the data, because you can simply transpose everything
         as in X-WH versus X^T - (H^T)(W^T)
      -- Everything needs to be non-negative
    References:
      -- Guangtun Ben Zhu, 2016
         A Vectorized Algorithm for Nonnegative Matrix Factorization with 
         Heteroskedastic Uncertainties and Missing Data
         AJ/PASP, (to be submitted)
      -- Blanton, M. and Roweis, S. 2007
         K-corrections and Filter Transformations in the Ultraviolet, Optical, and Near-infrared
         The Astronomical Journal, 133, 734
      -- Lee, D. D., & Seung, H. S., 2001
         Algorithms for non-negative matrix factorization
         Advances in neural information processing systems, pp. 556-562
    """

    def __init__(self, X, W=None, H=None, V=None, M=None, n_components=5):
        """
        Initialization
        
        Required Input:
          X -- the input data set
        Optional Input/Output:
          -- n_components: desired size of the basis set, default 5
          -- V: m x n matrix, the weight, (usually) the inverse variance
          -- M: m x n binary matrix, the mask, False means missing/undesired data
          -- H: n_components x n matrix, the H matrix, usually interpreted as the coefficients
          -- W: m x n_components matrix, the W matrix, usually interpreted as the basis set
        """

        # I'm making a copy for the safety of everything; should not be a bottleneck
        self.X = np.copy(X) 
        if (np.count_nonzero(self.X<0)>0):
            print("There are negative values in X. Setting them to be zero...", flush=True)
            self.X[self.X<0] = 0.

        self.n_components = n_components
        self.maxiters = 100
        self.tol = _smallnumber
        np.random.seed(10)
        if (W is None):
            self.W = np.random.rand(self.X.shape[0], self.n_components)
        else:
            if (W.shape != (self.X.shape[0], self.n_components)):
                raise ValueError("Initial W has wrong shape.")
            self.W = np.copy(W)
        if (np.count_nonzero(self.W<0)>0):
            print("There are negative values in W. Setting them to be zero...", flush=True)
            self.W[self.W<0] = 0.

        if (H is None):
            self.H = np.random.rand(self.n_components, self.X.shape[1])
        else:
            if (H.shape != (self.n_components, self.X.shape[1])):
                raise ValueError("Initial H has wrong shape.")
            self.H = np.copy(H)
        if (np.count_nonzero(self.H<0)>0):
            print("There are negative values in H. Setting them to be zero...", flush=True)
            self.H[self.H<0] = 0.

        if (V is None):
            self.V = np.ones(self.X.shape)
        else:
            if (V.shape != self.X.shape):
                raise ValueError("Initial V(Weight) has wrong shape.")
            self.V = np.copy(V)
        if (np.count_nonzero(self.V<0)>0):
            print("There are negative values in V. Setting them to be zero...", flush=True)
            self.V[self.V<0] = 0.

        if (M is None):
            self.M = np.ones(self.X.shape, dtype=np.bool)
        else:
            if (M.shape != self.X.shape):
                raise ValueError("M(ask) has wrong shape.")
            if (M.dtype != np.bool):
                raise TypeError("M(ask) needs to be boolean.")
            self.M = np.copy(M)

        # Set masked elements to be zero
        self.V[(self.V*self.M)<=0] = 0
        self.V_size = np.count_nonzero(self.V)

    @property
    def cost(self):
        """
        Total cost of a given set s
        """
        diff = self.X - np.dot(self.W, self.H)
        chi2 = np.einsum('ij,ij', self.V*diff, diff)/self.V_size
        return chi2

    def SolveNMF(self, W_only=False, H_only=False, maxiters=None, tol=None):
        """
        Construct the NMF basis
        Keywords:
            -- W_only: Only update W, assuming H is known
            -- H_only: Only update H, assuming W is known
               -- Only one of them can be set
        Optional Input:
            -- tol: convergence criterion, default 1E-5
            -- maxiters: allowed maximum number of iterations, default 1000
        Output: 
            -- chi2: reduced final cost
            -- time_used: time used in this run
        """


        if (maxiters is not None): 
            self.maxiters = maxiters
        if (tol is not None):
            self.tol = tol

        chi2 = self.cost
        oldchi2 = _largenumber

        if (W_only and H_only):
            return (chi2, 0.)

        V = np.copy(self.V)
        VT = V.T

        #XV = self.X*self.V
        XV = np.multiply(V, self.X)
        XVT = np.multiply(VT, self.X.T)

        niter = 0

        while (niter < self.maxiters) and ((oldchi2-chi2)/oldchi2 > self.tol):

            # Update H
            if (not W_only):
                H_up = np.dot(XVT, self.W)
                WHVT = np.multiply(VT, np.dot(self.W, self.H).T)
                H_down = np.dot(WHVT, self.W)
                self.H = self.H*H_up.T/H_down.T

            # Update W
            if (not H_only):
                W_up = np.dot(XV, self.H.T)
                WHV = np.multiply(V, np.dot(self.W, self.H))
                W_down = np.dot(WHV, self.H.T)
                self.W = self.W*W_up/W_down

            # chi2
            oldchi2 = chi2
            chi2 = self.cost

        return
    

