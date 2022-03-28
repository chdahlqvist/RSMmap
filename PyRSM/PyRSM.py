"""
PyRSM is a python package for exoplanets detection which applies the Regime Switching Model (RSM) framework on ADI (and ADI+SDI) sequences (see Dahlqvist et al. A&A, 2020, 633, A95).
The RSM map algorithm relies on one or several PSF subtraction techniques to process one or multiple ADI sequences before computing a final probability map. Considering the large
set of parameters needed for the computation of the RSM detection map (parameters for the selected PSF-subtraction techniques as well as the RSM algorithm itself), a parameter selection framework
called auto-RSM (Dahlqvist et al., 2021 in prep) is proposed to automatically select the optimal parametrization. The proposed multi-step parameter optimization framework can be divided into 
three main steps, (i) the selection of the optimal set of parameters for the considered PSF-subtraction techniques, (ii) the optimization of the RSM approach parametrization, and (iii) the 
selection of the optimal set of PSF-subtraction techniques and ADI sequences to be considered when generating the final detection map. 

The add_cube and add_model methods allows to consider several cubes and models to generate
the cube of residuals used to compute the RSM map. The cube should be provided by the same instrument
or rescaled to a unique pixel size. The class can be used with only ADI and ADI+SDI. A specific PSF should 
be provided for each cube. In the case of ADI+SDI a single psf should be provided per cube (typically the PSF
average over the set of frequencies Five different models and two forward model variants are available. 
Each model can be parametrized separately. Five different models and two forward model variants are available. 
The method like_esti allows the estimation of a cube of likelihoods containing for each pixel
and each frame the likelihood of being in the planetary or the speckle regime. These likelihood cubes
are then used by the probmap_esti method to provide the final probability map based on the RSM framework. 

The second set of methods regroups the four main methods used by the auto-RSM/auto-S/N framework.
The opti_model method allows the optimization of the PSF subtraction techniques parameters based on the 
minimisation of the average contrast. The opti_RSM method takes care of the optimization of the parameters 
of the RSM framework (all related to the computation of the likelihood associated to every pixels and frames). The
third method RSM_combination, relies on a greedy selection algorithm to define the optimal set of 
ADI sequences and PSF-subtraction techniques to consider when generating the final detection map using the RSM
approach. Finally, the opti_map method allows to compute the final RSM detection map. The optimization of
the parameters can be done using the reversed parallactic angles, blurring potential planetary signals while
keeping the main characteristics of the speckle noise. An S/N map based code is also proposed and encompasses
the opti_model, the RSM_combination and the opti_map methods. For the last two methods, the SNR 
parameter should be set to True.

The last set of methods regroups the methods allowing the computation of contrast curves and the 
characterization of a detected astrophysical signals. The self.contrast_curve method allows the computation
 of a contrast curve at a pre-defined completeness level (see Dahlqvist et al. 2021 for more details), 
 while the self.contrast_matrix method provided contrast curves for a range of completeness levels defined
 by the number of fake companion injected (completeness level from 1/n_fc to 1-1/n_fc with n_fc the number
 of fake companions). This last method provides a good representation of the contrast/completeness 
 distribution but requires a longer computation time. The self.target_charact() method allows the 
 estimation of the photometry and astrometry of a detected signal (see Dahlqvist et al. 2022 for more details)

Part of the code have been directly inspired by the VIP and PyKLIP packages for the estimation of the cube
of residuas and forward model PSF.
"""

__author__ = 'Carl-Henrik Dahlqvist'

import numpy as np
from scipy.optimize import curve_fit
from skimage.draw import disk 
import vip_hci as vip
from vip_hci.var import get_annulus_segments,frame_center,prepare_matrix
from vip_hci.preproc import frame_crop,cube_derotate,cube_crop_frames,cube_collapse
from vip_hci.metrics import cube_inject_companions, frame_inject_companion, normalize_psf
from hciplot import plot_frames 
from vip_hci.conf.utils_conf import pool_map, iterable
from multiprocessing import Pool, RawArray
import photutils
from scipy import stats
import pickle
import multiprocessing as mp
import sklearn.gaussian_process as gp
from scipy.stats import norm
import pyswarms.backend as P
from pyswarms.backend.topology import Star  
import scipy as sc
import lmfit
from scipy.optimize import minimize
import math
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
from .utils import (llsg_adisdi,loci_adisdi,do_pca_patch,_decompose_patch,
annular_pca_adisdi,NMF_patch,nmf_adisdi,LOCI_FM,KLIP_patch,perturb,KLIP,
get_time_series,poly_fit,interpolation,remove_outliers,check_delta_sep ,rot_scale,
bayesian_optimization,Hessian)



var_dict = {}

def init_worker(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape
    

            
class PyRSM:

    def __init__(self,fwhm,minradius,maxradius,pxscale=0.1,ncore=1,max_r_fm=None,opti_mode='full-frame',inv_ang=True,opti_type='Contrast',trunc=None,imlib='opencv', interpolation='lanczos4'):
        
        """
        Initialization of the PyRSM object on which the add_cube and add_model 
        functions will be applied to parametrize the model. The functions 
        like_esti and prob_esti will then allow the computation of the
        likelihood cubes and the final RSM map respectively.
        
        Parameters
        ----------
        fwhm: int
            Full width at half maximum for the instrument PSF
        minradius : int
            Center radius of the first annulus considered in the RSM probability
            map estimation. The radius should be larger than half 
            the value of the 'crop' parameter 
        maxradius : int
            Center radius of the last annulus considered in the RSM probability
            map estimation. The radius should be smaller or equal to half the
            size of the image minus half the value of the 'crop' parameter 
        pxscale : float
            Value of the pixel in arcsec/px. Only used for printing plots when
            ``showplot=True`` in like_esti. 
        ncore : int, optional
            Number of processes for parallel computing. By default ('ncore=1') 
            the algorithm works in single-process mode. 
        max_r_fm: int, optional
            Largest radius for which the forward model version of KLIP or LOCI
            are used, when relying on forward model versions of RSM. Forward model 
            versions of RSM have a higher performance at close separation, considering
            their computation time, their use should be restricted to small angular distances.
            Default is None, i.e. the foward model version are used for all considered
            angular distance.
        opti_mode: str, optional
            In the 'full-frame' mode, the parameter optimization is based on a reduced
            set of angular separations and a single global set of parameters is selected 
            (the one maximizing the global normalized average contrast). In 'annular' mode,
            a separate optimization is done for every consecutive annuli of width equal to 
            one FWHM and separated by a distance of one FWHM. For each annulus, a separate 
            optimal set of parameters is computed. Default is 'full-frame'.
        inv_ang: bool, optional
            If True, the sign of the parallactic angles of all ADI sequence is flipped for
            the entire optimization procedure. Default is True.
        opti_type: str, optional
            'Contrast' for an optimization based on the average contrast and 'RSM' for
            an optimization based on the ratio of the peak probability of the injected
            fake companion on the peak (noise) probability in the remaining of the 
            considered annulus (much higher computation time). Default is 'Contrast'.
        trunc: int, optional
            Maximum angular distance considered for the full-frame parameter optimization. Defaullt is None.
        imlib : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        interpolation : str, optional
            See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
        """
    
        self.ncore = ncore 
        self.minradius = minradius 
        self.maxradius = maxradius 
        self.fwhm = fwhm 
        self.pxscale = pxscale
        self.imlib=imlib
        self.interpolation=interpolation
        
        self.opti_mode=opti_mode
        self.inv_ang=inv_ang
        self.param_opti_mode=opti_type
        self.trunc=trunc
        
        if max_r_fm is not None:
            self.max_r=max_r_fm
        else:
            self.max_r=maxradius
          
        self.psf = []         
        self.cube=[]
        self.pa=[]
        self.scale_list=[]
        
        self.model = []    
        self.delta_rot = []
        self.delta_sep = []
        self.nsegments = [] 
        self.ncomp = []    
        self.rank = [] 
        self.tolerance = []
        self.asize=[]
        self.opti_bound=[]
        self.psf_fm=[]

        self.intensity = [] 
        self.distri = [] 
        self.distrifit=[]         
        self.var = [] 
        self.interval=[]
        self.crop=[]
        self.crop_range=[]
        
        self.snr_evolution=[]
        self.snr_combined_evolution=[]
        
        self.like_fin=[]
        self.flux_FMMF=[]
        self.distrisel=[]
        self.mixval=[]  
        self.fiterr=[]
        self.probmap=None
        
        self.param=None
        self.opti=False
        self.contrast=[]
        self.ini_esti=[]
        
        self.opti_sel=None
        self.threshold=None
        
        self.final_map=None
        
        
    def add_cube(self,psf, cube, pa, scale_list=None):
     
        """    
        Function used to add an ADI seuqence to the set of cubes considered for
        the RSM map estimation.
        
        Parameters
        ----------

        psf : numpy ndarray 2d
            2d array with the normalized PSF template, with an odd shape.
            The PSF image must be centered wrt to the array! Therefore, it is
            recommended to run the function ``normalize_psf`` to generate a 
            centered and flux-normalized PSF template.
        cube : numpy ndarray, 3d or 4d
            Input cube (ADI sequences), Dim 1 = temporal axis, Dim 2-3 = spatial axis
            Input cube (ADI + SDI sequences), Dim 1 = wavelength, Dim 2=temporal axis
            Dim 3-4 = spatial axis     
        pa : numpy ndarray, 1d
            Parallactic angles for each frame of the ADI sequences. 
        scale_list: numpy ndarray, 1d, optional
            Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
            scaling factors are the central channel wavelength divided by the
            shortest wavelength in the cube (more thorough approaches can be used
            to get the scaling factors). This scaling factors are used to re-scale
            the spectral channels and align the speckles. Default is None
        """
        if cube.shape[-1]%2==0:
            raise ValueError("Images should have an odd size")
            
        self.psf.append(psf)         
        self.cube.append(cube)
        self.pa.append(pa)
        self.scale_list.append(scale_list)
        
        self.like_fin.append([])
        self.flux_FMMF.append([])
        self.distrisel.append([])
        self.mixval.append([])
        self.fiterr.append([])
        self.psf_fm.append([]) 
              

    def add_method(self, model,delta_rot=0.5,delta_sep=0.1,asize=5,nsegments=1,ncomp=20,rank=5,tolerance=1e-2,interval=[5],intensity='Annulus',distri='A',var='ST',distrifit=False,crop_size=5, crop_range=1,opti_bound=None):

        
        """
        Function used to add a model to the set of post-processing techniques used to generate
        the cubes of residuals on which is based the computation of the likelihood of being 
        in either the planetary of the background regime. These likelihood matrices allow
        eventually the definition of the final RSM map.
        
        Parameters
        ----------
    
        model : str
            Selected ADI-based post-processing techniques used to 
            generate the cubes of residuals feeding the Regime Switching model.
            'APCA' for annular PCA, NMF for Non-Negative Matrix Factorization, LLSG
            for Local Low-rank plus Sparse plus Gaussian-noise decomposition, LOCI 
            for locally optimized combination of images and'KLIP' for Karhunen-Loeve
            Image Projection. There exitsts a foward model variant of KLIP and LOCI called 
            respectively 'FM KLIP' and 'FM LOCI'.
        delta_rot : float, optional
            Factor for tunning the parallactic angle threshold, expressed in FWHM.
            Default is 0.5 (excludes 0.5xFHWM on each side of the considered frame).
        delta_sep : float, optional
            The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
            Default is 0.1.
        asize : int, optional
            Width in pixels of each annulus.When a single. Default is 5. 
        n_segments : int, optional
            The number of segments for each annulus. Default is 1 as we work annulus-wise.
        ncomp : int, optional
            Number of components used for the low-rank approximation of the 
            speckle field with 'APCA', 'KLIP' and 'NMF'. Default is 20.
        rank : int, optional        
            Expected rank of the L component of the 'LLSG' decomposition. Default is 5.
        tolerance: float, optional
            Tolerance level for the approximation of the speckle field via a linear 
            combination of the reference images in the LOCI algorithm. Default is 1e-2.
        interval: list of float or int, optional
            List of values taken by the delta parameter defining, when mutliplied by the 
            standard deviation, the strengh of the planetary signal in the Regime Switching model.
            Default is [5]. The different delta paramaters are tested and the optimal value
            is selected via maximum likelmihood.
        intensity: str, optionnal
            If 'Pixel', the intensity parameter used in the RSM framework is computed
            pixel-wise via a gaussian maximum likelihood by comparing the set of observations
            and the PSF or the forward model PSF in the case of 'FM KLIP' and 'FM LOCI'.
            If 'Annulus', the intensity parameter is estimated annulus-wise and defined as
            a multiple of the annulus residual noise variance. If multiple multiplicative paramters
            are provided in PyRSM init (multi_factor), the multiplicative factor applied to the noise
            variance is selected via the maximisation of the total likelihood of the regime switching
            model for the selected annulus. Default is 'Annulus'.
        distri: str, optional
            Probability distribution used for the estimation of the likelihood 
            of both regimes (planetary or noise) in the Regime Switching framework.
            Default is Gaussian 'G' but four other possibilities exist, Laplacian 'L',
            Huber loss 'H', auto 'A' and mix 'M'. 
            
            'A': allow the automatic selection of the optimal distribution ('Laplacian'
            or 'Gaussian') depending on the fitness of these distributions compared to
            the empirical distribution of the residual noise in the considered annulus. 
            For each cubes and ADI-based post-processing techniques, the distribution 
            leading to the lowest fitness error is automatically selected. 
            
            'M': use both the 'Gaussian'and 'Laplacian' distribution to get closer to
            the empirical distribution by fitting a mix parameter providing the ratio
            of 'Laplacian' distribution compared to the 'Gaussian' one.
        var: str, optional
            Model used for the residual noise variance estimation. Five different approaches
            are proposed: 'ST', 'SM', 'FR', 'FM', and 'TE'. While all six can be used when 
            intensity='Annulus', only the last three can be used when intensity='Pixel'. 
            When using ADI+SDI dataset only 'FR' and 'FM' can be used. Default is 'ST'.
            
            'ST': consider every frame and pixel in the selected annulus with a 
            width equal to asize (default approach)
            
            'SM': consider for every pixels a segment of the selected annulus with 
            a width equal to asize. The segment is centered on the selected pixel and has
            a size of three FWHM. A mask of one FWHM is applied on the selected pixel and
            its surrounding. Every frame are considered.
            
            'FR': consider the pixels in the selected annulus with a width equal to asize
            but separately for every frame.
            
            'FM': consider the pixels in the selected annulus with a width 
            equal to asize but separately for every frame. Apply a mask one FWHM 
            on the selected pixel and its surrounding.
            
            'TE': rely on the method developped in PACO to estimate the 
            residual noise variance (take the pixels in a region of one FWHM arround 
            the selected pixel, considering every frame in the derotated cube of residuals 
            except for the selected frame)
            
        distrifit: bool, optional
            If true, the estimation of the mean and variance of the selected distribution
            is done via an best fit on the empirical distribution. If False, basic 
            estimation of the mean and variance using the set of observations 
            contained in the considered annulus, without taking into account the selected
            distribution.
            
        modtocube: bool, optional
            Parameter defining if the concatenated cube feeding the RSM model is created
            considering first the model or the different cubes. If 'modtocube=False',
            the function will select the first cube then test all models on it and move 
            to the next one. If 'modtocube=True', the model will select one model and apply
            it to every cubes before moving to the next model. Default is True.
        crop_size: int, optional
            Part of the PSF tempalte considered is the estimation of the RSM map
        crop_range: int, optional
            Range of crop sizes considered in the estimation of the RSM map, starting with crop_size
            and increasing the crop size incrementally by 2 pixels up to a crop size of 
            crop_size + 2 x (crop_range-1).
    
        opti_bound: list, optional
            List of boundaries used for the parameter optimization. 
                - For APCA: [[L_ncomp,U_ncomp],[L_nseg,U_nseg],[L_delta_rot,U_delta_rot]]
                  Default is [[15,45],[1,4],[0.25,1]]
                - For NMF: [[L_ncomp,U_ncomp]]
                  Default is [[2,20]]
                - For LLSG: [[L_ncomp,U_ncomp],[L_nseg,U_nseg]]
                  Default is [[1,10],[1,4]]
                - For LOCI: [[L_tolerance,U_tolerance],[L_delta_rot,U_delta_rot]]
                  Default is [[1e-3,1e-2],[0.25,1]]
                - For FM KLIP: [[L_ncomp,U_ncomp],[L_delta_rot,U_delta_rot]]
                  Default is [[15,45],[0.25,1]]
                - For FM LOCI: [[L_tolerance,U_tolerance],[L_delta_rot,U_delta_rot]]
                  Default is [[1e-3,1e-2],[0.25,1]]
            with L_ the lower bound and U_ the Upper bound.                
        """

        
        if crop_size+2*(crop_range-1)>=2*round(self.fwhm)+1:
            raise ValueError("Maximum cropsize should be lower or equal to two FWHM, please change accordingly either 'crop_size' or 'crop_range'")
            
        if any(var in myvar for myvar in ['ST','SM','FM','TE','FR'])==False:
            raise ValueError("'var' not recognized")
            
        if any(distri in mydistri for mydistri in ['G','L','A','M','H'])==False:
            raise ValueError("'distri' not recognized")
            
        if intensity=='Pixel' and any(var in myvar for myvar in ['FR','FM','TE'])==False:
            raise ValueError("'var' not recognized for intensity='Pixel'. 'var' should be 'FR','FM' or 'TE'")
            
        for c in range(len(self.cube)):
            if self.cube[c].ndim==4:
                if any(var in myvar for myvar in ['ST','SM','TE'])==True:
                    raise ValueError("'var' not recognized for ADI+SDI cube'. 'var' should be 'FR' or 'FM'")   
                if any(model == mymodel for mymodel in ['FM KLIP','FM LOCI','KLIP'])==True:
                    raise ValueError("ADI+SDI sequences can only be used with APCA, NMF, LLSG and LOCI")
                check_delta_sep(self.scale_list[c],delta_sep,self.minradius,self.fwhm,c)
        
        self.model.append(model)
        self.delta_rot.append(np.array([np.repeat(delta_rot,(len(self.cube)))]*(self.maxradius+asize)))
        self.delta_sep.append(np.array([np.repeat(delta_sep,(len(self.cube)))]*(self.maxradius+asize)))
        self.nsegments.append(np.array([np.repeat(nsegments,(len(self.cube)))]*(self.maxradius+asize)))
        self.ncomp.append(np.array([np.repeat(ncomp,(len(self.cube)))]*(self.maxradius+asize)))  
        self.rank.append(np.array([np.repeat(rank,(len(self.cube)))]*(self.maxradius+asize)))
        self.tolerance.append(np.array([np.repeat(tolerance,(len(self.cube)))]*(self.maxradius+asize)))
        self.interval.append(np.array([[interval]*(len(self.cube))]*(self.maxradius+asize)))
        self.asize.append(asize)
        self.opti_bound.append(opti_bound)
        
        self.snr_evolution.append(np.array([np.repeat(None,(len(self.cube)))]*(self.maxradius+asize)))
        self.snr_combined_evolution.append(np.array(np.repeat(None,(len(self.cube)))))
        
        self.distrifit.append(np.array([np.repeat(distrifit,(len(self.cube)))]*(self.maxradius+asize)))
        self.intensity.append(np.array([np.repeat(intensity,(len(self.cube)))]*(self.maxradius+asize)))
        self.var.append(np.array([np.repeat(var,(len(self.cube)))]*(self.maxradius+asize)))
        self.crop.append(np.array([np.repeat(crop_size,(len(self.cube)))]*(self.maxradius+asize)))        

        self.crop_range.append(crop_range)
        self.distri.append(distri)  


        for i in range(len(self.cube)):
            
            if model=='FM KLIP' or model=='FM LOCI':
                self.psf_fm[i].append(list([None]*(self.maxradius+1)))
            else:
                self.psf_fm[i].append(None) 
            self.like_fin[i].append(None)
            self.flux_FMMF[i].append(None)
            self.distrisel[i].append(None)
            self.mixval[i].append(None)
            self.fiterr[i].append(None)
            
            
                    
    def save_parameters(self,folder,name):
        
        with open(folder+name+'.pickle', "wb") as save:
            pickle.dump([self.model, self.delta_rot,self.nsegments,self.ncomp,self.rank,self.tolerance,self.asize,self.intensity,self.distri,self.distrifit,self.var,self.crop,self.crop_range,self.opti_sel,self.threshold,self.opti_mode,self.flux_opti,self.opti_theta,self.interval],save)
        
        
    def load_parameters(self,name):
        


        with open(name+'.pickle', "rb") as read:
            saved_param = pickle.load(read)

        
        self.model= saved_param[0]
        self.delta_rot = saved_param[1]
        self.nsegments = saved_param[2]
        self.ncomp = saved_param[3]   
        self.rank = saved_param[4]
        self.tolerance = saved_param[5]
        self.asize= saved_param[6]
        self.intensity = saved_param[7]
        self.distri = saved_param[8] 
        self.distrifit= saved_param[9]        
        self.var = saved_param[10]
        self.crop= saved_param[11]
        self.crop_range= saved_param[12]
        self.opti_sel=saved_param[13]
        self.threshold=saved_param[14]
        self.opti_mode=saved_param[15]
        self.flux_opti=saved_param[16]
        self.opti_theta=saved_param[17]
        self.interval=saved_param[18]
        
    def likelihood(self,ann_center,cuben,modn,mcube,cube_fc=None,verbose=True):
         
        if type(mcube) is not np.ndarray:
            mcube = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
        
        if mcube.ndim==4:
            z,n,y,x=mcube.shape
            
        else:
            n,y,x=mcube.shape
            z=1
        
        range_int=len(self.interval[modn][ann_center,cuben])
        
        likemap=np.zeros(((n*z)+1,x,y,range_int,2,self.crop_range[modn]))
        
        np.random.seed(10) 

        def likfcn(cuben,modn,mean,var,mixval,max_hist,mcube,ann_center,distrim,evals=None,evecs_matrix=None, KL_basis_matrix=None,refs_mean_sub_matrix=None,sci_mean_sub_matrix=None,resicube_klip=None,probcube=0,var_f=None, ind_ref_list=None,coef_list=None):

            np.random.seed(10) 
            
            phi=np.zeros(2)
            if self.cube[cuben].ndim==4:
                n_t=self.cube[cuben].shape[1]  

            ceny, cenx = frame_center(mcube[0])
            indicesy,indicesx=get_time_series(mcube,ann_center)

            range_int=len(self.interval[modn][ann_center,cuben])
                
            if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                
                if self.psf_fm[cuben][modn][ann_center] is not None:
                    psf_formod=True
                else:
                    
                    psf_formod=False
                
                psf_fm_out=np.zeros((len(indicesx),mcube.shape[0],int(2*round(self.fwhm)+1),int(2*round(self.fwhm)+1)))

            if (self.crop[modn][ann_center,cuben]+2*(self.crop_range[modn]-1))!=self.psf[cuben].shape[-1]:
                if len(self.psf[cuben].shape)==2:
                    psf_temp=frame_crop(self.psf[cuben],int(self.crop[modn][ann_center,cuben]+2*(self.crop_range[modn]-1)),cenxy=[int(self.psf[cuben].shape[1]/2),int(self.psf[cuben].shape[1]/2)],verbose=False)
                else:
                    psf_temp=cube_crop_frames(self.psf[cuben],int(self.crop[modn][ann_center,cuben]+2*(self.crop_range[modn]-1)),xy=(int(self.psf[cuben].shape[1]/2),int(self.psf[cuben].shape[1]/2)),verbose=False)
            else:
                psf_temp=self.psf[cuben]

            for i in range(0,len(indicesy)):

                psfm_temp=None
                cubind=0
                poscenty=indicesy[i]
                poscentx=indicesx[i]
                
                #PSF forward model computation for KLIP

                if self.model[modn]=='FM KLIP':
                    
                    an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
                    theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))    

                            
                    if psf_formod==False:
                            model_matrix=cube_inject_companions(np.zeros_like(mcube), self.psf[cuben], self.pa[cuben], flevel=1, plsc=0.1,rad_dists=an_dist, theta=theta, n_branches=1,verbose=False)

                            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * ann_center)))
                            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                            if pa_threshold >= mid_range - mid_range * 0.1:
                                pa_threshold = float(mid_range - mid_range * 0.1)

                            psf_map=np.zeros_like(model_matrix)
                            indices = get_annulus_segments(mcube[0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)


                            for b in range(0,n):
                                psf_map_temp = perturb(b,model_matrix[:, indices[0][0], indices[0][1]], self.ncomp[modn][ann_center,cuben],evals_matrix, evecs_matrix,
                                           KL_basis_matrix,sci_mean_sub_matrix,refs_mean_sub_matrix, self.pa[cuben], self.fwhm, pa_threshold, ann_center)
                                psf_map[b,indices[0][0], indices[0][1]]=psf_map_temp-np.mean(psf_map_temp)


                            psf_map_der = cube_derotate(psf_map, self.pa[cuben], imlib='opencv',interpolation='lanczos4')
                            psfm_temp=cube_crop_frames(psf_map_der,int(2*round(self.fwhm)+1),xy=(poscentx,poscenty),verbose=False)
                            psf_fm_out[i,:,:,:]=psfm_temp
                            
                    else:
                            psfm_temp=self.psf_fm[cuben][modn][ann_center][i,:,:,:]
                            psf_fm_out[i,:,:,:]=psfm_temp
                            
                #PSF forward model computation for LOCI
                            
                if self.model[modn]=='FM LOCI':
                    
                    an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
                    theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))  
                    
                        
                    if psf_formod==False:
                        
                            model_matrix=cube_inject_companions(np.zeros_like(mcube), self.psf[cuben], self.pa[cuben], flevel=1, plsc=0.1, 
                                 rad_dists=an_dist, theta=theta, n_branches=1,verbose=False)
                    
                            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)
                    
                            values_fc = model_matrix[:, indices[0][0], indices[0][1]]

                            cube_res_fc=np.zeros_like(model_matrix)
                        
                            matrix_res_fc = np.zeros((values_fc.shape[0], indices[0][0].shape[0]))
                        
                            for e in range(values_fc.shape[0]):
                            
                                recon_fc = np.dot(coef_list[e], values_fc[ind_ref_list[e]])
                                matrix_res_fc[e] = values_fc[e] - recon_fc
        
                            cube_res_fc[:, indices[0][0], indices[0][1]] = matrix_res_fc
                            cube_der_fc = cube_derotate(cube_res_fc-np.mean(cube_res_fc), self.pa[cuben], imlib='opencv', interpolation='lanczos4')
                            psfm_temp=cube_crop_frames(cube_der_fc,int(2*round(self.fwhm)+1),xy=(poscentx,poscenty),verbose=False)
                            psf_fm_out[i,:,:,:]=psfm_temp
                            
                    else:
                            psfm_temp=self.psf_fm[cuben][modn][ann_center][i,:,:,:]
                            psf_fm_out[i,:,:,:]=psfm_temp
                            
                #Flux parameter estimation via Gaussian maximum likelihood (matched filtering)
                            
                if self.intensity[modn][ann_center,cuben]=='Pixel':
                    
                    flux_esti=np.zeros((self.crop_range[modn]))
                
                    for v in range(0,self.crop_range[modn]):

                        cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                        num=[]
                        denom=[]
                    
                        for j in range(n*z): 
                            
                            if self.var[modn][ann_center,cuben]=='FR':
                                svar=var_f[j,v]
            
                            elif self.var[modn][ann_center,cuben]=='FM' :
                                svar=var_f[i,j,v]
                                    
                            elif self.var[modn][ann_center,cuben]=='TE':
                                svar=var_f[i,j,v]

                            if psfm_temp is not None:
                                psfm_temp2=psfm_temp[j]
                            else:
                                psfm_temp2=psf_temp
                        
                            if psfm_temp2.shape[1]==cropf:
                                psfm=psfm_temp2
                            else:
                                if len(psfm_temp2.shape)==2:
                                    psfm=frame_crop(psfm_temp2,cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
                                else:
                                    psfm=frame_crop(psfm_temp2[int(j//n_t)],cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
                                
                            num.append(np.multiply(frame_crop(mcube[j],cropf,cenxy=[poscentx,poscenty],verbose=False),psfm).sum()/svar)
                            denom.append(np.multiply(psfm,psfm).sum()/svar)
                        
                        flux_esti[v]=sum(num)/sum(denom)
                        probcube[n,indicesy[i],indicesx[i],0,0,v]=flux_esti[v]
                        
                # Reverse the temporal direction when moving from one patch to the next one to respect the temporal proximity of the pixels and limit potential non-linearity
                        
                if i%2==1:
                    range_sel=range(n*z)
                else:
                    range_sel=range(n*z-1,-1,-1)
                    
                # Likelihood computation for the patch i
                
                for j in range_sel: 
                    
                    for m in range(range_int):

                        if psfm_temp is not None:
                                psfm_temp2=psfm_temp[j]
                        else:
                                psfm_temp2=psf_temp
                                
                        for v in range(0,self.crop_range[modn]):
                            
                            cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                            if psfm_temp2.shape[1]==cropf:
                                psfm=psfm_temp2
                            else:
                                if len(psfm_temp2.shape)==2:
                                    psfm=frame_crop(psfm_temp2,cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
                                else:
                                    psfm=frame_crop(psfm_temp2[int(j//n_t)],cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
                                

                            if self.var[modn][ann_center,cuben]=='ST':
                                svar=var[v]
                                alpha=mean[v]
                                mv=mixval[v]
                                sel_distri=distrim[v]
                                maxhist=max_hist[v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var_f[v])
            
                            elif self.var[modn][ann_center,cuben]=='FR':
                                svar=var[j,v]
                                alpha=mean[j,v]
                                mv=mixval[j,v]
                                sel_distri=distrim[j,v]
                                maxhist=max_hist[j,v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var_f[j,v])
            
                            elif self.var[modn][ann_center,cuben]=='SM':
                                svar=var[i,v]
                                alpha=mean[i,v]
                                mv=mixval[i,v]
                                sel_distri=distrim[i,v] 
                                maxhist=max_hist[i,v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var_f[i,v])
            
                            elif self.var[modn][ann_center,cuben]=='FM' :
                                svar=var[i,j,v]
                                alpha=mean[i,j,v]
                                mv=mixval[i,j,v]
                                sel_distri=distrim[i,j,v] 
                                maxhist=max_hist[i,j,v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var_f[i,j,v])
                                    
                            elif self.var[modn][ann_center,cuben]=='TE':
                                svar=var[i,j,v]
                                alpha=mean[i,j,v]
                                mv=mixval[i,j,v]
                                sel_distri=distrim[i,j,v]  
                                maxhist=max_hist[i,j,v]
                                phi[1]=self.interval[modn][ann_center,cuben][m]*np.sqrt(var_f[i,j,v])
                                
                            if self.intensity[modn][ann_center,cuben]=='Pixel':
                                phi[1]=np.where(flux_esti[v]<=0,0,flux_esti[v]) 
                                
                            if self.intensity[modn][ann_center,cuben]=='Annulus' and (self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI'):
                                phi[1]=5*phi[1]
                            
                            
                            for l in range(0,2):

                                #Likelihood estimation
                                
                                ff=frame_crop(mcube[j],cropf,cenxy=[poscentx,poscenty],verbose=False)-phi[l]*psfm-alpha
                                if sel_distri==0:
                                        cftemp=(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)
                                elif sel_distri==1:
                                        cftemp=1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar))
                                elif sel_distri==2:
                                        cftemp=(mv*(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)+(1-mv)*1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar)))
                                elif sel_distri==3:
                                        abs_x=abs(ff)
                                        cftemp=maxhist*np.exp(-np.where(abs_x < svar, mv * abs_x**2,2*svar*mv*abs_x -mv*svar**2))

                                probcube[int(cubind),int(indicesy[i]),int(indicesx[i]),int(m),l,v]=cftemp.sum()
                
                    cubind+=1
            
            if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                return probcube,psf_fm_out
            else:
                return probcube
        
        
        if verbose==True:
            print("Radial distance: "+"{}".format(ann_center)) 

        #Estimation of the KLIP cube of residuals for the selected annulus
        
        evals_matrix=[]
        evecs_matrix=[]
        KL_basis_matrix=[]
        refs_mean_sub_matrix=[]
        sci_mean_sub_matrix=[]
        resicube_klip=None
        
        ind_ref_list=None
        coef_list=None
        
        if self.opti==True and cube_fc is not None:
            
            cube_test=cube_fc+self.cube[cuben]
        else:
            cube_test=self.cube[cuben]
            
            
        if self.model[modn]=='FM KLIP' or (self.opti==True and self.model[modn]=='KLIP'):
                       

            resicube_klip=np.zeros_like(self.cube[cuben])
            

            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * (ann_center))))
            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
            if pa_threshold >= mid_range - mid_range * 0.1:
                pa_threshold = float(mid_range - mid_range * 0.1)
            
            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)

            for k in range(0,self.cube[cuben].shape[0]):

                evals_temp,evecs_temp,KL_basis_temp,sub_img_rows_temp,refs_mean_sub_temp,sci_mean_sub_temp =KLIP_patch(k,cube_test[:, indices[0][0], indices[0][1]], self.ncomp[modn][ann_center,cuben], self.pa[cuben], self.asize[modn], pa_threshold, ann_center)
                resicube_klip[k,indices[0][0], indices[0][1]] = sub_img_rows_temp

                evals_matrix.append(evals_temp)
                evecs_matrix.append(evecs_temp)
                KL_basis_matrix.append(KL_basis_temp)
                refs_mean_sub_matrix.append(refs_mean_sub_temp)
                sci_mean_sub_matrix.append(sci_mean_sub_temp)

            mcube=cube_derotate(resicube_klip,self.pa[cuben],imlib=self.imlib, interpolation=self.interpolation)


        elif self.model[modn]=='FM LOCI':
            
            
            resicube, ind_ref_list,coef_list=LOCI_FM(cube_test, self.psf[cuben], ann_center, self.pa[cuben],None, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
            mcube=cube_derotate(resicube,self.pa[cuben])
            
        # Computation of the annular LOCI (used during the parameter optimization) 
        
        elif (self.opti==True and self.model[modn]=='LOCI'):
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            
            if scale_list is not None:
                resicube=np.zeros_like(cube_rot_scale)
                for i in range(int(max(scale_list)*ann_center/self.asize[modn])):
                    indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
                    resicube[:,indices[0][0], indices[0][1]]=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center,angle_list,scale_list, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],self.delta_sep[modn][ann_center,cuben])[0][:,indices[0][0], indices[0][1]]
            
            else:
                resicube, ind_ref_list,coef_list=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center, self.pa[cuben],None, self.fwhm, self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
 
            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)

        # Computation of the annular APCA (used during the parameter optimization) 

        elif (self.opti==True and self.model[modn]=='APCA'):
            
                  
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                
                pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * (ann_center+self.asize[modn]*i))))
                mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                if pa_threshold >= mid_range - mid_range * 0.1:
                    pa_threshold = float(mid_range - mid_range * 0.1)
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
    
                for k in range(0,cube_rot_scale.shape[0]):
                    for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                        resicube[k,indices[l][0], indices[l][1]],v_resi,data_shape=do_pca_patch(cube_rot_scale[:, indices[l][0], indices[l][1]], k,  angle_list,scale_list, self.fwhm, pa_threshold,self.delta_sep[modn][ann_center,cuben], ann_center+self.asize[modn]*i,
                           svd_mode='lapack', ncomp=self.ncomp[modn][ann_center,cuben],min_frames_lib=2, max_frames_lib=200, tol=1e-1,matrix_ref=None)
            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            
        # Computation of the annular NMF (used during the parameter optimization) 
                
        elif (self.opti==True and self.model[modn]=='NMF'):
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if self.opti_mode=='full-frame':
                matrix = prepare_matrix(cube_rot_scale, None, None, mode='fullfr',
                            verbose=False)
                res= NMF_patch(matrix, ncomp=self.ncomp[modn][ann_center,cuben], max_iter=50000,random_state=None)
                resicube=np.reshape(res,(cube_rot_scale.shape[0],cube_rot_scale.shape[1],cube_rot_scale.shape[2]))
            else:
                if scale_list is not None:
                    range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
                else:
                    range_adisdi=range(1)
                    
                for i in range_adisdi:
                    indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
                    
                    for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                        resicube[:,indices[l][0], indices[l][1]]= NMF_patch(cube_rot_scale[:, indices[l][0], indices[l][1]], ncomp=self.ncomp[modn][ann_center,cuben], max_iter=100,random_state=None)

            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)

        # Computation of the annular LLSG (used during the parameter optimization) 
                
        elif (self.opti==True and self.model[modn]=='LLSG'):

            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube_test,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
                for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                    resicube[:,indices[l][0], indices[l][1]]= _decompose_patch(indices,l, cube_rot_scale,self.nsegments[modn][ann_center,cuben],
                            self.rank[modn][ann_center,cuben], low_rank_ref=False, low_rank_mode='svd', thresh=1,thresh_mode='soft', max_iter=40, auto_rank_mode='noise', cevr=0.9,
                                 residuals_tol=1e-1, random_seed=10, debug=False, full_output=False).T

            mcube=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
          

        zero_test=abs(mcube.sum(axis=1).sum(axis=1))
        if np.min(zero_test)==0:
            mcube[np.argmin(zero_test),:,:]=mcube.mean(axis=0)
            
        # Fitness error computation for the noise distribution(s),
        # if automated probability distribution selection is activated (var='A')
        # the fitness errors allow the determination of the optimal distribution


        def vm_esti(modn,arr,var_e,mean_e):
            
            def gaus(x,x0,var):
                return 1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))
    
            def lap(x,x0,var):
                bap=np.sqrt(var/2)
                return (1/(2*bap))*np.exp(-np.abs(x-x0)/bap)
    
            def mix(x,x0,var,a):
                bap=np.sqrt(var/2)
                return a*(1/(2*bap))*np.exp(-np.abs(x-x0)/bap)+(1-a)*1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))
            
            def huber_loss(x,x0,delta,a):
                abs_x=abs(x-x0)
                
                huber_esti=np.exp(-a*np.where(abs_x < delta, 0.5*abs_x ** 2, delta*abs_x -0.5*delta**2))
                return huber_esti
            
            def te_f_mh(func1,func2,bin_m,hist,p0_1,bounds_1,p0_2,bounds_2,mean,var,distri):
               try: 
                   popt,pcov = curve_fit(func1,bin_m,hist,p0=p0_1,bounds=bounds_1)
                   fiter=sum(abs(func1(bin_m,*popt)-hist))
                   mean,var,a=popt
               except (RuntimeError, ValueError, RuntimeWarning):
                   try:
                       popt,pcov = curve_fit(func2,bin_m,hist,p0=p0_2,bounds=bounds_2)
                       fiter=sum(abs(func2(bin_m,*popt)-hist))
                       a=popt
                   except (RuntimeError, ValueError, RuntimeWarning):
                       a=0.01/max(abs(bin_m))
                       fiter=sum(abs(func2(bin_m,a)-hist))
               return mean,a,var,fiter,distri
                   
            def te_f_gl(func,bin_m,hist,p0_1,bounds_1,mean,var,distri):
               try: 
                   popt,pcov = curve_fit(func,bin_m,hist,p0=p0_1,bounds=bounds_1)
                   mean,var=popt
                   fiter=sum(abs(func(bin_m,*popt)-hist))
               except (RuntimeError, ValueError, RuntimeWarning): 
                   mean,var=[mean,var]
                   fiter=sum(abs(func(bin_m,mean,var)-hist))
               return mean,None,var,fiter,distri   

            def te_h(func1,func2,bin_m,hist,p0_1,bounds_1,p0_2,bounds_2,mean,delta,distri):
               
               try: 
                   popt,pcov = curve_fit(func1,bin_m,hist,p0=p0_1,bounds=bounds_1)
                   delta,a=popt
                   fiter=sum(abs(func1(bin_m,*popt)-hist))
               except (RuntimeError, ValueError, RuntimeWarning):
                   try:
                       popt,pcov = curve_fit(func2,bin_m,hist,p0=p0_2,bounds=bounds_2)
                       fiter=sum(abs(func2(bin_m,*popt)-hist))
                       a=popt
                   except (RuntimeError, ValueError, RuntimeWarning):
                        a=0.01/max(abs(bin_m))
                        fiter=sum(abs(func2(bin_m,a)-hist))
                        

               return mean,a,delta,fiter,distri
           
            def te_m(func,bin_m,hist,p0,bounds,mean,var,distri):
                try:
                   popt,pcov = curve_fit(func,bin_m,hist,p0=p0,bounds=bounds)
                   mixval=popt
                   return mean,mixval,var,sum(abs(func(bin_m,*popt)-hist)),distri
               
                except (RuntimeError, ValueError, RuntimeWarning):
                    return mean,0.5,var,sum(abs(func(bin_m,0.5)-hist)),distri
                
                   
            def te_gl(func,bin_m,hist,mean,var,distri):
               return mean,None,var,sum(abs(func(bin_m,mean,var)-hist)),distri 
           

            mixval_temp=None

            if self.model[modn]=='LLSG':
                hist, bin_edge =np.histogram(arr,bins=np.max([20,int(len(arr)/20)]),density=True)
            else:
                try:
                    hist, bin_edge =np.histogram(arr,bins='auto',density=True)
                except (IndexError, TypeError, ValueError,MemoryError):
                    hist, bin_edge =np.histogram(arr,bins=np.max([20,int(len(arr)/20)]),density=True)
                
            bin_mid=(bin_edge[0:(len(bin_edge)-1)]+bin_edge[1:len(bin_edge)])/2
            
            if self.distrifit[modn][ann_center,cuben]==False:
                
               fixmix = lambda binm, mv: mix(binm,mean_e,var_e,mv)    
               hl1 = lambda binm, delta,a: huber_loss(binm,mean_e,delta,a)
               hl2 = lambda binm, a: huber_loss(binm,mean_e,np.mean(abs(bin_mid)),a)

               if self.distri[modn]=='G':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_gl(gaus,bin_mid,hist,mean_e,var_e,0)

               elif self.distri[modn]=='L':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_gl(lap,bin_mid,hist,mean_e,var_e,1)

                   
               elif self.distri[modn]=='M':            
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_m(fixmix,bin_mid,hist,[0.5],[(0),(1)],mean_e,var_e,2)
                   
               elif self.distri[modn]=='H':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_h(hl1,hl2,bin_mid,hist/max(hist),[np.mean(abs(bin_mid)),0.01/max(abs(bin_mid))],[(min(abs(bin_mid)),0.0001/max(abs(bin_mid))),(max(abs(bin_mid)),2)],[0.01/max(abs(bin_mid))],[(0.0001),(2)],mean_e,1,3)
                   fiterr_temp=fiterr_temp*max(hist)

               elif self.distri[modn]=='A':
                    res=[]
                    res.append(te_gl(gaus,bin_mid,hist,mean_e,var_e,0))
                    res.append(te_gl(lap,bin_mid,hist,mean_e,var_e,1))
                    res.append(te_m(fixmix,bin_mid,hist,[0.5],[(0),(1)],mean_e,var_e,2))
                    res.append(te_h(hl1,hl2,bin_mid,hist/max(hist),[np.mean(abs(bin_mid)),0.01/max(abs(bin_mid))],[(min(abs(bin_mid)),0.0001/max(abs(bin_mid))),(max(abs(bin_mid)),2)],[0.01/max(abs(bin_mid))],[(0.0001),(2)],mean_e,1,3))
                    
                    fiterr=list([res[0][3],res[1][3],res[2][3],res[3][3]*max(hist)])
                    distrim_temp=fiterr.index(min(fiterr))
                    fiterr_temp=min(fiterr)
                   
                    mean_temp=res[distrim_temp][0]
                    mixval_temp=res[distrim_temp][1]
                    var_temp=res[distrim_temp][2]
                    fiterr_temp=res[distrim_temp][3]

            else:
                
               fixmix = lambda binm, mv: mix(binm,mean_e,var_e,mv)
               hl = lambda binm, a: huber_loss(binm,mean_e,np.mean(abs(bin_mid)),a)
               
               if self.distri[modn]=='G':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_f_gl(gaus,bin_mid,hist,[mean_e,var_e],[(mean_e-np.sqrt(var_e),0),(mean_e+np.sqrt(var_e),2*var_e)],mean_e,var_e,0)

               elif self.distri[modn]=='L':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_f_gl(lap,bin_mid,hist,[mean_e,var_e],[(mean_e-np.sqrt(var_e),0),(mean_e+np.sqrt(var_e),2*var_e)],mean_e,var_e,1)

               elif self.distri[modn]=='M':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_f_mh(mix,fixmix,bin_mid,hist,[mean_e,var_e,0.5],[(mean_e-np.sqrt(var_e),0,0),(mean_e+np.sqrt(var_e),4*var_e,1)],[0.5],[(0),(1)],mean_e,var_e,2)

               elif self.distri[modn]=='H':
                   mean_temp,mixval_temp,var_temp,fiterr_temp,distrim_temp=te_f_mh(huber_loss,hl,bin_mid,hist/max(hist),[mean_e,np.mean(abs(bin_mid)),0.01/max(abs(bin_mid))],[(mean_e-np.sqrt(var_e),min(abs(bin_mid)),0.0001/max(abs(bin_mid))),(mean_e+np.sqrt(var_e),max(abs(bin_mid)),2)],[0.01/max(abs(bin_mid))],[(0.0001),(2)],mean_e,1,3)
                   fiterr_temp=fiterr_temp*max(hist)
                   
               elif self.distri[modn]=='A':
                   
                   res=[]
                   res.append(te_f_gl(gaus,bin_mid,hist,[mean_e,var_e],[(mean_e-np.sqrt(var_e),1e-10),(mean_e+np.sqrt(var_e),2*var_e)],mean_e,var_e,0))
                   res.append(te_f_gl(lap,bin_mid,hist,[mean_e,var_e],[(mean_e-np.sqrt(var_e),1e-10),(mean_e+np.sqrt(var_e),2*var_e)],mean_e,var_e,1))
                   res.append(te_f_mh(mix,fixmix,bin_mid,hist,[mean_e,var_e,0.5],[(mean_e-np.sqrt(var_e),1e-10,0),(mean_e+np.sqrt(var_e),4*var_e,1)],[0.5],[(0),(1)],mean_e,var_e,2))                   
                   res.append(te_f_mh(huber_loss,hl,bin_mid,hist/max(hist),[mean_e,np.mean(abs(bin_mid)),0.01/max(abs(bin_mid))],[(mean_e-np.sqrt(var_e),min(abs(bin_mid)),0.0001/max(abs(bin_mid))),(mean_e+np.sqrt(var_e),max(abs(bin_mid)),2)],[0.01/max(abs(bin_mid))],[(0.0001),(2)],mean_e,1,3))
                   fiterr=list([res[0][3],res[1][3],res[2][3],res[3][3]*max(hist)])
                   distrim_temp=fiterr.index(min(fiterr))
                   fiterr_temp=min(fiterr)
                   
                   mean_temp=res[distrim_temp][0]
                   mixval_temp=res[distrim_temp][1]
                   var_temp=res[distrim_temp][2]
                   fiterr_temp=res[distrim_temp][3]
                
                
            
            return mean_temp,var_temp,fiterr_temp,mixval_temp,distrim_temp,max(hist)

        # Noise distribution parameter estimation considering the selected region (ST, F, FM, SM or T)
                                    
        var_f=None
        
        
        if self.var[modn][ann_center,cuben]=='ST':
            
            var=np.zeros(self.crop_range[modn])
            var_f=np.zeros(self.crop_range[modn])
            mean=np.zeros(self.crop_range[modn])
            mixval=np.zeros(self.crop_range[modn])
            fiterr=np.zeros(self.crop_range[modn])
            distrim=np.zeros(self.crop_range[modn])
            max_hist=np.zeros(self.crop_range[modn])
            
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)

                poscentx=indices[0][1]
                poscenty=indices[0][0]
                
                arr = np.ndarray.flatten(mcube[:,poscenty,poscentx])
            
                mean[v],var[v],fiterr[v],mixval[v],distrim[v],max_hist[v]=vm_esti(modn,arr,np.var(mcube[:,poscenty,poscentx]),np.mean(mcube[:,poscenty,poscentx]))
                
                var_f[v]=np.var(mcube[:,poscenty,poscentx])
            
        elif self.var[modn][ann_center,cuben]=='FR':
            
            var=np.zeros(((n*z),self.crop_range[modn]))
            var_f=np.zeros(((n*z),self.crop_range[modn]))
            mean=np.zeros(((n*z),self.crop_range[modn]))
            mixval=np.zeros(((n*z),self.crop_range[modn]))
            fiterr=np.zeros(((n*z),self.crop_range[modn]))
            distrim=np.zeros(((n*z),self.crop_range[modn]))
            max_hist=np.zeros(((n*z),self.crop_range[modn]))
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)

                poscentx=indices[0][1]
                poscenty=indices[0][0]
            
                for a in range((n*z)):
           
                    arr = np.ndarray.flatten(mcube[a,poscenty,poscentx])
            
                    var_f[a,v]=np.var(mcube[a,poscenty,poscentx])
                
                    if var_f[a,v]==0:
                        mean[a,v]=mean[a-1,v]
                        var[a,v]=var[a-1,v]
                        fiterr[a,v]=fiterr[a-1,v]
                        mixval[a,v]=mixval[a-1,v]
                        distrim[a,v]=distrim[a-1,v]
                        max_hist[a,v]=max_hist[a-1,v]
                        var_f[a,v]=var_f[a-1,v]
                    else:    
                    
                        mean[a,v],var[a,v],fiterr[a,v],mixval[a,v],distrim[a,v],max_hist[a,v]=vm_esti(modn,arr,np.var(mcube[a,poscenty,poscentx]),np.mean(mcube[a,poscenty,poscentx]))

        elif self.var[modn][ann_center,cuben]=='SM':
            
            indicesy,indicesx=get_time_series(mcube,ann_center)
            
            var=np.zeros((len(indicesy),self.crop_range[modn]))
            var_f=np.zeros((len(indicesy),self.crop_range[modn]))
            mean=np.zeros((len(indicesy),self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),self.crop_range[modn]))
            distrim=np.zeros((len(indicesy),self.crop_range[modn]))
            max_hist=np.zeros((len(indicesy),self.crop_range[modn]))
            size_seg=2
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
            
                for a in range(len(indicesy)):
            
                    if (a+int(cropf*3/2)+size_seg)>(len(indicesy)-1):
                        posup= a+int(cropf*3/2)+size_seg-len(indicesy)-1
                    else:
                        posup=a+int(cropf*3/2)+size_seg
               
                    indc=disk((indicesy[a], indicesx[a]),cropf/2)
           
                    radist_b=np.sqrt((indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))**2+(indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))**2)
           
                    if (indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))>=0:
                        ang_b= np.arccos((indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))/radist_b)/np.pi*180                        
                    else:
                        ang_b= 360-np.arccos((indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))/radist_b)/np.pi*180
          
                    radist_e=np.sqrt((indicesx[posup]-int(x/2))**2+(indicesy[posup]-int(y/2))**2)
           
                    if (indicesx[posup]-int(x/2))>=0:
                        ang_e= np.arccos((indicesy[posup]-int(y/2))/radist_e)/np.pi*180
                    else:
                        ang_e= 360-np.arccos((indicesy[posup]-int(y/2))/radist_e)/np.pi*180
                                     
                    if ang_e<ang_b:
                        diffang=(360-ang_b)+ang_e
                    else:
                        diffang=ang_e-ang_b

            
                    indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,int(360/diffang),ang_b)
                    positionx=[]
                    positiony=[]
           
                    for k in range(0,len(indices[0][1])):
                        if len(set(np.where(indices[0][1][k]==indc[1])[0]) & set(np.where(indices[0][0][k]==indc[0])[0]))==0:
                            positionx.append(indices[0][1][k])
                            positiony.append(indices[0][0][k])

        
                    arr = np.ndarray.flatten(mcube[:,positiony,positionx])
                    var_f[a,v]=np.var(mcube[:,positiony,positionx])
                
                    if var_f[a,v]==0:
                        mean[a,v]=mean[a-1,v]
                        var[a,v]=var[a-1,v]
                        fiterr[a,v]=fiterr[a-1,v]
                        mixval[a,v]=mixval[a-1,v]
                        distrim[a,v]=distrim[a-1,v]
                        max_hist[a,v]=max_hist[a-1,v]
                        var_f[a,v]=var_f[a-1,v]
                    else:    
                        mean[a,v],var[a,v],fiterr[a,v],mixval[a,v],distrim[a,v],max_hist[a,v]=vm_esti(modn,arr,np.var(mcube[:,positiony,positionx]),np.mean(mcube[:,positiony,positionx]))


                    
        elif self.var[modn][ann_center,cuben]=='FM' :
            
            indicesy,indicesx=get_time_series(mcube,ann_center)
            

            var=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            var_f=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            mean=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            distrim=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            max_hist=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)
            
                for a in range(len(indicesy)):
         
                    indc=disk((indicesy[a], indicesx[a]),3)
                    positionx=[]
                    positiony=[]
        
                    for k in range(0,len(indices[0][1])):
                        if len(set(np.where(indices[0][1][k]==indc[1])[0]) & set(np.where(indices[0][0][k]==indc[0])[0]))==0:
                            positionx.append(indices[0][1][k])
                            positiony.append(indices[0][0][k])
                
                    for b in range((n*z)):
           
                        arr = np.ndarray.flatten(mcube[b,positiony,positionx])
            
                        var_f[a,b,v]=np.var(mcube[b,positiony,positionx])
                        if var_f[a,b,v]==0:
                            mean[a,b,v]=mean[a,b-1,v]
                            var[a,b,v]=var[a,b-1,v]
                            fiterr[a,b,v]=fiterr[a,b-1,v]
                            mixval[a,b,v]=mixval[a,b-1,v]
                            distrim[a,b,v]=distrim[a,b-1,v]
                            max_hist[a,b,v]=max_hist[a,b-1,v]
                            var_f[a,b,v]=var_f[a,b-1,v]
                        else:    
                            mean[a,b,v],var[a,b,v],fiterr[a,b,v],mixval[a,b,v],distrim[a,b,v],max_hist[a,b,v]=vm_esti(modn,arr,np.var(np.asarray(mcube[b,positiony,positionx])),np.mean(np.asarray(mcube[b,positiony,positionx])))
            
                        
                            
        elif self.var[modn][ann_center,cuben]=='TE' :
        
            indicesy,indicesx=get_time_series(mcube,ann_center)
            
            var=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            var_f=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            mean=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))  
            distrim=np.zeros((len(indicesy),(n*z),self.crop_range[modn])) 
            max_hist=np.zeros((len(indicesy),(n*z),self.crop_range[modn]))
            
            if self.cube[cuben].ndim==4:
                pa_temp=np.hstack([self.pa[cuben]]*self.cube[cuben].shape[0])
            else:
                pa_temp=self.pa[cuben]
                
            mcube_derot=cube_derotate(mcube,-pa_temp)
            
            for v in range(0,self.crop_range[modn]):
                cropf=int(self.crop[modn][ann_center,cuben]+2*v)
                for a in range(0,len(indicesy)):
                    
                    
         
                    radist=np.sqrt((indicesx[a]-int(x/2))**2+(indicesy[a]-int(y/2))**2)
           
                    if (indicesy[a]-int(y/2))>=0:
                        ang_s= np.arccos((indicesx[a]-int(x/2))/radist)/np.pi*180
                    else:
                        ang_s= 360-np.arccos((indicesx[a]-int(x/2))/radist)/np.pi*180
                
                    for b in range((n*z)):
           
                        twopi=2*np.pi
                        sigposy=int(y/2 + np.sin((ang_s-pa_temp[b])/360*twopi)*radist)
                        sigposx=int(x/2+ np.cos((ang_s-pa_temp[b])/360*twopi)*radist)
           
           
                        y0 = int(sigposy - int(cropf/2))
                        y1 = int(sigposy + int(cropf/2)+1)  # +1 cause endpoint is excluded when slicing
                        x0 = int(sigposx - int(cropf/2))
                        x1 = int(sigposx + int(cropf/2)+1)
           
                   
                        mask = np.ones(mcube_derot.shape[0],dtype=bool)
                        mask[b]=False
                        mcube_sel=mcube_derot[mask,y0:y1,x0:x1]
           
           
                        arr = np.ndarray.flatten(mcube_sel)
                
                        var_f[a,b,v]=np.var(np.asarray(mcube_sel))
                        if var_f[a,b,v]==0:
                            mean[a,b,v]=mean[a,b-1,v]
                            var[a,b,v]=var[a,b-1,v]
                            fiterr[a,b,v]=fiterr[a,b-1,v]
                            mixval[a,b,v]=mixval[a,b-1,v]
                            distrim[a,b,v]=distrim[a,b-1,v]
                            max_hist[a,b,v]=max_hist[a,b-1,v]
                            var_f[a,b,v]=var_f[a,b-1,v]
                        else: 

                            mean[a,b,v],var[a,b,v],fiterr[a,b,v],mixval[a,b,v],distrim[a,b,v],max_hist[a,b,v]=vm_esti(modn,arr,np.var(np.asarray(mcube_sel)),np.mean(np.asarray(mcube_sel)))
         

        #Estimation of the cube of likelihoods
        #print(np.bincount(distrim.flatten(order='C').astype(int)))
        res=likfcn(cuben,modn,mean,var,mixval,max_hist,mcube,ann_center,distrim,evals_matrix,evecs_matrix, KL_basis_matrix,refs_mean_sub_matrix,sci_mean_sub_matrix,resicube_klip,likemap,var_f,ind_ref_list,coef_list)
        indicesy,indicesx=get_time_series(mcube,ann_center)
        if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
            return ann_center,res[0][:,indicesy,indicesx],res[1]
        else:
            return ann_center,res[:,indicesy,indicesx]
                       

        
    def lik_esti(self, sel_cube=None,showplot=False,verbose=True):
        
        """
        Function allowing the estimation of the likelihood of being in either the planetary regime 
        or the background regime for the different cubes. The likelihood computation is based on 
        the residual cubes generated  with the considered set of models.
        
        Parameters
        ----------
    
        showplot: bool, optional
            If True, provides the plots of the final residual frames for the selected 
            ADI-based post-processing techniques along with the final RSM map. Default is False.
        fulloutput: bool, optional
            If True, provides the selected distribution, the fitness erros and the mixval 
            (for distri='mix') for every annulus in respectively obj.distrisel, obj.fiterr
            and obj.mixval (the length of these lists are equall to maxradius - minradius, the
            size of the matrix for each annulus depends on the approach selected for the variance
            estimation, see var in add_model)
        verbose : bool, optional
            If True prints intermediate info. Default is True.
        """
        

        def init(probi):
            global probcube
            probcube = probi
            
         
        for i in range(len(self.model)):
            
            for j in range(len(self.cube)):
                
                #Computation of the cube of residuals
                if sel_cube is None or [j,i] in sel_cube:

                    if self.opti==False:
                
                        if self.model[i]=='APCA':
                            if verbose:
                                print("Annular PCA estimation") 
        
                            residuals_cube_, frame_fin = annular_pca_adisdi(self.cube[j], self.pa[j], self.scale_list[j], fwhm=self.fwhm, ncomp=self.ncomp[i][0,j], asize=self.asize[i], radius_int=self.minradius, radius_out=self.maxradius,
                                      delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j], svd_mode='lapack', n_segments=int(self.nsegments[i][0,j]), nproc=self.ncore,full_output=True,verbose=False)
                            if showplot:
                                plot_frames(frame_fin,title='APCA', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                        elif self.model[i]=='NMF':
                            if verbose:
                                print("NMF estimation") 
                            residuals_cube_, frame_fin = nmf_adisdi(self.cube[j], self.pa[j], self.scale_list[j], ncomp=self.ncomp[i][0,j], max_iter=5000, random_state=0, mask_center_px=None,full_output=True,verbose=False)
                            if showplot:
                                plot_frames(frame_fin,title='NMF', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                        elif self.model[i]=='LLSG':
                            if verbose:
                                print("LLSGestimation") 
        
                            residuals_cube_, frame_fin = llsg_adisdi(self.cube[j], self.pa[j],self.scale_list[j], self.fwhm, rank=self.rank[i][0,j],asize=self.asize[i], thresh=1,n_segments=int(self.nsegments[i][0,j]),radius_int=self.minradius, radius_out=self.maxradius, max_iter=40, random_seed=10, nproc=self.ncore,full_output=True,verbose=False)
                           
                            if showplot:
                                plot_frames(frame_fin,title='LLSG', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                        elif self.model[i]=='LOCI':
                            if verbose:
                                print("LOCI estimation") 
                            residuals_cube_,frame_fin=loci_adisdi(self.cube[j], self.pa[j],self.scale_list[j], fwhm=self.fwhm,asize=self.asize[i],radius_int=self.minradius, radius_out=self.maxradius, n_segments=int(self.nsegments[i][0,j]),tol=self.tolerance[i][0,j], nproc=self.ncore, optim_scale_fact=2,delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j],verbose=False,full_output=True)
                            if showplot:
                                plot_frames(frame_fin,title='LOCI', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
                
                        elif self.model[i]=='KLIP':
                            if verbose:
                                print("KLIP estimation") 
                            cube_out, residuals_cube_, frame_fin = KLIP(self.cube[j], self.pa[j], ncomp=self.ncomp[i][0,j], fwhm=self.fwhm, asize=self.asize[i],radius_int=self.minradius, radius_out=self.maxradius, 
                                      delta_rot=self.delta_rot[i][0,j],full_output=True,verbose=False)
                            if showplot:
                                plot_frames(frame_fin,title='KLIP', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
                            
                        elif self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                            residuals_cube_=np.zeros_like(self.cube[j])
                            frame_fin=np.zeros_like(self.cube[j][0])
                            
                        zero_test=abs(residuals_cube_.sum(axis=1).sum(axis=1))
                        if np.min(zero_test)==0:
                            residuals_cube_[np.argmin(zero_test),:,:]=residuals_cube_.mean(axis=0)
                    else:
                        residuals_cube_=np.zeros_like(rot_scale('ini',self.cube[j],None,self.pa[j],self.scale_list[j],self.imlib, self.interpolation)[0])
                        frame_fin=np.zeros_like(residuals_cube_[0])
                        
                    #Likelihood computation for the different models and cubes
                    
        
                    if self.model[i]=='FM KLIP' or self.model[i]=='FM LOCI':
                        max_rad=np.min([self.max_r+1,self.maxradius+1])
                    else:
                        max_rad=self.maxradius+1
                    like_temp=np.zeros(((residuals_cube_.shape[0]+1),residuals_cube_.shape[1],residuals_cube_.shape[2],len(self.interval[i][0,j]),2,self.crop_range[i]))    


                    if self.ncore==1:
            
                        for e in range(self.minradius,max_rad):
                            res=self.likelihood(e,j,i,residuals_cube_,None,verbose)

                            indicesy,indicesx=get_time_series(self.cube[0],res[0])
                            if self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                                like_temp[:,indicesy,indicesx,:,:,:]=res[1]
                                #self.psf_fm[j][i][res[0]]=res[2]
                            else:
                                like_temp[:,indicesy,indicesx,:,:,:]=res[1] 
                    else:
                        
                        X_shape=residuals_cube_.shape
                        X = RawArray('d', int(np.prod(X_shape)))
                        X_np = np.frombuffer(X).reshape(X_shape)
                        np.copyto(X_np, residuals_cube_)
                         
                        #Sometimes the parallel computation get stuck, to avoid that we impose a maximum computation
                        #time of 0.05 x number of frame x number of pixels in the last annulus x 2 x pi (0.05 second
                        #per treated pixel)
                        time_out=0.05*residuals_cube_.shape[0]*2*max_rad**np.pi
                        results=[]    
                        pool=Pool(processes=self.ncore, initializer=init_worker, initargs=(X, X_shape))           
                        for e in range(self.minradius,max_rad):
                            results.append(pool.apply_async(self.likelihood,args=(e,j,i,0,None,verbose)))
                        #[result.wait(timeout=time_out) for result in results]
                        
                        it=self.minradius
                        for result in results:
                            try:
                                res=result.get(timeout=time_out)
                                indicesy,indicesx=get_time_series(self.cube[0],res[0])
                                if self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1]
                                    #self.psf_fm[j][i][res[0]]=res[2]
                                else:
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1]  
                            except mp.TimeoutError:
                                time_out=0.1
                                pool.terminate()
                                pool.join()
                                res=self.likelihood(it,j,i,residuals_cube_,None,False)
                                indicesy,indicesx=get_time_series(self.cube[0],res[0])
                                if self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1]
                                    #self.psf_fm[j][i][res[0]]=res[2] 
                                else:
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1] 
                            it+=1 
                        pool.close()
                        pool.join()
                    like=[]
                    SNR_FMMF=[]
    
                    for k in range(self.crop_range[i]):
                        like.append(like_temp[0:residuals_cube_.shape[0],:,:,:,:,k])
                        SNR_FMMF.append(like_temp[residuals_cube_.shape[0],:,:,0,0,k])
                    
                    self.like_fin[j][i]=like
                
    def likfcn(self,ann_center,like_cube,estimator,ns):
        
        if type(like_cube) is not np.ndarray:
            like_cube = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])

        from vip_hci.var import frame_center  
        
        def forback(obs,Trpr,prob_ini):
            
            #Forward backward model relying on past and future observation to 
            #compute the probability based on a two-states Markov chain
    
            scalefact_fw=np.zeros(obs.shape[1])
            scalefact_bw=np.zeros(obs.shape[1])
            prob_fw=np.zeros((2,obs.shape[1]))
            prob_bw=np.zeros((2,obs.shape[1]))
            prob_fin=np.zeros((2,obs.shape[1]))
            prob_pre_fw=0
            prob_pre_bw=0
            lik=0
            
            for i in range(obs.shape[1]):
                if obs[:,i].sum()!=0:
                    j=obs.shape[1]-1-i
                    if i==0:
                        prob_cur_fw=np.dot(np.diag(obs[:,i]),Trpr).dot(prob_ini)
                        prob_cur_bw=np.dot(Trpr,np.diag(obs[:,j])).dot(prob_ini)
                    else:
                        prob_cur_fw=np.dot(np.diag(obs[:,i]),Trpr).dot(prob_pre_fw)
                        prob_cur_bw=np.dot(Trpr,np.diag(obs[:,j])).dot(prob_pre_bw)
        
                    scalefact_fw[i]=prob_cur_fw.sum()
                    if scalefact_fw[i]==0:
                        prob_fw[:,i]=0
                    else:
                        prob_fw[:,i]=prob_cur_fw/scalefact_fw[i]
                    prob_pre_fw=prob_fw[:,i]
        
                    scalefact_bw[j]=prob_cur_bw.sum()
                    if scalefact_bw[j]==0:
                        prob_bw[:,j]=0
                    else:
                        prob_bw[:,j]=prob_cur_bw/scalefact_bw[j]
                        
                    prob_pre_bw=prob_bw[:,j]
    
            scalefact_fw_tot=(scalefact_fw).sum()                
            scalefact_bw_tot=(scalefact_bw).sum()
    
    
            for k in range(obs.shape[1]):
                if (prob_fw[:,k]*prob_bw[:,k]).sum()==0:
                    prob_fin[:,k]=0
                else:
                    prob_fin[:,k]=(prob_fw[:,k]*prob_bw[:,k])/(prob_fw[:,k]*prob_bw[:,k]).sum()
    
            lik = scalefact_fw_tot+scalefact_bw_tot
    
            return prob_fin, lik
    
    
        def RSM_esti(obs,Trpr,prob_ini):
            
            #Original RSM approach involving a forward two-states Markov chain to compute the probabilities
    
            prob_fin=np.zeros((2,obs.shape[1]))
            prob_pre=0
            lik=0
    
            for i in range(obs.shape[1]):
                if obs[:,i].sum()!=0:
                    if i==0:
                        cf=obs[:,i]*np.dot(Trpr,prob_ini)
                    else:
                        cf=obs[:,i]*np.dot(Trpr,prob_pre)
        
                    
                    f=sum(cf)            
                    lik+=np.log(f)
                    prob_fin[:,i]=cf/f
                    prob_pre=prob_fin[:,i]
                else:
                    prob_fin[:,i]=[1,0]
    
            return prob_fin, lik
    
        probmap = np.zeros((like_cube.shape[0],like_cube.shape[1],like_cube.shape[2]))
        ceny, cenx = frame_center(like_cube[0,:,:,0,0])
        
        indicesy,indicesx=get_time_series(like_cube[:,:,:,0,0],ann_center)

        npix = len(indicesy)
        pini=[1-ns/(like_cube.shape[0]*(npix)),1/(like_cube.shape[0]*ns),ns/(like_cube.shape[0]*(npix)),1-1/(like_cube.shape[0]*ns)]
        prob=np.reshape([pini],(2, 2)) 

        Trpr= prob

        #Initialization of the Regime Switching model
        #I-prob
        mA=np.concatenate((np.diag(np.repeat(1,2))-prob,[np.repeat(1,2)]))
        #sol
        vE=np.repeat([0,1],[2,1])
        #mA*a=vE -> mA'mA*a=mA'*vE -> a=mA'/(mA'mA)*vE
            
        prob_ini=np.dot(np.dot(np.linalg.inv(np.dot(mA.T,mA)),mA.T),vE)

        cf=np.zeros((2,len(indicesy)*like_cube.shape[0],like_cube.shape[3]))
        totind=0
        for i in range(0,len(indicesy)):

            poscenty=indicesy[i]
            poscentx=indicesx[i]
                
            for j in range(0,like_cube.shape[0]):        

                    for m in range(0,like_cube.shape[3]):
                        
                        cf[0,totind,m]=like_cube[j,poscenty,poscentx,m,0]
                        cf[1,totind,m]=like_cube[j,poscenty,poscentx,m,1]
                    totind+=1
                    
        #Computation of the probability cube via the regime switching framework
        
        prob_fin=[] 
        lik_fin=[]
        for n in range(like_cube.shape[3]):
            if estimator=='Forward':
                prob_fin_temp,lik_fin_temp=RSM_esti(cf[:,:,n],Trpr,prob_ini)
            elif estimator=='Forward-Backward':
                prob_fin_temp,lik_fin_temp=forback(cf[:,:,n],Trpr,prob_ini)


            prob_fin.append(prob_fin_temp)
            lik_fin.append(lik_fin_temp)
        
        cub_id1=0   
        for i in range(0,len(indicesy)):
            cub_id2=0
            for j in range(like_cube.shape[0]):
                    probmap[cub_id2,indicesy[i],indicesx[i]]=prob_fin[lik_fin.index(max(lik_fin))][1,cub_id1]
                    cub_id1+=1
                    cub_id2+=1
                    
        return probmap[:,indicesy,indicesx],ann_center

    def probmap_esti(self,modthencube=True,ns=1,sel_crop=None, estimator='Forward',colmode='median',ann_center=None,sel_cube=None):
        
        """
        Function allowing the estimation of the final RSM map based on the likelihood computed with 
        the lik_esti function for the different cubes and different post-processing techniques 
        used to generate the speckle field model. The RSM map estimation may be based on a forward
        or forward-backward approach.
        
        Parameters
        ----------
        
        modtocube: bool, optional
            Parameter defining if the concatenated cube feeding the RSM model is created
            considering first the model or the different cubes. If 'modtocube=False',
            the function will select the first cube then test all models on it and move 
            to the next one. If 'modtocube=True', the model will select one model and apply
            it to every cubes before moving to the next model. Default is True.
        ns: float , optional
             Number of regime switches. Default is one regime switch per annulus but 
             smaller values may be used to reduce the impact of noise or disk structures
             on the final RSM probablity map.
        sel_crop: list of int or None, optional
            Selected crop sizes from proposed crop_range (crop size = crop_size + 2 x (sel_crop)).
            A specific sel_crop should be provided for each mode. Default is crop size = [crop_size]
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        ann_center:int, optional
            Selected annulus if the probabilities are computed for a single annulus 
            (Used by the optimization framework). Default is None
        sel_cube: list of arrays,optional
            List of selected PSF-subtraction techniques and ADI sequences used to generate 
            the final probability map. [[i1,j1],[i2,j2],...] with i1 the first considered PSF-subtraction
            technique and j1 the first considered ADI sequence, i2 the second considered PSF-subtraction
            technique, etc. Default is None whih implies that all PSF-subtraction techniques and all
            ADI sequences are used to compute the final probability map.
        """
            
        import numpy as np 
     

        if type(sel_crop)!=np.ndarray:
            sel_crop=np.zeros(len(self.model)*len(self.cube))
        if sel_cube==None:    
            if modthencube==True:
                for i in range(len(self.model)):
                    for j in range(len(self.cube)):
                        if (i+j)==0:
                            if self.like_fin[j][i][int(int(sel_crop[i]))].shape[3]==1:
                                like_cube=np.repeat(self.like_fin[j][i][int(sel_crop[i])],len(self.interval[i][0,j]),axis=3)
                            else:
                                like_cube=self.like_fin[j][i][int(sel_crop[i])]
                        else:
                            if self.like_fin[j][i][int(sel_crop[i])].shape[3]==1:
                                like_cube=np.append(like_cube,np.repeat(self.like_fin[j][i][int(sel_crop[i])],len(self.interval[i][0,j]),axis=3),axis=0)
                            else:
                                like_cube=np.append(like_cube,self.like_fin[j][i][int(sel_crop[i])],axis=0) 
            else:
                for i in range(len(self.cube)):
                    for j in range(len(self.model)):
                        if (i+j)==0:
                            if self.like_fin[i][j][int(sel_crop[j])].shape[3]==1:
                                like_cube=np.repeat(self.like_fin[i][j][int(sel_crop[j])],len(self.interval[i][0,j]),axis=3)
                            else:
                                like_cube=self.like_fin[i][j][int(sel_crop[j])]
                        else:
                            if self.like_fin[i][j][int(sel_crop[j])].shape[3]==1:
                                like_cube=np.append(like_cube,np.repeat(self.like_fin[i][j][int(sel_crop[j])],len(self.interval[i][0,j]),axis=3),axis=0)
                            else:
                                like_cube=np.append(like_cube,self.like_fin[i][j][int(sel_crop[j])],axis=0) 
        else:
             for i in range(len(sel_cube)):
                 if i==0:
                    if self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])].shape[3]==1:
                        like_cube=np.repeat(self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])],len(self.interval[sel_cube[i][1]][0,sel_cube[i][0]]),axis=3)
                    else:
                        like_cube=self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])]
                 else:
                    if self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])].shape[3]==1:
                        like_cube=np.append(like_cube,np.repeat(self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])],len(self.interval[sel_cube[i][1]][0,sel_cube[i][0]]),axis=3),axis=0)
                    else:
                        like_cube=np.append(like_cube,self.like_fin[sel_cube[i][0]][sel_cube[i][1]][int(sel_crop[i])],axis=0) 

        
        n,y,x,l_int,r_n =like_cube.shape 


        probmap = np.zeros((like_cube.shape[0],like_cube.shape[1],like_cube.shape[2]))
        if ann_center is not None:
            indicesy,indicesx=get_time_series(self.cube[0],ann_center)
            probmap[:,indicesy,indicesx]=self.likfcn(ann_center,like_cube,estimator,ns)[0]
        else:
            
            if self.ncore!=1:
                X_shape=like_cube.shape
                X = RawArray('d', int(np.prod(X_shape)))
                X_np = np.frombuffer(X).reshape(X_shape)
                np.copyto(X_np, like_cube)
                 
                #Sometimes the parallel computation get stuck, to avoid that we impose a maximum computation
                #time of 0.01 x number of frame x number of pixels iin the last annulus x 2 x pi (0.01 second
                #per treated pixel)
                time_out=0.01*like_cube.shape[0]*2*self.maxradius*np.pi
                results=[]    
                pool=Pool(processes=self.ncore, initializer=init_worker, initargs=(X, X_shape))           
                for e in range(self.minradius,self.maxradius+1):
                    results.append(pool.apply_async(self.likfcn,args=(e,0,estimator,ns)))
                #[result.wait(timeout=time_out) for result in results]
                
                it=self.minradius
                for result in results:
                    try:
                        res=result.get(timeout=time_out)
                        indicesy,indicesx=get_time_series(self.cube[0],res[1])
                        probmap[:,indicesy,indicesx]=res[0]
                        
                    except mp.TimeoutError:
                        time_out=0.1
                        pool.terminate()
                        pool.join()
                        res=self.likfcn(it,like_cube,estimator,ns)
                        indicesy,indicesx=get_time_series(self.cube[0],res[1])
                        probmap[:,indicesy,indicesx]=res[0]
                       
                    it+=1
                pool.close()
                pool.join()
            else:
                 for e in range(self.minradius,self.maxradius+1):
                     indicesy,indicesx=get_time_series(self.cube[0],e)
                     probmap[:,indicesy,indicesx]=self.likfcn(e,like_cube,estimator,ns)[0]
            
            
        if colmode == 'mean':
            self.probmap= np.nanmean(probmap, axis=0)
        elif colmode == 'median':
            self.probmap= np.nanmedian(probmap, axis=0)
        elif colmode == 'sum':
            self.probmap= np.sum(probmap, axis=0)
        elif colmode == 'max':
            self.probmap= np.max(probmap, axis=0)
            
            
    def opti_bound_sel(self): 
        
        
        """
        Function providing the optimal upper bound for the number of principal components (APCA, NMF, KLIP),
         and for the rank (LLSG) to be considered during the optimisation procedure (self.opti_model()). 
         The upper bound is defined as the principal component/rank associated to the highest contrast 
         (averaged over multiple angular separations). This definition allows to consider the range
         of values for which a change in the speckle noise treatment by the PSF-subtraction
         techniques is observable. The obtained upper bound is saved in self.opti_bound[model][cube][1].
         The obtained range self.opti_bound[model][cube][0:1] can then be divided in multiple subranges
         allowing to increase the diversity when computing the final detection map, allowing to increase
         the RSM map performance. This should help cancelling potential bright speckles as their evolution
         is driven by the number of principal components/rank used to generate the reference PSF, while the
         potential astrophysical signal should not evolve sensibly. The curves providing the evolution of 
         the averaged contrast with the number of principal components/rank is provided by 
         self.snr_combined_evolution[model][cube].
    
        """
        
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]

        self.opti_theta=np.zeros((len(self.cube),self.maxradius+5))
                
        for c in range(len(self.cube)):
                    
            for m in range(len(self.model)):
                self.param=[c,m,0]
                if self.model[m]!='LOCI' or self.model[m]!='FM LOCI':
                    
                    if self.model[m]=='FM KLIP' or self.model[m]=='FM LOCI':
                        max_rad=self.max_r+1
                    else:
                        max_rad=self.maxradius+1
                        
                    # Determination of the considered angular distances for the optimization process
                       
                    if self.opti_mode=='full-frame':
                        if self.trunc is not None:
                            max_rad=min(self.trunc*self.asize[m],max_rad)
                        if max_rad>self.minradius+3*self.asize[m]+self.asize[m]//2:
                            range_sel = list(range(self.minradius+self.asize[m]//2,self.minradius+3*self.asize[m]+self.asize[m]//2,self.asize[m]))
                            if max_rad>self.minradius+7*self.asize[m]:
                                range_sel.extend(list(range(self.minradius+3*self.asize[m]+self.asize[m]//2,self.minradius+7*self.asize[m],2*self.asize[m])))
                                range_sel.extend(list(range(self.minradius+7*self.asize[m]+self.asize[m]//2,max_rad-3*self.asize[m]//2-1,4*self.asize[m])))
                                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[m]*self.asize[m]-self.asize[m]//2-1)
                            else:
                                range_sel.extend(list(range(self.minradius+3*self.asize[m]+self.asize[m]//2,max_rad-self.asize[m]//2,2*self.asize[m])))
                                if max_rad==self.minradius+7*self.asize[m]:
                                    range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[m]*self.asize[m]-self.asize[m]//2-1)
                        else:
                            range_sel=list(range(self.minradius+self.asize[m]//2,max_rad-self.asize[m]//2,self.asize[m]))
                    elif self.opti_mode=='annular':
                        range_sel=range(self.minradius+self.asize[m]//2,max_rad-self.asize[m]//2,self.asize[m])
                    
                    for i in range_sel:  
                            
                            indicesy,indicesx=get_time_series(self.cube[c],i)
                            cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[c],None,self.pa[c],self.scale_list[c], self.imlib, self.interpolation)  
                            cube_derot=rot_scale('fin',self.cube[c],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                            apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                            fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                            fluxes = np.array(fluxes['aperture_sum'])
                            x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                            y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                            
                            ceny, cenx = frame_center(cube_derot[0])
                            
                            self.opti_theta[c,i]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))
        
                    if self.model[m]=='APCA' or self.model[m]=='KLIP' or self.model[m]=='FM KLIP':
                
                        step=4
                        
                    elif self.model[m]=='NMF':
                        
                        step=2
                        
                    elif self.model[m]=='LLSG':
                        
                        step=1
                        
                    snr_mean_evolution=np.zeros(len(range(self.opti_bound[m][0][0],self.opti_bound[m][0][1],step)))
            
                            
                    for l in range_sel: 
                        
                        self.param[2]=l
                        cuben=self.param[0]
                        modn=self.param[1]
                        ann_center=self.param[2]
                        
                        mean_snr=[] 
                        flux_temp=[]
                        
        
                        
                        for j in range(self.opti_bound[m][0][0],self.opti_bound[m][0][1],step):
                            
                            if self.model[modn]=='APCA':
                        
                                param=[j,self.nsegments[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben]]
                                
                            elif self.model[modn]=='NMF':
                                
                                param=j
                                
                            elif self.model[modn]=='LLSG':
                                
                                param=[j,self.nsegments[modn][ann_center,cuben]]
                                
                            elif self.model[modn]=='KLIP' or self.model[modn]=='FM KLIP':
                        
                               param=[j,self.delta_rot[modn][ann_center,cuben]]
                            
                            
                            flux_temp.append(self.contrast_esti(param)[1])
                        
                        
                        flux_mean=np.median(flux_temp)
                        
                        for j in range(self.opti_bound[m][0][0],self.opti_bound[m][0][1],step):
            
                        
                            if self.model[modn]=='APCA':
                        
                                self.ncomp[modn][ann_center,cuben]=j
        
                        
                            elif self.model[modn]=='NMF':
                                
                                self.ncomp[modn][ann_center,cuben]=j
                                
                        
                            elif self.model[modn]=='LLSG':
                                
                                self.rank[modn][ann_center,cuben]=j
                     
                                
                            elif self.model[modn]=='KLIP' or self.model[modn]=='FM KLIP':
                        
                                self.ncomp[modn][ann_center,cuben]=j
        
                                
                            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])
                                  
                            ceny, cenx = frame_center(self.cube[cuben])
                            fcy=[]
                            fcx=[]
                            cube_fc =self.cube[cuben]
                            ang_fc=range(int(self.opti_theta[cuben,ann_center]),int(360+self.opti_theta[cuben,ann_center]),int(360))
                            for i in range(len(ang_fc)):
                                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                                     self.pa[cuben], flux_mean, self.pxscale,
                                                     rad_dists=ann_center,
                                                     theta=ang_fc[i],
                                                     verbose=False)
                                y = int(ceny) + ann_center * np.sin(np.deg2rad(
                                                                       ang_fc[i]))
                                x = int(cenx) + ann_center * np.cos(np.deg2rad(
                                                                       ang_fc[i]))
                        
                                fcy.append(y)
                                fcx.append(x)
                        
                            
                            frame_fc=self.model_esti(modn,cuben,ann_center,cube_fc)[0]
                            
                            max_detect=[]
                            for i in range(len(ang_fc)):
                        
                        
                                indc = disk((fcy[i], fcx[i]),int(self.fwhm/2)+1)
                                
                                indices=get_annulus_segments(cube_fc[0], ann_center,1,1)
                            
                                snr_fwhm=[]
                               
                                for k in range(len(indc[0])):
                                    
                                    if len(np.intersect1d(np.where(indices[0][0]==indc[0][k])[0],np.where(indices[0][1]==indc[1][k])[0]))==1:
                                        
                                        snr_fwhm.append(vip.metrics.snr(frame_fc, (indc[1][k],indc[0][k]),self.fwhm))
                                
                                max_detect.append(max(snr_fwhm))
                           
                            mean_snr.append(np.mean(max_detect))
                            
                            
                        self.snr_evolution[m][l,c]=[mean_snr]
                        snr_mean_evolution+=mean_snr
                        

                    self.snr_combined_evolution[m][c]= [snr_mean_evolution/len(range_sel)] 
                    
                    snr_diff=np.diff(np.round(snr_mean_evolution/(10*len(range_sel)),1))
                    sum_snr_diff=[]
                    for n in range(len(snr_diff)):
                        sum_snr_diff.append(abs(snr_diff[n:len(snr_diff)]).sum()) 
                        
                    if len(np.where(np.asarray(sum_snr_diff)==0)[0])>=1:
                        
                        self.opti_bound[m][c][1]=(np.where(np.asarray(sum_snr_diff)==0)[0][0]+2)*step+self.opti_bound[m][0][0] #+1 because of np.diff and +1 as it starts at 0

                    else:
                        
                        self.opti_bound[m][c][1]=(np.argmax(snr_mean_evolution)+1)*step+self.opti_bound[m][0][0]

        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]                        
        
    def model_esti(self,modn,cuben,ann_center,cube): 
        
        """
        Function used during the optimization process to compute the cube of residuals
        for a given annulus whose center is defined by ann_center. The PSF-subtraction
        techniques index is given by modn, the ADI sequence index by cuben
        """  
               
        if self.model[modn]=='FM KLIP' or self.model[modn]=='KLIP':
                
            resicube=np.zeros_like(self.cube[cuben])

            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * ann_center)))
            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
            if pa_threshold >= mid_range - mid_range * 0.1:
                pa_threshold = float(mid_range - mid_range * 0.1)


            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)


            for k in range(0,self.cube[cuben].shape[0]):

                evals_temp,evecs_temp,KL_basis_temp,sub_img_rows_temp,refs_mean_sub_temp,sci_mean_sub_temp =KLIP_patch(k,cube[:, indices[0][0], indices[0][1]], self.ncomp[modn][ann_center,cuben], self.pa[cuben], self.crop[modn][ann_center,cuben], pa_threshold, ann_center)
                resicube[k,indices[0][0], indices[0][1]] = sub_img_rows_temp

            resicube_der=cube_derotate(resicube,self.pa[cuben])
            frame_fin=cube_collapse(resicube_der, mode='median')


        elif self.model[modn]=='FM LOCI':
            
            
            resicube, ind_ref_list,coef_list=LOCI_FM(cube, self.psf[cuben], ann_center, self.pa[cuben],None, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
            resicube_der=cube_derotate(resicube,self.pa[cuben])
            frame_fin=cube_collapse(resicube_der, mode='median')
            
        
        elif self.model[modn]=='LOCI':
            
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            
            if scale_list is not None:
                resicube=np.zeros_like(cube_rot_scale)
                for i in range(int(max(scale_list)*ann_center/self.asize[modn])):
                    indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
                    resicube[:,indices[0][0], indices[0][1]]=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center,angle_list,scale_list, self.asize[modn], self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],self.delta_sep[modn][ann_center,cuben])[0][:,indices[0][0], indices[0][1]]
            
            else:
                resicube, ind_ref_list,coef_list=LOCI_FM(cube_rot_scale, self.psf[cuben], ann_center, self.pa[cuben],None, self.fwhm, self.fwhm, self.tolerance[modn][ann_center,cuben],self.delta_rot[modn][ann_center,cuben],None)
 
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')


        elif self.model[modn]=='APCA':
            
                  
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                
                pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn][ann_center,cuben] * self.fwhm / (2 * (ann_center+self.asize[modn]*i))))
                mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                if pa_threshold >= mid_range - mid_range * 0.1:
                    pa_threshold = float(mid_range - mid_range * 0.1)
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
    
                for k in range(0,cube_rot_scale.shape[0]):
                    for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                        resicube[k,indices[l][0], indices[l][1]],v_resi,data_shape=do_pca_patch(cube_rot_scale[:, indices[l][0], indices[l][1]], k,  angle_list,scale_list, self.fwhm, pa_threshold,self.delta_sep[modn][ann_center,cuben], ann_center+self.asize[modn]*i,
                           svd_mode='lapack', ncomp=self.ncomp[modn][ann_center,cuben],min_frames_lib=2, max_frames_lib=200, tol=1e-1,matrix_ref=None)
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')

                
        elif self.model[modn]=='NMF':
            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if self.opti_mode=='full-frame':
                nfr = cube_rot_scale.shape[0]
                matrix = np.reshape(cube_rot_scale, (nfr, -1))
                res= NMF_patch(matrix, ncomp=self.ncomp[modn][ann_center,cuben], max_iter=5000,random_state=None)
                resicube=np.reshape(res,(cube_rot_scale.shape[0],cube_rot_scale.shape[1],cube_rot_scale.shape[2]))
            else:
                if scale_list is not None:
                    range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
                else:
                    range_adisdi=range(1)
                    
                for i in range_adisdi:
                    indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
                    
                    for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                        resicube[:,indices[l][0], indices[l][1]]= NMF_patch(cube_rot_scale[:, indices[l][0], indices[l][1]], ncomp=self.ncomp[modn][ann_center,cuben], max_iter=100,random_state=None)

            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')
            
                
        elif self.model[modn]=='LLSG':

            
            cube_rot_scale,angle_list,scale_list=rot_scale('ini',cube,None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)
            resicube=np.zeros_like(cube_rot_scale)
            if scale_list is not None:
                range_adisdi=range(int(max(scale_list)*ann_center/self.asize[modn]))
            else:
                range_adisdi=range(1)
                
            for i in range_adisdi:
                indices = get_annulus_segments(cube_rot_scale[0], ann_center-int(self.asize[modn]/2)+self.asize[modn]*i,int(self.asize[modn]),int(self.nsegments[modn][ann_center,cuben]))
    
                for l in range(self.nsegments[modn][ann_center,cuben]):
                    
                    resicube[:,indices[l][0], indices[l][1]]= _decompose_patch(indices,l, cube_rot_scale,self.nsegments[modn][ann_center,cuben],
                            self.rank[modn][ann_center,cuben], low_rank_ref=False, low_rank_mode='svd', thresh=1,thresh_mode='soft', max_iter=40, auto_rank_mode='noise', cevr=0.9,
                                 residuals_tol=1e-1, random_seed=10, debug=False, full_output=False).T
                        
            resicube_der=rot_scale('fin',self.cube[cuben],resicube,angle_list,scale_list,self.imlib, self.interpolation)
            frame_fin=cube_collapse(resicube_der, mode='median')
        
        return frame_fin,resicube_der      
    
        
    def contrast_esti(self,param): 
        
        """
        Function used during the PSF-subtraction techniques optimization process 
        to compute the average contrast for a given annulus via multiple injection 
        of fake companions, relying on the approach developed by Mawet et al. (2014), 
        Gonzales et al. (2017) and Dahlqvist et Al. (2021). The PSF-subtraction
        techniques index is given by self.param[1], the ADI sequence index by self.param[0]
        and the annulus center by self.param[2]. The parameters for the PSF-subtraction 
        technique are contained in param.
        """  

        
        cuben=self.param[0]
        modn=self.param[1]
        ann_center=self.param[2]
        

        if self.model[modn]=='APCA':

            self.ncomp[modn][ann_center,cuben]=int(param[0])
            self.nsegments[modn][ann_center,cuben]=int(param[1])
            self.delta_rot[modn][ann_center,cuben]=abs(param[2])

        elif self.model[modn]=='NMF':
            
            self.ncomp[modn][ann_center,cuben]=abs(int(param))
            

        elif self.model[modn]=='LLSG':
            
            self.rank[modn][ann_center,cuben]=int(param[0])
            self.nsegments[modn][ann_center,cuben]=int(param[1])

        elif self.model[modn]=='LOCI' or self.model[modn]=='FM LOCI':

            self.tolerance[modn][ann_center,cuben]=abs(param[0])
            self.delta_rot[modn][ann_center,cuben]=abs(param[1])              
            
        elif self.model[modn]=='KLIP' or self.model[modn]=='FM KLIP':

            self.ncomp[modn][ann_center,cuben]=abs(int(param[0]))
            self.delta_rot[modn][ann_center,cuben]=abs(param[1])
                        
        ceny, cenx = frame_center(self.cube[cuben])
        
        frame_nofc=self.model_esti(modn,cuben,ann_center,self.cube[cuben])[0]
            
        psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])


        
        # Noise computation using the approach proposed by Mawet et al. (2014)
        
        ang_step=360/((np.deg2rad(360)*ann_center)/self.fwhm)
        
        tempx=[]
        tempy=[]
        
        for l in range(int(((np.deg2rad(360)*ann_center)/self.fwhm))):
            newx = ann_center * np.cos(np.deg2rad(ang_step * l+self.opti_theta[cuben,ann_center]))
            newy = ann_center * np.sin(np.deg2rad(ang_step * l+self.opti_theta[cuben,ann_center]))
            tempx.append(newx)
            tempy.append(newy)
        
        tempx=np.array(tempx)
        
        tempy = np.array(tempy) +int(ceny)
        tempx = np.array(tempx) + int(cenx)
  
        apertures = photutils.CircularAperture(np.array((tempx, tempy)).T, round(self.fwhm/2))
        fluxes = photutils.aperture_photometry(frame_nofc, apertures)
        fluxes = np.array(fluxes['aperture_sum'])
         
        n_aper = len(fluxes)
        ss_corr = np.sqrt(1 + 1/(n_aper-1))
        sigma_corr = stats.t.ppf(stats.norm.cdf(5), n_aper)*ss_corr
        noise = np.std(fluxes)
        
        flux = sigma_corr*noise*2
        fc_map = np.ones((self.cube[cuben].shape[-1],self.cube[cuben].shape[-1])) * 1e-6
        fcy=[]
        fcx=[]
        cube_fc =self.cube[cuben]
        
        # Average contrast computation via multiple injections of fake companions
        
        ang_fc=range(int(self.opti_theta[cuben,ann_center]),int(360+self.opti_theta[cuben,ann_center]),int(360//min((len(fluxes)/2),8)))
        for i in range(len(ang_fc)):
            cube_fc = cube_inject_companions(cube_fc, psf_template,
                                 self.pa[cuben], flux, self.pxscale,
                                 rad_dists=ann_center,
                                 theta=ang_fc[i],
                                 verbose=False)
            y = int(ceny) + ann_center * np.sin(np.deg2rad(
                                                   ang_fc[i]))
            x = int(cenx) + ann_center * np.cos(np.deg2rad(
                                                   ang_fc[i]))
            if self.cube[cuben].ndim==4:
                fc_map = frame_inject_companion(fc_map, np.mean(psf_template,axis=0), y, x,
                                            flux)
            else:
                fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                            flux)
            fcy.append(y)
            fcx.append(x)

        
        frame_fc=self.model_esti(modn,cuben,ann_center,cube_fc)[0]
    
        contrast=[]
        
        for j in range(len(ang_fc)):
            apertures = photutils.CircularAperture(np.array(([fcx[j],fcy[j]])), round(self.fwhm/2))
            injected_flux = photutils.aperture_photometry(fc_map, apertures)['aperture_sum']
            recovered_flux = photutils.aperture_photometry((frame_fc - frame_nofc), apertures)['aperture_sum']
            throughput = float(recovered_flux / injected_flux)
    
            if throughput>0:
                contrast.append(flux / throughput)
                
        if len(contrast)!=0:
            contrast_mean=np.mean(contrast)
        else:
            contrast_mean=-1
            
            
        if self.param_opti_mode=='Contrast':
            
            return np.where(contrast_mean<0,0,1/contrast_mean), contrast_mean,param
        
        elif self.param_opti_mode=='RSM':
        
            # When self.param_opti_mode=='RSM', the average contrast is replaced by 
            #the ratio of peak probability of the fake companion injected at the median 
            #flux position with the average contrast defined previously and the peak (noise) 
            #probability in the remaining of the annulus
            
            if contrast_mean>0:
            
                self.flux_opti[cuben,modn,ann_center]=contrast_mean
            
                self.RSM_test(cuben,modn,ann_center,self.opti_theta[cuben,ann_center],contrast_mean)
                        
                contrast=[]
                for i in range(self.crop_range[modn]):
                    self.probmap_esti(modthencube=True,ns=1,sel_crop=[i], estimator='Forward',colmode='median',ann_center=ann_center,sel_cube=[[cuben,modn]])
                    contrast.append(self.perf_esti(cuben,modn,ann_center,self.opti_theta[cuben,ann_center]))
                    
                return contrast/(flux/throughput),flux/throughput,param
            else:
                return np.where(contrast_mean<0,0,1/contrast_mean), contrast_mean,param
        
        

    
    
    def bayesian_optimisation(self, loss_function, bounds,param_type, prev_res=None, opti_param={'opti_iter':15,'ini_esti':10, 'random_search':100} ,ncore=1):

        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `loss_function`.
        
        Parameters
        ----------
        
            loss_loss: function.
                Function to be optimised, in our case the contrast_esti function.
            bounds: numpy ndarray, 2d
                Lower and upper bounds of the parameters used to compute the cube of
                residuals allowing the estimation of the average contrast.
            param_type: list
                Type of parameters used by the PSF-subtraction technique to compute the cube
                of residuals allowing the estimation of the average contrast ('int' or 'float')
            prev_res: None or numpy ndarray, 2d
                Parmater sets and corresponding average contrasts generated at the previous 
                angular distance. Allow to smooth the transition from one annulus to another 
                during the optimization process for the annular mode
            param_optimisation: dict, optional
                Dictionnary regrouping the parameters used by the Bayesian. 'opti_iter' is the number of
                iterations,'ini_esti' number of sets of parameters for which the loss function is computed
                to initialize the Gaussian process, random_search the number of random searches for the
                selection of the next set of parameters to sample based on the maximisation of the 
                expected immprovement. Default is {'opti_iter':15,'ini_esti':10, 'random_search':100}
            ncore : int, optional
                Number of processes for parallel computing. By default ('ncore=1') 
                the algorithm works in single-process mode. 
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
            
        if prev_res==None:
            x_ini = []
            y_ini = []
        else:
            x_ini = prev_res[0]
            y_ini = prev_res[1]
            

        flux_fin=[]
    
        space_dim = bounds.shape[0]
        ann_center=self.param[2]
        modn=self.param[1]
        cuben=self.param[0]

        
        if self.opti_mode=='full-frame':
                params_m=[]
                for i in range(len(param_type)):
                    if param_type[i]=='int':
                        params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                    else:
                        params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                        
                if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                    
                # Determination of the considered angular distances for the optimization process
                
                if self.trunc is not None:
                    max_rad=min(self.trunc*self.asize[modn],max_rad)
                if max_rad>self.minradius+3*self.asize[modn]+self.asize[modn]//2:
                    range_sel = list(range(self.minradius+self.asize[modn]//2,self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.asize[modn]))
                    if max_rad>self.minradius+7*self.asize[modn]:
                        range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.minradius+7*self.asize[modn],2*self.asize[modn])))
                        range_sel.extend(list(range(self.minradius+7*self.asize[modn]+self.asize[modn]//2,max_rad-3*self.asize[modn]//2-1,4*self.asize[modn])))
                        range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
                    else:
                        range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,max_rad-self.asize[modn]//2,2*self.asize[modn])))
                        if max_rad==self.minradius+7*self.asize[modn]:
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
                else:
                    range_sel=list(range(self.minradius+self.asize[modn]//2,max_rad-self.asize[modn]//2,self.asize[modn]))
            
                it=0
                
                for j in range_sel:
                    
                    self.param[2]=j       
                    res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T))

                    res_mean_temp=[]
                    for res_temp in res_param:
                        res_mean_temp.append(res_temp[0])
                    self.mean_opti[cuben,modn,j]=np.mean(np.asarray(res_mean_temp))
                    
                    y_ini_temp=[]
                    for res_temp in res_param:
                        if j==self.minradius+self.asize[modn]//2:
                            x_ini.append(res_temp[2])
                        y_ini_temp.append(res_temp[0]/self.mean_opti[cuben,modn,j])


                    y_ini.append([y_ini_temp])
                    it+=1
                    print(self.model[modn]+' Gaussian process initialization: annulus {} done!'.format(j))                  
                y_ini=list(np.asarray(y_ini).sum(axis=0))

        else:

            self.param[2]=ann_center
            params_m=[]
            for i in range(len(param_type)):
                if param_type[i]=='int':
                    params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                else:
                    params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                    
                    
            res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T))
            
            res_mean_temp=[]
            for res_temp in res_param:
                res_mean_temp.append(res_temp[0])
            self.mean_opti[cuben,modn,self.param[2]]=np.mean(np.asarray(res_mean_temp))
                    
            for res_temp in res_param:
                if res_temp[0].prod()>0:
                    if prev_res is not None:
                        del x_ini[0]
                        del y_ini[0]
                    x_ini.append(res_temp[2])
                    y_ini.append(res_temp[0]/self.mean_opti[cuben,modn,self.param[2]])
                    

        # Creation the Gaussian process
        
        kernel = gp.kernels.RBF(1.0, length_scale_bounds=(0.5,5))
    
        model = gp.GaussianProcessRegressor( kernel,
                                           alpha=1e-2,                               
                                        n_restarts_optimizer=0,
                                        normalize_y=False)
        param_f=[]
        flux_f=[]
        optires=[]
        if self.param_opti_mode=='Contrast':
            crop_r=1
        else:
            crop_r=self.crop_range[self.param[1]]
            self.crop_range[self.param[1]]=1
        for j in range(crop_r):
            x_fin = []
            y_fin = []
        
            params_m=[]
            self.param[2]=ann_center
            
            if self.opti_mode=='full-frame':
                if self.param_opti_mode=='Contrast':
                    y_i=y_ini[0]
                else:
                    y_i=y_ini[0][:,j]
            else:               
                if self.param_opti_mode=='Contrast':
                    y_i=np.array(np.array(y_ini))
                else:
                    y_i=np.array(np.array(y_ini)[:,j])
            x_cop=x_ini.copy()
            y_cop=list(y_i.copy())
            
            if self.opti_mode=='full-frame':
                it2=0
                flux_fin=np.zeros((len(range_sel),opti_param['opti_iter']))
            
            for n in range(opti_param['opti_iter']):
                if len(x_fin)>0:
                    x_cop.append(x_fin[-1])
                    y_cop.append(y_fin[-1])
                    model.fit(np.array(x_cop), np.asarray(y_cop))
                else:
                    model.fit(np.array(x_ini),y_i)
        
                # Selection of next parameter set via maximisation of the expected improvement
                
                if opti_param['random_search']:
                    x_random=[]
                    for i in range(len(param_type)):
                        if param_type[i]=='int':
                            x_random.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['random_search'])))
                        else:
                            x_random.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['random_search'])))
                            
                    x_random=np.array(x_random).T
                    ei = -1 * expected_improvement(x_random, model, y_i, space_dim=space_dim) 
                    params_m = x_random[np.argmax(ei), :]
                    
                    # Duplicates break the GP
                    if any(np.abs(params_m - x_ini).sum(axis=1) > 1e-10):
            
                        for i in range(len(param_type)):
                            if param_type[i]=='int':
                                params_m[i]=np.random.random_integers(bounds[i, 0], bounds[i, 1],1)
                            else:
                                params_m[i]=np.random.uniform(bounds[i, 0], bounds[i, 1], 1)
                                
                if self.opti_mode=='full-frame':
                    it1=0
                    y_fin_temp=[]

                    for k in range_sel:
                        self.param[2]=k
                        res_temp =loss_function(params_m)
                        flux_fin[it1,it2]=res_temp[1]
                        if k==self.minradius+self.asize[modn]//2:
                            x_fin.append(res_temp[2])
                        y_fin_temp.append((res_temp[0]/self.mean_opti[cuben,modn,self.param[2]]))
                        it1+=1


                    y_fin.append(np.asarray(y_fin_temp).sum(axis=0))  
                    it2+=1
                    
                else:
                    res_temp = loss_function(params_m)
            
                    x_fin.append(res_temp[2])
                    y_fin.append(res_temp[0]/self.mean_opti[cuben,modn,self.param[2]])
                    flux_fin.append(res_temp[1])
             
                
            if self.opti_mode=='full-frame': 
                param_f.append(x_fin[np.argmax(np.array(y_fin))])
                flux_f.append(flux_fin[:,np.argmax(np.array(y_fin))])
                optires.append(max(y_fin))    
            else:   
                param_f.append(x_fin[np.argmax(np.array(y_fin))])
                flux_f.append(flux_fin[np.argmax(np.array(y_fin))])
                optires.append(max(y_fin))
      
        if self.param_opti_mode=='RSM': 
            self.crop[self.param[1]][ann_center,self.param[0]]=self.crop[self.param[1]][ann_center,self.param[0]]+optires.index(max(optires)) *2
            self.crop_range[self.param[1]]=crop_r
        
        if self.opti_mode=='full-frame': 
            print(self.model[modn]+' Bayesian optimization done!')  
            
 
        return param_f[optires.index(max(optires))],max(optires), [x_ini, y_ini],flux_f[optires.index(max(optires))]
    
    def pso_optimisation(self, loss_function, bounds,param_type, prev_res=None, opti_param={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':15,'ini_esti':10} ,ncore=1):

        """ bayesian_optimisation
        Uses Gaussian Processes to optimise the loss function `loss_function`.
        
        Parameters
        ----------
        
            loss_loss: function.
                Function to be optimised, in our case the contrast_esti function.
            bounds: numpy ndarray, 2d
                Lower and upper bounds of the parameters used to compute the cube of
                residuals allowing the estimation of the average contrast.
            param_type: list
                Type of parameters used by the PSF-subtraction technique to compute the cube
                of residuals allowing the estimation of the average contrast ('int' or 'float')
            prev_res: None or numpy ndarray, 2d
                Parmater sets and corresponding average contrasts generated at the previous 
                angular distance. Allow to smooth the transition from one annulus to another 
                during the optimization process for the annular mode
            param_optimisation: dict, optional
                Dictionnary regrouping the parameters used by the PSO optimization
                framework. 'w' is the inertia factor, 'c1' is the cognitive factor and
                'c2' is the social factor, 'n_particles' is the number of particles in the
                swarm (number of point tested at each iteration), 'opti_iter' the number of
                iterations,'ini_esti' number of sets of parameters for which the loss
                function is computed to initialize the PSO. {'c1': 1, 'c2': 1, 'w':0.5,
                'n_particles':10,'opti_iter':15,'ini_esti':10}
            ncore : int, optional
                Number of processes for parallel computing. By default ('ncore=1') 
                the algorithm works in single-process mode. 
        """

        if prev_res==None:
            x_ini = []
            y_ini = []
        else:
            x_ini = prev_res[0]
            y_ini = prev_res[1]
            
        ann_center=self.param[2]
        modn=self.param[1]
        cuben=self.param[0]

        
        if self.opti_mode=='full-frame':
                params_m=[]
                for i in range(len(param_type)):
                    if param_type[i]=='int':
                        params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                    else:
                        params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                        
                if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                    
                # Determination of the considered angular distances for the optimization process
                
                if self.trunc is not None:
                    max_rad=min(self.trunc*self.asize[modn],max_rad)
                if max_rad>self.minradius+3*self.asize[modn]+self.asize[modn]//2:
                    range_sel = list(range(self.minradius+self.asize[modn]//2,self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.asize[modn]))
                    if max_rad>self.minradius+7*self.asize[modn]:
                        range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,self.minradius+7*self.asize[modn],2*self.asize[modn])))
                        range_sel.extend(list(range(self.minradius+7*self.asize[modn]+self.asize[modn]//2,max_rad-3*self.asize[modn]//2-1,4*self.asize[modn])))
                        range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
                    else:
                        range_sel.extend(list(range(self.minradius+3*self.asize[modn]+self.asize[modn]//2,max_rad-self.asize[modn]//2,2*self.asize[modn])))
                        if max_rad==self.minradius+7*self.asize[modn]:
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[modn]*self.asize[modn]-self.asize[modn]//2-1)
                else:
                    range_sel=list(range(self.minradius+self.asize[modn]//2,max_rad-self.asize[modn]//2,self.asize[modn]))
            
                
                for j in range_sel:
                    
                    self.param[2]=j       
                    res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T))

                    res_mean_temp=[]
                    for res_temp in res_param:
                        res_mean_temp.append(1/res_temp[0])
                    self.mean_opti[cuben,modn,j]=np.mean(np.asarray(res_mean_temp))
                    
                    y_ini_temp=[]
                    for res_temp in res_param:
                        if j==self.minradius+self.asize[modn]//2:
                            x_ini.append(res_temp[2])
                        y_ini_temp.append((1/res_temp[0])/self.mean_opti[cuben,modn,j])


                    y_ini.append([y_ini_temp])
                    print(self.model[modn]+' PSO process initialization: annulus {} done!'.format(j))                  
                y_ini=list(np.asarray(y_ini).sum(axis=0))

        else:

            self.param[2]=ann_center
            params_m=[]
            for i in range(len(param_type)):
                if param_type[i]=='int':
                    params_m.append(np.random.random_integers(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                else:
                    params_m.append(np.random.uniform(bounds[i, 0], bounds[i, 1], (opti_param['ini_esti'])))
                    
                    
            res_param = pool_map(ncore, loss_function, iterable(np.array(params_m).T))
            
            res_mean_temp=[]
            for res_temp in res_param:
                res_mean_temp.append(1/res_temp[0])
            self.mean_opti[cuben,modn,self.param[2]]=np.mean(np.asarray(res_mean_temp))
                    
            for res_temp in res_param:
                if res_temp[0].prod()>0:
                    if prev_res is not None:
                        del x_ini[0]
                        del y_ini[0]
                    x_ini.append(res_temp[2])
                    y_ini.append((1/res_temp[0])/self.mean_opti[cuben,modn,self.param[2]])
                    

        # Initialization of the PSO algorithm
        x_i=np.array(x_ini)
        
        my_topology = Star() # The Topology Class
        my_options_start={'c1': opti_param['c1'], 'c2': opti_param['c2'], 'w':opti_param['w']}
        my_options_end = {'c1': opti_param['c1'], 'c2': opti_param['c2'], 'w': opti_param['w']}
                
        param_f=[]
        flux_f=[]
        optires=[]
        if self.param_opti_mode=='Contrast':
            crop_r=1
        else:
            crop_r=self.crop_range[self.param[1]]
            self.crop_range[self.param[1]]=1
        for j in range(crop_r):
        
            params_m=[]
            self.param[2]=ann_center
            
            if self.opti_mode=='full-frame':
                if self.param_opti_mode=='Contrast':
                    y_i=y_ini[0]
                else:
                    y_i=y_ini[0][:,j]
            else:               
                if self.param_opti_mode=='Contrast':
                    y_i=np.array(np.array(y_ini))
                else:
                    y_i=np.array(np.array(y_ini)[:,j])
            #print(y_i)
            ini_position=x_i[y_i<=np.sort(y_i)[opti_param['n_particles']-1],:]
            my_swarm = P.create_swarm(n_particles=opti_param['n_particles'], dimensions=bounds.shape[0], options=my_options_start, bounds=(list(bounds.T[0]),list(bounds.T[1])),init_pos=ini_position)
            my_swarm.bounds=(list(bounds.T[0]),list(bounds.T[1]))
            my_swarm.velocity_clamp=None
            my_swarm.vh=P.handlers.VelocityHandler(strategy="unmodified")
            my_swarm.bh=P.handlers.BoundaryHandler(strategy="periodic")
            my_swarm.current_cost=y_i[y_i<=np.sort(y_i)[opti_param['n_particles']-1]]
            my_swarm.pbest_cost = my_swarm.current_cost  
            
            iterations = opti_param['opti_iter']
            
            for n in range(iterations):

                y_i=[]
                # Part 1: Update personal best
                if n!=0:
                           
                    if self.opti_mode=='full-frame':
                        
                        for j in range_sel:
                            
                            self.param[2]=j       
                            res_param = pool_map(ncore, loss_function, iterable(np.array(my_swarm.position)))
                            
                            res_mean_temp=[]
                            for res_temp in res_param:
                                res_mean_temp.append(1/res_temp[0])
                            
                            self.mean_opti[cuben,modn,j]=np.mean(np.asarray(res_mean_temp))
                            
                            y_ini_temp=[]
                            for res_temp in res_param:
                                y_ini_temp.append((1/res_temp[0])/self.mean_opti[cuben,modn,j])
                            
                            y_i.append([y_ini_temp])
                            
                        my_swarm.current_cost=np.asarray(y_i).sum(axis=0)[0]
                        #print(my_swarm.position,np.asarray(y_i).sum(axis=0)[0])    
                    else:
                        self.param[2]=ann_center
     
                        res_param = pool_map(ncore, loss_function, iterable(np.array(my_swarm.position)))
                        
                        res_mean_temp=[]
                        for res_temp in res_param:
                            res_mean_temp.append(1/res_temp[0])
                        self.mean_opti[cuben,modn,self.param[2]]=np.mean(np.asarray(res_mean_temp))
                                
                        for res_temp in res_param:
                            if res_temp[0].prod()>0:
                                if prev_res is not None:
                                    del x_ini[0]
                                    del y_ini[0]
                                x_ini.append(res_temp[2])
                                y_ini.append((1/res_temp[0])/self.mean_opti[cuben,modn,self.param[2]])
                                y_i.append((1/res_temp[0])/self.mean_opti[cuben,modn,self.param[2]])
                                
                        my_swarm.current_cost=y_i
                
               
                my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store
    
                my_swarm.options={'c1': my_options_start['c1']+n*(my_options_end['c1']-my_options_start['c1'])/(iterations-1), 'c2': my_options_start['c2']+n*(my_options_end['c2']-my_options_start['c2'])/(iterations-1), 'w': my_options_start['w']+n*(my_options_end['w']-my_options_start['w'])/(iterations-1)}
                # Part 2: Update global best
                # Note that gbest computation is dependent on your topology
                if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
                    my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)
    
                # Let's print our output
    
                print('Iteration: {} | Best cost: {}'.format(n+1, my_swarm.best_cost))
    
            # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
                my_swarm.velocity = my_topology.compute_velocity(my_swarm, my_swarm.velocity_clamp, my_swarm.vh, my_swarm.bounds
                )
                my_swarm.position = my_topology.compute_position(my_swarm, my_swarm.bounds, my_swarm.bh
                )
                
            if self.opti_mode=='full-frame': 
                param_f.append(my_swarm.best_pos)
                flux_f.append(1/my_swarm.best_cost)
                optires.append(my_swarm.best_cost)    
            else:   
                param_f.append(my_swarm.best_pos)
                flux_f.append(1/my_swarm.best_cost)
                optires.append(my_swarm.best_cost)
      
        if self.param_opti_mode=='RSM': 
            self.crop[self.param[1]][ann_center,self.param[0]]=self.crop[self.param[1]][ann_center,self.param[0]]+optires.index(max(optires)) *2
            self.crop_range[self.param[1]]=crop_r
        
        if self.opti_mode=='full-frame': 
            print(self.model[modn]+' Bayesian optimization done!')  
            
 
        return param_f[optires.index(max(optires))],max(optires), [x_ini, y_ini],flux_f[optires.index(max(optires))]


    def opti_model(self,optimisation_model='PSO', param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':15,'ini_esti':10, 'random_search':100},filt=True):
             
        """
        Function allowing the optimization of the PSF-subtraction techniques parameters 
        relying either on the 'full-frame' or 'annular' mode (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        optimisation_model: str, optional
            Approach used for the paramters optimal selection via the maximization of the 
            contrast for APCA, LOCI, KLIP and KLIP FM. The optimization is done either
            via a Bayesian approach ('Bayesian') or using Particle Swarm optimization
            ('PSO'). Default is PSO.
        param_optimisation: dict, optional
            dictionnary regrouping the parameters used by the Bayesian or the PSO optimization
            framework. For the Bayesian optimization we have 'opti_iter' the number of iterations,
            ,'ini_esti' number of sets of parameters for which the loss function is computed to
            initialize the Gaussian process, random_search the number of random searches for the
            selection of the next set of parameters to sample based on the maximisation of the 
            expected immprovement. For the PSO optimization, 'w' is the inertia factor, 'c1' is
            the cognitive factor and 'c2' is the social factor, 'n_particles' is the number of
            particles in the swarm (number of point tested at each iteration), 'opti_iter' the 
            number of iterations,'ini_esti' number of sets of parameters for which the loss
            function is computed to initialize the PSO. {'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10
            ,'opti_iter':15,'ini_esti':10, 'random_search':100}
        filt: True, optional
            If True, a Hampel Filter is applied on the set of parameters for the annular mode
            in order to avoid outliers due to potential bright artefacts.
        """
            
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.max_r=self.maxradius
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.max_r=self.maxradius  
            
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
            
        self.opti_theta=np.zeros((len(self.cube),self.maxradius+5))
        self.contrast=np.zeros((len(self.cube),len(self.model),self.maxradius+5)) 
        self.flux_opti=np.zeros((len(self.cube),len(self.model),self.maxradius+5))
        self.mean_opti=np.zeros((len(self.cube),len(self.model),self.maxradius+5))
    
                           
        for k in range(len(self.cube)):
            
            for j in range(len(self.model)):
                simures=None
                self.ini=True
                
                if (self.opti_mode=='full-frame' or self.opti_mode=='annular') and self.max_r%self.asize[j]>0:
                    raise ValueError("For opti_mode equal to 'full-frame' or 'annular', max_r_fm should be a multiple of asize.")
                
                if self.model[j]=='FM KLIP' or self.model[j]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                    
                # Determination of the considered angular distances for the optimization process
                   
                if self.opti_mode=='full-frame':
                    if self.trunc is not None:
                        max_rad=min(self.trunc*self.asize[j],max_rad)
                    if max_rad>self.minradius+3*self.asize[j]+self.asize[j]//2:
                        range_sel = list(range(self.minradius+self.asize[j]//2,self.minradius+3*self.asize[j]+self.asize[j]//2,self.asize[j]))
                        if max_rad>self.minradius+7*self.asize[j]:
                            range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,self.minradius+7*self.asize[j],2*self.asize[j])))
                            range_sel.extend(list(range(self.minradius+7*self.asize[j]+self.asize[j]//2,max_rad-3*self.asize[j]//2-1,4*self.asize[j])))
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                        else:
                            range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,max_rad-self.asize[j]//2,2*self.asize[j])))
                            if max_rad==self.minradius+7*self.asize[j]:
                                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                    else:
                        range_sel=list(range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j]))
                elif self.opti_mode=='annular':
                    range_sel=range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j])
                
                for i in range_sel:  
                    
                    indicesy,indicesx=get_time_series(self.cube[k],i)
                    cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[k],None,self.pa[k],self.scale_list[k], self.imlib, self.interpolation)  
                    cube_derot=rot_scale('fin',self.cube[k],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                    apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                    fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                    fluxes = np.array(fluxes['aperture_sum'])
                    x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                    y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                    
                    ceny, cenx = frame_center(cube_derot[0])
                    
                    self.opti_theta[k,i]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))
                    
                for i in range_sel:                    
                        
                    if self.model[j]=='APCA':
                        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                            bounds=np.array([[15,45],[1,4],[0.25,1]])
                        else:
                            bounds=np.array(self.opti_bound[j])
                            
                        param_type=['int','int','float']
        
                        if self.opti_mode=='full-frame':
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.ncomp[j][:,k]=int(opti_param[0])
                            self.nsegments[j][:,k]=int(opti_param[1])
                            self.delta_rot[j][:,k]=opti_param[2] 
                            break
                        else:
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.pso_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation, ncore=self.ncore)
                               
                            self.ncomp[j][i,k]=int(opti_param[0])
                            self.nsegments[j][i,k]=int(opti_param[1])
                            self.delta_rot[j][i,k]=opti_param[2] 
                            print('APCA Bayesian optimization: annulus {} done!'.format(i))                  
                       
                    elif self.model[j]=='NMF':
                        
                        self.param=[k,j,i]
                        
                        optires=[]
                        flux=[]
                        test_param=[]
                        sel_param=[]
                        
                        if self.opti_bound[j] is None:
                            bounds=[2,20]
                        else:
                            bounds=self.opti_bound[j][0]
                        
                        if self.opti_mode=='full-frame':
                            param_range=range(bounds[0],bounds[1]+1)
                            flux=np.zeros((len(range_sel),len(param_range)))
                            it1=0
                            for h in range_sel:
                                
                                self.param=[k,j,h]
                                res_param=[]
                                
                                res_param = pool_map(self.ncore, self.contrast_esti, iterable(param_range))
                                
                                res_mean=[]
                                for res_temp in res_param:
                                    res_mean.append(res_temp[0])
                                if self.param_opti_mode=='Contrast':
                                    res_mean=np.mean(np.asarray(res_mean))
                                else:
                                    res_mean=[np.mean(np.asarray(res_mean)[:,k]) for k in range(len(res_mean[0]))]
                                
                                it2=0
                                for res_temp in res_param:
                                    flux[it1,it2]=res_temp[1]
                                    if h==self.minradius+self.asize[j]//2:
                                        sel_param.append(res_temp[2])
                                        optires.append(np.asarray(res_temp[0])/np.asarray(res_mean))
                                    else:
                                        if self.param_opti_mode=='Contrast':
                                            optires[it2]+=res_temp[0]/res_mean
                                        else:
                                            optires[it2]=[optires[it2][k] + res_temp[0][k]/res_mean[k] for k in range(len(res_temp[0]))]
                                
                                    it2+=1
                                it1+=1
                            print('NMF optimization done!')
                            optires=np.array(optires)
                            if self.param_opti_mode!='Contrast':
                                self.crop[j][:,k]=self.crop[j][i,k]+np.unravel_index(optires.argmax(), optires.shape)[1]*2
                            self.ncomp[j][:,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]]
                            self.contrast[k,j,range_sel]=optires.max()
                            self.flux_opti[k,j,range_sel]=flux[:,np.unravel_index(optires.argmax(), optires.shape)[0]]
                            break
                                    
                                
                        else: 
                            
                            if self.opti_bound[j] is None:
                                bounds=[2,20]
                            else:
                                bounds=self.opti_bound[j][0]
                            
                            param_range=range(bounds[0],bounds[1]+1)
                            
                            res_param = pool_map(self.ncore, self.contrast_esti, iterable(param_range))
    
                            for res_temp in res_param:
                                
                                    optires.append(res_temp[0])
                                    flux.append(res_temp[1])
                                    sel_param.append(res_temp[2])
                            optires=np.array(optires)
                            if self.param_opti_mode!='Contrast':
                                self.crop[j][i,k]=self.crop[j][i,k]+np.unravel_index(optires.argmax(), optires.shape)[1]*2
                            self.ncomp[j][i,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]]
                            self.contrast[k,j,i]=optires.max()
                            self.flux_opti[k,j,i]=flux[np.unravel_index(optires.argmax(), optires.shape)[0]]
                            print('NMF optimization: annulus {} done!'.format(i))
                        
                    
                    elif self.model[j]=='LLSG':
                        
                        self.param=[k,j,i]
                        
                        optires=[]
                        flux=[]
                        test_param=[]
                        sel_param=[]
                        
                        if self.opti_bound[j] is None:
                                bounds=[[1,10],[1,4]]
                        else:
                                bounds=self.opti_bound[j]
                        
                        for l in range(bounds[0][0],bounds[0][1]+1):
                            for m in range(bounds[1][0],bounds[1][1]+1):
                                test_param.append([l,m])
                                
                        if self.opti_mode=='full-frame': 
                            flux=np.zeros((len(range_sel),len(test_param)))
                            it1=0
                            for h in range_sel:
                                self.param=[k,j,h]
                                res_param = pool_map(self.ncore, self.contrast_esti, iterable(np.array(test_param)))
                                
                                res_mean=[]
                                for res_temp in res_param:
                                    res_mean.append(res_temp[0])
                                if self.param_opti_mode=='Contrast':
                                    res_mean=np.mean(np.asarray(res_mean))
                                else:
                                    res_mean=[np.mean(np.asarray(res_mean)[:,k]) for k in range(len(res_mean[0]))]
                                
                                it2=0
                                for res_temp in res_param:
                                    flux[it1,it2]=res_temp[1]
                                    if h==self.minradius+self.asize[j]//2:
                                        sel_param.append(res_temp[2])
                                        optires.append(np.asarray(res_temp[0])/np.asarray(res_mean))
                                    else:
                                        if self.param_opti_mode=='Contrast':
                                            optires[it2]+=res_temp[0]/res_mean
                                        else:
                                            optires[it2]=[optires[it2][k] + res_temp[0][k]/res_mean[k] for k in range(len(res_temp[0]))]
                                
                                    it2+=1
                                it1+=1
                            print('LLSG optimization done!')
                            optires=np.array(optires)
                            
                            if self.param_opti_mode!='Contrast':
                                self.crop[j][:,k]=self.crop[j][i,k]+np.unravel_index(optires.argmax(), optires.shape)[1]*2
                            self.rank[j][:,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]][0]
                            self.nsegments[j][:,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]][1]
                            self.contrast[k,j,range_sel]=optires.max()
                            self.flux_opti[k,j,range_sel]=flux[:,np.unravel_index(optires.argmax(), optires.shape)[0]]
                            break
                                    
                                
                        else:
                                
                            res_param = pool_map(self.ncore, self.contrast_esti, iterable(np.array(test_param)))
    
                            for res_temp in res_param:
                                
                                    optires.append(res_temp[0])
                                    flux.append(res_temp[1])
                                    sel_param.append(res_temp[2])
                            
                            optires=np.array(optires) 
                            if self.param_opti_mode!='Contrast':
                                self.crop[j][i,k]=self.crop[j][i,k]+np.unravel_index(optires.argmax(), optires.shape)[0]*2
                            self.rank[j][i,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]][0]
                            self.nsegments[j][i,k]=sel_param[np.unravel_index(optires.argmax(), optires.shape)[0]][1]
                            self.contrast[k,j,i]=optires.max()
                            self.flux_opti[k,j,i]=flux[np.unravel_index(optires.argmax(), optires.shape)[0]]
                            print('LLSG optimization: annulus {} done!'.format(i))
                      
                    elif self.model[j]=='LOCI':
                        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[1e-3,1e-2],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])

                        param_type=['float','float']
                        
                        if self.opti_mode=='full-frame':
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.tolerance[j][:,k]=opti_param[0]
                            self.delta_rot[j][:,k]=opti_param[1]

                            break
                        else:
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation, ncore=self.ncore)
                            self.tolerance[j][i,k]=opti_param[0]
                            self.delta_rot[j][i,k]=opti_param[1]
                            print('LOCI Bayesian optimization: annulus {} done!'.format(i))      
                    
                    elif self.model[j]=='FM LOCI':

                        opti_mode=np.copy(self.param_opti_mode)
                        self.param_opti_mode='Contrast'
                        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[1e-3,1e-2],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])

                        param_type=['float','float']
                        
                        if self.opti_mode=='full-frame':
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.tolerance[j][:,k]=opti_param[0]
                            self.delta_rot[j][:,k]=opti_param[1]

                            break
                        else:
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.tolerance[j][i,k]=opti_param[0]
                            self.delta_rot[j][i,k]=opti_param[1]
                            print('FM LOCI Bayesian optimization: annulus {} done!'.format(i))      
                        
                        self.param_opti_mode=opti_mode
                        
                    elif self.model[j]=='KLIP':
        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[15,45],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])

                        param_type=['int','float']

                        if self.opti_mode=='full-frame':
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.ncomp[j][:,k]=int(opti_param[0])
                            self.delta_rot[j][:,k]=opti_param[1]

                            break
                        else:
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.ncomp[j][i,k]=int(opti_param[0])
                            self.delta_rot[j][i,k]=opti_param[1]
                            print('KLIP Bayesian optimization: annulus {} done!'.format(i))   
                            
                    elif self.model[j]=='FM KLIP':
                        opti_mode=np.copy(self.param_opti_mode)
                        self.param_opti_mode='Contrast'
        
                        self.param=[k,j,i]
                        
                        if self.opti_bound[j] is None:
                                bounds=np.array([[15,45],[0.25,1]])
                        else:
                                bounds=np.array(self.opti_bound[j])
                                
                        param_type=['int','float']

                        if self.opti_mode=='full-frame':
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,range_sel],simures,self.flux_opti[k,j,range_sel]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.ncomp[j][:,k]=int(opti_param[0])
                            self.delta_rot[j][:,k]=opti_param[1]

                            break
                        else:
                            if optimisation_model=='Bayesian':
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.bayesian_optimisation( self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            else:
                                opti_param,self.contrast[k,j,i],simures,self.flux_opti[k,j,i]=self.pso_optimisation(self.contrast_esti, bounds,param_type, prev_res=simures,opti_param=param_optimisation,ncore=self.ncore)
                            self.ncomp[j][i,k]=int(opti_param[0])
                            self.delta_rot[j][i,k]=opti_param[1]
                            print('FM KLIP Bayesian optimization: annulus {} done!'.format(i))      
                        self.param_opti_mode=opti_mode

                    print('Model parameters selection: Cube {} : Model {} : Radius {} done!'.format(k, j,i))
                
                    
                if self.opti_mode=='annular' and filt==True:
                    self.ncomp[j][range_sel,k]=remove_outliers(self.ncomp[j][:,k],range_sel)
                    self.rank[j][range_sel,k]=remove_outliers(self.rank[j][:,k],range_sel)
                    self.nsegments[j][range_sel,k]=remove_outliers(self.nsegments[j][:,k],range_sel).astype(int)
                    self.nsegments[j][:,k]=np.where(self.nsegments[j][:,k]==0,1,self.nsegments[j][:,k])
                    self.delta_rot[j][range_sel,k]=remove_outliers(self.delta_rot[j][:,k],range_sel)
                    self.tolerance[j][range_sel,k]=remove_outliers(self.tolerance[j][:,k],range_sel)
                    
                
        for j in range(len(self.model)):
             if self.param_opti_mode=='RSM':
                self.crop_range[j]=1
                
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                    
                    
    def RSM_test(self,cuben,modn,ann_center,sel_theta,sel_flux,thresh=False):
        
        """
        Function computing the cube of likelihoods for a given PSF-subtraction 
        techniques 'modn', a given ADI sequence 'cuben' and a given angular distance 'ann_center'
        with or without the injection of a fake companion (respc. thresh=False and thresh=False). 
        Sel_theta indicate the azimuth of the injected fake companion and sel_flux the flux 
        associated to the fake companion. This function is used by the RSM optimization function (opti_RSM).
        """

        if thresh==False:
            
            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])
            
            cube_fc= cube_inject_companions(np.zeros_like(self.cube[cuben]), psf_template,
                                 self.pa[cuben], sel_flux, self.pxscale,
                                 rad_dists=ann_center,
                                 theta=sel_theta,
                                 verbose=False)
               
            result = self.likelihood(ann_center,cuben,modn,np.zeros_like(self.cube[cuben]),cube_fc,False)
                
        else:
                
            result = self.likelihood(ann_center,cuben,modn,np.zeros_like(self.cube[cuben]),None,False)
        
        like_temp=np.zeros(((rot_scale('ini',self.cube[cuben],None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)[0].shape[0]+1),self.cube[cuben].shape[-2],self.cube[cuben].shape[-1],len(self.interval[modn][ann_center,cuben]),2,self.crop_range[modn]))    
        
        indicesy,indicesx=get_time_series(self.cube[cuben],ann_center)          
        if self.model[modn]=='FM LOCI' or self.model[modn]=='FM KLIP':
            like_temp[:,indicesy,indicesx,:,:,:]=result[1]
            self.psf_fm[cuben][modn][result[0]]=result[2]
        else:
             like_temp[:,indicesy,indicesx,:,:,:]=result[1]
        
        
        like=[]
        flux_FMMF=[]

        for k in range(self.crop_range[modn]):
            like.append(like_temp[0:(like_temp.shape[0]-1),:,:,:,:,k])
            flux_FMMF.append(like_temp[(like_temp.shape[0]-1),:,:,0,0,k])
        
        self.like_fin[cuben][modn]=like 
        self.flux_FMMF[cuben][modn]=flux_FMMF 
                
                
    def perf_esti(self,cuben,modn,ann_center,sel_theta):
        
        """
        Function computing the performance index used for the RSM optimization 
        based on the cube of likelihoods generated by a PSF-subtraction 
        techniques 'modn', relying on the ADI sequence 'cuben'. The performance 
        index is defined as the ratio of the peak probability of
        an injected fake companion (injected at an angular distance 'ann_center'
        and an azimuth 'sel_theta') on the peak (noise) probability in the 
        remaining of the considered annulus. 
        This function is used by the RSM optimization function (opti_RSM).
        """

        ceny, cenx = frame_center(self.cube[cuben])
        twopi=2*np.pi
        sigposy=int(ceny + np.sin(sel_theta/360*twopi)*ann_center)
        sigposx=int(cenx+ np.cos(sel_theta/360*twopi)*ann_center)

        indc = disk((sigposy, sigposx),int(self.fwhm/2)+1)
        
        max_detect=self.probmap[indc[0],indc[1]].max()
        
        self.probmap[indc[0],indc[1]]=0
        
        indicesy,indicesx=get_time_series(self.cube[cuben],ann_center)
        
        if self.inv_ang==True:
            bg_noise=np.max(self.probmap[indicesy,indicesx])
        else:
            bg_noise=np.mean(self.probmap[indicesy,indicesx])
        
        return max_detect/bg_noise
    

        
    def opti_RSM_crop(self,ann_center,cuben,modn,estimator,colmode):
        
        """
        Function computing the performance index of the RSM detection map for a given range
        of crop sizes self.crop_range for the annulus ann_center on the cube of likelihoods generated 
        by a PSF-subtraction techniques 'modn', relying on the ADI sequence 'cuben'. 
        The detection map on which the perforance index is computed is using the
        estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') the sum the obtained probabilities along the time axis.
        This function is used by the RSM optimization function (opti_RSM).
        """
                            
        opti_res=[]
        self.RSM_test(cuben,modn,ann_center,self.opti_theta[cuben,ann_center],self.flux_opti[cuben,modn,ann_center])
        for l in range(self.crop_range[modn]):
            
            self.probmap_esti(modthencube=True,ns=1,sel_crop=[l], estimator=estimator,colmode=colmode,ann_center=ann_center,sel_cube=[[cuben,modn]])
            opti_res.append(self.perf_esti(cuben,modn,ann_center,self.opti_theta[cuben,ann_center]))

        return np.asarray(opti_res),ann_center
    

    def opti_RSM_var_annular(self,ann_center,cuben,modn,estimator,colmode):
        
        """
        Function computing the performance index of the RSM detection map for the different
        possible regions used to compute the noise mean and variance ('ST','FR','FM','SM','TE')
        and the two estimation mode (Gaussian maximum likelihood or variance base estimator, resp. 
        flux= True or False) for the annulus ann_center on the cube of likelihoods generated 
        by a PSF-subtraction techniques 'modn', relying on the ADI sequence 'cuben' and the annular 
        optimization mode. The detection map on which the performance index is computed uses the
        estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis.
        This function is used by the RSM optimization function (opti_RSM).
        """
        
        optires=np.zeros((4))
        if self.cube[cuben].ndim==4:
            var_list=[['FM',],['FM']]
        else:
            var_list=[['FM','TE','ST','SM'],['FM','TE']]
            
        if self.intensity[modn][ann_center,cuben]=='Pixel':
            n=1
        else:
            n=0
        for m in range(len(var_list[0])):
            if (n==1 and m<2) or n==0:
                self.var[modn][ann_center,cuben]=var_list[n][m]
                self.RSM_test(cuben,modn,ann_center,self.opti_theta[cuben,ann_center],self.flux_opti[cuben,modn,ann_center])
                self.probmap_esti(modthencube=True,ns=1,sel_crop=[0], estimator=estimator,colmode=colmode,ann_center=ann_center,sel_cube=[[cuben,modn]])
                optires[m]=self.perf_esti(cuben,modn,ann_center,self.opti_theta[cuben,ann_center])

        return optires,ann_center
    
    
    def opti_RSM_var_full(self,ann_center,cuben,modn,estimator,colmode):
        
        """
        Function computing the performance index of the RSM detection map for the different
        possible regions used to compute the noise mean and variance ('ST','FR','FM','SM','TE')
        and the two estimation mode (Gaussian maximum likelihood or variance base estimator, resp. 
        flux= True or False) for the annulus ann_center on the cube of likelihoods generated 
        by a PSF-subtraction techniques 'modn', relying on the ADI sequence 'cuben' and the full-frame 
        optimization mode. The detection map on which the performance index is computed uses the
        estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis.
        This function is used by the RSM optimization function (opti_RSM).
        """
        
        self.RSM_test(cuben,modn,ann_center,self.opti_theta[cuben,ann_center],self.flux_opti[cuben,modn,ann_center])
        self.probmap_esti(modthencube=True,ns=1,sel_crop=[0], estimator=estimator,colmode=colmode,ann_center=ann_center,sel_cube=[[cuben,modn]])
        optires=self.perf_esti(cuben,modn,ann_center,self.opti_theta[cuben,ann_center])
        

        return optires,ann_center
    
    
                  
    def opti_RSM(self,estimator='Forward',colmode='median'):
        
        
        """
        Function optimizing five parameters of the RSM algorithm, the crop size, the method used 
        to compute the intensity parameter (pixel-wise Gaussian maximum likelihood or annulus-wise, 
        variance based estimation), the region used for the computation of the noise mean and variance 
        ('ST', 'FR', 'FM', 'SM', 'TE', see Dalqvist et al. 2021 for the definition), and define
        if the noise mean and variance estimation should be performed empiricaly or via best fit.
        For the variance based estimation of the intensity parameter, an additional parameter, 
        the multiplicative factor, is also optimized. The detection map on which the performance index
        is computed uses the estimator ('Forward' or 'Forward-Bakward') probability computation mode and
        the colmode ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis.
        
        Parameters
        ----------
        
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        """
        
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.max_r=self.maxradius
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.max_r=self.maxradius  
        
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
        
        self.distri[:]=np.repeat('A',len(self.model))
        self.crop_noflux=self.crop.copy()
        self.crop_flux=self.crop.copy()
        self.opti_theta=np.zeros((len(self.cube),self.maxradius+5))


        for j in range(len(self.model)):
            for i in range(len(self.cube)):
                
                # Determination of the considered angular distances for the optimization process
                
                if self.model[j]=='FM KLIP' or self.model[j]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                               
                interval=int(self.interval[j][0,i])
                res_interval_crop=np.zeros((max_rad,interval,self.crop_range[j]))
                
                if self.opti_mode=='full-frame':
                    if self.trunc is not None:
                        max_rad=min(self.trunc*self.asize[j],max_rad)
                    if max_rad>self.minradius+3*self.asize[j]+self.asize[j]//2:
                        range_sel = list(range(self.minradius+self.asize[j]//2,self.minradius+3*self.asize[j]+self.asize[j]//2,self.asize[j]))
                        if max_rad>self.minradius+7*self.asize[j]:
                            range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,self.minradius+7*self.asize[j],2*self.asize[j])))
                            range_sel.extend(list(range(self.minradius+7*self.asize[j]+self.asize[j]//2,max_rad-3*self.asize[j]//2-1,4*self.asize[j])))
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                        else:
                            range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,max_rad-self.asize[j]//2,2*self.asize[j])))
                            if max_rad==self.minradius+7*self.asize[j]:
                                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                    else:
                        range_sel=list(range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j]))
                elif self.opti_mode=='annular':
                    range_sel=range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j])
                    
                    
                    
                # Computation of the median flux position in the original ADI sequence which will be used during the RSM optimization
                for k in range_sel:  
                    
                    indicesy,indicesx=get_time_series(self.cube[i],k)
                    cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[i],None,self.pa[i],self.scale_list[i], self.imlib, self.interpolation)  
                    cube_derot=rot_scale('fin',self.cube[i],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                    apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                    fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                    fluxes = np.array(fluxes['aperture_sum'])
                    x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                    y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                    
                    ceny, cenx = frame_center(cube_derot[0])
                    
                    self.opti_theta[i,k]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))

                # Step-1: Selection of the optimal crop size, the intensity parameter estimator, and the multiplicative factor for the variance based estimator   
                
                
                
                self.distrifit[j][:,i]=False
                
                self.var[j][:,i]='FR'

                for k in range(1,interval+1):
                    self.interval[j][:,i]=[k]
            
                    res_param=pool_map(self.ncore, self.opti_RSM_crop, iterable(range_sel),i,j,estimator,colmode)
                    
                    for res_temp in res_param:
                        res_interval_crop[res_temp[1],k-1,:]=res_temp[0]
                    
                if self.opti_mode=='full-frame':
                    interval_crop_sum=np.asarray(res_interval_crop).sum(axis=0) 
                    self.interval[j][:,i]=np.unravel_index(interval_crop_sum.argmax(), interval_crop_sum.shape)[0]+1
                    self.crop_noflux[j][:,i]=self.crop[j][0,i]+2*np.unravel_index(interval_crop_sum.argmax(), interval_crop_sum.shape)[1]
                
                if self.opti_mode=='annular':
                    
                    interval_crop_inter=np.copy(res_interval_crop)
                    for n in range(interval):
                        for m in range(self.crop_range[j]):
                            interval_crop_inter[range_sel[0]:(range_sel[-1]+1),n,m]=interpolation(interval_crop_inter[:,n,m],range_sel)
                    for l in range(self.minradius,max_rad):
                        self.interval[j][l,i]=np.unravel_index(interval_crop_inter[l,:,:].argmax(), interval_crop_inter[l,:,:].shape)[0]+1
                        self.crop_noflux[j][l,i]=self.crop[j][0,i]+2*np.unravel_index(interval_crop_inter[l,:,:].argmax(), interval_crop_inter[l,:,:].shape)[1]          
    
                print('Interval selected: Cube {}, Model {}'.format(i,j))
        
                self.intensity[j][:,i]='Pixel'
        
                res_param=pool_map(self.ncore, self.opti_RSM_crop, iterable(range_sel),i,j,estimator,colmode)
             
                crop_sel=np.zeros((max_rad,self.crop_range[j]))
                
                for res_temp in res_param:
                    crop_sel[res_temp[1],:]=res_temp[0]
                    
                if self.opti_mode=='full-frame':
                    crop_sum=np.asarray(crop_sel).sum(axis=0) 
                    self.crop_flux[j][:,i]=self.crop[j][0,i]+2*crop_sum.argmax()
                
                if self.opti_mode=='annular':
                        for m in range(self.crop_range[j]):
                            crop_sel[range_sel[0]:(range_sel[-1]+1),m]=interpolation(crop_sel[:,m],range_sel)
                        for l in range(self.minradius,max_rad):
                            self.crop_flux[j][l,i]=self.crop[j][0,i]+2*crop_sel[l,:].argmax()
                            
                print('Crop size selected: Cube {} : Model {}'.format(i,j))
                
                # Step-2: selection of the optimal region to compute the noise mean and variance
                
                optires=np.zeros((5))
                var_sel=np.zeros((max_rad,5))
                if self.opti_mode=='full-frame':
                    if interval_crop_sum.max()>crop_sum.max():
                        self.intensity[j][:,i]='Annulus'
                        self.crop[j][:,i]=self.crop_noflux[j][:,i]
                        optires[0]=interval_crop_sum.max()
                    else:
                        self.intensity[j][:,i]='Pixel'
                        self.crop[j][:,i]=self.crop_flux[j][:,i]
                        optires[0]=crop_sum.max()

                if self.opti_mode=='annular':
                    for l in range(self.minradius,max_rad):
                        if interval_crop_inter[l,:,:].max()>crop_sel[l,:].max():
                            self.intensity[j][l,i]='Annulus'
                            self.crop[j][l,i]=self.crop_noflux[j][l,i]
                            var_sel[l,0]=interval_crop_inter[l,:,:].max()
                        else:
                            self.intensity[j][l,i]='Pixel'
                            self.crop[j][l,i]=self.crop_flux[j][l,i]
                            var_sel[l,0]=crop_sel[l,:].max()
                
                
                crop_range_temp=np.copy(self.crop_range[j])
                self.crop_range[j]=1
                
                fit_sel=np.zeros((max_rad,2))
            
                if self.opti_mode=='full-frame':
                    
                    if self.cube[i].ndim==4:
                        var_list=[['FR','FM',],['FR','FM']]
                    else:
                        var_list=[['FR','FM','ST','SM'],['FR','FM']]
                        
                    if self.intensity[j][0,i]=='Pixel':
                        n=1
                    else:
                        n=0
                        
                    for m in range(1,len(var_list[0])):
                        
                        if (n==1 and m<2) or n==0: # m>3 if TE is used
                            self.var[j][:,i]=var_list[n][m]
                            res_param=pool_map(self.ncore, self.opti_RSM_var_full, iterable(range_sel),i,j,estimator,colmode)
                            opti_temp=0
                            for res_temp in res_param:
                                opti_temp+=res_temp[0]
                            optires[m]=opti_temp
                                  
                    self.var[j][:,i]=['FR','FM','ST','SM'][optires.argmax()]
                    fit_sel[0,0]=optires.max()
                    
                    
                if self.opti_mode=='annular':   
                    res_param=pool_map(self.ncore, self.opti_RSM_var_annular, iterable(range_sel),i,j,estimator,colmode)
                 
                    
                    for res_temp in res_param:
                        var_sel[res_temp[1],1:5]=res_temp[0]
                        
                    for m in range(5):
                        var_sel[range_sel[0]:(range_sel[-1]+1),m]=interpolation(var_sel[:,m],range_sel)
                    for l in range(self.minradius,max_rad):
                        self.var[j][l,i]=['FR','FM','ST','SM'][var_sel[l,:].argmax()]
                        if self.intensity[j][l,i]=='Pixel' and (self.var[j][l,i]=='ST' or self.var[j][l,i]=='SM'):
                            self.var[j][l,i]='FR'
                        fit_sel[l,0]=var_sel[l,:].max()

                print('Variance estimation method selected: Cube {} : Model {}'.format(i,j))
                
                # Step-3: definition of the approacch to compute the noise meaan and variance, either empirically or via best fit
                  
                self.distrifit[j][:,i]=True
                res_param=pool_map(self.ncore, self.opti_RSM_var_full, iterable(range_sel),i,j,estimator,colmode)
                
                for res_temp in res_param:
                    fit_sel[res_temp[1],1]=res_temp[0]

                if self.opti_mode=='full-frame':
                    fit_sum=np.asarray(fit_sel).sum(axis=0) 
                    self.distrifit[j][:,i]==[False,True][fit_sum.argmax()] 

                if self.opti_mode=='annular':
                        fit_sel[range_sel[0]:(range_sel[-1]+1),1]=interpolation(fit_sel[:,1],range_sel)
                        for l in range(self.minradius,max_rad):
                            self.distrifit[j][l,i]=[False,True][fit_sel[l,:].argmax()]
                
                print('Fit method selected: Cube {} : Model {}'.format(i,j))
                
                self.crop_range[j]=crop_range_temp
                                
            self.crop_range[j]=1
            
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
     
        
    def opti_combi_full(self,l,estimator,colmode,op_sel,SNR=False,ini=False):
        
        """
        Function computing the performance index of the RSM detection map for the tested sets of
        likelihhod cubes/residuals cubes used by the optimal subset selection function in the case of the full-frame 
        optimization mode. The performance index is computed based on the set of likelihood cubes op_seli
        for the radial distance l, relying either on the RSM algorithm or S/N map to compute the detection map.
        When using the RSM algorithm to gennerated the final probability map, the detection map is computed 
        using the estimator ('Forward' or 'Forward-Bakward') probability computation mode and the colmode
        ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis. 
        This function is used by the optimal subset of likelihood cubes selection function (RSM_combination).
        """
        
        op_seli=op_sel.copy()
        if SNR==True:
            
            mod_sel=[]
            
            for k in range(len(op_seli)):
                i=op_seli[k][0]
                j=op_seli[k][1]
                if (self.model[j]!='FM KLIP' and self.model[j]!='FM LOCI') or (self.model[j]=='FM KLIP' and l<self.max_r) or (self.model[j]=='FM LOCI' and l<self.max_r):
                    mod_sel.append(op_seli[k])
            if len(mod_sel)>0:
                return self.contrast_multi_snr(l,mod_sel=mod_sel)
            else:
                return 0
        else:
                
            mod_deli=[]
            
            if self.contrast_sel=='Max':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmax(), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Median':
                opti_pos=np.unravel_index(np.argmin(abs(self.flux_opti[:,:,l]-np.median(self.flux_opti[:,:,l]))), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Min':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmin(), self.flux_opti[:,:,l].shape)
                
            opti_theta=self.opti_theta[opti_pos[0],l]        
            for k in range(len(op_seli)):
                i=op_seli[k][0]
                j=op_seli[k][1]
                if (self.model[j]=='FM KLIP' and l>=self.max_r) or (self.model[j]=='FM LOCI' and l>=self.max_r) or (self.model[j]=='FM KLIP' and ini==True) or (self.model[j]=='FM LOCI' and ini==True):
                    mod_deli.append(k)

            
            if len(mod_deli)>0:
                for index in sorted(mod_deli, reverse=True): 
                    del op_seli[index]  
                
            if len(op_seli)>0:
                self.probmap_esti(ns=1, estimator=estimator,colmode=colmode,ann_center=l,sel_cube=op_seli)
                return self.perf_esti(i,j,l,opti_theta)
            else:
                return 0
    
    
    
    
    def opti_combi_annular(self,k,estimator,colmode,SNR=False): 
        
        """
        Function computing the performance index of the RSM detection map for the tested sets of
        likelihhod cubes/residuals cubes used by the optimal subset selection function in the case of the annular 
        optimization mode. The performance index is computed based on the set of likelihood cubes op_sel
        for the radial distance l, relying either on the RSM algorithm or S/N map to compute the detection map.
        When using the RSM algorithm to gennerated the final probability map, the detection map is computed 
        using the estimator ('Forward' or 'Forward-Bakward') probability computation mode and the colmode
        ('mean', 'median' or 'max') to sum the obtained probabilities along the time axis. 
        This function is used by the optimal subset of likelihood cubes selection function (RSM_combination).
        """
    
        op_sel=[]
        res_sep=[]
        sel_cube=[]
        if self.contrast_sel=='Max':
            opti_pos=np.unravel_index(self.flux_opti[:,:,k].argmax(), self.flux_opti[:,:,k].shape)  
        elif self.contrast_sel=='Median':
            opti_pos=np.unravel_index(np.argmin(abs(self.flux_opti[:,:,k]-np.median(self.flux_opti[:,:,k]))), self.flux_opti[:,:,k].shape)
        elif self.contrast_sel=='Min':
            opti_pos=np.unravel_index(self.flux_opti[:,:,k].argmin(), self.flux_opti[:,:,k].shape)
         
        if self.combination=='Top-Down':
                
                mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
                it=0
                for i in range(len(self.cube)):
                    for j in range(len(self.model)):
                        mod_sel[it]=[i,j]
                        it+=1
    
                for i in range(len(self.cube)):
                    for j in range(len(self.model)):
                        
                        if SNR==True:
                            if (self.model[j]=='FM KLIP' and k>=self.max_r) or (self.model[j]=='FM LOCI' and k>=self.max_r):
                                del mod_sel[mod_sel.index([i,j])]

                        else:
                            if (self.model[j]!='FM KLIP' and self.model[j]!='FM LOCI') or  (self.model[j]=='FM KLIP' and k<self.max_r) or (self.model[j]=='FM LOCI' and k<self.max_r):
                                self.RSM_test(i,j,k,self.opti_theta[opti_pos[0],k] ,self.flux_opti[opti_pos[0],opti_pos[1],k])
                            else:
                                del mod_sel[mod_sel.index([i,j])]
                                
                if SNR==True:  
                    if len(mod_sel)>0:
                        res_opti=self.contrast_multi_snr(k,mod_sel=mod_sel)
                    else:
                        res_opti==0
                else:
                    self.probmap_esti(ns=1, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=mod_sel)
                    res_opti=self.perf_esti(i,j,k,self.opti_theta[opti_pos[0],k])

                prev_res_opti=0
                while res_opti>prev_res_opti:
                    prev_res_opti=res_opti
                    res_temp=[]
                    for i in range(len(mod_sel)):
                        temp_sel=mod_sel.copy()
                        del temp_sel[i]
                        if SNR==True:
                            res_temp.append(self.contrast_multi_snr(k,mod_sel=temp_sel))
                        else:
                            self.probmap_esti(ns=1, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=temp_sel) 
                            res_temp.append(self.perf_esti(i,j,k,self.opti_theta[opti_pos[0],k]) )
                    res_opti=max(res_temp)
                    if res_opti>prev_res_opti: 
                        del mod_sel[np.argmax(res_temp)]
                    
            
                op_sel=mod_sel
    
        elif self.combination=='Bottom-Up':
            
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    
                    if SNR==True:
                        res_sep.append(self.contrast_multi_snr(k,mod_sel=[[i,j]]))
                        sel_cube.append([i,j])
                    else:
                        
                        if (self.model[j]!='FM KLIP' and self.model[j]!='FM LOCI') or  (self.model[j]=='FM KLIP' and k<self.max_r) or (self.model[j]=='FM LOCI' and k<self.max_r):
                                 self.RSM_test(i,j,k,self.opti_theta[opti_pos[0],k] ,self.flux_opti[opti_pos[0],opti_pos[1],k])
                                 self.probmap_esti(ns=1, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=[[i,j]])
                                 res_sep.append(self.perf_esti(i,j,k,self.opti_theta[opti_pos[0],k]))
                                 sel_cube.append([i,j])
                
            op_sel.append(sel_cube[np.argmax(np.array(res_sep))])
            opti_res=max(res_sep)
            del sel_cube[np.argmax(np.array(res_sep))]
            prev_opti_res=0
            while opti_res>prev_opti_res and len(sel_cube)>0:
                res_temp=[]
                mod_del=[]
                for l in range(len(sel_cube)):
                    op_sel.append(sel_cube[l])
                    if SNR==True:
                        res_temp.append(self.contrast_multi_snr(k,mod_sel=op_sel))
                    else:
                        self.probmap_esti(ns=1, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=op_sel)
                        res_temp.append(self.perf_esti(i,j,k,self.opti_theta[opti_pos[0],k]))
                    del op_sel[len(op_sel)-1]
                    
                    if res_temp[-1]<opti_res:
                            mod_del.append(l)
                    
                if max(res_temp)>opti_res:
                    prev_opti_res=opti_res
                    opti_res=max(res_temp)
                    op_sel.append(sel_cube[np.argmax(np.array(res_temp))])
                    mod_del.append(np.argmax(np.array(res_temp)))
                    if len(mod_del)>0:
                        for index in sorted(mod_del, reverse=True): 
                            del sel_cube[index]
                else:
                    prev_opti_res=opti_res
        
        print('Greedy selection: Radius {} done!'.format(k))  
        return op_sel
    
    def RSM_test_multi(self,cuben,modn,range_sel):
        
        """
        Function computing the cube of likelihoods for a given PSF-subtraction 
        techniques 'modn', a given ADI sequence 'cuben' and a given angular distance 'ann_center'
        with or without the injection of a fake companion (respc. thresh=False and thresh=False). 
        Sel_theta indicate the azimuth of the injected fake companion and sel_flux the flux 
        associated to the fake companion. This function is used by the RSM optimization function (opti_RSM).
        """
        like_temp=np.zeros(((rot_scale('ini',self.cube[cuben],None,self.pa[cuben],self.scale_list[cuben],self.imlib, self.interpolation)[0].shape[0]+1),self.cube[cuben].shape[-2],self.cube[cuben].shape[-1],len(self.interval[modn][np.argmax(self.interval[modn][:,cuben]),cuben]),2,self.crop_range[modn]))    
            
        for l in range_sel:
            
            if self.contrast_sel=='Max':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmax(), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Median':
                opti_pos=np.unravel_index(np.argmin(abs(self.flux_opti[:,:,l]-np.median(self.flux_opti[:,:,l]))), self.flux_opti[:,:,l].shape)
            elif self.contrast_sel=='Min':
                opti_pos=np.unravel_index(self.flux_opti[:,:,l].argmin(), self.flux_opti[:,:,l].shape)
                
            flux_opti=self.flux_opti[opti_pos[0],opti_pos[1],l]
            opti_theta=self.opti_theta[opti_pos[0],l]    
                
            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])

            
            cube_fc= cube_inject_companions(np.zeros_like(self.cube[cuben]), psf_template,
                                 self.pa[cuben], flux_opti, self.pxscale,
                                 rad_dists=l,
                                 theta=opti_theta,
                                 verbose=False)
               
            result = self.likelihood(l,cuben,modn,np.zeros_like(self.cube[cuben]),cube_fc,False)
                

            indicesy,indicesx=get_time_series(self.cube[cuben],l)          
            if self.model[modn]=='FM LOCI' or self.model[modn]=='FM KLIP':
                like_temp[:,indicesy,indicesx,:,:,:]=result[1]
                self.psf_fm[cuben][modn][result[0]]=result[2]
            else:
                 like_temp[:,indicesy,indicesx,:,:,:]=result[1]
        
        
        like=[]
        flux_FMMF=[]

        for k in range(self.crop_range[modn]):
            like.append(like_temp[0:(like_temp.shape[0]-1),:,:,:,:,k])
            flux_FMMF.append(like_temp[(like_temp.shape[0]-1),:,:,0,0,k])
        
        self.like_fin[cuben][modn]=like 
        self.flux_FMMF[cuben][modn]=flux_FMMF 
    
                                         
    def opti_combination(self,estimator='Forward',colmode='median',threshold=True,contrast_sel='Max',combination='Bottom-Up',SNR=False):  

        """
        Function selecting the sets of likelihhod cubes/residuals cubes maximizing the annulus-wise 
        perfomance index for the annular or the global performance index for full-frame optimization
        mode, for respectively the auto-RSM and auto-S/N frameworks.
        
        Parameters
        ----------
        

        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        threshold: bool, optional
            When True a radial treshold is computed on the final detection map with parallactic angles reversed.
            For a given angular separation, the radial threshold is defined as the maximum probability observed 
            within the annulus. The radia thresholds are checked for outliers and smoothed via a Hampel filter.
            Only used when relying on the auto-RSM framework. Default is True.
        contrast_sel: str,optional
            Contrast and azimuth definition for the optimal likelihood cubes/ residuall cubes selection.
            If 'Max' ('Min' or 'Median'), the largest (smallest or median) contrast obtained during the 
            PSF-subtraction techniques optimization will be chosen along the corresponding 
            azimuthal position for the likelihood cubes selection. Default is 'Max'.
        combination: str,optional
            Type of greedy selection algorithm used for the selection of the optimal set of cubes 
            of likelihoods/cubes of residuals (either 'Bottom-Up' or 'Top-Down'). For more details
            see Dahlqvist et al. (2021). Default is 'Bottom-Up'.
        SNR: bool,optional
            If True, the auto-S/N framework is used, resulting in an optimizated final S/N map when using 
            subsequently the opti_map. If False the auto-RSM framework is used, providing an optimized
            probability map when using subsequently the opti_map.
        """
        
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.maxradius=self.max_r
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.maxradius=self.max_r  
        
        self.opti=True
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
        self.opti_sel=list(np.zeros((self.maxradius+1)))
        self.threshold=np.zeros((self.maxradius+1))            
        self.contrast_sel=contrast_sel
        self.combination=combination
        
                
        for j in range(len(self.model)):
            for i in range(len(self.cube)):
                
                # Determination of the considered angular distances for the optimization process
                
                if self.model[j]=='FM KLIP' or self.model[j]=='FM LOCI':
                    max_rad=self.max_r+1
                else:
                    max_rad=self.maxradius+1
                
                if self.opti_mode=='full-frame':
                    if self.trunc is not None:
                        max_rad=min(self.trunc*self.asize[j],max_rad)
                    if max_rad>self.minradius+3*self.asize[j]+self.asize[j]//2:
                        range_sel = list(range(self.minradius+self.asize[j]//2,self.minradius+3*self.asize[j]+self.asize[j]//2,self.asize[j]))
                        if max_rad>self.minradius+7*self.asize[j]:
                            range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,self.minradius+7*self.asize[j],2*self.asize[j])))
                            range_sel.extend(list(range(self.minradius+7*self.asize[j]+self.asize[j]//2,max_rad-3*self.asize[j]//2-1,4*self.asize[j])))
                            range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                        else:
                            range_sel.extend(list(range(self.minradius+3*self.asize[j]+self.asize[j]//2,max_rad-self.asize[j]//2,2*self.asize[j])))
                            if max_rad==self.minradius+7*self.asize[j]:
                                range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[j]*self.asize[j]-self.asize[j]//2-1)
                    else:
                        range_sel=list(range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j]))
                elif self.opti_mode=='annular':
                    range_sel=range(self.minradius+self.asize[j]//2,max_rad-self.asize[j]//2,self.asize[j])
                    
                    
                    
                # Computation of the median flux position in the original ADI sequence which will be used during the RSM optimization
                for k in range_sel:  
                    
                    indicesy,indicesx=get_time_series(self.cube[i],k)
                    cube_derot,angle_list,scale_list=rot_scale('ini',self.cube[i],None,self.pa[i],self.scale_list[i], self.imlib, self.interpolation)  
                    cube_derot=rot_scale('fin',self.cube[i],cube_derot,angle_list,scale_list, self.imlib, self.interpolation)
                    apertures = photutils.CircularAperture(np.array((indicesx, indicesy)).T, round(self.fwhm/2))
                    fluxes = photutils.aperture_photometry(cube_derot.sum(axis=0), apertures)
                    fluxes = np.array(fluxes['aperture_sum'])
                    x_sel=indicesx[np.argsort(fluxes)[len(fluxes)//2]]
                    y_sel=indicesy[np.argsort(fluxes)[len(fluxes)//2]]
                    
                    ceny, cenx = frame_center(cube_derot[0])
                    
                    self.opti_theta[i,k]=np.degrees(np.arctan2(y_sel-ceny, x_sel-cenx))
                    
                if SNR==False:   
                    self.RSM_test_multi(i,j,range_sel)
        
        # Selection of the optimal set of cubes of likelihoods / cubes of residuals via top-down or bottom-up greedy selection
        ncore_temp=np.copy(self.ncore)
        self.ncore=1
        if self.opti_mode=='full-frame':
            
            # Determination of the considered angular distances for the optimization process

            if self.trunc is not None:
                max_rad=min(self.trunc*self.asize[0],self.maxradius+1)
            else:
                max_rad=self.maxradius+1
            if max_rad>self.minradius+3*self.asize[0]+self.asize[0]//2:
                range_sel = list(range(self.minradius+self.asize[0]//2,self.minradius+3*self.asize[0]+self.asize[0]//2,self.asize[0]))
                if max_rad>self.minradius+7*self.asize[0]:
                    range_sel.extend(list(range(self.minradius+3*self.asize[0]+self.asize[0]//2,self.minradius+7*self.asize[0],2*self.asize[0])))
                    range_sel.extend(list(range(self.minradius+7*self.asize[0]+self.asize[0]//2,max_rad-3*self.asize[0]//2-1,4*self.asize[0])))
                    range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[0]*self.asize[0]-self.asize[0]//2-1)
                else:
                    range_sel.extend(list(range(self.minradius+3*self.asize[0]+self.asize[0]//2,max_rad-self.asize[0]//2,2*self.asize[0])))
                    if max_rad==self.minradius+7*self.asize[0]:
                        range_sel.append(self.minradius+(max_rad-self.minradius)//self.asize[0]*self.asize[0]-self.asize[0]//2-1)
            else:
                range_sel=list(range(self.minradius+self.asize[0]//2,max_rad-self.asize[0]//2,self.asize[0]))

            if self.combination=='Top-Down':
                mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
                it=0
                for i in range(len(self.cube)):
                    for j in range(len(self.model)):
                        mod_sel[it]=[i,j]
                        it+=1
    
                results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,mod_sel,SNR)
                res_opti=sum(results) 
                prev_res_opti=0
                print('Initialization done!')
                while res_opti>prev_res_opti:
                    prev_res_opti=res_opti
                    res_temp=[]
                    for i in range(len(mod_sel)):
                        temp_sel=mod_sel.copy()
                        del temp_sel[i]
                        
                        results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,temp_sel,SNR)
                        res_temp.append(sum(results))
                    res_opti=max(res_temp)
                    if res_opti>prev_res_opti: 
                        del mod_sel[np.argmax(res_temp)]
                    
                    print('Round done!') 
    
            
                self.opti_sel=list([mod_sel]*(self.maxradius+1))


                #if self.contrast_sel!='Min':
                #    self.contrast_sel='Min'
                #    for n in range(len(mod_sel)):
                #        self.RSM_test_multi(mod_sel[n][0],mod_sel[n][1],range_sel)
                #    results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,mod_sel,SNR)
                #    self.opti_combi_res=np.sum(results)
                #else:    
                #     self.opti_combi_res=res_opti
                     
                print('Greedy selection done!') 
                
            elif self.combination=='Bottom-Up':
                op_sel=[]
                res_sep=[]
                sel_cube=[]
                for i in range(len(self.cube)):
                    for j in range(len(self.model)):

                                results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,[[i,j]],SNR,True)
                                
                                res_sep.append(sum(results))
                                sel_cube.append([i,j])
                
                print('Initialization done!')
                op_sel.append(sel_cube[np.argmax(np.array(res_sep))])
                opti_res=max(res_sep)
                del sel_cube[np.argmax(np.array(res_sep))]
                prev_opti_res=0
                while opti_res>prev_opti_res and len(sel_cube)>0:
                    res_temp=[]
                    mod_del=[]
                    for l in range(len(sel_cube)):
                        op_sel.append(sel_cube[l])
                        
                        results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,op_sel,SNR)
                        
                        res_temp.append(sum(results))
                        if sum(results)<opti_res:
                            mod_del.append(l)
    
                        del op_sel[len(op_sel)-1]
                        
                    if max(res_temp)>opti_res:
                        prev_opti_res=opti_res
                        opti_res=max(res_temp)
                        op_sel.append(sel_cube[np.argmax(np.array(res_temp))])
                        mod_del.append(np.argmax(np.array(res_temp)))
                    else:
                        prev_opti_res=opti_res
                    if len(mod_del)>0:
                        for index in sorted(mod_del, reverse=True): 
                            del sel_cube[index]
                        
                    print('Round done!')
                
                self.opti_sel=list([op_sel]*(self.maxradius+1))
                
                #if self.contrast_sel!='Min':
                #    self.contrast_sel='Min'
                #    for n in range(len(op_sel)):
                #        self.RSM_test_multi(op_sel[n][0],op_sel[n][1],range_sel)
                #    results=pool_map(self.ncore, self.opti_combi_full,iterable(range_sel),estimator,colmode,op_sel,SNR)
                #    self.opti_combi_res=np.sum(results)
                #else:    
                #     self.opti_combi_res=opti_res
                print('Greedy selection done!') 
             
        elif self.opti_mode=='annular':
            
            range_sel=range(self.minradius+self.asize[0]//2,self.maxradius+1-self.asize[0]//2,self.asize[0])

            results=pool_map(self.ncore, self.opti_combi_annular,iterable(range_sel),estimator,colmode,SNR)
             
            it=0
            for result in results:

            
                for l in range(self.asize[0]):
                    self.opti_sel[(self.minradius+it*self.asize[0]):(self.minradius+(it+1)*self.asize[0])]=self.asize[0]*[result]
          
                it+=1
                
        # Computation of the radial thresholds

        if threshold==True and SNR==False: 
           # if sum(self.threshold[(self.max_r+1):(self.maxradius+1)])==0:
            #    range_sel= range(self.minradius,self.max_r+1)
            #else:

            self.threshold_esti(estimator=estimator,colmode=colmode,Full=False)            
                
        self.ncore=ncore_temp
        if self.inv_ang==True:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
                

    def threshold_esti(self,estimator='Forward',colmode='median',Full=False): 
      
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.maxradius=self.max_r
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.maxradius=self.max_r  
        
        if self.opti_sel==None:
            self.opti_sel=list([[[0,0]]]*(self.maxradius+1))  
            
        if Full==True:
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1
            self.opti_sel=list([mod_sel]*(self.maxradius+1))
        
        range_sel= range(self.minradius,self.maxradius+1)
        self.opti=False
        
        if self.inv_ang==False:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
        self.opti_map(estimator=estimator,colmode=colmode,threshold=False,Full=False)
        
        self.final_map_inv=np.copy(self.final_map)
        
                
        for k in range_sel:
            indicesy,indicesx=get_time_series(self.cube[0],k)
            self.threshold[k]=np.max(self.final_map[indicesy,indicesx])
        
        if self.opti_mode=='full-frame':
            self.threshold=poly_fit(self.threshold,range_sel,3)
            
        if self.inv_ang==False:
            for n in range(len(self.cube)):
                self.pa[n]=-self.pa[n]
                
        print('Threshold determination done!')
        
        
    def opti_map(self,estimator='Forward',colmode='median',ns=1,threshold=True,Full=False,SNR=False,verbose=True): 
        
        
        """
        Function computing the final detection map using the optimal set of parameters for the
        PSF-subtraction techniques (and for the RSM algorithm in the case of auto-RSM) and the
        optimal set of cubes of likelihoods/ cubes of residuals for respectively the auto-RSM 
        and auto-S/N frameworks.
        
        Parameters
        ----------
        
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        ns: float , optional
            Number of regime switches. Default is one regime switch per annulus but 
            smaller values may be used to reduce the impact of noise or disk structures
            on the final RSM probablity map.
        threshold: bool, optional
            When True the radial treshold is computed during the RSM_combination is applied on the
            final detection map with the original parallactic angles. Only used when relying on the auto-RSM
            framework. Default is True.
        Full: bool,optional
            If True, the entire set of ADI-sequences and PSF-subtraction techniques are used to 
            generate the final detection map. If performed after RSM_combination, the obtained optimal set
            is repkaced by the entire set of cubes. Please make ure you have saved the optimal set
            via the save_parameter function. Default is 'False'.
        SNR: bool,optional
            If True, the auto-S/N framework is used, resulting in an optimizated final S/N map when using 
            subsequently the opti_map. If False the auto-RSM framework is used, providing an optimized
            probability map when using subsequently the opti_map.
        """
        
        if (any('FM KLIP'in mymodel for mymodel in self.model) or any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==1:
            self.maxradius=self.max_r
        elif(any('FM KLIP'in mymodel for mymodel in self.model) and any('FM LOCI'in mydistri for mydistri in self.model)) and len(self.model)==2:
            self.maxradius=self.max_r  
            
        self.final_map=np.zeros((self.cube[0].shape[-2],self.cube[0].shape[-1]))
        
        if self.opti_sel==None:
            self.opti_sel=list([[[0,0]]]*(self.maxradius+1))
          
        if type(self.threshold)!=np.ndarray:
            self.threshold=np.zeros((self.maxradius+1))   
            
        if Full==True:
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1
            self.opti_sel=list([mod_sel]*(self.maxradius+1))

        # Computation of the final detection map for the auto-RSM or auto-S/N for the annular case  
          
        if self.opti_mode=='annular':
            
            
            if SNR==True:
                self.final_map=self.SNR_esti_annular(sel_cube=self.opti_sel,verbose=verbose)
            else:
                self.opti=False
                it=0
                for k in range(self.minradius+self.asize[0]//2,self.maxradius+1-self.asize[0]//2,self.asize[0]):
                    for m in range(len(self.opti_sel[k])):
      
                        residuals_cube_=np.zeros_like(rot_scale('ini',self.cube[self.opti_sel[k][m][0]],None,self.pa[self.opti_sel[k][m][0]],self.scale_list[self.opti_sel[k][m][0]],self.imlib, self.interpolation)[0])
                        for l in range(k-self.asize[self.opti_sel[k][m][1]],k+self.asize[self.opti_sel[k][m][1]]+1,self.asize[self.opti_sel[k][m][1]]):
                            indices = get_annulus_segments(residuals_cube_[0], l-int(self.asize[self.opti_sel[k][m][1]]/2),int(self.asize[self.opti_sel[k][m][1]]),1)
    
                            residuals_cube_temp=self.model_esti(self.opti_sel[k][m][1],self.opti_sel[k][m][0],l,self.cube[self.opti_sel[k][m][0]])[1]
                            residuals_cube_[:,indices[0][0],indices[0][1]]=residuals_cube_temp[:,indices[0][0],indices[0][1]]
    
    
                        range_sel=range((self.minradius+it*self.asize[self.opti_sel[k][m][1]]),(self.minradius+(it+1)*self.asize[self.opti_sel[k][m][1]]))
                        
                        like_temp=np.zeros(((residuals_cube_.shape[0]+1),self.cube[self.opti_sel[k][m][0]].shape[-2],self.cube[self.opti_sel[k][m][0]].shape[-1],len(self.interval[self.opti_sel[k][m][1]][k,self.opti_sel[k][m][0]]),2,self.crop_range[self.opti_sel[k][m][1]]))    
                    
                    
                        #Sometimes the parallel computation get stuck, to avoid that we impose a maximum computation
                        #time of 0.01 x number of frame x number of pixels in the last annulus of range_sel x 2 x pi (0.01 second
                        #per treated pixel)
                        time_out=0.01*residuals_cube_.shape[0]*2*range_sel[-1]*np.pi
                        results=[]    
                        pool=Pool(processes=self.ncore)           
                        for e in range_sel:
                            results.append(pool.apply_async(self.likelihood,args=(e,self.opti_sel[k][m][0],self.opti_sel[k][m][1],residuals_cube_,None,True)))
                        [result.wait(timeout=time_out) for result in results]
                        
                        it1=k-self.asize[self.opti_sel[k][m][1]]
                        for result in results:
                            try:
                                res=result.get(timeout=1)
                                indicesy,indicesx=get_time_series(self.cube[0],res[0])
                                if self.model[self.opti_sel[k][m][1]]=='FM LOCI' or self.model[self.opti_sel[k][m][1]]=='FM KLIP':
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1]
                                    self.psf_fm[self.opti_sel[k][m][0]][self.opti_sel[k][m][1]][res[0]]=res[2]
                                else:
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1] 
                            except mp.TimeoutError:
                                pool.terminate()
                                pool.join()
                                res=self.likelihood(it1,self.opti_sel[k][m][0],self.opti_sel[k][m][1],residuals_cube_,None,True)
                                indicesy,indicesx=get_time_series(self.cube[0],res[0])
                                if self.model[self.opti_sel[k][m][1]]=='FM LOCI' or self.model[self.opti_sel[k][m][1]]=='FM KLIP':
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1]
                                    self.psf_fm[self.opti_sel[k][m][0]][self.opti_sel[k][m][1]][res[0]]=res[2]
                                else:
                                    like_temp[:,indicesy,indicesx,:,:,:]=res[1]
                            it1+=1 
                    
                                
                        like=[]
                    
                        for n in range(self.crop_range[self.opti_sel[k][m][1]]):
                            like.append(like_temp[0:residuals_cube_.shape[0],:,:,:,:,n])
                        
                        self.like_fin[self.opti_sel[k][m][0]][self.opti_sel[k][m][1]]=like   
                    
                    for l in range_sel:
                        indicesy,indicesx=get_time_series(self.cube[0],l)
                        self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=l,sel_cube=self.opti_sel[l])  
                
                        if threshold==True:
                            self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]-self.threshold[l]
                        else:
                            self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]
                    it+=1
                    
        # Computation of the final detection map for the auto-RSM or auto-S/N for the full-frame case  

        elif self.opti_mode=='full-frame':
            
            if SNR==True:
                self.final_map=self.SNR_esti_full(sel_cube=self.opti_sel[0],verbose=verbose)
            else:
            
                range_sel= range(self.minradius,self.maxradius+1)
                self.opti=False
                mod_sel=self.opti_sel[0].copy()
                self.lik_esti(sel_cube=mod_sel,verbose=verbose)
                
                
                if 'FM KLIP' not in self.model and 'FM LOCI' not in self.model:
                    self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=None,sel_cube=mod_sel) 
                    for k in range_sel:
                        indicesy,indicesx=get_time_series(self.cube[0],k) 
                        if threshold==True:
                            self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]-self.threshold[k]
                        else:
                            self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]       
                else:
                    for k in range_sel:
                        
                        if k>=self.max_r:
                            
                            try:
                                del mod_sel[list(np.asarray(mod_sel)[:,1]).index(self.model.index('FM KLIP'))]
                            except (ValueError,IndexError):
                                pass
                            try:
                                del mod_sel[list(np.asarray(mod_sel)[:,1]).index(self.model.index('FM LOCI'))] 
                            except (ValueError,IndexError):
                                pass
        
                            if len(mod_sel)==0:
                                break
                            self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=mod_sel) 
                                
                        else:
                            self.probmap_esti(ns=ns, estimator=estimator,colmode=colmode,ann_center=k,sel_cube=mod_sel) 
        
        
                        indicesy,indicesx=get_time_series(self.cube[0],k)                    
                        if threshold==True:
                            self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]-self.threshold[k]
                        else:
                            self.final_map[indicesy,indicesx]=self.probmap[indicesy,indicesx]
                    
        if verbose:
            print('Final RSM map computation done!')

        
        
        
    def contrast_multi_snr(self,ann_center,mod_sel=[[0,0]]): 
        
        """
        Function computing the performance index of the S/N detection map (the contrast) when using of multiple cubes
        of residuals to generate the final S/N map (auto-S/N framweork). The final S/N map is obtained 
        by averaging the S/N maps of the selected cubes of residuals. The performance index is computed for a
        radial distane ann_center using the set of cubes of residuals mod_sel. [[i1,j1],[i2,j2],...] with i1
        the first considered PSF-subtraction technique and j1 the first considered ADI sequence, i2 the
        second considered PSF-subtraction technique, etc. This function is used by the optimal subset of
        likelihood cubes selection function (RSM_combination).
        """
        
        ceny, cenx = frame_center(self.cube[0])
        
        init_angle = 0
        ang_step=360/((np.deg2rad(360)*ann_center)/self.fwhm)
        
        tempx=[]
        tempy=[]
        
        for l in range(int(((np.deg2rad(360)*ann_center)/self.fwhm))):
            newx = ann_center * np.cos(np.deg2rad(ang_step * l+init_angle))
            newy = ann_center * np.sin(np.deg2rad(ang_step * l+init_angle))
            tempx.append(newx)
            tempy.append(newy)
        
        tempx=np.array(tempx)
        
        tempy = np.array(tempy) +int(ceny)
        tempx = np.array(tempx) + int(cenx)
      
        apertures = photutils.CircularAperture(np.array((tempx, tempy)).T, round(self.fwhm/2))
        
        flux=np.zeros(len(mod_sel))
        injected_flux=np.zeros((len(mod_sel),min(len(apertures)//2,8)))
        recovered_flux=np.zeros((len(mod_sel),min(len(apertures)//2,8)))

        for k in range(len(mod_sel)):
            cuben=mod_sel[k][0]
            modn=mod_sel[k][1]
        
            frame_nofc=self.model_esti(modn,cuben,ann_center,self.cube[cuben])[0]
                
            apertures = photutils.CircularAperture(np.array((tempx, tempy)).T, round(self.fwhm/2))

            fluxes = photutils.aperture_photometry(frame_nofc, apertures)
            fluxes = np.array(fluxes['aperture_sum'])

            n_aper = len(fluxes)
            ss_corr = np.sqrt(1 + 1/(n_aper-1))
            sigma_corr = stats.t.ppf(stats.norm.cdf(5), n_aper)*ss_corr

            noise = np.std(fluxes)
        
            flux[k] = sigma_corr*noise

            psf_template = normalize_psf(self.psf[cuben], fwhm=self.fwhm, verbose=False,size=self.psf[cuben].shape[1])
            
            fc_map = np.ones((self.cube[cuben].shape[-2],self.cube[cuben].shape[-1])) * 1e-6
            fcy=[]
            fcx=[]
            cube_fc =self.cube[cuben]
            ang_fc=range(int(init_angle),int(360+init_angle),int(360//min((len(fluxes)//2),8)))
            for i in range(len(ang_fc)):
                cube_fc = cube_inject_companions(cube_fc, psf_template,
                                     self.pa[cuben], flux[k], self.pxscale,
                                     rad_dists=ann_center,
                                     theta=ang_fc[i],
                                     verbose=False)
                y = int(ceny) + ann_center * np.sin(np.deg2rad(
                                                       ang_fc[i]))
                x = int(cenx) + ann_center * np.cos(np.deg2rad(
                                                       ang_fc[i]))
                if self.cube[cuben].ndim==4:
                    fc_map = frame_inject_companion(fc_map, np.mean(psf_template,axis=0), y, x,
                                            flux[k])
                else:
                    fc_map = frame_inject_companion(fc_map, psf_template, y, x,
                                            flux[k])
                fcy.append(y)
                fcx.append(x)
    
            
            frame_fc=self.model_esti(modn,cuben,ann_center,cube_fc)[0]
            
            
            for j in range(len(ang_fc)):
                apertures = photutils.CircularAperture(np.array(([fcx[j],fcy[j]])), round(self.fwhm/2))
                injected_flux[k,j] = photutils.aperture_photometry(fc_map, apertures)['aperture_sum']
                recovered_flux[k,j] = photutils.aperture_photometry((frame_fc - frame_nofc), apertures)['aperture_sum']
                
        contrast=[]        
        for j in range(len(ang_fc)):
            
            recovered_flux_conso=0
            injected_flux_conso=0
            if len(mod_sel)==1:
                recovered_flux_conso=recovered_flux[0,j]
                injected_flux_conso=injected_flux[0,j]
            else:
                for k in range(len(mod_sel)):
                    temp_list=np.array(range(len(mod_sel)))
                    temp_list=np.delete(temp_list,k)
                    recovered_flux_conso+=recovered_flux[k,j]*np.prod(flux[temp_list])
                    injected_flux_conso+=injected_flux[k,j]*np.prod(flux[temp_list])
            
            throughput = float(recovered_flux_conso / injected_flux_conso)
            
            if np.prod(flux)/throughput>0:
                contrast.append(np.mean(flux) / throughput)
                
        if len(contrast)!=0:
            contrast_mean=np.mean(contrast)
        else:
            contrast_mean=-1
            
        return np.where(contrast_mean<0,0,1/contrast_mean)
    
    
    
    
    def SNR_esti_full(self, sel_cube=[[0,0]],verbose=True):
        
        """
        Function computing the final S/N detection map, in the case of the full-frame optimization mode, 
        after optimization of the PSF-subtraction techniques and the optimal selection of the residual cubes.
        The final S/N map is obtained by averaging the S/N maps of the selected cubes of residuals provided
        by  mod_sel, [[i1,j1],[i2,j2],...] with i1 the first considered PSF-subtraction technique and j1 the
        first considered ADI sequence, i2 the second considered PSF-subtraction technique, etc.
        This function is used by the final detection map computation function (opti_map).
        """
        
        snr_temp=[]
         
        for k in range(len(sel_cube)):
            
                j=sel_cube[k][0]
                i=sel_cube[k][1]
                #Computation of the SNR maps

                if self.model[i]=='APCA':
                    print("Annular PCA estimation") 
                    residuals_cube_, frame_fin = annular_pca_adisdi(self.cube[j], self.pa[j], self.scale_list[j], fwhm=self.fwhm, ncomp=self.ncomp[i][0,j], asize=self.asize[i], 
                              delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j], svd_mode='lapack', n_segments=int(self.nsegments[i][0,j]), nproc=self.ncore,full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='NMF':
                    print("NMF estimation") 
                    residuals_cube_, frame_fin = nmf_adisdi(self.cube[j], self.pa[j], self.scale_list[j], ncomp=self.ncomp[i][0,j], max_iter=100, random_state=0, mask_center_px=None,full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='LLSG':
                    print("LLSGestimation") 

                    residuals_cube_, frame_fin = llsg_adisdi(self.cube[j], self.pa[j],self.scale_list[j], self.fwhm, rank=self.rank[i][0,j],asize=self.asize[i], thresh=1,n_segments=int(self.nsegments[i][0,j]), max_iter=40, random_seed=10, nproc=self.ncore,full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='LOCI':
                    print("LOCI estimation") 
                    residuals_cube_,frame_fin=loci_adisdi(self.cube[j], self.pa[j],self.scale_list[j], fwhm=self.fwhm,asize=self.asize[i], n_segments=int(self.nsegments[i][0,j]),tol=self.tolerance[i][0,j], nproc=self.ncore, optim_scale_fact=2,delta_rot=self.delta_rot[i][0,j],delta_sep=self.delta_sep[i][0,j],verbose=False,full_output=True)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
            
                elif self.model[i]=='KLIP':
                    print("KLIP estimation") 
                    cube_out, residuals_cube_, frame_fin = KLIP(self.cube[j], self.pa[j], ncomp=self.ncomp[i][0,j], fwhm=self.fwhm, asize=self.asize[i], 
                              delta_rot=self.delta_rot[i][0,j],full_output=True,verbose=False)
                    snr_temp.append(vip.metrics.snrmap(frame_fin, fwhm=self.fwhm, approximated=False, plot=False,nproc=self.ncore, verbose=False))
                
        return np.array(snr_temp).mean(axis=0)
           
        

    def SNR_esti_annular(self,sel_cube=[[0,0]],verbose=True):
        
        """
        Function computing the final S/N detection map, in the case of the annular optimization mode, 
        after optimization of the PSF-subtraction techniques and the optimal selection of the residual cubes.
        The final S/N map is obtained by averaging the S/N maps of the selected cubes of residuals provided
        by  mod_sel, [[i1,j1],[i2,j2],...] with i1 the first considered PSF-subtraction technique and j1 the
        first considered ADI sequence, i2 the second considered PSF-subtraction technique, etc.
        This function is used by the final detection map computation function (opti_map).
        """
              
        self.opti_sel=sel_cube
        snrmap_array = np.zeros((self.cube[0].shape[-2],self.cube[0].shape[-1]))
        
        for k in range(self.minradius+self.asize[0]//2,self.maxradius+1-self.asize[0]//2,self.asize[0]):
                snr_temp=[]
                for m in range(len(self.opti_sel[k])):
                    snrmap_array_temp = np.zeros((self.cube[0].shape[-2],self.cube[0].shape[-1]))
                    
                    residuals_cube_=self.model_esti(self.opti_sel[k][m][1],self.opti_sel[k][m][0],k,self.cube[self.opti_sel[k][m][0]])[1]

                    mask = get_annulus_segments(residuals_cube_[0], k-int(self.asize[self.opti_sel[k][m][1]]/2),int(self.asize[self.opti_sel[k][m][1]]), mode="mask")[0]
                    mask = np.ma.make_mask(mask)
                    yy, xx = np.where(mask)
                    coords = zip(xx, yy)   
                    res = pool_map(self.ncore, vip.metrics.snr, residuals_cube_.mean(axis=0), iterable(coords), self.fwhm, True,None, False)
                    res = np.array(res)
                    yy = res[:, 0]
                    xx = res[:, 1]
                    snr_value = res[:, -1]
                    snrmap_array_temp[yy.astype('int'), xx.astype('int')] = snr_value
                    snr_temp.append(snrmap_array_temp)
                snrmap_array[yy.astype('int'), xx.astype('int')]=np.asarray(snr_temp).mean(axis=0)[yy.astype('int'), xx.astype('int')]
        
        return snrmap_array
    


    def estimate_probmax_fc(self,a,b,level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor=[1],estimator='Forward', colmode='median'):
        
        ncore_temp=np.copy(self.ncore)
        self.ncore=1
        self.minradius=a-1
        self.maxradius=a+1
        
        for j in range(len(cube)):
    
            cubefc = cube_inject_companions(cube[j], self.psf[j], self.pa[j], flevel=level*scaling_factor[j], plsc=self.pxscale, 
                                    rad_dists=a, theta=b/n_fc*360, n_branches=1,verbose=False)
            self.cube[j]=cubefc
        
        self.opti_map(estimator=estimator,colmode=colmode,threshold=threshold,verbose=False) 
                
        rsm_fin = np.where(abs(np.nan_to_num(self.final_map))>0.000001,0,probmap)+np.nan_to_num(self.final_map)
    
        rsm_fin=rsm_fin+abs(rsm_fin.min())
    
        n,y,x=cubefc.shape
        twopi=2*np.pi
        sigposy=int(y/2 + np.sin(b/n_fc*twopi)*a)
        sigposx=int(x/2+ np.cos(b/n_fc*twopi)*a)
    
        indc = disk((sigposy, sigposx),3)
        max_target=np.nan_to_num(rsm_fin[indc[0],indc[1]]).max()
        rsm_fin[indc[0],indc[1]]=0
        max_map=np.nan_to_num(rsm_fin).max()
        
        print("Distance: "+"{}".format(a)+" Position: "+"{}".format(b)+" Level: "+"{}".format(level)+" Probability difference: "+"{}".format(max_target-max_map))
        self.ncore=ncore_temp
        
        return max_target-max_map,b
    

    def contrast_curve(self,an_dist,ini_contrast,probmap,inv_ang=False,threshold=False,psf_oa=None,estimator='Forward',colmode='median',n_fc=20,completeness=0.9):
        
        """
        Function allowing the computation of contrast curves using the RSM framework
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        an_dist: list or ndarray
            List of angular separations for which a contrast has to be estimated.
        ini_contrast: list or ndarray
            Initial contrast for the range of angular separations included in an_dist.
            The number of initial contrasts shoul be equivalent to the number of angular
            separations.
        probmap: numpy 2d ndarray
            Detection map provided by the RSM algorithm via opti_map or probmap_esti.
        inv_ang: bool, optional
            If True, the sign of the parallactic angles of all ADI sequence is flipped for
            the computation of the contrast. Default is False.
        threshold: bool, optional 
            If an angular separation based threshold has been used when generating the
            detection map, the same set of thresholds should be considered as well during
            the contrast computation. Default is False.
        psf_oa: bool, optional, optional
            Saturated PSF of the host star used to compute the scaling factor allowing the 
            conversion between contrasts and fluxes for the injection of fake companions
            during the computation of the contrast. If no Saturated PSF is provided, the 
            ini_contrast should be provided in terms of flux instead of contrast. 
            Default is None.
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        n_fc: int, optional
            Number of azimuths considered for the computation of the True positive rate/completeness,
            (number of fake companions injected separately). The number of azimuths is defined 
            such that the selected completeness is reachable (e.g. 95% of completeness requires at least
            20 fake companion injections). Default 20.
        completeness: float, optional
            The completeness level to be achieved when computing the contrasts, i.e. the True positive
            rate reached at the threshold associated to the first false positive (the first false 
            positive is defined as the brightest speckle present in the entire detection map). Default 95.
            
        Returns
        ----------
        1D numpy ndarray containg the contrasts for the considered angular distance at the selected
        completeness level.            
        """
        
        if type(probmap) is not np.ndarray:
            if type(self.final_map) is not np.ndarray:
                raise ValueError("A detection map should be provided")
            else:
                probmap=np.copy(self.final_map)
        if threshold==True and type(self.threshold) is not np.ndarray:
            raise ValueError("If threshold is true a threshold should be provided via self.threshold")
            
        if (100*completeness)%(100/n_fc)>0:
            n_fc=int(100/math.gcd(int(100*completeness), 100))
        
        minradius_temp=np.copy(self.minradius)
        maxradius_temp=np.copy(self.maxradius)
        cube=self.cube.copy()
        scaling_factor=[]
        
        for i in range(len(self.psf)):
       
            if psf_oa is not None:
                indices = get_annulus_segments(self.psf[i],0,round((self.fwhm)/2),1) 
                indices_oa = get_annulus_segments(psf_oa[i],0,round((self.fwhm)/2),1) 
                scaling_factor.append(psf_oa[i][indices_oa[0][0], indices_oa[0][1]].sum()/self.psf[i][indices[0][0], indices[0][1]].sum())
                
            else:
                scaling_factor.append(1)
        
        contrast_curve=np.zeros((len(an_dist)))
        ini_contrast_in=np.copy(ini_contrast)
        
        for k in range(0,len(an_dist)):
        
            a=an_dist[k]
            level=ini_contrast[k]
            pos_detect=[]
            
            detect_bound=[None,None]
            level_bound=[None,None]
            
            while len(pos_detect)==0:
                pos_detect=[] 
                pos_non_detect=[]
                val_detect=[] 
                val_non_detect=[] 
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(range(0,n_fc)),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                for res_i in res:
                    
                   if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       val_detect.append(res_i[0])
                   else:
                       pos_non_detect.append(res_i[1])
                       val_non_detect.append(res_i[0])
                        
                        
                if len(pos_detect)==0:
                    level=level*1.5
                    
            if len(pos_detect)>round(completeness*n_fc):
                detect_bound[1]=len(pos_detect)
                level_bound[1]=level
            elif len(pos_detect)<round(completeness*n_fc):
                detect_bound[0]=len(pos_detect)
                level_bound[0]=level            
                pos_non_detect_temp=pos_non_detect.copy()
                val_non_detect_temp=val_non_detect.copy()
                pos_detect_temp=pos_detect.copy()
                val_detect_temp=val_detect.copy()
            
            perc=len(pos_detect)/n_fc  
            print("Initialization done: current precentage "+"{}".format(perc))    
                
            while (detect_bound[0]==None or detect_bound[1]==None) and len(pos_detect)!=round(completeness*n_fc):
                
                if detect_bound[0]==None:
                    
                    level=level*0.5
                    pos_detect=[] 
                    pos_non_detect=[]
                    val_detect=[] 
                    val_non_detect=[] 
                    
                    res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(range(0,n_fc)),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                    for res_i in res:
                        
                       if res_i[0]>0:
                           pos_detect.append(res_i[1])
                           val_detect.append(res_i[0])
                       else:
                           pos_non_detect.append(res_i[1])
                           val_non_detect.append(res_i[0])
                        
    
                    if len(pos_detect)>round(completeness*n_fc) and level_bound[1]>level:
                        detect_bound[1]=len(pos_detect)
                        level_bound[1]=level
                    elif len(pos_detect)<round(completeness*n_fc):
                        detect_bound[0]=len(pos_detect)
                        level_bound[0]=level 
                        pos_non_detect_temp=pos_non_detect.copy()
                        val_non_detect_temp=val_non_detect.copy()
                        pos_detect_temp=pos_detect.copy()
                        val_detect_temp=val_detect.copy()
                        
                elif detect_bound[1]==None:
                    
                    level=level*1.5
                    
                    res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                    it=len(pos_non_detect)-1        
                    for res_i in res:
                        
                        if res_i[0]>0:
                           pos_detect.append(res_i[1])
                           val_detect.append(res_i[0])
                           del pos_non_detect[it]
                           del val_non_detect[it]
                        it-=1
                           
                    if len(pos_detect)>round(completeness*n_fc):
                        detect_bound[1]=len(pos_detect)
                        level_bound[1]=level
                    elif len(pos_detect)<round(completeness*n_fc)  and level_bound[0]<level:
                        detect_bound[0]=len(pos_detect)
                        level_bound[0]=level
                        pos_non_detect_temp=pos_non_detect.copy()
                        val_non_detect_temp=val_non_detect.copy()
                        pos_detect_temp=pos_detect.copy()
                        val_detect_temp=val_detect.copy()
                        
            if len(pos_detect)!=round(completeness*n_fc):
                print("Boundaries defined "+"{}".format(detect_bound[0]/n_fc)+" {}".format(detect_bound[1]/n_fc)) 
            
                pos_non_detect=pos_non_detect_temp.copy()
                val_non_detect=val_non_detect_temp.copy()
                pos_detect=pos_detect_temp.copy()
                val_detect=val_detect_temp.copy()
            
            while len(pos_detect)!=round(completeness*n_fc):
                                
                level=level_bound[0]+(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])*(completeness*n_fc-detect_bound[0])
            
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
            
                it=len(pos_non_detect)-1      
                for res_i in res:
                    
                    if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       val_detect.append(res_i[0])
                       del pos_non_detect[it]
                       del val_non_detect[it]
                    it-=1
                       
                if len(pos_detect)>round(completeness*n_fc):
                    detect_bound[1]=len(pos_detect)
                    level_bound[1]=level
                elif len(pos_detect)<round(completeness*n_fc)  and level_bound[0]<level:
                    detect_bound[0]=len(pos_detect)
                    level_bound[0]=level
                    pos_non_detect_temp=pos_non_detect.copy()
                    val_non_detect_temp=val_non_detect.copy()
                    pos_detect_temp=pos_detect.copy()
                    val_detect_temp=val_detect.copy()               
                
                if len(pos_detect)!=round(completeness*n_fc):
                    
                    pos_non_detect=pos_non_detect_temp.copy()
                    val_non_detect=val_non_detect_temp.copy()
                    pos_detect=pos_detect_temp.copy()
                    val_detect=val_detect_temp.copy()
    
                    print("Current boundaries "+"{}".format(detect_bound[0]/n_fc)+" {}".format(detect_bound[1]/n_fc)) 
    
          
            print("Distance: "+"{}".format(a)+" Final contrast "+"{}".format(level))  
            contrast_curve[k]=level           
            ini_contrast=ini_contrast_in*(contrast_curve[0:(k+1)]/ini_contrast_in[0:(k+1)]).sum()/(k+1)
                
        self.minradius=minradius_temp
        self.maxradius=maxradius_temp
        self.cube=cube
        
        return contrast_curve                              
                                                   
    def contrast_matrix(self,an_dist,ini_contrast,probmap=None,inv_ang=False,threshold=False,psf_oa=None,estimator='Forward',colmode='median',n_fc=20):
    
        """
        Function allowing the computation of three dimensional contrast curves using the RSM framework,
        with contrasts computed for multiple completeness level, allowing the reconstruction of the
        contrast/completeness distribution for every considered angular separations.
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        an_dist: list or ndarray
            List of angular separations for which a contrast has to be estimated.
        ini_contrast: list or ndarray
            Initial contrast for the range of angular separations included in an_dist.
            The number of initial contrasts shoul be equivalent to the number of angular
            separations.
        probmap: numpy 2d ndarray
            Detection map provided by the RSM algorithm via opti_map or probmap_esti.
        inv_ang: bool, optional
            If True, the sign of the parallactic angles of all ADI sequence is flipped for
            the computation of the contrast. Default is False.
        threshold: bool, optional 
            If an angular separation based threshold has been used when generating the
            detection map, the same set of thresholds should be considered as well during
            the contrast computation. Default is False.
        psf_oa: bool, optional, optional
            Saturated PSF of the host star used to compute the scaling factor allowing the 
            conversion between contrasts and fluxes for the injection of fake companions
            during the computation of the contrast. If no Saturated PSF is provided, the 
            ini_contrast should be provided in terms of flux instead of contrast. 
            Default is None.
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
        n_fc: int, optional
            Number of azimuths considered for the computation of the True positive rate/completeness,
            (number of fake companions injected separately). The range of achievable completenness 
            depends on the number of considered azimuths (the minimum completeness is defined as
            1/n_fc an the maximum is 1-1/n_fc). Default 20.
        
        Returns
        ----------
        2D numpy ndarray providing the contrast with the first axis associated to the angular distance
        and the second axis associated to the completeness level.
        """
        
        if type(probmap) is not np.ndarray:
            if type(self.final_map) is not np.ndarray:
                raise ValueError("A detection map should be provided")
            else:
                probmap=np.copy(self.final_map)
        if threshold==True and type(self.threshold) is not np.ndarray:
            raise ValueError("If threshold is true a threshold should be provided via self.threshold")
        
        minradius_temp=np.copy(self.minradius)
        maxradius_temp=np.copy(self.maxradius)
        cube=self.cube.copy()
        scaling_factor=[]
        
        for i in range(len(self.psf)):
       
            if psf_oa is not None:
                indices = get_annulus_segments(self.psf[i],0,round((self.fwhm)/2),1) 
                indices_oa = get_annulus_segments(psf_oa[i],0,round((self.fwhm)/2),1) 
                scaling_factor.append(psf_oa[i][indices_oa[0][0], indices_oa[0][1]].sum()/self.psf[i][indices[0][0], indices[0][1]].sum())
                
            else:
                scaling_factor.append(1)
        
        contrast_matrix=np.zeros((len(an_dist),n_fc+1))
        detect_pos_matrix=[[]]*(n_fc+1)
        ini_contrast_in=np.copy(ini_contrast)
        
        for k in range(0,len(an_dist)):
        
            a=an_dist[k]
            level=ini_contrast[k]
            pos_detect=[] 
            detect_bound=[None,None]
            level_bound=[None,None]
            
            while len(pos_detect)==0:
                pos_detect=[] 
                pos_non_detect=[]
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(range(0,n_fc)),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                
                for res_i in res:
                    
                   if res_i[0]>0:
                       pos_detect.append(res_i[1])
                   else:
                       pos_non_detect.append(res_i[1])
                        
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
                if len(pos_detect)==0:
                    level=level*1.5
    
            perc=len(pos_detect)/n_fc  
            
            print("Initialization done: current data point "+"{}".format(perc)) 
           
            while contrast_matrix[k,0]==0:
                
                level=level*0.75
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
            
                it=len(pos_detect)-1        
                for res_i in res:
                    
                    if res_i[0]<0:
                       pos_non_detect.append(res_i[1])
                       del pos_detect[it]
                    it-=1
    
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
    
            print("Lower boundary found") 
            
            level=contrast_matrix[k,np.where(contrast_matrix[k,:]>0)[0][-1]]
            
            pos_detect=[] 
            pos_non_detect=list(np.arange(0,11))
            
            while contrast_matrix[k,n_fc]==0:
                
                level=level*1.25
                
                res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
            
                it=len(pos_non_detect)-1        
                for res_i in res:
                    
                    if res_i[0]>0:
                       pos_detect.append(res_i[1])
                       del pos_non_detect[it]
                    it-=1
    
                contrast_matrix[k,len(pos_detect)]=level
                detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
    
            print("Upper boundary found") 
    
            missing=np.where(contrast_matrix[k,:]==0)[0]
            computed=np.where(contrast_matrix[k,:]>0)[0]
            while len(missing)>0:
                
                detect_bound[0]= computed[np.argmax((computed-missing[0])[computed<missing[0]])]
                level_bound[0]=contrast_matrix[k,detect_bound[0]]
                detect_bound[1]= -np.sort(-computed)[np.argmax(np.sort((missing[0]-computed))[np.sort((missing[0]-computed))<0])]
                level_bound[1]=contrast_matrix[k,detect_bound[1]]
                it=0    
                while len(pos_detect)!=missing[0]:
                    
                    if np.argmin([len(detect_pos_matrix[detect_bound[1]][0]),len(detect_pos_matrix[detect_bound[0]][1])])==0:
                        

                        pos_detect=list(np.sort(detect_pos_matrix[detect_bound[1]][0]))
                        pos_non_detect=list(np.sort(detect_pos_matrix[detect_bound[1]][1]))
                        
                        level=level_bound[1]+(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])*(missing[0]-detect_bound[1])
                        
                        res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                    
                        it=len(pos_detect)-1      
                        for res_i in res:
                            
                            if res_i[0]<0:
                               pos_non_detect.append(res_i[1])
                               del pos_detect[it]
                            it-=1   
    
                    else:
                        
                        pos_detect=list(np.sort(detect_pos_matrix[detect_bound[0]][0]))
                        pos_non_detect=list(np.sort(detect_pos_matrix[detect_bound[0]][1]))
                                   
                        level=level_bound[0]+(level_bound[1]-level_bound[0])/(detect_bound[1]-detect_bound[0])*(missing[0]-detect_bound[0])
                        
                        res=pool_map(self.ncore, self.estimate_probmax_fc,a,iterable(-np.sort(-np.array(pos_non_detect))),level,cube,n_fc,inv_ang,threshold,probmap,scaling_factor,estimator=estimator, colmode=colmode)
                    
                        it=len(pos_non_detect)-1      
                        for res_i in res:
                            
                            if res_i[0]>0:
                               pos_detect.append(res_i[1])
                               del pos_non_detect[it]
                            it-=1
                        
                    if len(pos_detect)>missing[0]:
                        detect_bound[1]=len(pos_detect)
                        level_bound[1]=level
                    elif len(pos_detect)<missing[0]  and level_bound[0]<level:
                        detect_bound[0]=len(pos_detect)
                        level_bound[0]=level
        
                    contrast_matrix[k,len(pos_detect)]=level
                    detect_pos_matrix[len(pos_detect)]=[list(pos_detect.copy()),list(pos_non_detect.copy())]
                    
                    if len(pos_detect)==missing[0]:
    
                        print("Data point "+"{}".format(len(pos_detect)/n_fc)+" found. Still "+"{}".format(len(missing)-i-1)+" data point(s) missing") 
                    
                computed=np.where(contrast_matrix[k,:]>0)[0]
                missing=np.where(contrast_matrix[k,:]==0)[0]
                
            ini_contrast=ini_contrast_in*(contrast_matrix[k,int(n_fc/2)]/ini_contrast_in[0:(k+1)]).sum()/(k+1)
           
        self.minradius=minradius_temp
        self.maxradius=maxradius_temp
        self.cube=cube
        
        return contrast_matrix 
    

    def multi_estimate_flux_negfc(self,param,level=None,expected_pos=None,cube=None,ns=1,loss_func='value',scaling_factor=[1],opti='photo',norma_factor=[1,1]):
        
        args=(level,expected_pos,cube,ns,loss_func,scaling_factor,'interval',norma_factor)  
        res_param = pool_map(self.ncore, self.estimate_flux_negfc, iterable(np.array(param)),*args)
        return np.array(res_param)
        
    def estimate_flux_negfc(self,param,level,expected_pos,cube,ns,loss_func='value',scaling_factor=[1],opti='photo',norma_factor=[1,1]):
    
        ncore_temp=np.copy(self.ncore)
        self.ncore=1
        
        if opti=='photo':
            level=param
        elif opti=='bayesian':
            level=param[0]
            expected_pos=[param[1],param[2]]
        elif opti=='interval':
            level=param[0]*norma_factor[0]
            expected_pos=[param[1],param[2]]
    
        ceny, cenx = frame_center(cube[0])
    
        an_center=np.sqrt((expected_pos[0]-ceny)**2+(expected_pos[1]-cenx)**2)
    
        a=int(an_center)
    
        expected_theta=np.degrees(np.arctan2(expected_pos[0]-ceny, expected_pos[1]-cenx))
    
        self.minradius=a-round(self.fwhm)//2
        self.maxraddius=a+round(self.fwhm)//2
    
        for j in range(len(cube)):
    
            cubefc = cube_inject_companions(cube[j], self.psf[j],self.pa[j], flevel=-level*scaling_factor[j], plsc=self.pxscale, 
                                    rad_dists=an_center, theta=expected_theta, n_branches=1,verbose=False)
            self.cube[j]=cubefc    

        self.opti_map(estimator='Forward-Backward',ns=ns,colmode='median',threshold=False,verbose=False) 
        #plot_frames(self.final_map)
    
        indices=get_annulus_segments(cube[0][0],a-int(self.fwhm)//2,int(self.fwhm)+1,1) 
    
        indc = disk((expected_pos[0], expected_pos[1]),2*int(self.fwhm)+1)
        indc2 = disk((expected_pos[0], expected_pos[1]),3*int(self.fwhm)+1)
        
        ind_circle=np.array([x for x in set(tuple(x) for x in np.array(indices[0]).T) & set(tuple(x) for x in np.array(indc).T)]).T
        
        ind_circle2=np.array([x for x in set(tuple(x) for x in np.array(indices[0]).T) & set(tuple(x) for x in np.array(indc2).T)]).T
    
        del_list=-np.sort(-np.array([np.intersect1d(np.where(ind_circle2[0]==ind_circle[0][x])[0],np.where(ind_circle2[1]==ind_circle[1][x])[0]) for x in range(len(ind_circle[1]))]).reshape(1,-1))[0]
      
        indicey=  list(ind_circle2[0]) 
        indicex= list(ind_circle2[1])
    
        for k in range(len(del_list)):
    
            del indicex[del_list[k]]
            del indicey[del_list[k]]        
    
        values_t=self.final_map[ind_circle[0],ind_circle[1]]
        values=self.final_map[indicey,indicex]
    
        mean_values = np.mean(values)
        #values_t=abs(values_t-mean_values)
        values=abs(values-mean_values)
        
        self.ncore=ncore_temp
        
        if loss_func=='value':
            if opti=='bayesian':
                return param,1/np.mean(values_t)
            elif opti=='interval':
                return np.mean(values_t)*norma_factor[1]
            else:
                return np.mean(values_t)
        elif loss_func=='prob':
            mean_values = np.mean(values)
            bap=np.sqrt((np.std(values)**2)/2)
            if opti=='bayesian':
                return param,1/(np.sum(np.abs(values_t-mean_values)/bap)) 
            else:               
                return np.sum(np.abs(values_t-mean_values)/bap)
    
        
    def target_charact(self,expected_pos,psf_oa=None, ns=1, loss_func='value',optimisation_model='PSO',param_optimisation={'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10,'opti_iter':15,'ini_esti':10,'random_search':100},photo_bound=[1e-5,1e-4,10],ci_esti=None,first_guess=False):
    
        
        """
        Planet characterization algorithm relying on the RSM framework and the negative
        fake companion approach. The function provides both the estimated astrometry
        and photometry of a planetary candidate as well as the associated error.
        (for more details see Dahlqvist et al. 2021)
        
        Parameters
        ----------
        expected_pos: list or ndarray
            (Y,X) position of the detected planetary candidate.
        psf_oa: bool, optional
            Saturated PSF of the host star used to compute the scaling factor allowing the 
            conversion between contrasts and fluxes. If no Saturated PSF is provided, the 
            photometry will be provided in terms of flux not contrast. 
            Default is None.
        ns: float , optional
            Number of regime switches. Default is one regime switch per annulus but 
            smaller values may be used to reduce the impact of noise or disk structures
            on the final RSM probablity map. The number of regime switches my be increase
            in the case of faint sources to ease their characterization. Default is 1.
        loss_func: str, optional
            Loss function used for the computation of the source astrometry and photometry.
            If 'value', it relies on the minimization of the average probability within a 2
            FWHM aperture centered on the expected position of the source. If 'prob', it
            considers the entire annulus for the computation of the background noise statistics
            and use a Gaussian distribution to determine the probability that the probabilities
            associated with the planetary candidates in the detection map belongs to the 
            background noise distribution. Default is 'value'.
        optimisation_model: str, optional
            Approach used for the astrometry and photometry estimation (minimisation 
            of the probabilities within a 2 FWHM aperture centered on the detected source).
            The optimization is done either via a Bayesian approach ('Bayesian') or using
            Particle Swarm optimization('PSO'). Default is PSO.
        param_optimisation: dict, optional
            dictionnary regrouping the parameters used by the Bayesian or the PSO optimization
            framework. For the Bayesian optimization we have 'opti_iter' the number of iterations,
            ,'ini_esti' number of sets of parameters for which the loss function is computed to
            initialize the Gaussian process, random_search the number of random searches for the
            selection of the next set of parameters to sample based on the maximisation of the 
            expected immprovement. For the PSO optimization, 'w' is the inertia factor, 'c1' is
            the cognitive factor and 'c2' is the social factor, 'n_particles' is the number of
            particles in the swarm (number of point tested at each iteration), 'opti_iter' the 
            number of iterations,'ini_esti' number of sets of parameters for which the loss
            function is computed to initialize the PSO. {'c1': 1, 'c2': 1, 'w':0.5,'n_particles':10
            ,'opti_iter':15,'ini_esti':10,'random_search':100}
        photo_bound: list, optional
            Photometry range considered during the estimation. The range, expressed in terms of
            contrast (or flux if psf_oa is not provided), is given by the first two number, while
            the third number gives the number of tested values within this range. Default [1e-5,1e-4,10].
        ci_esti: str, optional
            Parameters determining if a confidence interval should be computed for the photometry
            and astrometry.The erros associated with the photometry and astrometry can be estimated
            via the inversion of the hessian matrix ('hessian') or via the BFGS minimisation approach 
            ('BFGS') which allows to further improve the precision of the estimates but requires 
            more computation time. Default is None, implying no computation of confidence intervals.
        first_guess: boolean, optional
            Define if an initialisation of the algorrithm is done via a standard negfc (using the VIP
            function firstguess) before applying the PSO or Bayesian optimisation. This initialisation
            is useful when the target is very bright. It relies on PCA approach, SNR ratio maps and
            negative fake companion injection to estimate the photometry and astrometry. Default is
            False.
        
        Returns
        ----------
        List containing, the photometry, the astrometry expressed in pixel and in angular distance 
        (mas) and angular position (°), as well as the errors associated with the photometry and
        astrometry at 68% (one standard deviation) and at 95% if ci_esti is 'hessian' or 'BFGS'.
        In the first case the order is:
        [photometry,centroid position y,centroid position x, radial distance, angular position]
        and in the second case:
        [photometry,centroid position y,centroid position x, photometry standar error, position y
        standard error,position x standard error, photometry 95% error, position y 95% error,
        position x 95% error  radial distance, angular position]
        
        """ 
        
        if self.opti_sel==None:
            mod_sel=[[0,0]]*(len(self.cube)*len(self.model))
            it=0
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    mod_sel[it]=[i,j]
                    it+=1
            self.opti_sel=list([mod_sel]*(self.maxradius+1))
        
        minradius_temp=np.copy(self.minradius)
        maxradius_temp=np.copy(self.maxradius)
        cube=self.cube.copy()
        scaling_factor=[]
    
        for i in range(len(self.psf)):
    
            if psf_oa is not None:
                indices = get_annulus_segments(self.psf[i],0,round((self.fwhm)/2),1) 
                indices_oa = get_annulus_segments(psf_oa[i],0,round((self.fwhm)/2),1) 
                scaling_factor.append(psf_oa[i][indices_oa[0][0], indices_oa[0][1]].sum()/self.psf[i][indices[0][0], indices[0][1]].sum())
    
            else:
                scaling_factor.append(1)
    
        ceny, cenx = frame_center(cube[0])
    
        an_center=np.sqrt((expected_pos[0]-ceny)**2+(expected_pos[1]-cenx)**2)
    
        a=int(an_center)
    
        self.minradius=a-round(self.fwhm)
        self.maxradius=a+round(self.fwhm)
    
        self.opti_map(estimator='Forward-Backward',ns=ns,colmode='median',threshold=False,verbose=False) 
        #plot_frames(self.final_map)
        indc = disk((expected_pos[0], expected_pos[1]),2*int(self.fwhm)+1)
        centroid_y,centroid_x=np.where(self.final_map == np.amax(self.final_map[indc[0],indc[1]]))
        print("Peak value: "+"cartesian Y {}".format(centroid_y[0])+", X {}".format(centroid_x[0]))
    
        psf_subimage, suby, subx = vip.var.get_square(self.final_map, 19,
                                                  centroid_y[0],centroid_x[0], position=True)       
        y, x = np.indices(psf_subimage.shape)
    
        y=y.flatten()
        x=x.flatten()
        psf_subimage=psf_subimage.flatten()
    
        model_fit = lmfit.models.Gaussian2dModel()
        params = model_fit.guess(psf_subimage, x, y)
        result = model_fit.fit(psf_subimage, x=x, y=y, params=params)
    
        centroid_x=result.params['centerx'].value+ subx        
        centroid_y=result.params['centery'].value+ suby
        erry=result.params['centery'].stderr
        errx=result.params['centerx'].stderr
    
        print("First guess astrometry: "+"cartesian Y {}".format(centroid_y)+", X {}".format(centroid_x)+", Error Y {}".format(erry)+", Error X {}".format(errx))  
    
        expected_pos=np.copy([centroid_y,centroid_x]) 
    
        from vip_hci.negfc import firstguess
        #from vip_hci.negfc import firstguess_from_coord 
        if first_guess==True:
            ini_flux=[]
            ini_posy=[]
            ini_posx=[]
            for i in range(len(self.psf)):
                step=(np.log10(photo_bound[1])-np.log10(photo_bound[0]))/(photo_bound[2]*2)
                flux_list=np.power(np.repeat(10,photo_bound[2]*2+1),
            np.array(np.arange(np.log10(photo_bound[0]),np.log10(photo_bound[1])+step,step))[:photo_bound[2]*2+1])
                temp_res = firstguess(cube[i], self.pa[i], self.psf[i], ncomp=20, plsc=self.pxscale,
                                       planets_xy_coord=[(centroid_x,centroid_y)], fwhm=self.fwhm, 
                                      f_range=flux_list*scaling_factor[i], annulus_width=5, aperture_radius=2,
                                           imlib = 'opencv',interpolation= 'lanczos4', simplex=True, 
                                       plot=False, verbose=False)
    
                ini_flux.append(temp_res[2]/scaling_factor[i])
                posy_fc= ceny+temp_res[0] * np.sin(np.deg2rad(temp_res[1]))
                posx_fc = cenx+temp_res[0] * np.cos(np.deg2rad(temp_res[1]))
                ini_posy.append(posy_fc)
                ini_posx.append(posx_fc)    
            curr_flux=np.mean(ini_flux)
            centroid_y=np.mean(ini_posy)
            centroid_x=np.mean(ini_posx)
            lower_b_flux=10**(np.log10(curr_flux)-step)
            upper_b_flux=10**(np.log10(curr_flux)+step)
            
            print("Initial astrometry: "+"cartesian Y {}".format(centroid_y)+", X {}".format(centroid_x))
    
        if first_guess==False or abs(expected_pos[0]-centroid_y)>2 or abs(expected_pos[1]-centroid_x)>2:
            step=(np.log10(photo_bound[1])-np.log10(photo_bound[0]))/photo_bound[2]
            flux_list=np.power(np.repeat(10,photo_bound[2]+1),
        np.array(np.arange(np.log10(photo_bound[0]),np.log10(photo_bound[1])+step,step))[:photo_bound[2]+1])
            args=(0,expected_pos,cube,ns, 'value',scaling_factor,'photo')   
            res_list=pool_map(self.ncore, self.estimate_flux_negfc, iterable(flux_list.T),*args)
    
            curr_flux=flux_list[np.argmin(np.array(res_list))]
            if np.argmin(np.array(res_list))==(len(flux_list)-1):
                print('Upper bound reached, consider an extension of the upper bound.')
                upper_b_flux=10**(np.log10(photo_bound[1])+step)
            else:
                upper_b_flux=flux_list[np.argmin(np.array(res_list))+1]
            if np.argmin(np.array(res_list))==0:
                print('Lower bound reached, consider an extension of the lower bound.')
                lower_b_flux=10**(np.log10(photo_bound[0])-step)
            else:
                lower_b_flux=flux_list[np.argmin(np.array(res_list))-1]

    
        print("Inital estimated photometry {}".format(curr_flux))
        
        if abs(expected_pos[0]-centroid_y)>2 or abs(expected_pos[1]-centroid_x)>2:
            centroid_y,centroid_x=expected_pos
        else:
            expected_pos=[centroid_y,centroid_x]

    
        if optimisation_model=='Bayesian':
    
            opti_param,max_val=bayesian_optimization(self.estimate_flux_negfc,np.array([[lower_b_flux,upper_b_flux],[expected_pos[0]-1,expected_pos[0]+1],[expected_pos[1]-1,expected_pos[1]+1]]), ['float','float','float'],args=(0,expected_pos,cube,ns,loss_func,scaling_factor,'bayesian'),n_random_esti=param_optimisation['ini_esti'], n_iters=param_optimisation['opti_iter'],multi_search=self.ncore,random_search=True,n_restarts=param_optimisation['random_search'],ncore=self.ncore)
    
            expected_pos=opti_param[1:3]
            curr_flux=opti_param[0]
            print("Initial estimated astrometry {}".format(expected_pos))
            print("Initial estimated photometry {}".format(curr_flux))
            centroid_y,centroid_x=opti_param[1:3]
    
    
        elif optimisation_model=='PSO':
    
            my_topology = Star() # The Topology Class
            my_options_start={'c1': param_optimisation['c1'], 'c2': param_optimisation['c2'], 'w':param_optimisation['w']}
            my_options_end = {'c1': param_optimisation['c1'], 'c2': param_optimisation['c2'], 'w': param_optimisation['w']}
    
             # The Swarm Class
            kwargs=dict(level=curr_flux/10**(int(np.log10(curr_flux))-1),expected_pos=expected_pos,cube=cube,ns=ns,loss_func=loss_func,scaling_factor=scaling_factor,opti='interval',norma_factor=[10**(int(np.log10(curr_flux))-1),1])
    
            bounds=([lower_b_flux/10**(int(np.log10(curr_flux))-1),expected_pos[0]-1,expected_pos[1]-1],[upper_b_flux/10**(int(np.log10(curr_flux))-1),expected_pos[0]+1,expected_pos[1]+1])
            params=[]
            for i in range(len(bounds[0])):
    
                params.append(np.random.uniform(bounds[0][i], bounds[1][i], (param_optimisation['ini_esti'])))
    
            params=np.append(np.array(params).T,[[curr_flux/10**(int(np.log10(curr_flux))-1),expected_pos[0],expected_pos[1]]],0)
    
    
            ini_cost=self.multi_estimate_flux_negfc(params,**kwargs)
    
            ini_position=params[ini_cost<np.sort(ini_cost)[param_optimisation['n_particles']],:]
    
            my_swarm = P.create_swarm(n_particles=param_optimisation['n_particles'], dimensions=3, options=my_options_start, bounds=([lower_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y-1,centroid_x-1],[upper_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y+1,centroid_x+1]),init_pos=ini_position)
            my_swarm.bounds=([lower_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y-1,centroid_x-1],[upper_b_flux/10**(int(np.log10(curr_flux))-1),centroid_y+1,centroid_x+1])
            my_swarm.velocity_clamp=None
            my_swarm.vh=P.handlers.VelocityHandler(strategy="unmodified")
            my_swarm.bh=P.handlers.BoundaryHandler(strategy="periodic")
            my_swarm.current_cost=ini_cost[ini_cost<np.sort(ini_cost)[param_optimisation['n_particles']]]
            my_swarm.pbest_cost = my_swarm.current_cost
    
            iterations = param_optimisation['opti_iter'] # Set 100 iterations
    
            for i in range(iterations):
                # Part 1: Update personal best
                if i!=0:
                    my_swarm.current_cost = self.multi_estimate_flux_negfc(my_swarm.position,**kwargs) # Compute current c        ost
    
                my_swarm.pbest_pos, my_swarm.pbest_cost = P.compute_pbest(my_swarm) # Update and store
    
                my_swarm.options={'c1': my_options_start['c1']+i*(my_options_end['c1']-my_options_start['c1'])/(iterations-1), 'c2': my_options_start['c2']+i*(my_options_end['c2']-my_options_start['c2'])/(iterations-1), 'w': my_options_start['w']+i*(my_options_end['w']-my_options_start['w'])/(iterations-1)}
                # Part 2: Update global best
                # Note that gbest computation is dependent on your topology
                if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
                    my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)
    
                # Let's print our output
    
                print('Iteration: {} | my_swarm.best_cost: {}'.format(i+1, my_swarm.best_cost))
    
            # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
                my_swarm.velocity = my_topology.compute_velocity(my_swarm, my_swarm.velocity_clamp, my_swarm.vh, my_swarm.bounds
                )
                my_swarm.position = my_topology.compute_position(my_swarm, my_swarm.bounds, my_swarm.bh
                )
    
            expected_pos=my_swarm.best_pos[1:3]
            centroid_y,centroid_x=expected_pos
            curr_flux=my_swarm.best_pos[0]*10**(int(np.log10(curr_flux))-1)
            max_val=my_swarm.best_cost
    
    
        print("Optimal astrometry {}".format(i+1)+" {}".format(expected_pos))
        print("Optimal photometry {}".format(i+1)+" {}".format(curr_flux))
    
        if ci_esti is None:
            
            self.minradius=minradius_temp
            self.maxradius=maxradius_temp
            self.cube=cube
            
            ceny, cenx = frame_center(cube[0])
            rad_dist= np.sqrt((centroid_y-ceny)**2+(centroid_x-cenx)**2)*self.pxscale*1000
            theta=np.degrees(np.arctan2(centroid_y-ceny, centroid_x-cenx))
            print("Estimated astrometry: "+"radial distance {}".format(rad_dist)+" mas, Angle {}".format(theta))
    
            
            return curr_flux,centroid_y,centroid_x,rad_dist,theta
        else:
            print("Confidence interval computation")  
    
            int95=[1,1,1]
            if ci_esti=='hessian':
                power_curr_flux=10**(int(np.log10(curr_flux))-1)
                fun = lambda param: self.estimate_flux_negfc(param,curr_flux/power_curr_flux,expected_pos,cube,ns,loss_func,scaling_factor,'interval',[power_curr_flux,10**(-round(np.log10(max_val)))])   
                opti_param, hessian = Hessian([curr_flux/power_curr_flux,expected_pos[0],expected_pos[1]],fun,step=0.05)   
                curr_flux,expected_pos[0],expected_pos[1]=opti_param
                curr_flux*=power_curr_flux
                std=np.nan_to_num(np.sqrt(np.diag(np.linalg.inv(hessian))))
                int95=sc.stats.norm.ppf(0.95)*std
            elif ci_esti=='BFGS' or any(int95==0):
                res=minimize(self.estimate_flux_negfc,x0=[opti_param[0]/10**(int(np.log10(curr_flux))-1),opti_param[1],opti_param[2]], method='L-BFGS-B',bounds=((lower_b_flux/10**(int(np.log10(curr_flux))-1),upper_b_flux/10**(int(np.log10(curr_flux))-1)),(opti_param[1]-0.2,opti_param[1]+0.2),(opti_param[2]-0.2,opti_param[2]+0.2)),args=(curr_flux/10**(int(np.log10(curr_flux))-1),expected_pos,cube,ns,loss_func,scaling_factor,'interval',[10**(int(np.log10(curr_flux))-1),10**(-round(np.log10(max_val)))]),options={'ftol':5e-3,'gtol':5e-4, 'eps': 1e-2, 'maxiter': 10, 'finite_diff_rel_step': [1e-2,1e-2,1e-2]})
                std=np.nan_to_num(np.sqrt(np.diag(res['hess_inv'].todense())))
                int95=sc.stats.norm.ppf(0.95)*std
                curr_flux=res.x[0]*10**(int(np.log10(curr_flux))-1)
                expected_pos=res.x[1:3]
    
            err_phot=std[0]*10**(int(np.log10(curr_flux))-1)
            int95[0]*=10**(int(np.log10(curr_flux))-1)
            erry=std[1]
            errx=std[2]
            centroid_y,centroid_x=expected_pos
    
            print("Estimated photometry: "+" {}".format(curr_flux)+", Error {}".format(err_phot))
    
    
            print("Estimated astrometry: "+"Cartesian Y {}".format(centroid_y)+", X {}".format(centroid_x)+", Error Y {}".format(erry)+", Error X {}".format(errx))  
    
            ceny, cenx = frame_center(cube[0])
            rad_dist= np.sqrt((centroid_y-ceny)**2+(centroid_x-cenx)**2)*self.pxscale*1000
            theta=np.degrees(np.arctan2(centroid_y-ceny, centroid_x-cenx))
            print("Estimated astrometry: "+"radial distance {}".format(rad_dist)+" mas, Angle {}".format(theta))
    
            self.minradius=minradius_temp
            self.maxradius=maxradius_temp
            self.cube=cube
            
            return curr_flux,centroid_y,centroid_x,err_phot,erry,errx,int95[0],int95[1],int95[2],rad_dist,theta
            



                                               
        

