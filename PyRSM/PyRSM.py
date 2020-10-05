"""
Regime Switching Model (RSM) detection map. The package contains The PyRSM class
containing four main functions. The add_cube and add_model functions allow to 
consider several cubes and models to generate the cube of residuals used to compute 
the RSM map. The cube should be provided by the same instrument or rescaled to a unique
pixel size. The class can be used with ADI and ADI+IFS. In the case of IFS data 
the data should be rescaled for each wavelength and included separately in the model 
via the add_cube function. A specific PSF should be provided for each cube. Five different 
models and two forward model versions are available. Each model can be parametrized separately.
The function like_esti allows the estimation of a cube of likelihoods containing for
each pixel and each frame the likelihood of being in the planetary and speckle regime.
These likelihoods cubes are then used by the probmap_esti function to provide the final
probability map based on the RSM framework. Part of the code have been directly inspired 
by the VIP and PyKLIP packages for the estimation of the cube of residuals and forward model PSF.
"""

__author__ = 'Carl-Henrik Dahlqvist'

import numpy as np
import numpy.linalg as la
import scipy as sp
from scipy.optimize import curve_fit
from skimage.draw import circle 
import multiprocessing as mp
import ctypes as ct
import vip_hci as vip
from vip_hci.var import get_annulus_segments,frame_center
from vip_hci.preproc import frame_crop,cube_derotate,cube_crop_frames, check_pa_vector,cube_collapse
from vip_hci.metrics import cube_inject_companions
from vip_hci.preproc.derotation import _define_annuli,_find_indices_adi
from hciplot import plot_frames 


class PyRSM:

    def __init__(self,fwhm,minradius,maxradius,interval=[1],pxscale=0.1,ncore=1):
        
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
            Radius of center of the first annulus considered in the RSM probability
            map estimation. The radius should be larger than half 
            the value of the 'crop' parameter 
        maxradius : int
            Radius of the center of the last annulus considered in the RSM probability
            map estimation. The radius should be smaller or equal to half the
            size of the image minus half the value of the 'crop' parameter 
        interval: list of float or int, optional
            List of values taken by the delta parameter defining, when mutliplied by the 
            standard deviation, the strengh of the planetary signal in the Regime Switching model.
            Default is [1]. The different delta paramaters are tested and the optimal value
            is selected via maximum likelmihood.
        pxscale : float
            Value of the pixel in arcsec/px. Only used for printing plots when
            ``showplot=True`` in like_esti. 
        ncore : int, optional
            Number of processes for parallel computing. By default ('ncore=1') 
            the algorithm works in single-process mode.  
        """
        
        if minradius<np.where(round(fwhm)%2==1,round(fwhm),round(fwhm)+1):
            raise ValueError("'minradius' should be larger or equal to one FWHM")
    
        self.ncore = ncore 
        self.minradius = minradius 
        self.maxradius = maxradius 
        self.interval=interval
        self.fwhm = fwhm 
        self.pxscale = pxscale 
          
        self.psf = []         
        self.cube=[]
        self.pa=[]

        self.model = []    
        self.delta_rot = []
        self.nsegments = [] 
        self.ncomp = []    
        self.rank = [] 
        self.tolerance = []
        self.asize=[]
        self.psf_fm=[]

        self.flux = [] 
        self.distri = [] 
        self.distrifit=[]         
        self.var = [] 

        self.crop=[]
        self.crop_range=[]
        
        self.like_fin=[]
        self.distrisel=[]
        self.mixval=[]  
        self.fiterr=[]
        self.probmap=None
        
        
    def add_cube(self,psf, cube, pa):
     
        """    
        Function used to add an ADI seuqence to the set of cubes considered for
        the RSM map estimation.
        
        Parameters
        ----------
        cube : numpy ndarray, 3d
            Input cube (ADI sequences), Dim 1 = temporal axis, Dim 2-3 = spatial axis
        angs : numpy ndarray, 1d
            Parallactic angles for each frame of the ADI sequences. 
        psf : numpy ndarray 2d
            2d array with the normalized PSF template, with an odd shape.
            The PSF image must be centered wrt to the array! Therefore, it is
            recommended to run the function ``normalize_psf`` to generate a 
            centered and flux-normalized PSF template.
        """
    
        if self.maxradius>int((cube.shape[1]-1)/2)-np.where(round(self.fwhm)%2==1,round(self.fwhm),round(self.fwhm)+1):
            raise ValueError("'maxradius' should be smaller or equal to half the size of a frame minus one FWHM")
        if type(cube)==np.ndarray:
            if cube.ndim<3:
                raise TypeError('`cube` must be a numpy 3d array')
        else:
            raise TypeError('`cube` must be a numpy 3d array')
            
        if type(psf)==np.ndarray:
            if cube.ndim<2:
                raise TypeError('`psf` must be a numpy 2d array')
        else:
            raise TypeError('`psf` must be a numpy 2d array')
            
        if type(pa) is not np.ndarray:
            raise TypeError('`pa` must be a numpy 1d array')
    
        self.psf.append(psf)         
        self.cube.append(cube)
        self.pa.append(pa)
        
        self.like_fin.append([])
        self.distrisel.append([])
        self.mixval.append([])
        self.fiterr.append([])
        self.psf_fm.append([])                

    def add_method(self, model,delta_rot=0.5,asize=5,nsegments=1,ncomp=20,rank=5,tolerance=1e-2,flux=False,distri='Gaussian',var='Full',distrifit=False,crop_size=5, crop_range=1):

        
        """
        Function used to add a model to the set of post-processing techniques used to generate
        the cubes of residuals on which is based the computation of the likelihood of being 
        in either the planetary or the speckle regime. These likelihoods matrices allow
        eventually the definition of the final RSM map.
        
        Parameters
        ----------
    
        model : str
            Selected ADI-based post-processing techniques used to 
            generate the cubes of residuals feeding the Regime Switching model.
            'APCA' for annular PCA, NMF for Non-Negative Matrix Factorization, LLSG
            for Local Low-rank plus Sparse plus Gaussian-noise decomposition, LOCI 
            for locally optimized combination of images and'KLIP' for Karhunen-Loeve
            Image Projection. There exists a foward model version of KLIP and LOCI called 
            respectively 'FM KLIP' and 'FM LOCI'.
        delta_rot : int, optional
            Factor for tunning the parallactic angle threshold, expressed in FWHM.
            Default is 0.5 (excludes 0.5xFHWM on each side of the considered frame).
        asize : int, optional
            Width in pixels of each annulus. Default is 5. 
        n_segments : int, optional
            The number of segments in each annulus. Default is 1, working annulus-wise.
        ncomp : int, optional
            Number of components used for the low-rank approximation of the 
            speckle field with 'APCA', 'KLIP', 'NMF' and 'FM KLIP'. Default is 20.
        rank : int, optional        
            Expected rank of the L component of the 'LLSG' decomposition. Default is 5.
        tolerance: float, optional
            Tolerance level for the approximation of the speckle field via a linear 
            combination of the reference images in the LOCI algorithm. Default is 1e-2.
        flux: boolean, optionnal
            If true the flux parameter within the regime switching framework is estimated
            via a gaussian maximum likelihood by comparing the set of observations
            and the PSF or the forward model PSF in the case of 'FM KLIP' and 'FM LOCI'.
            If False, the flux parameter is defined as a multiple of the annulus residual
            noise variance. The multiplicative parameter is selected via the maximisation
            of the total likelihood of the regime switching model for the selected annulus.
            Default is False.
        distri: str, optional
            Probability distribution used for the estimation of the likelihood 
            of both regimes (planetary or speckle) in the Regime Switching framework.
            Default is 'Gaussian' but three other possibilities exist, 'Laplacian',
            'auto' and 'mix'. 
            
            'auto': allow the automatic selection of the optimal distribution ('Laplacian'
            or 'Gaussian') depending on the fitness of these distributions compared to
            the empirical distribution of the residual noise in the considered annulus. 
            For each cubes and ADI-based post-processing techniques, the distribution 
            leading to the lowest fitness error is automatically selected. 
            
            'mix': use both the 'Gaussian'and 'Laplacian' distribution to get closer to
            the empirical distribution by fitting a mix parameter providing the ratio
            of 'Laplacian' distribution compared to the 'Gaussian' one.
        var: str, optional
            Model used for the residual noise variance estimation. Six different approaches
            are proposed: 'Full', 'Annulus', 'Segment with mask', 'Time', 'Time with mask',
            'Background patch'. While all six can be used when flux=False, only the last
            three can be used when flux=True. Default is 'Full'.
            
            'Full': consider every frame and pixel in the selected annulus with a 
            width equal to asize (default approach)
            
            'Annulus': consider separately the different sub-annuli with a width
            equal to one pixel contained in the selected annulus with a width equal
            to asize. For each sub-annulus, every pixel and frame are considered
            
            'Segment with mask': consider for every pixels a segment of the selected annulus with 
            a width equal to asize. The segment is centered on the selected pixel and has
            a size of three FWHM. A mask with a diameter of one FWHM is applied on the
            selected pixel. Every frame are considered.
            
            'Time': consider the pixels in the selected annulus with a width equal to asize
            but separately for every frame.
            
            'Time with mask': consider the pixels in the selected annulus with a width 
            equal to asize but separately for every frame. Apply a mask wqith a diameter of
            one FWHM on the selected pixel.
            
            'Background patch': rely on the method developped in PACO to estimate the 
            residual noise variance (take the pixels in a region of one FWHM arround 
            the selected pixel, considering every frame in the derotated cube of residuals 
            except for the selected frame)
        distrifit: bool, optional
            If true, the estimation of the mean and variance of the selected distribution
            is done via an best fit on the empirical distribution. If False, basic 
            empirical estimation of the mean and variance using the set of observations 
            contained in the considered annulus, without taking into account the selected
            distribution.
        crop_size: int, optional
            Part of the PSF tempalte considered in the estimation of the RSM map
        crop_range: int, optional
            Range of crop sizes considered in the estimation of the RSM map, starting with crop_size
            and increasing the crop size incrementally by 2 pixels up to a crop size of 
            crop_size + 2 x (crop_range-1).  
        """
        if crop_size+2*(crop_range-1)>=2*round(self.fwhm)+1:
            raise ValueError("Maximum cropsize should be lower or equal to two FWHM, please change accordingly either 'crop_size' or 'crop_range'")
            
        if any(var in myvar for myvar in ['Full','Annulus','Segment with mask','Time','Time with mask','Background patch'])==False:
            raise ValueError("'var' not recognized")
        
        if (model=='FM KLIP' or model=='FM LOCI') and any(var in myvar for myvar in ['Time','Time with mask','Background patch'])==False:
            raise ValueError("'var' not recognized for forward model. 'var' should be 'Time','Time with mask' or 'Background patch'")
            
        if any(distri in mydistri for mydistri in ['auto','Gaussian','Laplacian','mix'])==False:
            raise ValueError("'distri' not recognized")
            
        if flux==True and any(var in myvar for myvar in ['Time','Time with mask','Background patch'])==False:
            raise ValueError("'var' not recognized for flux=True. 'var' should be 'Time','Time with mask' or 'Background patch'")
            
        self.model.append(model)
        self.delta_rot.append(delta_rot)
        self.nsegments.append(nsegments)
        self.ncomp.append(ncomp)   
        self.rank.append(rank)
        self.tolerance.append(tolerance)
        self.asize.append(asize)


        self.flux.append(flux)
        self.distri.append(distri) 
        self.distrifit.append(distrifit) 
        self.var.append(var) 
        self.crop.append(crop_size)
        self.crop_range.append(crop_range)       

        for i in range(len(self.cube)):
            
            self.psf_fm[i].append([])
            self.like_fin[i].append([])
            self.distrisel[i].append([])
            self.mixval[i].append([])
            self.fiterr[i].append([])
        
    def likelihood(self,cuben,modn,mcube,ann_center,verbose=True,fulloutput=False):
         

        
        if self.flux[modn]:
            range_int=1
        else:
            range_int=len(self.interval)

        n,y,x=mcube.shape 
        liketemp = np.frombuffer(probcube.get_obj())
        likemap=liketemp.reshape((n+1,x,y,range_int,2,self.crop_range[modn]))


        def likfcn(cuben,modn,mean,var,mixval,mcube,ann_center,distrim,evals=None,evecs_matrix=None, KL_basis_matrix=None,refs_mean_sub_matrix=None,sci_mean_sub_matrix=None,resicube_klip=None,probcube=0,var_f=None, ind_ref_list=None,coef_list=None):

            phi=np.zeros(2)
            n,y,x=mcube.shape 
            ceny, cenx = frame_center(mcube[0])
            
            indicesy,indicesx=get_time_series(mcube,ann_center)

            range_int=len(self.interval)
                
            if self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
                
                if len(self.psf_fm[cuben][modn])!=0:
                    psf_formod=True
                else:
                    
                    psf_formod=False
                psf_fm_out=np.zeros((len(indicesx),mcube.shape[0],2*round(self.fwhm)+1,2*round(self.fwhm)+1))
            if (self.crop[modn]+2*(self.crop_range[modn]-1))!=self.psf[cuben].shape[1]:
                    psfm=frame_crop(self.psf[cuben],(self.crop[modn]+2*(self.crop_range[modn]-1)),cenxy=[int(self.psf[cuben].shape[1]/2),int(self.psf[cuben].shape[1]/2)],verbose=False)
            else:
                    psfm=self.psf[cuben]

            for i in range(0,len(indicesy)):

                psfm_temp=None
                poscenty=indicesy[i]
                poscentx=indicesx[i]
                cubind=0
                
                #PSF forward model computation for KLIP

                if self.model[modn]=='FM KLIP':
                    
                    an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
                    theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))    

                            
                    if psf_formod==False:
                            
                            model_matrix=cube_inject_companions(np.zeros_like(mcube), self.psf[cuben], self.pa[cuben], flevel=1, plsc=0.1,rad_dists=an_dist, theta=theta, n_branches=1,verbose=False)

                            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn] * self.fwhm / (2 * ann_center)))
                            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
                            if pa_threshold >= mid_range - mid_range * 0.1:
                                pa_threshold = float(mid_range - mid_range * 0.1)

                            psf_map=np.zeros_like(model_matrix)
                            indices = get_annulus_segments(mcube[0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)


                            for b in range(0,n):
                                psf_map_temp = perturb(b,model_matrix[:, indices[0][0], indices[0][1]], self.ncomp[modn],evals_matrix, evecs_matrix,
                                           KL_basis_matrix,sci_mean_sub_matrix,refs_mean_sub_matrix, self.pa[cuben], self.fwhm, pa_threshold, ann_center)
                                psf_map[b,indices[0][0], indices[0][1]]=psf_map_temp-np.mean(psf_map_temp)



                            psf_map_der = cube_derotate(psf_map, self.pa[cuben], imlib='opencv',interpolation='lanczos4')
                            psfm_temp=cube_crop_frames(psf_map_der,2*round(self.fwhm)+1,xy=(poscentx,poscenty),verbose=False)
                            psf_fm_out[i,:,:,:]=psfm_temp
                            
                    else:
                            psfm_temp=self.psf_fm[cuben][modn][ann_center-self.minradius][i,:,:,:]
                            psf_fm_out[i,:,:,:]=psfm_temp
                #PSF forward model computation for LOCI
                            
                if self.model[modn]=='FM LOCI':
                    
                    an_dist = np.sqrt((poscenty-ceny)**2 + (poscentx-cenx)**2)
                    theta = np.degrees(np.arctan2(poscenty-ceny, poscentx-cenx))  
                    
                        
                    if psf_formod==False:
                        
                            model_matrix=cube_inject_companions(np.zeros_like(mcube), self.psf[cuben], self.pa[cuben], flevel=1, plsc=0.1, 
                                 rad_dists=an_dist, theta=theta, n_branches=1,verbose=False)
                    
                            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.fwhm/2),int(self.fwhm),1)
                    
                            values_fc = model_matrix[:, indices[0][0], indices[0][1]]

                            cube_res_fc=np.zeros_like(model_matrix)
                        
                            matrix_res_fc = np.zeros((values_fc.shape[0], indices[0][0].shape[0]))
                        
                            for e in range(values_fc.shape[0]):
                            
                                recon_fc = np.dot(coef_list[e], values_fc[ind_ref_list[e]])
                                matrix_res_fc[e] = values_fc[e] - recon_fc
        
                            cube_res_fc[:, indices[0][0], indices[0][1]] = matrix_res_fc
                            cube_der_fc = cube_derotate(cube_res_fc-np.mean(cube_res_fc), self.pa[cuben], imlib='opencv', interpolation='lanczos4')
                            psfm_temp=cube_crop_frames(cube_der_fc,2*round(self.fwhm)+1,xy=(poscentx,poscenty),verbose=False)
                            psf_fm_out[i,:,:,:]=psfm_temp
                    else:
                            psfm_temp=self.psf_fm[cuben][modn][ann_center-self.minradius][i,:,:,:]
                            psf_fm_out[i,:,:,:]=psfm_temp
                #Flux parameter estimation via Gaussian maximum likelihood (matched filtering)
                            
                if self.flux[modn]:
                    flux_esti=np.zeros((self.crop_range[modn]))
                
                    for v in range(0,self.crop_range[modn]):

                        cropf=self.crop[modn]+2*v
                        num=[]
                        denom=[]
                    
                        for j in range(n): 
                            
                            if self.var[modn]=='Time':
                                svar=var_f[j,v]
            
                            elif self.var[modn]=='Time with mask' :
                                svar=var_f[i,j,v]
                                    
                            elif self.var[modn]=='Background patch':
                                svar=var_f[i,j,v]

                            if psfm_temp is not None:
                                psfm_temp2=psfm_temp[j]
                            else:
                                psfm_temp2=psfm
                        
                            if psfm_temp2.shape[0]==cropf:
                                psfm=psfm_temp2
                            else:
                                psfm=frame_crop(psfm_temp2,cropf,cenxy=[int(psfm_temp2.shape[0]/2),int(psfm_temp2.shape[0]/2)],verbose=False)
  
                            num.append(np.multiply(frame_crop(mcube[j],cropf,cenxy=[poscentx,poscenty],verbose=False),psfm).sum()/svar)
                            denom.append(np.multiply(psfm,psfm).sum()/svar)
                        
                        flux_esti[v]=sum(num)/sum(denom)
                        probcube[n,indicesy[i],indicesx[i],0,0,v]=sum(num)/np.sqrt(sum(denom))
                
                for j in range(n):        

                    for m in range(range_int):

                        if psfm_temp is not None:
                                psfm_temp2=psfm_temp[j]
                        else:
                                psfm_temp2=psfm
                                
                        for v in range(0,self.crop_range[modn]):
                            
                            
                            cropf=self.crop[modn]+2*v
                            if psfm_temp2.shape[0]==cropf:
                                psfm=psfm_temp2
                            else:
                                psfm=frame_crop(psfm_temp2,cropf,cenxy=[int(psfm_temp2.shape[1]/2),int(psfm_temp2.shape[1]/2)],verbose=False)
           
                            if self.var[modn]=='Full':
                                svar=var[v]
                                alpha=mean[v]
                                mv=mixval[v]
                                sel_distri=distrim[v]
                                phi[1]=self.interval[m]*np.sqrt(svar)
            
                            elif self.var[modn]=='Time':
                                svar=var[j,v]
                                alpha=mean[j,v]
                                mv=mixval[j,v]
                                sel_distri=distrim[j,v]
                                phi[1]=self.interval[m]*np.sqrt(svar)
            
                            elif self.var[modn]=='Segment with mask':
                                svar=var[i,v]
                                alpha=mean[i,v]
                                mv=mixval[i,v]
                                sel_distri=distrim[i,v] 
                                phi[1]=self.interval[m]*np.sqrt(svar)
            
                            elif self.var[modn]=='Time with mask' :
                                svar=var[i,j,v]
                                alpha=mean[i,j,v]
                                mv=mixval[i,j,v]
                                sel_distri=distrim[i,j,v] 
                                phi[1]=self.interval[m]*np.sqrt(svar)
                                    
                            elif self.var[modn]=='Background patch':
                                svar=var[i,j,v]
                                alpha=mean[i,j,v]
                                mv=mixval[i,j,v]
                                sel_distri=distrim[i,j,v]  
                                phi[1]=self.interval[m]*np.sqrt(svar)
                                
                            elif self.var[modn]=='Annulus' :
                                    phi[1]=self.interval[m]*np.sqrt(var[(5-int(cropf/2)):(5+int(cropf/2)+1),j].mean())
                                    svar=np.transpose(np.reshape(cropf*list(var[(5-int(cropf/2)):(5+int(cropf/2)+1),j]),(cropf, cropf)))
                                    alpha=np.transpose(np.reshape(cropf*list(mean[(5-int(cropf/2)):(5+int(cropf/2)+1),j]),(cropf, cropf)))
                                    mv=np.transpose(np.reshape(cropf*list(mixval[(5-int(cropf/2)):(5+int(cropf/2)+1),j]),(cropf, cropf)))
                                    sel_distri=int(distrim[(5-int(cropf/2)):(5+int(cropf/2)+1),j].mean())

                            if self.flux[modn]==True:
                                phi[1]=np.where(flux_esti[v]<=0,0,flux_esti[v]) 
                            
                            
                            for l in range(0,2):

                                #Likelihood estimation
                                
                                ff=frame_crop(mcube[cubind],cropf,cenxy=[poscentx,poscenty],verbose=False)-phi[l]*psfm-alpha

                                if sel_distri==0:
                                        cftemp=(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)
                                elif sel_distri==1:
                                        cftemp=1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar))
                                elif sel_distri==2:
                                        cftemp=(mv*(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)+(1-mv)*1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar)))


                                probcube[int(cubind),int(indicesy[i]),int(indicesx[i]),int(m),l,v]=cftemp.sum()

                    cubind+=1

            return probcube,psf_fm_out
        


        if verbose==True:
            print("Radial distance: "+"{}".format(ann_center)) 


        #Estimation of the KLIP cube of residuals for the selected annulus
        
        evals_matrix=[]
        evecs_matrix=[]
        KL_basis_matrix=[]
        refs_mean_sub_matrix=[]
        sci_mean_sub_matrix=[]
        resicube_klip=None
        
        if self.model[modn]=='FM KLIP':
                

            resicube_klip=np.zeros_like(self.cube[cuben])
        

            pa_threshold = np.rad2deg(2 * np.arctan(self.delta_rot[modn] * self.fwhm / (2 * ann_center)))
            mid_range = np.abs(np.amax(self.pa[cuben]) - np.amin(self.pa[cuben])) / 2
            if pa_threshold >= mid_range - mid_range * 0.1:
                pa_threshold = float(mid_range - mid_range * 0.1)


            indices = get_annulus_segments(self.cube[cuben][0], ann_center-int(self.asize[modn]/2),int(self.asize[modn]),1)


            for k in range(0,self.cube[cuben].shape[0]):

                evals_temp,evecs_temp,KL_basis_temp,sub_img_rows_temp,refs_mean_sub_temp,sci_mean_sub_temp =KLIP_patch(k,self.cube[cuben][:, indices[0][0], indices[0][1]], self.ncomp[modn], self.pa[cuben], self.crop[modn], pa_threshold, ann_center)
                resicube_klip[k,indices[0][0], indices[0][1]] = sub_img_rows_temp

                evals_matrix.append(evals_temp)
                evecs_matrix.append(evecs_temp)
                KL_basis_matrix.append(KL_basis_temp)
                refs_mean_sub_matrix.append(refs_mean_sub_temp)
                sci_mean_sub_matrix.append(sci_mean_sub_temp)

            mcube=cube_derotate(resicube_klip,self.pa[cuben])


        #Estimation of the LOCI cube of residuals and optimal factors for the PSF forward model estimation

        ind_ref_list=None
        coef_list=None
    
        if self.model[modn]=='FM LOCI':
            
            
            mcube, ind_ref_list,coef_list=LOCI_FM(self.cube[cuben], self.psf[cuben], ann_center, self.pa[cuben], self.fwhm, self.tolerance[modn],self.delta_rot[modn])
    

        # Probability definition for the determination of the sellection of the optimal distribution
        # and the computation of the fitness errors

        def gaus(x,x0,var):
            return 1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))

        def lap(x,x0,var):
            bap=np.sqrt(var/2)
            return (1/(2*bap))*np.exp(-np.abs(x-x0)/bap)

        def mix(x,x0,var,a):
            bap=np.sqrt(var/2)
            return a*(1/(2*bap))*np.exp(-np.abs(x-x0)/bap)+(1-a)*1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))

        def vm_esti(modn,arr,var_e,mean_e):
            
            import numpy as np

            mixval_temp=None
            hist, bin_edge =np.histogram(arr,bins='auto',density=True)
            bin_mid=(bin_edge[0:(len(bin_edge)-1)]+bin_edge[1:len(bin_edge)])/2
            if self.distrifit[modn]==False:

               if self.distri[modn]=='Gaussian':
                    fiterr_temp=sum(abs(gaus(bin_mid,mean_e,var_e)-hist))
                    mean_temp=mean_e
                    var_temp=var_e
                    distrim_temp=0

               elif self.distri[modn]=='Laplacian':
                   fiterr_temp=sum(abs(lap(bin_mid,mean_e,var_e)-hist))
                   mean_temp=mean_e
                   var_temp=var_e
                   distrim_temp=1

               elif self.distri[modn]=='mix':
                   fixmix = lambda binm, mv: mix(binm,mean_e,var_e,mv)
                   popt,pcov = curve_fit(fixmix,bin_mid,hist,p0=[0.5],bounds=[(0),(1)])
                   mixval_temp=popt[0]
                   fiterr_temp=sum(abs(mix(bin_mid,mean_e,var_e,*popt)-hist))
                   distrim_temp=2
                   mean_temp=mean_e
                   var_temp=var_e 

               elif self.distri[modn]=='auto':
                   fiterrg=sum(abs(gaus(bin_mid,mean_e,var_e)-hist))
                   fiterrl=sum(abs(lap(bin_mid,mean_e,var_e)-hist))

                   if fiterrg>fiterrl:
                       distrim_temp=1
                       fiterr_temp=fiterrl
                       mean_temp=mean_e
                       var_temp=var_e

                   else:
                       distrim_temp=0
                       fiterr_temp=fiterrg
                       mean_temp=mean_e
                       var_temp=var_e

            else:
               if self.distri[modn]=='Gaussian':
                   distrim_temp=0
                   try:
                       popt,pcov = curve_fit(gaus,bin_mid,hist,p0=[mean_e,var_e],bounds=[(-2*abs(mean_e),0),(2*abs(mean_e),4*var_e)])
                       mean_temp=popt[0]
                       var_temp=popt[1]
                       fiterr_temp=sum(abs(gaus(bin_mid,*popt)-hist))
                   except RuntimeError:
                       fiterr_temp=sum(abs(gaus(bin_mid,mean_e,var_e)-hist)) 
               elif self.distri[modn]=='Laplacian':
                   distrim_temp=1
                   try:
                       popt,pcov = curve_fit(lap,bin_mid,hist,p0=[mean_e,var_e],bounds=[(-2*abs(mean_e),0),(2*abs(mean_e),4*var_e)])
                       mean_temp=popt[0]
                       var_temp=popt[1] 
                       fiterr_temp=sum(abs(lap(bin_mid,*popt)-hist))
                   except RuntimeError:
                       fiterr_temp=sum(abs(lap(bin_mid,mean_e,var_e)-hist))
            
               elif self.distri[modn]=='mix':
                   distrim_temp=2
                   try:
                       popt,pcov = curve_fit(mix,bin_mid,hist,p0=[mean_e,var_e,0.5],bounds=[(-2*abs(mean_e),0,0),(2*abs(mean_e),4*var_e,1)])
                       mean_temp=popt[0]
                       var_temp=popt[1] 
                       mixval_temp=popt[2]
                       fiterr_temp=sum(abs(mix(bin_mid,*popt)-hist))
                       
                   except RuntimeError:
                       fixmix = lambda binm, mv: mix(binm,mean_e,var_e,mv)
                       popt,pcov = curve_fit(fixmix,bin_mid,hist,p0=[0.5],bounds=[(0),(1)])
                       mean_temp=mean_e
                       var_temp=var_e
                       mixval_temp=popt[0]
                       fiterr_temp=sum(abs(mix(bin_mid,mean_temp,var_temp,*popt)-hist))

               elif self.distri[modn]=='auto':

                   try:

                       poptg,pcovg = curve_fit(gaus,bin_mid,hist,p0=[mean_e,var_e],bounds=[(-2*abs(mean_e),0),(2*abs(mean_e),4*var_e)])
                       poptl,pcovl = curve_fit(lap,bin_mid,hist,p0=[mean_e,var_e],bounds=[(-2*abs(mean_e),0),(2*abs(mean_e),4*var_e)])
                       fiterrg=sum(abs(gaus(bin_mid,*poptg)-hist))
                       fiterrl=sum(abs(lap(bin_mid,*poptl)-hist))

                       if fiterrg>fiterrl:
                           distrim_temp=1
                           mean_temp=poptl[0]
                           var_temp=poptl[1]
                           fiterr_temp=fiterrl
                       else:
                           distrim_temp=0
                           mean_temp=poptg[0]
                           var_temp=poptg[1] 
                           fiterr_temp=fiterrg

                   except RuntimeError:
                        fiterrg=sum(abs(gaus(bin_mid,mean_e,var_e)-hist))
                        fiterrl=sum(abs(lap(bin_mid,mean_e,var_e)-hist))

                        if fiterrg>fiterrl:
                            distrim_temp=1
                            fiterr_temp=fiterrl
                            mean_temp=mean_e
                            var_temp=var_e

                        else:
                            distrim_temp=0
                            fiterr_temp=fiterrg
                            mean_temp=mean_e
                            var_temp=var_e
                

            return mean_temp,var_temp,fiterr_temp,mixval_temp,distrim_temp

        # Variance computation
                                    
        var_f=None
        
        if self.var[modn]=='Full':
            
            var=np.zeros(self.crop_range[modn])
            mean=np.zeros(self.crop_range[modn])
            mixval=np.zeros(self.crop_range[modn])
            fiterr=np.zeros(self.crop_range[modn])
            distrim=np.zeros(self.crop_range[modn])
            
            
            for v in range(0,self.crop_range[modn]):
                cropf=self.crop[modn]+2*v
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)

                poscentx=indices[0][1]
                poscenty=indices[0][0]
                
                arr = np.ndarray.flatten(mcube[:,poscentx,poscenty])
            
                mean[v],var[v],fiterr[v],mixval[v],distrim[v]=vm_esti(modn,arr,np.var(mcube[:,poscentx,poscenty]),np.mean(mcube[:,poscentx,poscenty]))
                
            
        elif self.var[modn]=='Time':
            var=np.zeros((n,self.crop_range[modn]))
            var_f=np.zeros((n,self.crop_range[modn]))
            mean=np.zeros((n,self.crop_range[modn]))
            mixval=np.zeros((n,self.crop_range[modn]))
            fiterr=np.zeros((n,self.crop_range[modn]))
            distrim=np.zeros((n,self.crop_range[modn]))
            
            for v in range(0,self.crop_range[modn]):
                cropf=self.crop[modn]+2*v
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)

                poscentx=indices[0][1]
                poscenty=indices[0][0]
            
                for a in range(n):
           
                    arr = np.ndarray.flatten(mcube[a,poscentx,poscenty])
                    
                    mean[a,v],var[a,v],fiterr[a,v],mixval[a,v],distrim[a,v]=vm_esti(modn,arr,np.var(mcube[a,poscentx,poscenty]),np.mean(mcube[a,poscentx,poscenty]))

                    if self.flux[modn]:
                    
                        var_f[a,v]=np.var(mcube[a,poscentx,poscenty])
                    
        elif self.var[modn]=='Segment with mask':
            
            indicesy,indicesx=get_time_series(mcube,ann_center)
            
            var=np.zeros((len(indicesy),self.crop_range[modn]))
            mean=np.zeros((len(indicesy),self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),self.crop_range[modn]))
            distrim=np.zeros((len(indicesy),self.crop_range[modn]))
            
            size_seg=2
            
            for v in range(0,self.crop_range[modn]):
                cropf=self.crop[modn]+2*v
            
                for a in range(len(indicesy)):
            
                    if (a+int(cropf*3/2)+size_seg)>(len(indicesy)-1):
                        posup= a+int(cropf*3/2)+size_seg-len(indicesy)-1
                    else:
                        posup=a+int(cropf*3/2)+size_seg
               
                    indc=circle(indicesy[a], indicesx[a],3)
           
                    radist_b=np.sqrt((indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))**2+(indicesy[a-int(cropf*3/2)-size_seg-1]-int(y/2))**2)
           
                    if (indicesy[a-int(cropf*3/2)]-int(y/2))>=0:
                        ang_b= np.arccos((indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))/radist_b)/np.pi*180
                    else:
                        ang_b= 360-np.arccos((indicesx[a-int(cropf*3/2)-size_seg-1]-int(x/2))/radist_b)/np.pi*180
           
                    radist_e=np.sqrt((indicesx[posup]-int(x/2))**2+(indicesy[posup]-int(y/2))**2)
           
                    if (indicesy[posup]-int(y/2))>=0:
                        ang_e= np.arccos((indicesx[posup]-int(x/2))/radist_b)/np.pi*180
                    else:
                        ang_e= 360-np.arccos((indicesx[posup]-int(x/2))/radist_b)/np.pi*180
                                     
                    if ang_e>ang_b:
                        diffang=(360-ang_e)+ang_b
                    else:
                        diffang=ang_b-ang_e

            
                    indices = get_annulus_segments(mcube[0], radist_e-int(cropf/2),cropf,int(360/diffang),ang_b)
                    positionx=[]
                    positiony=[]
           
                    for k in range(0,len(indices[0][1])):
                        if len(set(np.where(indices[0][1][k]==indc[1])[0]) & set(np.where(indices[0][0][k]==indc[0])[0]))==0:
                            positionx.append(indices[0][1][k])
                            positiony.append(indices[0][0][k])

        
                    arr = np.ndarray.flatten(mcube[:,positionx,positiony])

                    mean[a,v],var[a,v],fiterr[a,v],mixval[a,v],distrim[a,v]=vm_esti(modn,arr,np.var(mcube[:,positionx,positiony]),np.mean(mcube[:,positionx,positiony]))

            
        elif self.var[modn]=='Annulus' :
            var=np.zeros((2*round(self.fwhm)+1,n))
            mean=np.zeros((2*round(self.fwhm)+1,n))
            mixval=np.zeros((2*round(self.fwhm)+1,n))
            fiterr=np.zeros((2*round(self.fwhm)+1,n))
            distrim=np.zeros((2*round(self.fwhm)+1,n))
            
            for a in range(0,2*round(self.fwhm)+1):
         
                indices = get_annulus_segments(mcube[0], ann_center-int(2*round(self.fwhm)+1/2)+a,1,1)
                
                for b in range(n):
           
                    arr = np.ndarray.flatten(mcube[b,indices[0][1],indices[0][0]])
                    
                    mean[a,b],var[a,b],fiterr[a,b],mixval[a,b],distrim[a,b]=vm_esti(modn,arr,np.var(mcube[b,indices[0][1],indices[0][0]]),np.mean(mcube[b,indices[0][1],indices[0][0]]))

            
            
        elif self.var[modn]=='Time with mask' :
            
            indicesy,indicesx=get_time_series(mcube,ann_center)
            

            var=np.zeros((len(indicesy),n,self.crop_range[modn]))
            var_f=np.zeros((len(indicesy),n,self.crop_range[modn]))
            mean=np.zeros((len(indicesy),n,self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),n,self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),n,self.crop_range[modn]))
            distrim=np.zeros((len(indicesy),n,self.crop_range[modn]))
            
            for v in range(0,self.crop_range[modn]):
                cropf=self.crop[modn]+2*v
                indices = get_annulus_segments(mcube[0], ann_center-int(cropf/2),cropf,1)
            
                for a in range(0,len(indicesy)):
         
                    indc=circle(indicesy[a], indicesx[a],3)
                    positionx=[]
                    positiony=[]
        
                    for k in range(0,len(indices[0][1])):
                        if len(set(np.where(indices[0][1][k]==indc[1])[0]) & set(np.where(indices[0][0][k]==indc[0])[0]))==0:
                            positionx.append(indices[0][1][k])
                            positiony.append(indices[0][0][k])
                
                    for b in range(n):
           
                        arr = np.ndarray.flatten(mcube[b,positionx,positiony])
                    
                        mean[a,b,v],var[a,b,v],fiterr[a,b,v],mixval[a,b,v],distrim[a,b,v]=vm_esti(modn,arr,np.var(np.asarray(mcube[b,positionx,positiony])),np.mean(np.asarray(mcube[b,positionx,positiony])))
            
                        if self.flux[modn]:
                    
                            var_f[a,b,v]=np.var(mcube[b,positionx,positiony])
                            
        elif self.var[modn]=='Background patch' :
        
            indicesy,indicesx=get_time_series(mcube,ann_center)
            
            var=np.zeros((len(indicesy),n,self.crop_range[modn]))
            var_f=np.zeros((len(indicesy),n,self.crop_range[modn]))
            mean=np.zeros((len(indicesy),n,self.crop_range[modn]))
            mixval=np.zeros((len(indicesy),n,self.crop_range[modn]))
            fiterr=np.zeros((len(indicesy),n,self.crop_range[modn]))  
            distrim=np.zeros((len(indicesy),n,self.crop_range[modn])) 
            
            mcube_derot=cube_derotate(mcube,-self.pa[cuben])
            
            for v in range(0,self.crop_range[modn]):
                cropf=self.crop[modn]+2*v
                for a in range(0,len(indicesy)):
         
                    radist=np.sqrt((indicesx[a]-int(x/2))**2+(indicesy[a]-int(y/2))**2)
           
                    if (indicesy[a]-int(y/2))>=0:
                        ang_s= np.arccos((indicesx[a]-int(x/2))/radist)/np.pi*180
                    else:
                        ang_s= 360-np.arccos((indicesx[a]-int(x/2))/radist)/np.pi*180
                
                    for b in range(n):
           
                        twopi=2*np.pi
                        sigposy=int(y/2 + np.sin((ang_s+self.pa[cuben][b])/360*twopi)*radist)
                        sigposx=int(x/2+ np.cos((ang_s+self.pa[cuben][b])/360*twopi)*radist)
           
           
                        y0 = int(sigposy - int(cropf/2)-1)
                        y1 = int(sigposy + int(cropf/2))  # +1 cause endpoint is excluded when slicing
                        x0 = int(sigposx - int(cropf/2)-1)
                        x1 = int(sigposx + int(cropf/2))
           
                    #indc=circle(indicesy[j], indicesx[j],3)
                        mask = np.ones(mcube_derot.shape[0],dtype=bool)
                        mask[b]=0
                        mcube_sel=mcube_derot[:,x0:x1,y0:y1]
                        mcube_sel=mcube_sel[mask,:]
           
           
                        arr = np.ndarray.flatten(mcube_sel)

                        mean[a,b,v],var[a,b,v],fiterr[a,b,v],mixval[a,b,v],distrim[a,b,v]=vm_esti(modn,arr,np.var(np.asarray(mcube_sel)),np.mean(np.asarray(mcube_sel)))

                        if self.flux[modn]:
                    
                            var_f[a,b,v]=np.var(np.asarray(mcube_sel))

        #Estimation of the final probability map

        likemap,psf_fm_out=likfcn(cuben,modn,mean,var,mixval,mcube,ann_center,distrim,evals_matrix,evecs_matrix, KL_basis_matrix,refs_mean_sub_matrix,sci_mean_sub_matrix,resicube_klip,likemap,var_f,ind_ref_list,coef_list)

        
        if fulloutput:
            if self.distri[modn]=='mix':
                return psf_fm_out,distrim,fiterr,mixval
            else:   
                return psf_fm_out,distrim,fiterr
        elif self.model[modn]=='FM KLIP' or self.model[modn]=='FM LOCI':
            return psf_fm_out
        
    def lik_esti(self, showplot=False,fulloutput=False,verbose=True):
        
        """
        Function allowing the estimation of the likelihood of being in either the planetary regime 
        or the speckle regime for the different cubes. The likelihood computation is based on 
        the residual cubes generated with the considered set of models.
        
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
            
                if self.model[i]=='APCA':
                    print("Annular PCA estimation") 
                    cube_out, residuals_cube_, frame_fin = vip.pca.pca_annular(self.cube[j], self.pa[j], fwhm=self.fwhm, ncomp=self.ncomp[i], asize=self.asize[i], 
                              delta_rot=self.delta_rot[i], svd_mode='lapack', n_segments=int(self.nsegments[i]), nproc=self.ncore,full_output=True,verbose=False)
                    if showplot:
                        plot_frames(frame_fin,title='APCA', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)

                elif self.model[i]=='NMF':
                    print("NMF estimation") 
                    H_resh, recon_resh, array_out, residuals_cube_, frame_fin = vip.nmf.nmf(self.cube[j], self.pa[j], ncomp=self.ncomp[i], max_iter=100, random_state=0, mask_center_px=None,full_output=True,verbose=False)
                    if showplot:
                        plot_frames(frame_fin,title='NMF_cube', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)

                elif self.model[i]=='LLSG':
                    print("LLSGestimation") 
                    list_l, list_s, list_g, f_l, frame_fin, f_g = vip.llsg.llsg(self.cube[j], self.pa[j], self.fwhm, rank=self.rank[i],asize=self.asize[i], thresh=1,n_segments=int(self.nsegments[i]), max_iter=40, random_seed=10, nproc=self.ncore,full_output=True,verbose=False)
                    res_s=np.array(list_s)
                    residuals_cube_=res_s[0]
                    if showplot:
                        plot_frames(frame_fin,title='LLSG', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)

                elif self.model[i]=='LOCI':
                    print("LOCI estimation") 
                    residuals_cube, residuals_cube_,frame_fin=vip.leastsq.xloci(self.cube[j], self.pa[j], fwhm=self.fwhm,asize=self.asize[i], n_segments=self.nsegments[i],tol=self.tolerance[i], nproc=1, optim_scale_fact=2,delta_rot=self.delta_rot[i],verbose=False,full_output=True)
                    if showplot:
                        plot_frames(frame_fin,title='LOCI', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
        
                elif self.model[i]=='KLIP':
                    print("KLIP estimation") 
                    cube_out, residuals_cube_, frame_fin = KLIP(self.cube[j], self.pa[j], ncomp=self.ncomp[i], fwhm=self.fwhm, asize=self.asize[i], 
                              delta_rot=self.delta_rot[i],nframes=30,full_output=True,verbose=False)
                    if showplot:
                        plot_frames(frame_fin,title='KLIP', colorbar=True,ang_scale=True, axis=False,pxscale=self.pxscale,ang_legend=True,show_center=True)
                    
                elif self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                        residuals_cube_=np.zeros_like(self.cube[j])
                        frame_fin=np.zeros_like(self.cube[j][0])
                    
                #Likelihood computation for the different models and cubes
                
                probcube = mp.Array(ct.c_double, (residuals_cube_.shape[0]+1)*residuals_cube_.shape[1]*residuals_cube_.shape[2]*len(self.interval)*self.crop_range[i]*2) 

                pool = mp.Pool(processes=self.ncore,initializer=init, initargs=(probcube,))
                results=[]
    
                for ann_center in range(self.minradius,self.maxradius):
        
                    result = pool.apply_async(self.likelihood, args=(j,i,residuals_cube_,ann_center,verbose,fulloutput))
                    results.append(result)
                [result.wait() for result in results]

                pool.close()
                for result in results:
                    if fulloutput:
                        self.psf_fm[j][i].append(result.get()[0])
                        self.distrisel[j][i].append(result.get()[1])
                        self.fiterr[j][i].append(result.get()[2])
                        if self.distri[j][i]=='mix':
                            self.mixval[j][i].append(result.get()[3])
                    elif self.model[i]=='FM LOCI' or self.model[i]=='FM KLIP':
                        self.psf_fm[j][i].append(result.get())
     
                probtemp = np.frombuffer(probcube.get_obj())
                like_temp=probtemp.reshape(((residuals_cube_.shape[0]+1),residuals_cube_.shape[1],residuals_cube_.shape[2],len(self.interval),2,self.crop_range[i]))

                
                like=[]
                SNR_FMMF=[]

                for k in range(self.crop_range[i]):
                    like.append(like_temp[0:self.cube[j].shape[0],:,:,:,:,k])
                    SNR_FMMF.append(like_temp[self.cube[j].shape[0],:,:,0,0,k])
                
                self.like_fin[j][i]=like
                
                like=[]
                SNR_FMMF=[]

                for k in range(self.crop_range[i]):
                    like.append(like_temp[0:self.cube[j].shape[0],:,:,:,:,k])
                    SNR_FMMF.append(like_temp[self.cube[j].shape[0],:,:,0,0,k])
                
                self.like_fin[j][i]=like
                
                


    def probmap_esti(self,modthencube=True,ns=1,sel_crop=None, estimator='Forward',colmode='median'):
        
        """
        Function allowing the estimation of the final RSM map based on the likelihood computed with 
        the lik_esti function for the different cubes and different post-processing techniques 
        used to generate the speckle field model. The RSM map estimation may be based on a forward
        or forward-backward approach.
        
        Parameters
        ----------
        
        modthencube: bool, optional
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
            Selected crop sizes from proposed crop_range (selected crop size = crop_size + 2 x (sel_crop-1)).
            A specific sel_crop should be provided for each mode. Default is None which is equivalent to
            selected crop size = [crop_size]
        estimator: str, optional
            Approach used for the probability map estimation either a 'Forward' model
            (approach used in the original RSM map algorithm) which consider only the 
            past observations to compute the current probability or 'Forward-Backward' model
            which relies on both past and future observations to compute the current probability
        colmode:str, optional
            Method used to generate the final probability map from the three-dimensionnal cube
            of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
            the 'median' or the 'max' value of the probabilities along the time axis. Default is 'median'.
        """
            
        if any(estimator in myesti for myesti in ['Forward','Forward-Backward'])==False:
            raise ValueError("'estimator' should be either 'Forward' or 'Forward-Backward'")
            
        if any(colmode in mycl for mycl in ['mean','median','max'])==False:
            raise ValueError("'estimator' should be either 'mean', 'median' or 'max'")            
        
        if sel_crop==None:
            sel_crop=np.zeros(len(self.model))
            
        if modthencube==True:
            for i in range(len(self.model)):
                for j in range(len(self.cube)):
                    if (i+j)==0:
                        if self.like_fin[j][i][int(int(sel_crop[i]))].shape[3]==1:
                            like_cube=np.repeat(self.like_fin[j][i][int(sel_crop[i])],self.interval,axis=3)
                        else:
                            like_cube=self.like_fin[j][i][int(sel_crop[i])]
                    else:
                        if self.like_fin[j][i][int(sel_crop[i])].shape[3]==1:
                            like_cube=np.append(like_cube,np.repeat(self.like_fin[j][i][int(sel_crop[i])],self.interval,axis=3),axis=0)
                        else:
                            like_cube=np.append(like_cube,self.like_fin[j][i][int(sel_crop[i])],axis=0) 
        else:
            for i in range(len(self.cube)):
                for j in range(len(self.model)):
                    if (i+j)==0:
                        if self.like_fin[i][j][int(sel_crop[j])].shape[3]==1:
                            like_cube=np.repeat(self.like_fin[i][j][int(sel_crop[j])],self.interval,axis=3)
                        else:
                            like_cube=self.like_fin[i][j][int(sel_crop[j])]
                    else:
                        if self.like_fin[i][j][int(sel_crop[j])].shape[3]==1:
                            like_cube=np.append(like_cube,np.repeat(self.like_fin[i][j][int(sel_crop[j])],self.interval,axis=3),axis=0)
                        else:
                            like_cube=np.append(like_cube,self.like_fin[i][j][int(sel_crop[j])],axis=0) 
                    
        n,y,x,l_int,r_n =like_cube.shape 
        probmap = np.zeros((like_cube.shape[0],like_cube.shape[1],like_cube.shape[2]))
    

        def forback(obs,Trpr,prob_ini):
            
            #Forward-backward model relying on past and future observations to 
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
                j=obs.shape[1]-1-i
                if i==0:
                    prob_cur_fw=np.dot(np.diag(obs[:,i]),Trpr).dot(prob_ini)
                    prob_cur_bw=np.dot(Trpr,np.diag(obs[:,j])).dot(prob_ini)
                else:
                    prob_cur_fw=np.dot(np.diag(obs[:,i]),Trpr).dot(prob_pre_fw)
                    prob_cur_bw=np.dot(Trpr,np.diag(obs[:,j])).dot(prob_pre_bw)
    
                scalefact_fw[i]=prob_cur_fw.sum()
                prob_fw[:,i]=prob_cur_fw/scalefact_fw[i]
                prob_pre_fw=prob_fw[:,i]
    
                scalefact_bw[j]=prob_cur_bw.sum()
                prob_bw[:,j]=prob_cur_bw/scalefact_bw[j]
                prob_pre_bw=prob_bw[:,j]
    
            scalefact_fw_tot=(scalefact_fw).sum()                
            scalefact_bw_tot=(scalefact_bw).sum()
    
    
            for k in range(obs.shape[1]):
    
                prob_fin[:,k]=(prob_fw[:,k]*prob_bw[:,k])/(prob_fw[:,k]*prob_bw[:,k]).sum()
    
            lik = scalefact_fw_tot+scalefact_bw_tot
    
            return prob_fin, lik
    
    
        def RSM_esti(obs,Trpr,prob_ini):
            
            #Original RSM approach involving a forward two-states Markov chain
    
            prob_fin=np.zeros((2,obs.shape[1]))
            prob_pre=0
            lik=0
    
            for i in range(obs.shape[1]):
                if i==0:
                    cf=obs[:,i]*np.dot(Trpr,prob_ini)
                else:
                    cf=obs[:,i]*np.dot(Trpr,prob_pre)
    
                f=sum(cf)            
                lik+=np.log(f)
                prob_fin[:,i]=cf/f
                prob_pre=prob_fin[:,i]
    
            return prob_fin, lik
    
        def likfcn(like_cube,ann_center,ns):
    

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
    
            cf=np.zeros((2,len(indicesy)*like_cube.shape[0],len(self.interval)))
            totind=0
            for i in range(0,len(indicesy)):
    
                poscenty=indicesy[i]
                poscentx=indicesx[i]
                    
                for j in range(0,like_cube.shape[0]):        
    
                        for m in range(0,len(self.interval)):
                            
                            cf[0,totind,m]=like_cube[j,poscenty,poscentx,m,0]
                            cf[1,totind,m]=like_cube[j,poscenty,poscentx,m,1]
                        totind+=1
                        
            #Computation of the probability cube via the regime switching framework
            
            prob_fin=[] 
            lik_fin=[]
            for n in range(len(self.interval)):
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
        
        for i in range(self.maxradius-self.minradius):
            ann_center=self.minradius+i
            likfcn(like_cube,ann_center,ns)
            
        self.probmap=cube_collapse(probmap, mode=colmode)
        

def get_time_series(mcube,ann_center):
    
        #Definition and ordering (anti-clockwise) of the pixels composing the selected annulus
             

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
    

    
    #Function allowing the estimation of the PSF forward model when relying on KLIP
    #for the computation of the speckle field. The code is based on the PyKLIP library
    # considering only the ADI case with a singlle number of principal components considered.
    #For more details about the code, consider the PyKLIP library or the originall articles
    # (Pueyo, L. 2016, ApJ, 824, 117 or
    # Ruffio, J.-B., Macintosh, B., Wang, J. J., & Pueyo, L. 2017, ApJ, 842)
    
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

    
    #Function allowing the estimation of the cube of residualls after
    #the subttraction of the speckle field modeled via the KLIP framework 
    
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


               
    #Function allowing the computation via KLIP of the speckle field for a 
    #given sub-region of the original ADI sequence. Code inspired by the PyKLIP librabry
    
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


def LOCI_FM(cube, psf, ann_center, angle_list, asize, Tol,delta_rot):


    
    #Computation of the optimal factors weigthing the linear combination of reference
    #frames used to obtain the modeled speckle field for each frame and allowing the 
    #determination of the forward modeled PSF. Estimation of the cube 
    #of residuals based on the modeled speckle field.


    cube_res = np.zeros_like(cube)
    ceny, cenx = frame_center(cube[0])
    radius_int=ann_center-int(1.5*asize)
            
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
        
        ayxyx = [pa_threshold, indices[0][0], indices[0][1],
                   ind_opt[0][0], ind_opt[0][1]]
                
        matrix_res, ind_ref, coef, yy, xx = _leastsq_patch(ayxyx,
                         angle_list,cube,ann_center,'manhattan', 100,
                         'lstsq', Tol,formod=True,psf=psf)
        
        if ann==1:
            ind_ref_list=ind_ref
            coef_list=coef
        
        cube_res[:, yy, xx] = matrix_res


    cube_der = cube_derotate(cube_res, angle_list, imlib='opencv', interpolation='lanczos4')
    
    return cube_der, ind_ref_list,coef_list
            

def _leastsq_patch(ayxyx, angles,cube, nann,metric, dist_threshold,
                   solver, tol,formod=False,psf=None):


    
    #Function allowing the estimation of the optimal factors for the modeled speckle field
    #estimation via the LOCI framework. The code has been developped based on the VIP 
    #python function LOCI.
    
    pa_threshold, yy, xx, yy_opti, xx_opti = ayxyx
    
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
        ind_fr_i = _find_indices_adi(angles, i, pa_threshold,truncate=False)
        if len(ind_fr_i) > 0:
            A = values_opt[ind_fr_i]
            b = values_opt[i]
            if solver == 'lstsq':
                coef = np.linalg.lstsq(A.T, b, rcond=tol)[0]
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


