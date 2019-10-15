"""
Regime Switching Model (RSM) detection map. The package contains two functions.
The RSM function allows the estimation of the RSM detection map when feeded with 
cube of residuals. The RSMmap function includes the entire estimation procedure 
with first the estimation of the cube of residuals and then the estimation of 
the RSM detection map using the RSM function. The package necessitate the VIP 
package, as well as the hciplot package. The RSMmap function can be used with 
either ADI or IFS+ADI. In the case of IFS+ADI data sets, a list of the ADI 
sequences for each wavelength should be provided along with a list of the 
parallactic angles and of the PSF.

"""

__author__ = 'Carl-Henrik Dahlqvist'




def RSM(mcube,psf,inner_radius,nmod,cubesize,distri='Gaussian',distrifit=True,interval=5,ns=1,crop=5,fulloutput=True,verbose=True):
    
    """
    Probability map generation via a Regime Switching model using as input cubes 
    of residuals provided by one or several different ADI-based post processing techniques.
    If not used via RSMmap, a multiprocessing array called probcube should be created 
    and initialized before calling the RSM function to get back the RSM detection map.
    For more details see the RSMmap function.

    Parameters
    ----------
    mcube : numpy ndarray, 3d
        Input stacked cubes of residuals. One or several ADI sequences of the same object
        and multiple post-processing techniques may be used with RSM.
    psf : numpy ndarray 2d or list of numpy ndarray 2d 
        2d array with the normalized PSF template, with an odd shape.
        The PSF image must be centered wrt to the array! Therefore, it is
        recommended to run the function ``normalize_psf`` to generate a 
        centered and flux-normalized PSF template. In the case of multiple
        ADI sequences of the same object, several PSFd may be providing, the 
        third axis accounting for these different PSFs.
    inner_radius : int
        Inner radius of the considered annulus. The set of pixels for which the
        probability is estimated are situated in the middle of this annulus of 
        width equal to 'crop'.
    nmod : int
        Number of ADI-based post-processing techniques used for the estimation 
        of the cubes of residuals.
    cubesize: int or numpy ndarray, 1d,optional
        In the case of a single ADI sequence, cubesize provide the number of frames 
        composing the ADI sequence via an integer. In the case of multiple ADI sequence
        it provides the size of every cubes of residuals provided in resicube via a 
        numpy 1d ndarray.
    distri: str, optional
        Probability distribution used for the estimation of the likelihood 
        of both regimes(planetary or noise) in the Regime Switching model.
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
        
    distrifit: bool, optional
        If true, the estimation of the mean and variance of the selected distribution
        is done via an optimal fit on the empirical distribution. If False, basic 
        estimation of the mean and variance using the set of observations 
        contained in the considered annulus, without taking into account the selected
        distribution.
    interval: int, optional
        Maximum value for the delta parameter defining, when mutliplied by the 
        variance, the strengh of the planetary signal in the Regime Switching model.
        Default is 5. Values ranging between 1 and 'interval' are tested 
        with a step of 1. 
    ns: float , optional
         Number of regime switches. Default is one regime switch per annulus but 
         smaller values may be used to reduce the impact of noise or disk structures
         on the final RSM probablity map.
    crop: int, optional
        Part of the PSF tempalte considered is the estimation of the RSM map
    fulloutput: bool, optional
        Whether to get the final RSM map only or with the total log-likelihood, 
        the optimal delta, the selected distribution and the fit errors for every 
        cubes of residuals.Default is True.
    verbose : bool, optional
        If True prints intermediate info. Default is True.
        

    Returns
    -------
    likres : 
        Total log-likelihood used to select the optimal delta
    delta:
        Optimal delta providing when multiplied by the residual noise variance,
        the strength of the planetary signal in the Regime Swicting Model (2d numpy array).The 
        first dimension of the array provides the model/cube while the second one gives the 
        radial distance.
    distrisel:
        Selected distribution (string array) for each cube of residuals and each ADI sequences,
        Provided when using the automatic selection (distri='auto') of the 
        distribution.The first dimension of the array provides the model/cube while the second
        one gives the radial distance.
    fiterr: 
        Differences between the selected distribution "distri" and the empirical 
        residuals distribution estimated via np.histogram with automatic binning,
        summed over the entire set of bin values (2d numpy array).The first dimension
        of the array provides the model/cube while the second one gives the radial distance.
    mixratio:
        The ratio of Laplacian distribution when using the distri='mix' option,
        with (1-mixratio) providing the ratio of Gaussian distribution (2d numpy array).
        The ratiocis optimally selected based on the fitness of the resulting mixed distribution
        compared to the empirical probability distribution.The first dimension
        of the array provides the model/cube while the second one gives the radial distance.      

    """
    
    
    import numpy as np
    from vip_hci.var import get_annulus_segments
    from vip_hci.preproc import frame_crop
    from scipy.optimize import curve_fit
    
    if type(mcube)==np.ndarray:
        if mcube.ndim<3:
            raise TypeError('`cube` must be a numpy 3d array or a list of numpy 3d arrays')

    if mcube.shape[1]%2==0:
            raise TypeError('`cube` spatial dimension should be odd')
            
            
    if type(psf)==np.ndarray:
        if psf.ndim<2:
            raise TypeError('`psf` must be a numpy 2d array or a list of numpy 2d arrays')
    elif type(psf)==list:
        for i in range(0,len(psf)):
            if psf[i].ndim<2:
                raise TypeError('`psf` must be a numpy 2d array or a list of numpy 2d arrays')
    
    if not type(inner_radius)==int:
        raise TypeError('`inner_radius` must be an integer')
    if not type(nmod)==int:
        raise TypeError('`nmod` must be an integer')
    if not (type(cubesize)==np.ndarray or type(cubesize)==int):
        raise TypeError('`cubesize` must be a numpy 1d array')        
        
    n,y,x=mcube.shape 
    probtemp = np.frombuffer(probcube.get_obj())
    probmap=probtemp.reshape((n,x,y))
    
    #Probability/Likelihood estimation via a Regime Switching Model

    def likfcn(param,mean,var,mixval,ns,mcube,psf,inner_radius,cubeseq,distrim,crop,cubenum,probcube=0):

        phi=np.zeros(2)               

        #Definition and ordering (anti-clockwise) of the pixels composing the selected annulus

        ann_center = inner_radius + int(crop / 2)
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
        
        #prob noise to noise, planetary to noise, noise to planetary, planetary to planetary
        npix = len(indicesy)
        pini=[1-ns/(mcube.shape[0]*(npix)),1/(mcube.shape[0]*ns),ns/(mcube.shape[0]*(npix)),1-1/(mcube.shape[0]*ns)]
        prob=np.reshape([pini],(2, 2)) 
              
        Trpr= prob
        
        #Initialization of the Regime Switching model
        #I-prob
        mA=np.concatenate((np.diag(np.repeat(1,2))-prob,[np.repeat(1,2)]))
        #sol
        vE=np.repeat([0,1],[2,1])
        #mA*a=vE -> mA'mA*a=mA'*vE -> a=mA'/(mA'mA)*vE
        vProb= np.dot(np.dot(np.linalg.inv(np.dot(mA.T,mA)),mA.T),vE)
        likv = 0
        
        if type(psf)==np.ndarray:
            if crop!=psf.shape[1]:
                psfm=frame_crop(psf,crop,cenxy=[int(psf.shape[1]/2),int(psf.shape[1]/2)],verbose=False)
            else:
                psfm=psf

        for i in range(0,len(indicesy)):

            poscenty=indicesy[i]
            poscentx=indicesx[i]
            cf=np.zeros(2)
            cubind=0
            for k in range(0,cubenum):
                
                if type(psf)==list:
                    if (k%nmod)==0:
                        if crop!=psf[0].shape[1]:
                            psfm=frame_crop(psf[k//nmod],crop,cenxy=[int(psf[0].shape[1]/2),int(psf[0].shape[1]/2)],verbose=False)
                        else:
                            psfm=psf[k//nmod]
                
                for j in range(0,cubeseq[k]):

                    probitemp=np.dot(Trpr,vProb)
            
                    svar=var[k]
                    phi[1]=param*np.sqrt(var[k])
                    alpha=mean[k]
                    

                    for l in range(0,2):
                        
                        #Likelihood estimation
                        
                        ff=frame_crop(mcube[cubind],crop,cenxy=[poscentx,poscenty],verbose=False)-phi[l]*psfm-alpha
                        
                        if distrim[k]=='Gaussian':
                            cftemp=(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)*probitemp[l]
                        elif distrim[k]=='Laplacian':
                            cftemp=1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar))*probitemp[l]
                        else:
                            cftemp=(mixval[k]*(1/np.sqrt(2 * np.pi*svar))*np.exp(-0.5*np.multiply(ff,ff)/svar)+(1-mixval[k])*1/(np.sqrt(2*svar))*np.exp(-abs(ff)/np.sqrt(0.5*svar)))*probitemp[l]

                        cf[l]=cftemp.sum()/psfm.shape[0]

                    f=sum(cf)            
                    lik=np.log(f)
                    vProb=cf/f
                    likv+=lik
                    
                    if type(probcube)==np.ndarray:
                        probcube[cubind,poscenty,poscentx]=vProb[1]
                    cubind+=1
        if type(probcube)==np.ndarray:
            return likv, probcube
        else:
            return likv

    # probability definition for the determination of the optimal distribution and fitness errors
   
    def gaus(x,x0,var):
        return 1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))

    def lap(x,x0,var):
        bap=np.sqrt(var/2)
        return (1/(2*bap))*np.exp(-np.abs(x-x0)/bap)

    def mix(x,x0,var,a):
        bap=np.sqrt(var/2)
        return a*(1/(2*bap))*np.exp(-np.abs(x-x0)/bap)+(1-a)*1/np.sqrt(2 * np.pi*var)*np.exp(-(x-x0)**2/(2*var))

  
    if verbose==True:
        print("Radial distance: "+"{}".format(inner_radius)) 

    


    if type(cubesize)==np.ndarray:
        cubeseq=cubesize
    else:
        cubeseq=[cubesize] * nmod
        
    cubenum=len(cubeseq)

    var=np.zeros(cubenum)
    mean=np.zeros(cubenum)
    mixval=np.zeros(cubenum)
    fiterr=np.zeros(cubenum)
    distrim=np.repeat(distri, cubenum)
    indices = get_annulus_segments(mcube[0], inner_radius,crop,1,90)

    poscentx=indices[0][1]
    poscenty=indices[0][0]


    startp =0 
    
  

    
    for c in range(cubenum):
        
        # Empirical estimation of the mean and variance of the probability distribution of the quasi-static speckles
        
       arr = np.ndarray.flatten(mcube[startp:(startp+cubeseq[c]),poscentx,poscenty])
       hist, bin_edge =np.histogram(arr,bins='auto',density=True)
       bin_mid=(bin_edge[0:(len(bin_edge)-1)]+bin_edge[1:len(bin_edge)])/2
       
       var[c]=np.var(np.asarray(mcube[startp:(startp+cubeseq[c]),poscentx,poscenty]))     
       mean[c]=np.mean(np.asarray(mcube[startp:(startp+cubeseq[c]),poscentx,poscenty]))
       
       # Estimation of the fitness errors, optimal distributions and mix parameter
        
       if distrifit==False:
          
           if distri=='Gaussian':
               fiterr[c]=sum(abs(gaus(bin_mid,mean[c],var[c])-hist)) 

           elif distri=='Laplacian':
               fiterr[c]=sum(abs(lap(bin_mid,mean[c],var[c])-hist))

           elif distri=='mix':
               fixmix = lambda binm, mv: mix(binm,mean[c],var[c],mv)
               popt,pcov = curve_fit(fixmix,bin_mid,hist,p0=[0.5],bounds=[(0),(1)])
               mixval[c]=popt[0]
               fiterr[c]=sum(abs(mix(bin_mid,mean[c],var[c],*popt)-hist))
               
           if distri=='auto':
               fiterrg=sum(abs(gaus(bin_mid,mean[c],var[c])-hist))
               fiterrl=sum(abs(lap(bin_mid,mean[c],var[c])-hist))
          
               if fiterrg>fiterrl:
                   distrim[c]='Laplacian'
                   fiterr[c]=fiterrl

               else:
                  distrim[c]='Gaussian'
                  fiterr[c]=fiterrg
               
       else:
           if distri=='Gaussian':
               popt,pcov = curve_fit(gaus,bin_mid,hist,p0=[mean[c],var[c]],bounds=[(-2*abs(mean[c]),0),(2*abs(mean[c]),4*var[c])])
               mean[c]=popt[0]
               var[c]=popt[1]

               fiterr[c]=sum(abs(gaus(bin_mid,*popt)-hist))                
           elif distri=='Laplacian':
               popt,pcov = curve_fit(lap,bin_mid,hist,p0=[mean[c],var[c]],bounds=[(-2*abs(mean[c]),0),(2*abs(mean[c]),4*var[c])])
               mean[c]=popt[0]
               var[c]=popt[1] 

               fiterr[c]=sum(abs(lap(bin_mid,*popt)-hist))
           elif distri=='mix':
               popt,pcov = curve_fit(mix,bin_mid,hist,p0=[mean[c],var[c],0.5],bounds=[(-2*abs(mean[c]),0,0),(2*abs(mean[c]),4*var[c],1)])
               mean[c]=popt[0]
               var[c]=popt[1] 
               mixval[c]=popt[2]
               fiterr[c]=sum(abs(mix(bin_mid,*popt)-hist))

           elif distri=='auto':
               poptg,pcovg = curve_fit(gaus,bin_mid,hist,p0=[mean[c],var[c]],bounds=[(-2*abs(mean[c]),0),(2*abs(mean[c]),4*var[c])])
               poptl,pcovl = curve_fit(lap,bin_mid,hist,p0=[mean[c],var[c]],bounds=[(-2*abs(mean[c]),0),(2*abs(mean[c]),4*var[c])])
               fiterrg=sum(abs(gaus(bin_mid,*poptg)-hist))
               fiterrl=sum(abs(lap(bin_mid,*poptl)-hist))

               if fiterrg>fiterrl:
                   distrim[c]='Laplacian'
                   mean[c]=poptl[0]
                   var[c]=poptl[1]
                   fiterr[c]=fiterrl
               else:
                   distrim[c]='Gaussian'
                   mean[c]=poptg[0]
                   var[c]=poptg[1] 
                   fiterr[c]=fiterrg

      
       startp+=cubeseq[c]
       

    restemp=np.zeros(interval)
    
    # Definition of the optimal beta parameter
    
    for h in range(0,interval):

        param=h+1

        restemp[h]=likfcn(param,mean,var,mixval,ns,mcube,psf,inner_radius,cubeseq,distrim,crop,cubenum)


    #Estimation of the final probability map
    
    likres,probmap=likfcn((restemp.argmax()+1),mean,var,mixval,ns,mcube,psf,inner_radius,cubeseq,distrim,crop,cubenum,probmap)

    if fulloutput:
        if distri=='mix':
            return likres,(restemp.argmax()+1),fiterr,distrim,mixval
        else:   
            return likres,(restemp.argmax()+1),fiterr,distrim




def RSMmap(cube,angs,psf,pxscale,minradius,maxradius,fwhm,resicube=None,cubesize=None, model=['APCA','NMF','LLSG'],modtocube=True,distri='Gaussian',distrifit=True,interval=5,ns=1, paramod=[20,20,5], asize=[5,0,5], n_segments=[1,0,1],delta_rot=0.5,crop=5,numcore=1,colmode='median',showplot=True,fulloutput=True,verbose=True):

    """
    Probability map generation via a Regime Switching model using as input cubes 
    of residuals provided by one or several different ADI-based post-processing techniques.
    
    Parameters
    ----------
    cube : numpy ndarray, 3d or list of numpy ndarray, 3d
        Input cube or cubes. One or several ADI sequences of the same object
        may be used with RSMmap. Dim 1 = temporal axis, Dim 2-3 = spatial axis
    angs : numpy ndarray, 1d or 2d
        Corresponding parallactic angle for each frame and each cube if multiple 
        ADI sequences or used. 
    psf : numpy ndarray 2d or list of numpy ndarray 2d 
        2d array with the normalized PSF template, with an odd shape.
        The PSF image must be centered wrt to the array! Therefore, it is
        recommended to run the function ``normalize_psf`` to generate a 
        centered and flux-normalized PSF template. In the case of multiple
        ADI sequences of the same object, several PSFs may be providing, the 
        third axis accounting for these different PSFs.
    pxscale : float
        Value of the pixel in arcsec/px. Only used for printing plots when
        ``showplot=True``.        
    minradius : int
        Inner radius of the first annulus considered in the RSM probability
        map estimation
    maxradius : int
        Inner radius of the last annulus considered in the RSM probability
        map estimation. The radius should be smaller or equal to half the
        size of the image minus the value of the 'crop' parameter 
    fwhm: int
        Full width at half maximum for the PSF template
    resicube: numpy ndarray, 3d, optional
        Cubes of residuals provided by one or several post-processing techniques
        for one or several ADI sequences. The cubes of residuals should be stacked
        along the time axis. Dim 1 = temporal axis, Dim 2-3 = spatial axis. The 
        temporal axis should be a multiple of the number of frames composing the cube(s).
        Default is None. If resicube is not None, the RSMmap function uses directly 
        the RSM function to estimate based on the cubes of residuals the RSM detection map.
    cubesize: int or numpy ndarray, 1d,optional
        To be used when a cube of residuals is provided via resicube. In the case of a single
        ADI sequence, cubesize provide the number of frames composing the ADI sequence via an 
        integer. In the case of multiple ADI sequence it provides the size of every cubes of 
        residuals provided in resicube via a numpy 1d ndarray.
    model : str array, optional
        Set of selected ADI-based post-processing techniques used to 
        generate the cubes of residuals feeding the Regime Switching model.
        'PCA' for principal component analysis, 'APCA' for annular PCA, NMF for
        Non-Negative Matrix Factorization and LLSG for Local Low-rank plus 
        Sparse plus Gaussian-noise decomposition and LOCI for locally optimized 
        combination of images. Default is ['APCA','NMF','LLSG'].
    modtocube: bool, optional
        Parameter defining if the concatenated cube feeding the RSM model is created
        considering first the model or the different cubes. If 'modtocube=False',
        the function will select the first cube then test all models on it and move 
        to the next one. If 'modtocube=True', the model will select one model and apply
        it to every cubes before moving to the next model. Default is True.
    distri: str, optional
        Probability distribution used for the estimation of the likelihood 
        of both regimes (planetary or noise) in the Regime Switching model.
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
        
    distrifit: bool, optional
        If true, the estimation of the mean and variance of the selected distribution
        is done via an optimal fit on the empirical distribution. If False, basic 
        estimation of the mean and variance using the set of observations 
        contained in the considered annulus, without taking into account the selected
        distribution.
    interval: int, optional
        Maximum value for the delta parameter defining, when mutliplied by the 
        standard deviation, the strengh of the planetary signal in the Regime Switching model.
        Default is 5. Values ranging between 1 and 'interval' are tested 
        with a step of 1. 
     ns: float , optional
         Number of regime switches. Default is one regime switch per annulus but 
         smaller values may be used to reduce the impact of noise or disk structures
         on the final RSM probablity map.
    paramod : int or list of ints, optional
        Number of components used for the low-rank approximation of the 
        datacube with 'PCA', 'APCA' and 'NMF'; expected rank of the L component of the
        'LLSG' decomposition; the value taken by optim_scale_fact in 'LOCIX'.
        When a single integer is given it is used for all models.
        If an array is provided, it must contain a value for each model even if the model
        does not use this parameter (put simply 0).Default is [20,20,5].
    asize : int or list of ints or None, optional
        If ``asize`` is None then each annulus will have a width of ``2*fwhm``.
        If an integer then it is the width in pixels of each annulus.When a single
        integer is given it is used for all models. If an array is provided, it must
        contain a value for each model even if the model does not use this parameter
        (put simply 0). Default is [5,0,5]. 
    n_segments : int or list of ints, optional
        The number of segments for each annulus. When a single integer is given
        it is used for all models. If an array is provided, it must contain a value
        for each model even if the model does not use this parameter (put simply 0).
        Default is [1,0,1] as we are working annulus-wise.
    delta_rot : int, optional
        Factor for tunning the parallactic angle threshold, expressed in FWHM.
        Default is 0.5 (excludes 0.5xFHWM on each side of the considered frame).
    crop: int, optional
        Part of the PSF tempalte considered is the estimation of the RSM map
    numcore : int, optional
        Number of processes for parallel computing. By default ('numcore=1') 
        the algorithm works in single-process mode.  
    colmode: {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the intermediate RSM maps for producing 
        a final RSM map. Default is 'median'.
    showplot: bool, optional
        If True, provide the plots of the final residual frames for the selected 
        ADI-based post-processing techniques along the final RSM map. Default is True.
    fulloutput: bool, optional
        Whether to get the final RSM map only or with the optimal delta, the selected 
        distribution and the fit errors for every cubes of residuals. 
        Default is True.
    verbose : bool, optional
        If True prints intermediate info. Default is True.
        

    Returns
    -------
    probmapfin : 
        Final probability map (2d numpy array) providing the probability of oberving a planetary
        signal for every pixels inside the maximum radius 'maxradius".
    delta:
        Optimal delta providing, when multiplied by the residual noise standrad deviation,
        the strength of the planetary signal in the Regime Swicting Model (2d numpy array).The 
        first dimension of the array provides the model/cube while the second one gives the 
        radial distance.
    distrisel:
        Selected distribution (string array) for each cube of residuals and each ADI sequences,
        Provided when using the automatic selection (distri='auto') of the 
        distribution.The first dimension of the array provides the model/cube while the second
        one gives the radial distance.
    fiterr: 
        Differences between the selected distribution "distri" and the empirical 
        residuals distribution estimated via np.histogram with automatic binning,
        summed over the entire set of bin values (2d numpy array).The first dimension
        of the array provides the model/cube while the second one gives the radial distance.
    frame_mod:
        Set of final frames provided by the different ADI-based post-processing techniques.
        (3d numpy array). The first dimension represents the selected model and the two 
        others the two spatial dimensions
    mixratio:
        The ratio of Laplacian distribution when using the distri='mix' option,
        with (1-mixratio) providing the ratio of Gaussian distribution (2d numpy array).
        The ratio is optimally selected based on the fitness of the resulting mixed distribution
        compared to the empirical probability distribution.The first dimension
        of the array provides the model/cube while the second one gives the radial distance.

    """
     

    import vip_hci as vip
    import numpy as np
    import multiprocessing as mp
    import ctypes as c
    from hciplot import plot_frames 
    
    if type(cube)==np.ndarray:
        if cube.ndim<3:
            raise TypeError('`cube` must be a numpy 3d array or a list of numpy 3d arrays')
    elif type(cube)==list:
        for i in range(0,len(cube)):
            if cube[i].ndim<3:
                raise TypeError('`cube` must be a numpy 3d array or a list of numpy 3d arrays')
    else:
         raise TypeError('`cube` must be a numpy 3d array or a list of numpy 3d arrays')      

    if type(psf)==np.ndarray:
        if psf.ndim<2:
            raise TypeError('`psf` must be a numpy 2d array or a list of numpy 2d arrays')
    elif type(psf)==list:
        for i in range(0,len(psf)):
            if psf[i].ndim<2:
                raise TypeError('`psf` must be a numpy 2d array or a list of numpy 2d arrays')
    else:
        raise TypeError('`psf` must be a numpy 2d array or a list of numpy 2d arrays')
    

    if type(cube)==list:
        if not type(angs)==list :
            
                raise TypeError('When `cube` is a list `angs` and `psf` must be list of numpy arrays')
        elif not (len(psf)==len(cube) and len(angs)==len(cube)):
                raise TypeError('When `cube` is a list `angs` and `psf` should have the same size')
                
                
    if not type(pxscale)==float:
        raise TypeError('`pxscale` must be a float')
    if not type(minradius)==int:
        raise TypeError('`minradius` must be an integer')
    if not type(maxradius)==int:
        raise TypeError('`maxradius` must be an integer')
    if not (type(fwhm)==int or type(fwhm)==float) :
        raise TypeError('`fwhm` must be a float or an integer')        
        
        
    if not type(paramod)==list:
        paramod=[paramod]*len(model)
    if not type(asize)==list:
        asize=[asize]*len(model)        
    if not type(n_segments)==list:
        n_segments=[n_segments]*len(model)        
        
        
        
    def init(probi):
        global probcube
        probcube = probi
    

    #Cubes of residuals estimation

    def Resesti(subcube,angs,model=['APCA'],fwhm=5, paramod=20, asize=5, n_segments=1,delta_rot=0.5,numcore=1):    
        

        
        if model=='APCA':
                print("Annular PCA estimation") 
                cube_out, residuals_cube_, frame_fin = vip.pca.pca_annular(subcube, angs, fwhm=fwhm, ncomp=paramod, asize=asize, 
                              delta_rot=delta_rot, svd_mode='lapack', full_output=True, n_segments=n_segments, nproc=numcore,verbose=False)
                if showplot:
                    plot_frames(frame_fin,title='Annular PCA', colorbar=True,ang_scale=True, axis=False,pxscale=pxscale,ang_legend=True,show_center=True)

        elif model=='NMF':
                print("NMF estimation") 
                H_resh, recon_resh, array_out, residuals_cube_, frame_fin = vip.nmf.nmf(subcube, angs, ncomp=paramod, max_iter=100, random_state=0, mask_center_px=None,full_output=True,verbose=False)
                if showplot:
                    plot_frames(frame_fin,title='NMF', colorbar=True,ang_scale=True, axis=False,pxscale=pxscale,ang_legend=True,show_center=True)

        elif model=='LLSG':
                print("LLSGestimation") 
                list_l, list_s, list_g, f_l, frame_fin, f_g = vip.llsg.llsg(subcube, angs, fwhm, rank=paramod,asize=asize, thresh=1,n_segments=n_segments, max_iter=40, random_seed=10, nproc=numcore,full_output=True,verbose=False)
                res_s=np.array(list_s)
                residuals_cube_=res_s[0]
                if showplot:
                    plot_frames(frame_fin,title='LLSG', colorbar=True,ang_scale=True, axis=False,pxscale=pxscale,ang_legend=True,show_center=True)

        elif model=='PCA':
                print("PCA estimation") 
                frame_fin, pcs, recon, residuals_cube, residuals_cube_ = vip.pca.pca(subcube, angs, ncomp=paramod, fwhm=fwhm,full_output=True,verbose=False)
                if showplot:
                    plot_frames(frame_fin,title='PCA', colorbar=True,ang_scale=True, axis=False,pxscale=pxscale,ang_legend=True,show_center=True)

        elif model=='LOCI':
                print("LOCI estimation") 
                residuals_cube, residuals_cube_,frame_fin=vip.leastsq.xloci(subcube, angs, scale_list=None, fwhm=fwhm,asize=asize, n_segments=n_segments, nproc=numcore, optim_scale_fact=paramod,verbose=False,full_output=True)
                if showplot:
                    plot_frames(frame_fin,title='LOCI', colorbar=True,ang_scale=True, axis=False,pxscale=pxscale,ang_legend=True,show_center=True)
                
        return residuals_cube_,frame_fin

    if not type(resicube)==np.ndarray:

        if type(cube)==np.ndarray:
        
            n_segments_index=0
            asize_index=0
            paramod_index=0
        
            for j in range(0,len(model)):
                    if n_segments_index==0:
                        resicube,frame_mod=Resesti(cube,angs,model=model[j],fwhm=fwhm, paramod=paramod[paramod_index], asize=asize[asize_index], n_segments=n_segments[n_segments_index],delta_rot=delta_rot,numcore=numcore)
                    else:
                        resicube_temp,frame_temp=Resesti(cube,angs,model=model[j],fwhm=fwhm, paramod=paramod[paramod_index], asize=asize[asize_index], n_segments=n_segments[n_segments_index],delta_rot=delta_rot,numcore=numcore)
                        resicube=np.concatenate((resicube,resicube_temp),axis=0)
                        if n_segments_index==1:
                            frame_mod=np.append([frame_mod],[frame_temp],axis=0)
                        else:
                            frame_mod=np.append(frame_mod,[frame_temp],axis=0)
                    n_segments_index+=1
                    asize_index+=1
                    paramod_index+=1
            cubesize=cube[:,:,:].shape[0]

        elif modtocube==False:

            for i in range(0,len(cube)):
                n_segments_index=0
                asize_index=0
                paramod_index=0

                for j in range(0,len(model)):
                    if i==0 and j==0:
                        resicube,frame_mod=Resesti(cube[i],angs[i],model=model[j],fwhm=fwhm, paramod=paramod[paramod_index], asize=asize[asize_index], n_segments=n_segments[n_segments_index],delta_rot=delta_rot,numcore=numcore)
                        cubesize=cube[i].shape[0]
                    else:
                        resicube_temp,frame_temp=Resesti(cube[i],angs[i],model=model[j],fwhm=fwhm, paramod=paramod[paramod_index], asize=asize[asize_index], n_segments=n_segments[n_segments_index],delta_rot=delta_rot,numcore=numcore)
                        resicube=np.concatenate((resicube,resicube_temp),axis=0)
                        cubesize=np.append(cubesize,cube[i].shape[0])
                        if i==0 and j==1:
                            frame_mod=np.append([frame_mod],[frame_temp],axis=0)
                        else:
                            frame_mod=np.append(frame_mod,[frame_temp],axis=0)
                    n_segments_index+=1
                    asize_index+=1
                    paramod_index+=1
        else:      
            n_segments_index=0
            asize_index=0
            paramod_index=0
            for i in range(0,len(model)):

                for j in range(0,len(cube)):
                    if i==0 and j==0:
                        resicube,frame_mod=Resesti(cube[j],angs[j],model=model[i],fwhm=fwhm, paramod=paramod[paramod_index], asize=asize[asize_index], n_segments=n_segments[n_segments_index],delta_rot=delta_rot,numcore=numcore)
                        cubesize=cube[i].shape[0]
                    else:
                        resicube_temp,frame_temp=Resesti(cube[j],angs[j],model=model[i],fwhm=fwhm, paramod=paramod[paramod_index], asize=asize[asize_index], n_segments=n_segments[n_segments_index],delta_rot=delta_rot,numcore=numcore)
                        resicube=np.concatenate((resicube,resicube_temp),axis=0)
                        cubesize=np.append(cubesize,cube[i].shape[0])
                        if i==0 and j==1:
                            frame_mod=np.append([frame_mod],[frame_temp],axis=0)
                        else:
                            frame_mod=np.append(frame_mod,[frame_temp],axis=0)
                n_segments_index+=1
                asize_index+=1
                paramod_index+=1

    
    nmod=len(model)
    probcube = mp.Array(c.c_double, resicube.shape[0]*resicube.shape[1]*resicube.shape[2])

    fiterr=[]
    delta=[]
    distrisel=[]
    mixratio=[]
    
    #Probability map estimation
    

    pool = mp.Pool(processes=numcore,initializer=init, initargs=(probcube,))
    results=[]
    for inner_radius in range(minradius,maxradius):
        result = pool.apply_async(RSM, args=(resicube,psf,inner_radius,nmod,cubesize,distri,distrifit,interval,ns,crop,fulloutput,verbose))
        results.append(result)
    [result.wait() for result in results]

    pool.close()
    for result in results:
        delta.append(result.get()[1])
        fiterr.append(result.get()[2])
        distrisel.append(result.get()[3])
        if distri=='mix':
            mixratio.append(result.get()[4])
     
          
    probtemp = np.frombuffer(probcube.get_obj())
    probmapfin=probtemp.reshape(((resicube.shape[0]),resicube.shape[1],resicube.shape[2]))
    probmapfin=vip.preproc.cube_collapse(probmapfin, mode=colmode)
    plot_frames(probmapfin)

    

    if fulloutput:
        
        if distri=='mix':
            return probmapfin,delta,distrisel,fiterr,frame_mod,resicube,mixratio
        else:
            return probmapfin,delta,distrisel,fiterr,frame_mod,resicube
    else:   
        return probmapfin

