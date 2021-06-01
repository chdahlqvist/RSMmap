# PyRSM 

**PyRSM** is a python package for exoplanets detection which applies the **Regime Switching Model** (RSM) framework on ADI (and ADI+SDI) sequences (see Dahlqvist et al. A&A, 2020, 633, A95).The RSM map algorithm relies on one or several **PSF subtraction techniques** to process one or multiple **ADI sequences** (or ADI+SDI) before computing a final probability map. Considering the large set of parameters needed for the computation of the RSM detection map (parameters for the selected PSF-subtraction techniques as well as the RSM algorithm itself), a parameter selection framework called **auto-RSM** (Dahlqvist et al., 2021 in prep) is proposed to automatically select the optimal parametrization. The proposed multi-step parameter optimization framework can be divided into three main steps, (i) the selection of the optimal set of parameters for the considered PSF-subtraction techniques, (ii) the optimization of the RSM approach parametrization, and (iii) the selection of the optimal set of PSF-subtraction techniques and ADI sequences to be considered when generating the final detection map. 

The *add_cube* and *add_model* functions allows to consider several ADI or ADI+SDI sequences and models to generate
the cube of residuals used to compute the RSM map. The cube should be provided by the same instrument
or rescaled to a unique pixel size. A specific PSF should be provided for each cube. In the case of ADI+SDI, 
a single psf should be provided per cube (typically the PSF averaged over the set of frequencies, as the package 
does not acomodate yet 3-dimensionnal PSF). Five different models and two forward model variants are available. 
Each model can be parametrized separately. The function *like_esti* allows the estimation of a cube of 
likelihoods containing for each pixel and each frame the likelihood of being in the planetary or the speckle 
regime. These likelihood cubes are then used by the *probmap_esti* function to provide the final probability 
map based on the RSM framework. 

The second set of funtions regroups the four main functions used by the auto-RSM/auto-S/N framework.
The *opti_model* function allows the optimization of the PSF subtraction techniques parameters based on the 
minimisation of the average annulus-wise contrast. The *opti_RSM* function takes care of the optimization of the parameters 
of the RSM framework (all related to the computation of the likelihood associated to every pixels and frames). The
third function *opti_combination*, relies on a greedy selection algorithm to define the optimal set of 
ADI sequences and PSF-subtraction techniques to consider when generating the final detection map using the RSM
approach. Finally, the *opti_map* function allows to compute the final RSM detection map. The optimization of
the parameters can be done using the reversed parallactic angles, blurring potential planetary signals while
keeping the main characteristics of the speckle noise. An S/N map based code is also proposed and encompasses
the *opti_model*, the *opti_combination* and the *opti_map* functions. For the last two functions, the SNR 
parameter should be set to True.

A jupyter notebook tutorial as well as a dataset of Beta Pictoris B is provided in the folder example to test the PyRSM class. A list of parameters for the PyRSM class and for the main functions are given below:

## Setup and tutorial

The package may be installed via pip install using the command:

>pip install https://github.com/chdahlqvist/RSMmap/releases/download/0.2.0/PyRSM.tar.gz

A jupyter notebook tutorial as well as a test dataset of Beta Pictoris B is provided [here](https://github.com/chdahlqvist/RSMmap/tree/master/Example).


## PyRSM class

* fwhm: int
    Full width at half maximum for the instrument PSF
* minradius : int
    Center radius of the first annulus considered in the RSM probability
    map estimation. The radius should be larger than half 
    the value of the 'crop' parameter 
* maxradius : int
    Center radius of the last annulus considered in the RSM probability
    map estimation. The radius should be smaller or equal to half the
    size of the image minus half the value of the 'crop' parameter 
* pxscale : float
    Value of the pixel in arcsec/px. Only used for printing plots when
    'showplot=True' in like_esti. 
* ncore : int, optional
    Number of processes for parallel computing. By default ('ncore=1') 
    the algorithm works in single-process mode. 
* max_r_fm: int, optional
    Largest radius for which the forward model version of KLIP or LOCI
    are used, when relying on forward model versions of RSM. Forward model 
    versions of RSM have a higher performance at close separation, considering
    their computation time, their use should be restricted to small angular distances.
    Default is None, i.e. the foward model version are used for all considered
    angular distance.
* opti_mode: str, optional
    In the 'full-frame' mode, the parameter optimization is based on a reduced
    set of angular separations and a single global set of parameters is selected 
    (the one maximizing the global normalized average contrast). In 'annular' mode,
    a separate optimization is done for every consecutive annuli of width equal to 
    one FWHM and separated by a distance of one FWHM. For each annulus, a separate 
    optimal set of parameters is computed. Default is 'full-frame'.
* inv_ang: bool, optional
    If True, the sign of the parallactic angles of all ADI sequence is flipped for
    the entire optimization procedure. Default is True.
* opti_type: str, optional
    'Contrast' for an optimization based on the average contrast and 'RSM' for
    an optimization based on the ratio of the peak probability of the injected
    fake companion on the peak (noise) probability in the remaining of the 
    considered annulus (much higher computation time). Default is 'Contrast'.
* trunc: int, optional
    Maximum angular distance considered for the full-frame parameter optimization. Defaullt is None.
* imlib : str, optional
    See the documentation of the 'vip_hci.preproc.frame_rotate' function.
* interpolation : str, optional
    See the documentation of the 'vip_hci.preproc.frame_rotate' function. 


## add_cube

* psf : numpy ndarray 2d
    2d array with the normalized PSF template, with an odd shape.
    The PSF image must be centered wrt to the array! Therefore, it is
    recommended to run the function 'normalize_psf' to generate a 
    centered and flux-normalized PSF template.
* cube : numpy ndarray, 3d or 4d
    Input cube (ADI sequences), Dim 1 = temporal axis, Dim 2-3 = spatial axis
    Input cube (ADI + SDI sequences), Dim 1 = temporal axis, Dim 2=wavelength
    Dim 3-4 = spatial axis     
* pa : numpy ndarray, 1d
    Parallactic angles for each frame of the ADI sequences. 
* scale_list: numpy ndarray, 1d, optional
    Scaling factors in case of IFS data (ADI+mSDI cube). Usually, the
    scaling factors are the central channel wavelength divided by the
    shortest wavelength in the cube (more thorough approaches can be used
    to get the scaling factors). This scaling factors are used to re-scale
    the spectral channels and align the speckles. Default is None
    
## add_model

* model : str
    Selected ADI-based post-processing techniques used to 
    generate the cubes of residuals feeding the Regime Switching model.
    'APCA' for annular PCA, NMF for Non-Negative Matrix Factorization, LLSG
    for Local Low-rank plus Sparse plus Gaussian-noise decomposition, LOCI 
    for locally optimized combination of images and'KLIP' for Karhunen-Loeve
    Image Projection. There exitsts a foward model variant of KLIP and LOCI called 
    respectively 'FM KLIP' and 'FM LOCI'.
* delta_rot : float, optional
    Factor for tunning the parallactic angle threshold, expressed in FWHM.
    Default is 0.5 (excludes 0.5xFHWM on each side of the considered frame).
* delta_sep : float, optional
    The threshold separation in terms of the mean FWHM (for ADI+mSDI data).
    Default is 0.1.
* asize : int, optional
    Width in pixels of each annulus.When a single. Default is 5. 
* n_segments : int, optional
    The number of segments for each annulus. Default is 1 as we work annulus-wise.
* ncomp : int, optional
    Number of components used for the low-rank approximation of the 
    speckle field with 'APCA', 'KLIP' and 'NMF'. Default is 20.
* rank : int, optional        
    Expected rank of the L component of the 'LLSG' decomposition. Default is 5.
* tolerance: float, optional
    Tolerance level for the approximation of the speckle field via a linear 
    combination of the reference images in the LOCI algorithm. Default is 1e-2.
* intensity: str, optionnal
    If 'Pixel', the intensity parameter used in the RSM framework is computed
    pixel-wise via a gaussian maximum likelihood by comparing the set of observations
    and the PSF or the forward model PSF in the case of 'FM KLIP' and 'FM LOCI'.
    If 'Annulus', the intensity parameter is estimated annulus-wise and defined as
    a multiple of the annulus residual noise variance. If multiple multiplicative paramters
    are provided in PyRSM init (multi_factor), the multiplicative factor applied to the noise
    variance is selected via the maximisation of the total likelihood of the regime switching
    model for the selected annulus. Default is 'Annulus'.
* interval: list of float or int, optional
    List of values taken by the delta parameter defining, when mutliplied by the 
    standard deviation, the strengh of the planetary signal in the Regime Switching model.
    Default is [1]. The different delta paramaters are tested and the optimal value
    is selected via maximum likelmihood.
* distri: str, optional
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
* var: str, optional
    Model used for the residual noise variance estimation. Five different approaches
    are proposed: 'ST', 'SM', 'F', 'FM', and 'T'. While all six can be used when 
    intensity='Annulus', only the last three can be used when intensity='Pixel'. 
    When using ADI+SDI dataset only 'F' and 'FM' can be used. Default is 'ST'.
    
    'ST': consider every frame and pixel in the selected annulus with a 
    width equal to asize (default approach)
    
    'SM': consider for every pixels a segment of the selected annulus with 
    a width equal to asize. The segment is centered on the selected pixel and has
    a size of three FWHM. A mask of one FWHM is applied on the selected pixel and
    its surrounding. Every frame are considered.
    
    'F': consider the pixels in the selected annulus with a width equal to asize
    but separately for every frame.
    
    'FM': consider the pixels in the selected annulus with a width 
    equal to asize but separately for every frame. Apply a mask one FWHM 
    on the selected pixel and its surrounding.
    
    'T': rely on the method developped in PACO to estimate the 
    residual noise variance (take the pixels in a region of one FWHM arround 
    the selected pixel, considering every frame in the derotated cube of residuals 
    except for the selected frame)
    
* distrifit: bool, optional
    If true, the estimation of the mean and variance of the selected distribution
    is done via an best fit on the empirical distribution. If False, basic 
    estimation of the mean and variance using the set of observations 
    contained in the considered annulus, without taking into account the selected
    distribution.
    
* modtocube: bool, optional
    Parameter defining if the concatenated cube feeding the RSM model is created
    considering first the model or the different cubes. If 'modtocube=False',
    the function will select the first cube then test all models on it and move 
    to the next one. If 'modtocube=True', the model will select one model and apply
    it to every cubes before moving to the next model. Default is True.
* crop_size: int, optional
    Part of the PSF tempalte considered is the estimation of the RSM map
* crop_range: int, optional
    Range of crop sizes considered in the estimation of the RSM map, starting with crop_size
    and increasing the crop size incrementally by 2 pixels up to a crop size of 
    crop_size + 2 x (crop_range-1).
* ini_esti: int, optional
    Number of loss function computations (average contrast) to initialize the Gaussian 
    process used during the Bayesian optimization of the PSF-subtraction technique parameters
    (APCA, LOCI, KLIP FM and LOCI FM). Default is 40.
* opti_bound: list, optional
    List of boundaries used for the parameter optimization. 
        - For APCA: [[L_ncomp,U_ncomp],[L_nseg,U_nseg],[L_delta_rot,U_delta_rot]]
          Default is [[15,45],[1,4],[0.25,1.5]]
        - For NMF: [[L_ncomp,U_ncomp]]
          Default is [[2,20]]
        - For LLSG: [[L_ncomp,U_ncomp],[L_nseg,U_nseg]]
          Default is [[1,10],[1,4]]
        - For LOCI: [[L_tolerance,U_tolerance],[L_delta_rot,U_delta_rot]]
          Default is [[1e-3,1e-2],[0.25,1.5]]
        - For FM KLIP: [[L_ncomp,U_ncomp],[L_delta_rot,U_delta_rot]]
          Default is [[15,45],[0.25,1.5]]
        - For FM LOCI: [[L_tolerance,U_tolerance],[L_delta_rot,U_delta_rot]]
          Default is [[1e-3,1e-2],[0.25,1.5]]
    with L_ the lower bound and U_ the Upper bound.   
    
    
## like_esti

* showplot: bool, optional
    If True, provides the plots of the final residual frames for the selected 
    ADI-based post-processing techniques along with the final RSM map. Default is False.
* fulloutput: bool, optional
    If True, provides the selected distribution, the fitness erros and the mixval 
    (for distri='mix') for every annulus in respectively obj.distrisel, obj.fiterr
    and obj.mixval (the length of these lists are equall to maxradius - minradius, the
    size of the matrix for each annulus depends on the approach selected for the variance
    estimation, see var in add_model)
* verbose : bool, optional
    If True prints intermediate info. Default is True.
    
## prob_esti

* modtocube: bool, optional
    Parameter defining if the concatenated cube feeding the RSM model is created
    considering first the model or the different cubes. If 'modtocube=False',
    the function will select the first cube then test all models on it and move 
    to the next one. If 'modtocube=True', the model will select one model and apply
    it to every cubes before moving to the next model. Default is True.
* ns: float , optional
     Number of regime switches. Default is one regime switch per annulus but 
     smaller values may be used to reduce the impact of noise or disk structures
     on the final RSM probablity map.
* sel_crop: list of int or None, optional
    Selected crop sizes from proposed crop_range (crop size = crop_size + 2 x (sel_crop)).
    A specific sel_crop should be provided for each mode. Default is crop size = [crop_size]
* estimator: str, optional
    Approach used for the probability map estimation either a 'Forward' model
    (approach used in the original RSM map algorithm) which consider only the 
    past observations to compute the current probability or 'Forward-Backward' model
    which relies on both past and future observations to compute the current probability
* colmode:str, optional
    Method used to generate the final probability map from the three-dimensionnal cube
    of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
    the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
* ann_center:int, optional
    Selected annulus if the probabilities are computed for a single annulus 
    (Used by the optimization framework). Default is None
* sel_cube: list of arrays,optional
    List of selected PSF-subtraction techniques and ADI sequences used to generate 
    the final probability map. [[i1,j1],[i2,j2],...] with i1 the first considered PSF-subtraction
    technique and j1 the first considered ADI sequence, i2 the second considered PSF-subtraction
    technique, etc. Default is None whih implies that all PSF-subtraction techniques and all
    ADI sequences are used to compute the final probability map.
    
## opti_model

* maxiter: int, optional
    Maximum number of iterations of the Bayesian optimization algorithm for 
    APCA, LOCI, KLIP FM and LOCI FM. Default is 40.
* filt: True, optional
    If True, a Hampel Filter is applied on the set of parameters for the annular mode
    in order to avoid outliers due to potential bright artefacts.
    
## opti_RSM

* estimator: str, optional
    Approach used for the probability map estimation either a 'Forward' model
    (approach used in the original RSM map algorithm) which consider only the 
    past observations to compute the current probability or 'Forward-Backward' model
    which relies on both past and future observations to compute the current probability
* colmode:str, optional
    Method used to generate the final probability map from the three-dimensionnal cube
    of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
    the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
    
## opti_combination

* estimator: str, optional
    Approach used for the probability map estimation either a 'Forward' model
    (approach used in the original RSM map algorithm) which consider only the 
    past observations to compute the current probability or 'Forward-Backward' model
    which relies on both past and future observations to compute the current probability
* colmode:str, optional
    Method used to generate the final probability map from the three-dimensionnal cube
    of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
    the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
* threshold: bool, optional
    When True a radial treshold is computed on the final detection map with parallactic angles reversed.
    For a given angular separation, the radial threshold is defined as the maximum probability observed 
    within the annulus. The radia thresholds are checked for outliers and smoothed via a Hampel filter.
    Only used when relying on the auto-RSM framework. Default is True.
* contrast_sel: str,optional
    Contrast and azimuth definition for the optimal likelihood cubes/ residuall cubes selection.
    If 'Max' ('Min' or 'Median'), the largest (smallest or median) contrast obtained during the 
    PSF-subtraction techniques optimization will be chosen along the corresponding 
    azimuthal position for the likelihood cubes selection. Default is 'Max'.
* combination: str,optional
    Type of greedy selection algorithm used for the selection of the optimal set of cubes 
    of likelihoods/cubes of residuals (either 'Bottom-Up' or 'Top-Down'). For more details
    see Dahlqvist et al. (2021). Default is 'Bottom-Up'.
* SNR: bool,optional
    If True, the auto-S/N framework is used, resulting in an optimizated final S/N map when using 
    subsequently the opti_map. If False the auto-RSM framework is used, providing an optimized
    probability map when using subsequently the opti_map.
    
## opti_map

* estimator: str, optional
    Approach used for the probability map estimation either a 'Forward' model
    (approach used in the original RSM map algorithm) which consider only the 
    past observations to compute the current probability or 'Forward-Backward' model
    which relies on both past and future observations to compute the current probability
* colmode:str, optional
    Method used to generate the final probability map from the three-dimensionnal cube
    of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
    the 'median' of the 'max' value of the probabilities along the time axis. Default is 'median'.
* threshold: bool, optional
    When True the radial treshold is computed during the RSM_combination is applied on the
    final detection map with the original parallactic angles. Only used when relying on the auto-RSM
    framework. Default is True.
* Full: bool,optional
    If True, the entire set of ADI-sequences and PSF-subtraction techniques are used to 
    generate the final detection map. If performed after RSM_combination, the obtained optimal set
    is repkaced by the entire set of cubes. Please make ure you have saved the optimal set
    via the save_parameter function. Default is 'False'.
* SNR: bool,optional
    If True, the auto-S/N framework is used, resulting in an optimizated final S/N map when using 
    subsequently the opti_map. If False the auto-RSM framework is used, providing an optimized
    probability map when using subsequently the opti_map.
