# PyRSM 

**PyRSM** is a python package for exoplanets detection which applies the **Regime Switching Model** (RSM) framework on ADI (and potentialy ADI+IFS) sequences (see Dahlqvist et al., A&A, 2020, 633, A95).
The RSM map algorithm relies on one or several **PSF subtraction techniques** to process one or multiple **ADI sequences** before computing a final probability map.

The package contains a class PyRSM regrouping four main methods. The *add_cube* and *add_model* methods allow to consider several ADI sequences and models (PSF subtraction techniques) to generate the cube of residuals used to compute the RSM map. The cube should be provided by the same instrument or rescaled to a unique pixel size. The class can be used with ADI and ADI+IFS. In the case of IFS data the data should be rescaled and cropped for each wavelength (using for example the vip cube_rescaling_wavelengths function) and included separately in the model via the *add_cube* method. A specific PSF should be provided for each cube. Five different models and two forward model versions are available (APCA, NMF, LLSG, KLIP, LOCI, FM KLIP and FM LOCI). Each model can be parametrized separately. The method *like_esti* allows the estimation of a cube of likelihoods containing for each pixel and each frame the likelihood of being in the planetary and speckle regime. These likelihoods cubes are then used by the *probmap_esti* method to provide the final probability map based on the RSM 
framework.

## Setup and tutorial

The package may be installed via pip install using the command:

>pip install https://github.com/chdahlqvist/RSMmap/archive/0.1.1.tar.gz

A jupyter notebook tutorial as well as a test dataset of Beta Pictoris B is provided [here](https://github.com/chdahlqvist/RSMmap/tree/master/Example). A list of parameters for the PyRSM class and for its four methods are given below.

## PyRSM class

* fwhm: int

    Full width at half maximum for the instrument PSF
* minradius : int

    Radius of center of the first annulus considered in the RSM probability
    map estimation. The radius should be larger than half 
    the value of the 'crop' parameter 
* maxradius : int

    Radius of the center of the last annulus considered in the RSM probability
    map estimation. The radius should be smaller or equal to half the
    size of the image minus half the value of the 'crop' parameter 
* interval: list of float or int, optional

    List of values taken by the delta parameter defining, when mutliplied by the 
    standard deviation, the strengh of the planetary signal in the Regime Switching model.
    Default is 1. The different delta paramaters are tested and the optimal value
    is selected via maximum likelmihood.
* pxscale : float

    Value of the pixel in arcsec/px. Only used for printing plots when
    showplot=True in like_esti. 
* ncore : int, optional

    Number of processes for parallel computing. By default ('ncore=1') 
    the algorithm works in single-process mode.  


## add_cube

* cube : numpy ndarray, 3d

    Input cube (ADI sequences), Dim 1 = temporal axis, Dim 2-3 = spatial axis
* angs : numpy ndarray, 1d

    Parallactic angles for each frame of the ADI sequences. 
* psf : numpy ndarray 2d

    2d array with the normalized PSF template, with an odd shape.
    The PSF image must be centered wrt to the array! Therefore, it is
    recommended to run the function "normalize_psf" to generate a 
    centered and flux-normalized PSF template.
    
## add_model

* model : str

    Selected ADI-based post-processing techniques used to 
    generate the cubes of residuals feeding the Regime Switching model.
    'APCA' for annular PCA, NMF for Non-Negative Matrix Factorization, LLSG
    for Local Low-rank plus Sparse plus Gaussian-noise decomposition, LOCI 
    for locally optimized combination of images and'KLIP' for Karhunen-Loeve
    Image Projection. There exists a foward model version of KLIP and LOCI called 
    respectively 'FM KLIP' and 'FM LOCI'.
* delta_rot : int, optional

    Factor for tunning the parallactic angle threshold, expressed in FWHM.
    Default is 0.5 (excludes 0.5xFHWM on each side of the considered frame).
* asize : int, optional

    Width in pixels of each annulus. Default is 5. 
* n_segments : int, optional

    The number of segments in each annulus. Default is 1, working annulus-wise.
* ncomp : int, optional

    Number of components used for the low-rank approximation of the 
    speckle field with 'APCA', 'KLIP', 'NMF' and 'FM KLIP'. Default is 20.
* rank : int, optional  

    Expected rank of the L component of the 'LLSG' decomposition. Default is 5.
* tolerance: float, optional

    Tolerance level for the approximation of the speckle field via a linear 
    combination of the reference images in the LOCI algorithm. Default is 1e-2.
* flux: boolean, optionnal

    If true the flux parameter within the regime switching framework is estimated
    via a gaussian maximum likelihood by comparing the set of observations
    and the PSF or the forward model PSF in the case of 'FM KLIP' and 'FM LOCI'.
    If False, the flux parameter is defined as a multiple of the annulus residual
    noise variance. The multiplicative parameter is selected via the maximisation
    of the total likelihood of the regime switching model for the selected annulus.
    Default is False.
* distri: str, optional

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
* var: str, optional

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
* distrifit: bool, optional

    If true, the estimation of the mean and variance of the selected distribution
    is done via an best fit on the empirical distribution. If False, basic 
    empirical estimation of the mean and variance using the set of observations 
    contained in the considered annulus, without taking into account the selected
    distribution.
* crop_size: int, optional

    Part of the PSF tempalte considered in the estimation of the RSM map
* crop_range: int, optional

    Range of crop sizes considered in the estimation of the RSM map, starting with crop_size
    and increasing the crop size incrementally by 2 pixels up to a crop size of 
    crop_size + 2 x (crop_range-1).  
    
    
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

* modthencube: bool, optional

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

    Selected crop sizes from proposed crop_range (selected crop size = crop_size + 2 x (sel_crop-1)).
    A specific sel_crop should be provided for each mode. Default is None which is equivalent to
    selected crop size = crop_size
* estimator: str, optional

    Approach used for the probability map estimation either a 'Forward' model
    (approach used in the original RSM map algorithm) which consider only the 
    past observations to compute the current probability or 'Forward-Backward' model
    which relies on both past and future observations to compute the current probability
* colmode:str, optional

    Method used to generate the final probability map from the three-dimensionnal cube
    of probabilities generated by the RSM approach. It is possible to chose between the 'mean',
    the 'median' or the 'max' value of the probabilities along the time axis. Default is 'median'.
