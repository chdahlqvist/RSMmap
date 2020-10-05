
import vip_hci as vip
import os
from hci_plot import plot_frames
import sys
sys.path.append('../../')   # append the path to PyRSM
import PyRSM


os.chdir('/Users/..')

psfnaco = './naco_betapic_psf.fits'
cube = './naco_betapic_cube.fits'
angle = './naco_betapic_pa.fits'

angs = vip.fits.open_fits(angle)
cube_orig = vip.fits.open_fits(cube)
psf = vip.fits.open_fits(psfnaco)
pxscale_naco = vip.conf.VLT_NACO['plsc']

# Measure the FWHM by fitting a 2d Gaussian to the core of the PSF

fit = vip.var.fit_2dgaussian(psf, crop=True, cropsize=9, debug=True)
fwhm = float((fit.fwhm_y+fit.fwhm_x)/2)

# Normalize the PSF flux to one in the FWHM aperture

psfn = vip.metrics.normalize_psf(psf, fwhm, size=19)
psf=  vip.preproc.frame_crop(psfn,11)

# Center the cube (odd shape)

centy,centx=vip.var.frame_center(cube_orig[0])
cube_recentered, shy1, shx1 = vip.preproc.cube_recenter_2dfit(cube_orig, 
                                                        xy= (int(centx)+1,int(centy)+1),
                                                        fwhm=fwhm, nproc=1, subi_size=5, 
                                                        model='gauss', negative=True, full_output=True,
                                                        debug=False)

# Create PyRSM class object

d=PyRSM(fwhm,minradius=5,maxradius=45,pxscale=pxscale_naco,ncore=1)

# Add a cube

d.add_cube(psf,cube_recentered, angs)

# Add several methods

d.add_method('APCA', flux=True, distri='auto', ncomp=20, var='Time', delta_rot=0.5, asize=5)
d.add_method('NMF', flux=True, distri='auto', ncomp=20, var='Time')
d.add_method('LLSG', flux=True, distri='auto', rank=5, var='Time', delta_rot=0.5, asize=5)

# Estimate the cube of likelihoods

d.lik_esti(verbose=True)     

# Estimate final RSM map
     
d.probmap_esti(estimator='Forward-Backward',colmode='median')

# Plot final probability map

plot_frames(d.probmap)