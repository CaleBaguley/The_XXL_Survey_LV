import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import scipy.ndimage as ndimage
import copy as cp
import matplotlib.gridspec as gridspec

# Astropy imports
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb

# paramiters
object_id_col = 'Index'

Ra_col   = 'EXT_RA'
Dec_col  = 'EXT_DEC'

cutout_size = 4. # arcminuets

fits_files_parent_folder = "V4.3_Lotto_run/"

mosaic_source_file = "../../../../XXLN_mosaics/mosaics/XXLN_0.5_2.0_clean.fits"

output_files = "V4.3_Lotto_run/Paper_cutouts/{}.png"

minimum = [-0.05,-0.1]
stretch = [0.4,0.8]
Q=[3,2]

# sources to produce cutouts for
sources    = [ 7729    ,  8577    ,  28661   ,  30628   ,  65216   ,  1740    ] 
Ra         = [ 37.50285,  37.78788,  37.91137,  36.48753,  31.10109,  36.90965]
Dec        = [-4.42678 , -4.35591 , -4.57767 , -5.64082 , -6.32278 , -3.30016 ]
confidence = ["0.20"   , "0.289"  , "0.139"  , "0.50"   , "0.220"  , "0.530"  ]
conf_error = ["0.02"   , "0.008"  , "0.003"  , "0.01"   , "0.005"  , "0.007"  ]

source_file_name = ['Point_source_example', 'Background_example', 'Cluster_example', 'Nearby_group_example', 'Contaminated_example', 'C1_example']

# Crop image to center from current size to new smaller size
def Crop_to_Center(image, new_image_size, current_image_size):
    
    size_to_pix = len(image)/current_image_size
    image_center = len(image)/2
    min_cut = int(image_center - (new_image_size/2)*size_to_pix)
    max_cut = int(image_center + (new_image_size/2)*size_to_pix)
    
    return image[min_cut:max_cut, min_cut:max_cut]

# --- Optical data functions -------------------------------------------------------------------------------------------------

#Load in optical data
def load_hsc_images(source_id, folder = ''):
    
    #source_id = int(source_id)
    
    # load G band
    file_name = '{}Fits_files/{}_G.fits'.format(folder, source_id)
    fits_data = fits.open(file_name)
    image_G = fits_data[1].data
    image_G -= np.median(image_G)

    # load R band
    file_name = '{}Fits_files/{}_R.fits'.format(folder, source_id)
    fits_data = fits.open(file_name)
    image_R = fits_data[1].data
    image_R -= np.median(image_R)

    # load I band
    file_name = '{}Fits_files/{}_I.fits'.format(folder, source_id)
    fits_data = fits.open(file_name)
    image_I = fits_data[1].data
    image_I -= np.median(image_I)

    # load Z band
    file_name = '{}Fits_files/{}_Z.fits'.format(folder, source_id)
    fits_data = fits.open(file_name)
    image_Z = fits_data[1].data
    image_Z -= np.median(image_Z)
    
    return image_G, image_R, image_I, image_Z

#Create colour images from indevidual optical bands
def create_colour_images(G, R, I, Z, minimum = [0,0], stretch = [0.8,1.5], Q=[5,3]):

    imageGRI = make_lupton_rgb(I, R, G, minimum=minimum[0], stretch=stretch[0], Q=Q[0])
    imageRIZ = make_lupton_rgb(Z, I, R, minimum=minimum[1], stretch=stretch[1], Q=Q[1])
    
    return imageGRI, imageRIZ

# --- X-ray  data functions --------------------------------------------------------------------------------------------------

# Cutout image of source from X-ray data
def XXL_Cutout(ra, dec, field, wcs, edge_size = 5.):
    
    # Create cutout properties
    position = SkyCoord(ra, dec, unit="deg")
    size = units.Quantity((edge_size, edge_size), units.arcmin)
    
    # Return cutout as array
    return Cutout2D(field[0].data, position, size, wcs=wcs).data

# Calculate contours for ploting X-ray data
def XXL_Contours(image, bins = 500, contour_fractions = [   0.692, 0.841,   0.933, 0.977, 0.99375, 0.9985, 0.999767]):
    
    # min and max values in image, note min value limited to minimum of 0.001 to avoid divide by zero
    min_val = max(np.amin(image),0.001)
    max_val = np.amax(image)
    
    # bin data over logarithmic bins
    hist, bin_edges = np.histogram(image, bins = 10**np.linspace(np.log10(min_val),np.log10(max_val),bins))
    
    # make hist cumulative
    for i in range(1,len(hist)):
        hist[i] += hist[i-1]
    
    # normalise data
    hist = hist/hist[-1]
    
    # calculate threshold for each contour
    contours = []
    i = 0
    for frac in contour_fractions:
        # iterate till end of current hist value is larger than contour fraction
        while( i<len(hist) and hist[i] < frac ):
            i += 1
        
        # append bin center to contour
        contours.append((bin_edges[i]+bin_edges[i+1])/2)
    
    
    i = 0
    while i+1 < len(contours) and contours[i] < contours[i+1]:
        i += 1
    
    return contours[:i]
        
    
    
# --- Ploting functions ------------------------------------------------------------------------------------------------------

def Plot_Cutout(image, axs, title, size = 5., contour_image = None, contours = None, colours = None, cmap = None, y=1.05):
    
    # Plot image
    axs.imshow(image, origin = 'lower', extent = [-size/2, size/2, -size/2, size/2], cmap = cmap)
    
    # Only plot contours if given
    if(contour_image is not None):
        # get positions of each pixle relative to image
        positions = np.linspace(-size/2, size/2, len(contour_image))
        
        # if no contour colours are given use a grey scale with higher values being whiter
        if(colours is None):
            colours = np.linspace(0.6,1,len(contours)).astype(str)
        
        # plot contours
        axs.contour(positions, positions, contour_image, contours, colors = colours, origin = 'lower', alpha = 1, linewidths = 0.3)
    
    # Set plot properties
    axs.set_aspect('equal')
    axs.tick_params(bottom=True, top=True, left=True, right=True)

# --- Main calculation -------------------------------------------------------------------------------------------------------

# -- Get and setup XXL field image ----------------

# Load the XXL mosaics
print("Loading XXL field")
XXLN_field = fits.open(mosaic_source_file)

# Get the world cordinate system for the XXL field
wcs = WCS(XXLN_field[0].header)

# -- create images -------------------------------

for i in range(len(sources)):

    # -- Get X-ray images --
    
    # Get 5 by 5 cutout
    imageXXL = XXL_Cutout(Ra[i], Dec[i], XXLN_field, wcs, edge_size = cutout_size)
    
    # Apply power law to raw image to make low counts easier to see
    imageXXL_pow = 1-np.power(imageXXL+1,-0.7)
    
    # Smooth X-ray data
    imageXXL_smoothed = ndimage.gaussian_filter(imageXXL, sigma = 2, mode = 'constant')
    
    # Calculate contour values for ploting smoothed X-ray image
    #contours = XXL_Contours(imageXXL_smoothed)
    contours = np.median(imageXXL_smoothed)+np.power(2,np.linspace(-5,5,11))
    
    
    # -- Get optical images --
    
    # Load optical bands into seperate arrays
    G, R, I, Z = load_hsc_images(sources[i], folder = fits_files_parent_folder)
    
    #Combine optical bands into two images
    GRI, RIZ = create_colour_images(G, R, I, Z, minimum = minimum, stretch = stretch, Q=Q)
    
    # Cut image down
    GRI = Crop_to_Center(GRI, cutout_size, 5)
    
    # --- Ploting ---
    plt.figure()
    
    plt.imshow(GRI, origin = 'lower', extent = [-cutout_size/2, cutout_size/2, -cutout_size/2, cutout_size/2])
    
    # get positions of each pixle relative to image
    positions = np.linspace(-cutout_size/2, cutout_size/2, len(imageXXL_smoothed))
    
    
    # plot contours
    plt.contour(positions, positions, imageXXL_smoothed, contours, colors = 'white', origin = 'lower', alpha = 1, linewidths = 0.3)
    
    # Set plot properties
    plt.gca().set_aspect('equal')
    #plt.title("{}, {}".format(Ra[i], Dec[i]))
    plt.tick_params(bottom=True, top=True, left=True, right=True)
    
    plt.xticks(np.arange(-cutout_size/2, 0.5+cutout_size/2, 0.5))
    plt.yticks(np.arange(-cutout_size/2, 0.5+cutout_size/2, 0.5))
    
    plt.title("{}{}{}".format(confidence[i], u"\u00B1", conf_error[i]), fontsize = 20)
    
    print('saving {} to {}'.format(sources[i], output_files.format(source_file_name[i])))
    
    plt.savefig(output_files.format(source_file_name[i]))
    #plt.show()
    
    
    
    
