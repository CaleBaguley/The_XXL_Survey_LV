import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize as Norm_colours
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

# Astropy imports
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

# NOTE: Assumes that cutout overlaps a minimum of 4 bins
#       i.e: cutout size / bin size >= 1

XXL_region = [[33.56, 37.84],[-6.12, -3.18]]
Bin_num = [428, 296]

parent_folder = 'MBremer2018/'
source_file = 'source.csv'

save_file = 'Number_Counts_i_cut_24.csv'

i_mag_cut = 24

# paramiters
object_id_col = 'index'
Ra_col   = 'RA'
Dec_col  = 'DEC'

# cutout sizes
cutout_sizes = np.asarray([5,2,1])/60

# Colour col names
Band_cols = ['g_undeblended_kronflux_mag', 'r_undeblended_kronflux_mag', 'i_undeblended_kronflux_mag', 'z_undeblended_kronflux_mag'][::-1]
Band_err_cols = ['g_undeblended_kronflux_magerr', 'r_undeblended_kronflux_magerr', 'i_undeblended_kronflux_magerr', 'z_undeblended_kronflux_magerr'][::-1]
#Band_cols = ['g_cmodel_mag', 'r_cmodel_mag', 'i_cmodel_mag', 'z_cmodel_mag'][::-1]
#Band_err_cols = ['g_cmodel_magerr', 'r_cmodel_magerr', 'i_cmodel_magerr', 'z_cmodel_magerr'][::-1]
Colours = ['G', 'R', 'I', 'Z'][::-1]

# Cuts
cuts = [['i_undeblended_kronflux_mag',25], ['z_undeblended_kronflux_mag',27]]
#cuts = [['i_cmodel_mag',25], ['z_cmodel_mag',27]]

# Aditional plotting
Plot_Colour_Mag = True
Plot_Hsc_Objects_Over_Cutout = True

# --- Ra and Dec to bin indecies ------------------------------------------------------------------------------------------------------------

# Calcualte size of bins
Bin_size = [(XXL_region[0][1]-XXL_region[0][0])/Bin_num[0], (XXL_region[1][1]-XXL_region[1][0])/Bin_num[1]]

def Bin_ra_index(ra):
    return (0.5+(ra  - XXL_region[0][0])/Bin_size[0]).astype(int)

def Bin_dec_index(dec):
    return (0.5+(dec - XXL_region[1][0])/Bin_size[1]).astype(int)
    
# --- Plotting colour info ------------------------------------------------------------------------------------------------------------------

def Plot_Colour_Corner(Background_Objects_index, Forground_Objects_index, Object_data, title):
    
    fig, axs = plt.subplots(len(Band_cols), len(Band_cols))
    
    # Plot scatter over bottom left triangle with histograms along diagonal
    for i in range(len(Band_cols)):
        
        n, bins, patches = axs[i][i].hist(Object_data[Band_cols[i]][Background_Objects_index], bins = 20  , density = True, histtype = 'step', color = 'grey')
        axs[i][i].hist(                   Object_data[Band_cols[i]][Forground_Objects_index ], bins = bins, density = True, histtype = 'step', color = 'blue')
        
        x_min = np.nanmin(Object_data[Band_cols[i]][Forground_Objects_index])
        x_max = np.nanmax(Object_data[Band_cols[i]][Forground_Objects_index])
        
        x_range = x_max - x_min
        
        x_min -= 0.1*x_range
        x_max += 0.1*x_range
        
        axs[i][i].set_xlim(x_min, x_max)
        
        axs[-1][i].set_xlabel("{}".format(Colours[i]))
        
        for j in range(i):
            
            axs[j][i].axis('off')
            
            axs[i][j].scatter(Object_data[Band_cols[j]][Background_Objects_index],
                              Object_data[Band_cols[i]][Background_Objects_index] - Object_data[Band_cols[j]][Background_Objects_index],
                              c = 'grey', alpha = 0.5, s = 2)
                              
            axs[i][j].errorbar(Object_data[Band_cols[j]][Forground_Objects_index], # x
                               Object_data[Band_cols[i]][Forground_Objects_index] - Object_data[Band_cols[j]][Forground_Objects_index], # y
                               xerr = Object_data[Band_err_cols[j]][Forground_Objects_index], # x error
                               yerr = np.sqrt(np.square(Object_data[Band_err_cols[i]][Forground_Objects_index]) + np.square(Object_data[Band_err_cols[j]][Forground_Objects_index])), # y error
                               fmt = '.', c = 'blue')
            
            
            axs[i][j].set_ylabel("{} - {}".format(Colours[i], Colours[j]))
            
            x_min = np.nanmin(Object_data[Band_cols[j]][Forground_Objects_index])
            x_max = np.nanmax(Object_data[Band_cols[j]][Forground_Objects_index])
            
            x_range = x_max - x_min
            
            x_min -= 0.1*x_range
            x_max += 0.1*x_range
            
            y_min = np.nanmin(Object_data[Band_cols[i]][Forground_Objects_index] - Object_data[Band_cols[j]][Forground_Objects_index])
            y_max = np.nanmax(Object_data[Band_cols[i]][Forground_Objects_index] - Object_data[Band_cols[j]][Forground_Objects_index])
            
            y_range = y_max - y_min
            
            y_min -= 0.1*y_range
            y_max += 0.1*y_range
            
            axs[i][j].set_xlim(x_min, x_max)
            axs[i][j].set_ylim(y_min, y_max)
            
            axs[i][j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
    return fig
    

# --- Prepair hsc data ----------------------------------------------------------------------------------------------------------------------

print("Loading hsc data")
hsc_data = pd.read_csv("../../../../Data/hsc objects/Sample_2.csv")
print("Finished loading hsc data")

hsc_data.replace([-np.inf, np.inf], np.nan, inplace = True) 

# Add any flux or colour cuts to reduce hsc data here before working.
if(cuts is not None):
    for current in cuts:
        hsc_data.where(hsc_data[current[0]] < current[1], inplace = True)

# Get object ra and dec
hsc_ras  = hsc_data['ra'].to_numpy()
hsc_decs = hsc_data['dec'].to_numpy()

#create spacial bins for hsc objects
Bins = [[[] for j in range(Bin_num[1])] for i in range(Bin_num[0])]

#calculat bin for each source
bin_ra_index  = Bin_ra_index(hsc_ras)
bin_dec_index = Bin_dec_index(hsc_decs)

# Add object indecies to bins
for i in range(len(bin_ra_index)):
    # Check if object falls into bin
    if( 0 <= bin_ra_index[i] and bin_ra_index[i] < Bin_num[0] and 0 <= bin_dec_index[i] and bin_dec_index[i] < Bin_num[1]):
        Bins[bin_ra_index[i]][bin_dec_index[i]].append(i)


Bin_counts = np.zeros((Bin_num[0],Bin_num[1]),int)

for i in range(Bin_num[0]):
    for j in range(Bin_num[1]):
        Bin_counts[i,j] = len(Bins[i][j])

# --- Prepair cuttout location data -----------------------------------------------------------------------------------------------------------

# Tell pandas to treat infinite values as nan
pd.set_option("mode.use_inf_as_na", True)

print("Loading cutout data")
cutout_data = pd.read_csv('{}{}'.format(parent_folder,source_file))

# Get Ra and Dec of cutouts reddy
cutout_ra = cutout_data[Ra_col].to_numpy()
cutout_dec = cutout_data[Dec_col].to_numpy()

# Array to hold number of objects within cutout
counts = np.zeros((len(cutout_ra),len(cutout_sizes)))

#Used to convert count_ra and cutout_sizes to apropriate len(cutout_ra) by len(cutout_sizes) matices for calcualting bin values
A = np.ones((len(cutout_ra),len(cutout_sizes)))

# Each column contains one of the cutout_sizes
Formated_cutout_sizes = A*cutout_sizes

# Each row contains one of the cutout ra or dec 
Formated_cutout_ra  = (cutout_ra*A.transpose()).transpose()
Formated_cutout_dec = (cutout_dec*A.transpose()).transpose()

#Calculate the bin limits for each cutout
cutout_ra_bin_min  = Bin_ra_index(Formated_cutout_ra - Formated_cutout_sizes/2)
cutout_ra_bin_max  = Bin_ra_index(Formated_cutout_ra + Formated_cutout_sizes/2)
cutout_dec_bin_min = Bin_dec_index(Formated_cutout_dec - Formated_cutout_sizes/2)
cutout_dec_bin_max = Bin_dec_index(Formated_cutout_dec + Formated_cutout_sizes/2)

print("Matching hsc objects to cutouts")

# Holds id's of hsc sources matched to cutout region
matched_hsc_ids = [[[] for j in range(len(cutout_sizes))] for i in range(len(cutout_ra))]

# Loop over all cutouts
for cutout_index in range(len(cutout_ra)):

    ra = cutout_ra[cutout_index]
    dec = cutout_dec[cutout_index]

    # Loop over all cutout sizes
    for size_index, size in enumerate(cutout_sizes):
        
        # Check cutout doesn't extend outside of available area
        if(0 <= cutout_ra_bin_min[cutout_index, size_index] and cutout_ra_bin_max[cutout_index, size_index] < Bin_num[0] and 0 <= cutout_dec_bin_min[cutout_index, size_index] and cutout_dec_bin_max[cutout_index, size_index] < Bin_num[1]):
        
            # NOTE: This proces asumes that a cutout overlaps with a minimum of 4 bins
            
            # Grab all hsc source from bins wholy within cutout area
            # No need to test if object within cutout as bin is wholy within
            for i in range(cutout_ra_bin_min[cutout_index, size_index] + 1, cutout_ra_bin_max[cutout_index, size_index]):
                for j in range(cutout_dec_bin_min[cutout_index, size_index] + 1, cutout_dec_bin_max[cutout_index, size_index]):
                    
                    # Loop over all objects in bin and add index to matched list
                    for current in Bins[i][j]:
                        matched_hsc_ids[cutout_index][size_index].append(current)
            
            
            # Loop along top and bottom
            # No nead to check ra as bins already wholy within ra range
            for i in range(cutout_ra_bin_min[cutout_index, size_index] + 1, cutout_ra_bin_max[cutout_index, size_index]):
            
                # Loop through objects in bottom bin
                for current in Bins[i][cutout_dec_bin_min[cutout_index, size_index]]:
                    # If in cutout add to matched list
                    # Only need to check if object is above cut due to top of bin bening in cutout
                    if(dec - size/2 < hsc_decs[current]):
                        matched_hsc_ids[cutout_index][size_index].append(current)
                
                # Loop through objects in top bin
                for current in Bins[i][cutout_dec_bin_max[cutout_index, size_index]]:
                    # If in cutout add to matched list
                    # Only need to check if object is below cut due to bottom of bin bening in cutout
                    if(hsc_decs[current] < dec + size/2):
                        matched_hsc_ids[cutout_index][size_index].append(current)
                
            
            # Loop down sides
            # No nead to check dec as bins already wholy within dec range
            for j in range(cutout_dec_bin_min[cutout_index, size_index] + 1, cutout_dec_bin_max[cutout_index, size_index]):
            
                # Loop through objects on left bin
                for current in Bins[cutout_ra_bin_min[cutout_index, size_index]][j]:
                    # If in cutout add to matched list
                    # Only need to check if object is above cut due to top of bin bening in cutout
                    if(ra - size/2 < hsc_ras[current]):
                        matched_hsc_ids[cutout_index][size_index].append(current)
                
                # Loop through objects on right bin
                for current in Bins[cutout_ra_bin_max[cutout_index, size_index]][j]:
                    # If in cutout add to matched list
                    # Only need to check if object is above cut due to top of bin bening in cutout
                    if(hsc_ras[current] < ra + size/2):
                        matched_hsc_ids[cutout_index][size_index].append(current)
            # Corners
            
            # Bottom Left
            for current in Bins[cutout_ra_bin_min[cutout_index, size_index]][cutout_dec_bin_min[cutout_index, size_index]]:
                # If in cutout add to matched list
                # Only need to check against min ra and dec of cutout
                if(ra - size/2 < hsc_ras[current] and dec - size/2 < hsc_decs[current]):
                    matched_hsc_ids[cutout_index][size_index].append(current)
            
            # Bottom Right
            for current in Bins[cutout_ra_bin_max[cutout_index, size_index]][cutout_dec_bin_min[cutout_index, size_index]]:
                # If in cutout add to matched list
                # Only need to check against max ra and min dec of cutout
                if(hsc_ras[current] < ra + size/2 and dec - size/2 < hsc_decs[current]):
                    matched_hsc_ids[cutout_index][size_index].append(current)
            
            # Top Left
            for current in Bins[cutout_ra_bin_min[cutout_index, size_index]][cutout_dec_bin_max[cutout_index, size_index]]:
                # If in cutout add to matched list
                # Only need to check against min ra and max dec of cutout
                if(ra - size/2 < hsc_ras[current] and hsc_decs[current] < dec + size/2):
                    matched_hsc_ids[cutout_index][size_index].append(current)
            
            # Top Right
            for current in Bins[cutout_ra_bin_max[cutout_index, size_index]][cutout_dec_bin_max[cutout_index, size_index]]:
                # If in cutout add to matched list
                # Only need to check against max ra and dec of cutout
                if(hsc_ras[current] < ra + size/2 and hsc_decs[current] < dec + size/2):
                    matched_hsc_ids[cutout_index][size_index].append(current)
            
            
            # Get number of objects in cutout
            counts[cutout_index, size_index] = len(matched_hsc_ids[cutout_index][size_index])
            
        # If cutout does extend outside area set count to -1
        else:
            counts[cutout_index, size_index] = -1
            
# Calculate object densities
cutout_areas = np.asarray([(60*size)**2 for size in cutout_sizes])

# A is used as previously to make cutout_areas an len(cutout_ra) by len(cutout_sizes) with each column containing an area
densities = counts / (A*cutout_areas)

# --- Ouput results to file --------------------------------------------------------------------------------------------------------------------------------------------       

print("Saving to file")

# create data frame for output
Data_out = pd.DataFrame({object_id_col: cutout_data[object_id_col]})

# Add counts to data frame
# names of count columns
names = ["{} arcmin count".format(60*size) for size in cutout_sizes]

# Loop over sizes adding counts for each as a column
for i in range(len(cutout_sizes)):
    Data_out.insert(1+i, names[i], counts[:,i], allow_duplicates = True)

# Add densities to data frame
# names of densities column
names = ["{} arcmin densities".format(60*size) for size in cutout_sizes]

for i in range(len(cutout_sizes)):
    Data_out.insert(4+i, names[i], densities[:,i], allow_duplicates = True)

# Save to output file
Data_out.to_csv("{}{}".format(parent_folder, save_file))


# --- Plot colour data ---------------------------------------------------------------------------------------------------

if(Plot_Colour_Mag):
    print("Creating colour magnitude plots")

    for i in range(len(cutout_data)):
        if(counts[i,0] > 0 and counts[i,2] > 0):
            fig = Plot_Colour_Corner(matched_hsc_ids[i][0], matched_hsc_ids[i][2], hsc_data, "")
            
            source_id = cutout_data[object_id_col][i]
            ra  = cutout_data[Ra_col ][i]
            dec = cutout_data[Dec_col][i]
            
            title = "Source {}; Ra {:.5f}, Dec {:.5f} (undeblended kronflux magnitude)".format(source_id, ra, dec)
            fig.suptitle(title)
            
            fig.set_size_inches(18,7.8)
            
            # save fig and clear buffer
            
            fig.savefig("{}/Colour_Mag_Images/{:03d}_{}.png".format(parent_folder, i+1, cutout_data[object_id_col][i]), format='png', dpi = 300)
            plt.close(fig)

# --- Plot hsc objects over cutout images ------------------------------------------------------------------------------

# -- General functions --

# Crop image to center from current size to new smaller size
def Crop_to_Center(image, new_image_size, current_image_size):
    
    size_to_pix = len(image)/current_image_size
    image_center = len(image)/2
    min_cut = int(image_center - (new_image_size/2)*size_to_pix)
    max_cut = int(image_center + (new_image_size/2)*size_to_pix)
    
    return image[min_cut:max_cut, min_cut:max_cut]

#Load in optical data
def load_hsc_images(source_id, folder = ''):

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

# Cutout image of source from X-ray data
def XXL_Cutout(ra, dec, field, wcs, edge_size = 5.):
    
    # Create cutout properties
    position = SkyCoord(ra, dec, unit="deg")
    size = units.Quantity((edge_size, edge_size), units.arcmin)
    
    # Return cutout as array
    return Cutout2D(field[0].data, position, size, wcs=wcs).data
    
def Plot_Cutout(image, axs, title, size = 5., contour_image = None, contours = None, colours = None, cmap = None, y=1.05):
    
    # Plot image
    axs.imshow(image, origin = 'lower', extent = [-size/2, size/2, -size/2, size/2], cmap = cmap)
    
    # Only plot contours if given
    if(contour_image is not None):
        # get positions of each pixle relative to image
        positions = np.linspace(-size/2, size/2, len(contour_image))
        
        # if no contour colours are given use a grey scale with higher values being whiter
        if(colours is None):
            colours = np.linspace(0.6, 1,len(contours)).astype(str)
        
        # plot contours
        axs.contour(positions, positions, contour_image, contours, colors = colours, origin = 'lower', alpha = 1, linewidths = 0.3)
    
    # Set plot properties
    axs.set_aspect('equal')
    axs.set_title(title, y=y)
    axs.tick_params(bottom=True, top=True, left=True, right=True)


# -- Plotting code --

if(Plot_Hsc_Objects_Over_Cutout):

    # Load the XXL mosaics
    print("Loading XXL field")
    XXLN_field = fits.open("../../../../XXLN_mosaics/mosaics/XXLN_0.5_2.0_clean.fits")

    # Get the world cordinate system for the XXL field
    wcs = WCS(XXLN_field[0].header)

    print("Creating colour magnitude with object positions")
    for i in range(len(cutout_data[object_id_col])):
        if(counts[i,0] > 0 and counts[i,2] > 0):
            # get current object data
            source_id = cutout_data[object_id_col][i]
            ra  = cutout_data[Ra_col ][i]
            dec = cutout_data[Dec_col][i]
            
            # --- X-ray data analysis ---
        
            # Get 5 by 5 cutout
            imageXXL = XXL_Cutout(ra, dec, XXLN_field, wcs)
            
            # Check if the X-ray image contais data
            is_Xray_data = np.amax(imageXXL)>0
            
            # If there is X-ray data process it
            if(is_Xray_data):
            
                # Smooth X-ray data
                imageXXL_smoothed = ndimage.gaussian_filter(imageXXL, sigma = 2, mode = 'constant')
                
                # Calculate contour values for ploting smoothed X-ray image
                #contours = XXL_Contours(imageXXL_smoothed)
                contours = np.median(imageXXL_smoothed)+np.power(2,np.linspace(-5,5,11))
                
                # Crop to central 2by2
                # This is done here instead of before smoothing and contour calculations to match lotto images
                imageXXL_smoothed = Crop_to_Center(imageXXL_smoothed, 1.2, 5)
                
            # --- Optical data analysis ---
            
            # Load optical bands into seperate arrays
            G, R, I, Z = load_hsc_images(source_id, folder = parent_folder)
            
            
            # Reduce to central region
            R = Crop_to_Center(R, 1.2, 5)
            I = Crop_to_Center(I, 1.2, 5)
            Z = Crop_to_Center(Z, 1.2, 5)
            
            # Use power to make background more obveouse
            R = 1-np.power(R+1,-0.7)
            I = 1-np.power(I+1,-0.7)
            Z = 1-np.power(Z+1,-0.7)
            
            # --- Ploting ---
            fig, axs = plt.subplots(2,2, gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [1, 1]})
            
           
            
            # Plot cutouts with X-ray contours
            Plot_Cutout(I, axs[0][0], "I", size = 1.2, contour_image = imageXXL_smoothed, contours = contours, colours = 'red', cmap = 'Greys')
            Plot_Cutout(Z, axs[0][1], "Z", size = 1.2, contour_image = imageXXL_smoothed, contours = contours, colours = 'red', cmap = 'Greys')
            
            # Get Ra and Dec relative to image in arc minutes
            hsc_ras_image  = -60*(hsc_ras[matched_hsc_ids[i][2]]  - ra )
            hsc_decs_image =  60*(hsc_decs[matched_hsc_ids[i][2]] - dec)
            
            # Get magnitudes
            hsc_R_mag = hsc_data[Band_cols[2]][matched_hsc_ids[i][2]]
            hsc_I_mag = hsc_data[Band_cols[1]][matched_hsc_ids[i][2]].to_numpy(dtype = float)
            hsc_Z_mag = hsc_data[Band_cols[0]][matched_hsc_ids[i][2]]
            
            # Get magnitude errors
            hsc_R_mag_err = hsc_data[Band_err_cols[2]][matched_hsc_ids[i][2]]
            hsc_I_mag_err = hsc_data[Band_err_cols[1]][matched_hsc_ids[i][2]]
            hsc_Z_mag_err = hsc_data[Band_err_cols[0]][matched_hsc_ids[i][2]]
            
            cmap_name = 'spring'
            
            # Plot hsc objects over cutouts
            axs[0][0].scatter(hsc_ras_image, hsc_decs_image, c = hsc_I_mag, fc = 'none', cmap = cmap_name, s = 100)
            axs[0][1].scatter(hsc_ras_image, hsc_decs_image, c = hsc_I_mag, fc = 'none', cmap = cmap_name, s = 100)
            
            # Plot R - I colour magnitude diagram
            axs[1][0].scatter(hsc_data[Band_cols[1]][matched_hsc_ids[i][0]], hsc_data[Band_cols[2]][matched_hsc_ids[i][0]] - hsc_data[Band_cols[1]][matched_hsc_ids[i][0]], c = 'grey', alpha = 0.5, s = 2, zorder = 0)
            axs[1][0].errorbar(hsc_I_mag, hsc_R_mag - hsc_I_mag, xerr = hsc_I_mag_err, yerr = np.sqrt(np.square(hsc_R_mag_err) + np.square(hsc_I_mag_err)), fmt = 'none', c = 'black', zorder = 1)
            axs[1][0].scatter( hsc_I_mag, hsc_R_mag - hsc_I_mag, c = hsc_I_mag, cmap = cmap_name, s = 4, zorder = 2)
            
            axs[1][0].set_xlabel(Colours[1])
            axs[1][0].set_ylabel("{} - {}".format(Colours[2], Colours[1]))
            
            x_min = np.nanmin(hsc_I_mag)
            x_max = np.nanmax(hsc_I_mag)
            
            x_range = x_max - x_min
            
            x_min -= 0.1*x_range
            x_max += 0.1*x_range
            
            y_min = np.nanmin(hsc_R_mag - hsc_I_mag)
            y_max = np.nanmax(hsc_R_mag - hsc_I_mag)
            
            y_range = y_max - y_min
            
            y_min -= 0.1*y_range
            y_max += 0.1*y_range
            
            axs[1][0].set_xlim(x_min, x_max)
            axs[1][0].set_ylim(y_min, y_max)
            
            # Plot I - Z colour magnitude diagram
            axs[1][1].scatter(hsc_data[Band_cols[0]][matched_hsc_ids[i][0]], hsc_data[Band_cols[1]][matched_hsc_ids[i][0]] - hsc_data[Band_cols[0]][matched_hsc_ids[i][0]], c = 'grey', alpha = 0.5, s = 2, zorder = 0)
            axs[1][1].errorbar(hsc_Z_mag, hsc_I_mag - hsc_Z_mag, xerr = hsc_Z_mag_err, yerr = np.sqrt(np.square(hsc_I_mag_err) + np.square(hsc_Z_mag_err)), fmt = 'none', c = 'black', zorder = 1)
            axs[1][1].scatter( hsc_Z_mag, hsc_I_mag - hsc_Z_mag, c = hsc_I_mag, cmap = cmap_name, s = 4, zorder = 2)
            
            axs[1][1].set_xlabel(Colours[0])
            axs[1][1].set_ylabel("{} - {}".format(Colours[1], Colours[0]))
            
            x_min = np.nanmin(hsc_Z_mag)
            x_max = np.nanmax(hsc_Z_mag)
            
            x_range = x_max - x_min
            
            x_min -= 0.1*x_range
            x_max += 0.1*x_range
            
            y_min = np.nanmin(hsc_I_mag - hsc_Z_mag)
            y_max = np.nanmax(hsc_I_mag - hsc_Z_mag)
            
            y_range = y_max - y_min
            
            y_min -= 0.1*y_range
            y_max += 0.1*y_range
            
            axs[1][1].set_xlim(x_min, x_max)
            axs[1][1].set_ylim(y_min, y_max)
            
            title = "Source {}; Ra {:.5f}, Dec {:.5f} (undeblended kronflux magnitude)".format(source_id, ra, dec)
            fig.suptitle(title)
            
            fig.set_size_inches(14.5,10)
            
            # save fig and clear buffer
            fig.savefig("{}/Colour_Mag_Images/{:03d}_{}_Positions.png".format(parent_folder, i+1, cutout_data[object_id_col][i]), format='png', dpi = 400)
            plt.close(fig)
            
        



# Uncoment for testing purposes
'''    
print(Data_out)

for i in [0,2,3]:
    plt.scatter(60*hsc_ras[matched_hsc_ids[i][0]], 60*hsc_decs[matched_hsc_ids[i][0]])
    plt.scatter(60*hsc_ras[matched_hsc_ids[i][1]], 60*hsc_decs[matched_hsc_ids[i][1]])
    plt.scatter(60*hsc_ras[matched_hsc_ids[i][2]], 60*hsc_decs[matched_hsc_ids[i][2]])

    plt.plot([60*cutout_ra[i] -2.5, 60*cutout_ra[i] +2.5, 60*cutout_ra[i] +2.5, 60*cutout_ra[i] -2.5, 60*cutout_ra[i] -2.5],
             [60*cutout_dec[i]-2.5, 60*cutout_dec[i]-2.5, 60*cutout_dec[i]+2.5, 60*cutout_dec[i]+2.5, 60*cutout_dec[i]-2.5])

    plt.plot([60*cutout_ra[i] -1, 60*cutout_ra[i] +1, 60*cutout_ra[i] +1, 60*cutout_ra[i] -1, 60*cutout_ra[i] -1],
             [60*cutout_dec[i]-1, 60*cutout_dec[i]-1, 60*cutout_dec[i]+1, 60*cutout_dec[i]+1, 60*cutout_dec[i]-1])

    plt.plot([60*cutout_ra[i] -0.5, 60*cutout_ra[i] +0.5, 60*cutout_ra[i] +0.5, 60*cutout_ra[i] -0.5, 60*cutout_ra[i] -0.5],
             [60*cutout_dec[i]-0.5, 60*cutout_dec[i]-0.5, 60*cutout_dec[i]+0.5, 60*cutout_dec[i]+0.5, 60*cutout_dec[i]-0.5])

    plt.xlim(60*cutout_ra[i] -3, 60*cutout_ra[i] +3)
    plt.ylim(60*cutout_dec[i]-3, 60*cutout_dec[i]+3)

    plt.axis('equal')

    plt.show()
'''














