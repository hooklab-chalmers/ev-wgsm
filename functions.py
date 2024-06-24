import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import skimage as sci
import tifffile as tf
import pandas as pd
import cv2
import ast
import re
from skimage.transform import warp, AffineTransform
from mie_coated import *
from scipy.ndimage import (gaussian_filter, binary_dilation, generate_binary_structure)
from scipy.signal import convolve2d
from skimage.morphology import (erosion, dilation, opening, closing)
from skimage.morphology import disk  
from skimage.segmentation import expand_labels
from skimage import measure
import scipy.optimize as opt
from scipy.optimize import curve_fit
#from stardist.models import StarDist2D 
#from csbdeep.utils import normalize

### File writing/opening

def write_files(data, date, folder, filename):
    data.to_csv(f'Results/{date}{folder}/{filename}.csv', sep=';')
    data.to_pickle(f'Results/{date}{folder}/{filename}.pkl')

def open_file(date, folder, filename):
    return pd.read_pickle(f'Results/{date}{folder}/{filename}.pkl')

def open_csv(date, filename):
    return pd.read_csv(f'Results/{date}/{filename}.csv', sep=';', index_col=[0], \
        converters={'Coordinates': check_for_tuple, 'CD9 Coordinates': check_for_tuple, 'CD41 Coordinates': check_for_tuple, \
           'Biotin Coordinates': check_for_tuple,\
                'Iodixanol series': convert_string_to_array, 'Glycerol series': convert_string_to_array, 
                'Iodixanol series normalized': convert_string_to_array, 'Glycerol series normalized': convert_string_to_array,
                'Fluorescence time series': convert_string_to_array, 'Scattering time series': convert_string_to_array})

### Particle detection

def load_frames(dims, Ns_samples, Nf_samples, Nm_samples, Nm_start, positions, params, base_path, scat_frames, fluor_frames):
    height, width = dims[0], dims[1]
    sigma, r = params[0], params[1]
    
    I0 = np.zeros((Ns_samples, len(positions), height, width)).astype(np.uint16) #Reference frames
    I = np.zeros((Ns_samples, len(positions), height, width)).astype(np.uint16) #Particle frames
    Ibg = np.zeros((Ns_samples, len(positions), height, width)).astype(float) #Particle detection frames
    Ig = np.zeros((Nm_samples, len(positions), height, width)).astype(np.uint16) #Glycerol frames
    Ii = np.zeros((Nm_samples, len(positions), height, width)).astype(np.uint16) #Iodixanol frames
    if Nf_samples is not None:
        If = np.zeros((Nf_samples, len(positions), height, width)).astype(np.uint16) #Fluorescence frames
        Ifbg = np.zeros((Nf_samples, len(positions), height, width)).astype(float) #Fluorescence detection frames
    else:
        If = np.zeros((1, len(positions), height, width)).astype(np.uint16) #Fluorescence frames
        Ifbg = np.zeros((1, len(positions), height, width)).astype(float) #Fluorescence detection frames

    # Loop through the chip positions
    for i, pos in enumerate(positions):
        # Open aligned frames and store them in the arrays
        frames = tf.imread(f'{base_path}/{scat_frames}_pos{pos}.tif')

        I0[:, i] = frames[:Ns_samples]
        I[:, i] = frames[1:Ns_samples+1]

        Ig[:, i] = frames[Nm_start:Nm_start+Nm_samples]
        Ii[:, i] = frames[Nm_start+Nm_samples:]

        # Open fluorescence frames and store them in the array
        if Nf_samples is not None:
            frames = tf.imread(f'{base_path}/{fluor_frames}_pos{pos}.tif')
            If[:, i] = frames.copy()

    # Make the particle detection frames
    for i in range(Ibg.shape[0]):
        for j, pos in enumerate(positions):
            Ibg[i, j] = reference_removal(I[i, j], I0[i, j], sigma, r)
    if Nf_samples is not None:
        for i in range(Ifbg.shape[0]):
            for j, pos in enumerate(positions):
                Ifbg[i, j] = reference_removal_fluor(If[i, j], 20, 5)

    return I0, I, Ibg, Ig, Ii, If, Ifbg

def detect_particles(dims, Ibg, model, thresh, disk_sizes, thickening):
    height, width = dims[0], dims[1]
    L = np.zeros((Ibg.shape[0], Ibg.shape[1], height, width)).astype(int) #Particle labels
    Lbg = np.zeros((Ibg.shape[0], Ibg.shape[1], height, width)).astype(int) #Particle background labels
    if model == 'Max':
        for i in range(Ibg.shape[0]):
            for j in range(Ibg.shape[1]):
                p, lbg, sbg = locate_particles(Ibg[i, j], size_side = 12, dilate_r = 6, thres_max = thresh[i], thickening = thickening[0])
                L[i, j], Lbg[i, j] = label_particles(lbg, sbg, disk_sizes = disk_sizes, thickening = thickening[1])

    return L, Lbg

def bg_removal(I, sigma, r):
    footprint = disk(r)
    I = opening(I, footprint) 
    bg = gaussian_filter(I, sigma = sigma)
    return bg

def reference_removal(I, I0, sigma, r):
    bg0 = bg_removal(I0, sigma, r) 
    bg = bg_removal(I, sigma, r) 
    A = np.maximum(I/bg, 1)
    B = np.maximum(I0/bg0, 1)
    I_bg = np.maximum(A/B - 1, 0)
    I_bg[np.isnan(I_bg)] = 0
    return I_bg

def reference_removal_fluor(I, sigma, r):
    bg = bg_removal(I, sigma, r)
    A = np.maximum(I/bg, 1)
    B = np.maximum(A - 1, 0)
    B[np.isnan(B)] = 0
    return B

def label_bg(L_new, select_bg, disk_sizes, thickening):
    L_new = expand_labels(L_new, distance=thickening[0])
    select_inv = erosion(select_bg, disk(disk_sizes[0]))==0 #Inverted selection matrix, i.e the background
    select_inv = closing(select_inv, disk(disk_sizes[1])) #Noise removel
    L_new_dil = expand_labels(L_new, distance = disk_sizes[2]) #Area around each particle
    L_new_bg = L_new_dil*select_inv #Removing other particles
    L_new_bg_er = erosion(L_new_bg, disk(disk_sizes[3])) #Removing area around particles
    
    mask = binary_dilation(select_bg.astype(int), structure = disk(thickening[1]))
    inv_mask = erosion(mask, disk(disk_sizes[0]))==0
    L_new_bg_er = L_new_bg_er*inv_mask
    
    mask = ~np.isin(L_new, L_new_bg_er)

    # set the values in a that are in b to zero using the boolean mask
    L_new[mask] = 0

    return L_new, L_new_bg_er

def locate_particles(I_bg, size_side, dilate_r, thres_max, thickening):

    # Local maximum
    SE = disk(dilate_r)
    I_dil = dilation(I_bg, SE*10000)

    #Maximum threshold
    h = (1/9)*np.ones((3,3)); 
    imfilt = convolve2d(I_bg, np.rot90(h, 2), mode = 'same')
    max_p = ((I_dil==I_bg)*(imfilt > thres_max)).astype(int)

    label_I_bg = measure.label(max_p, connectivity = 2)
    label_I_bg = expand_labels(label_I_bg, distance=thickening)
    max_p_thick = label_I_bg.copy()
    max_p_thick[max_p_thick != 0] = 1
    select_bg = label_I_bg.astype(bool)
    
    return max_p_thick, label_I_bg, select_bg

def label_particles(L_new, select_bg, disk_sizes, thickening):

    select_inv = erosion(select_bg, disk(disk_sizes[0]))==0 #Inverted selection matrix, i.e the background
    select_inv = closing(select_inv, disk(disk_sizes[1])) #Noise removel
    L_new_dil = expand_labels(L_new, distance = disk_sizes[2]) #Area around each particle
    L_new_bg = L_new_dil*select_inv #Removing other particles
    L_new_bg_er = erosion(L_new_bg, disk(disk_sizes[3])) #Removing area around particles
    
    struct = generate_binary_structure(2, 1)
    mask = binary_dilation(select_bg.astype(int), structure = disk(thickening))
    inv_mask = erosion(mask, disk(disk_sizes[0]))==0
    L_new_bg_er = L_new_bg_er*inv_mask

    mask = ~np.isin(L_new, L_new_bg_er)

    # set the values in a that are in b to zero using the boolean mask
    L_new[mask] = 0
    
    return L_new, L_new_bg_er

###

#Check for ovelap for particles included df_list
def check_ovelap(df, df_list):
    if 'Overlap' not in df.columns:
        df['Overlap'] = np.inf
    for i, df_ref in enumerate(df_list):
        for index, particle in df.iterrows():
            j = particle['Position']
            c1 = particle['Coordinates']
            subdata = df_ref.query('Position == @j')
            c2 = np.stack(subdata['Coordinates'].values)
            distances = np.sqrt((c1[0] - c2[:, 0])**2 + (c1[1] - c2[:, 1])**2)
            closest_particle = np.nanargmin(distances)
            distance = distances[closest_particle]
            if df.at[index, 'Overlap'] > distance:
                df.at[index, 'Overlap'] = distance
            overlap = df.loc[index, 'Overlap']
            print(f'Overlap for #{index} with ref particle {i+1} = {overlap}')

### Update dataframes

def init_array(N_samples, N_pos, h, w):
    # Make an array with certain number of samples (N_samples), positions (N_pos) and images with height = h and width = w
    return np.empty((N_samples, N_pos, h, w)).astype(int)

def init_dataframe(i, columns, coords, saturated, Ints, Ints_after_AB1, Ints_after_AB2, convert_dict):
    df = pd.DataFrame(columns=columns)

    for j in range(Ints.shape[1]):

        data_to_add = pd.DataFrame(
            {
            'Particle': np.arange(1, len(Ints[i, j])+1),
            'Position': j,
            'Coordinates': coords[i, j],
            'Saturated': saturated[i, j],
            'Intensity1': Ints[i, j],
            'Intensity2': Ints_after_AB1[i, j],
            'Intensity3': Ints_after_AB2[i, j],
            })
        df = pd.concat([df, data_to_add], ignore_index=True)
    return df.astype(convert_dict)

def initialize_pnames(pnames, df):
    for pname in pnames:
        df[f'{pname} Particle'] = 0
        df[f'{pname} Distance'] = np.nan
        df[f'{pname} Coordinates'] = None
        df[f'{pname} Saturated'] = 0
        df[f'{pname} Intensity'] = np.nan

def update_particle_info(df, index, subdata, particle, pname):
    c1 = particle['Coordinates']
    c2 = np.stack(subdata['Coordinates'].values)
    distances = np.sqrt((c1[0] - c2[:, 0])**2 + (c1[1] - c2[:, 1])**2)
    closest_particle = np.nanargmin(distances)
    distance = distances[closest_particle]
    df.at[index, f'{pname} Particle'] = subdata.iloc[closest_particle]['Particle']
    df.at[index, f'{pname} Distance'] = distance
    df.at[index, f'{pname} Coordinates'] = subdata.iloc[closest_particle]['Coordinates']
    df.at[index, f'{pname} Saturated'] = subdata.iloc[closest_particle]['Saturated']
    df.at[index, f'{pname} Intensity'] = subdata.iloc[closest_particle]['Intensity']

def update_time_series(df, df_ab, pname, Ints_s, Ints_f, rates_s, rates_f, scat_increase, fluor_increase, saturated, saturated_fl, errors):
    df['Fluorescence time series'] = np.empty((len(df)), dtype=object)
    df['Scattering time series'] = np.empty((len(df)), dtype=object)
    df_ab['Fluorescence time series'] = np.empty((len(df_ab)), dtype=object)

    idx = df.query('Position == 0').index.values
    df.loc[idx, 'Scattering rate'] = rates_s
    df.loc[idx, 'Scattering increase'] = scat_increase
    if saturated is not None:
        df.loc[idx, 'Saturated rate'] = np.nanmax(saturated, axis=0)

    idx = df_ab.query('Position == 0').index.values
    df_ab.loc[idx, 'Fluorescence rate'] = rates_f
    df_ab.loc[idx, 'Fluorescence rate err'] = errors
    df_ab.loc[idx, 'Fluorescence increase'] = fluor_increase
    if saturated_fl is not None:
        df_ab.loc[idx, 'Saturated rate'] = np.nanmax(saturated_fl, axis=0)
    for index, particle in df_ab.iterrows():
        if particle['Position'] == 0:
            df_ab.at[index, 'Fluorescence time series'] = Ints_f[:, index]

    for index, particle in df.iterrows():
        if particle['Position'] == 0:
            idx = particle['Particle'] - 1
            df.at[index, 'Scattering time series'] = Ints_s[:, idx]
            idx = particle[f'{pname} Particle'] - 1
            df.at[index, 'Fluorescence time series'] = Ints_f[:, idx]
            if saturated_fl is not None:
                df.at[index, 'Saturated fl rate'] = np.nanmax(saturated_fl[:, idx])
            if pd.notnull(idx):
                df.at[index, 'Fluorescence rate'] = rates_f[int(idx)]
                df.at[index, 'Fluorescence rate err'] = errors[int(idx)]
                df.at[index, 'Fluorescence increase'] = fluor_increase[int(idx)]

### Fluorescence detection

def ECC(img_ref, img, warp_mode, number_of_iterations, termination_eps):
    # Find size of image1
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (img_ref, img, warp_matrix, warp_mode, criteria)
    return warp_matrix

### Calculate medium-exchange-related stuff

def calculate_medium_exchange(ptype, df, Ii, Ig, L, Lbg):
    Ints_i = np.empty((Ii.shape[0], Ii.shape[1]), dtype='object')
    Ints_g = np.empty((Ig.shape[0], Ig.shape[1]), dtype='object')

    df['Iodixanol series'] = np.empty((len(df)), dtype=object)
    df['Glycerol series'] = np.empty((len(df)), dtype=object)

    for i in range(Ii.shape[0]):
        for j in range(Ii.shape[1]):
            Ints_i[i, j], _, _ = calculate_intensities(Ii[i, j], L[ptype, j], Lbg[ptype, j])
            Ints_g[i, j], _, _ = calculate_intensities(Ig[i, j], L[ptype, j], Lbg[ptype, j])

    for index, particle in df.iterrows():
        j = particle['Position']
        pid = particle['Particle']-1
        subdata_i = np.empty(Ints_i.shape[0])
        subdata_g = np.empty(Ints_g.shape[0])
        for i in range(Ints_i.shape[0]):
            subdata_i[i] = Ints_i[i, j][pid]
            subdata_g[i] = Ints_g[i, j][pid]
        df.at[index, 'Iodixanol series'] = subdata_i
        df.at[index, 'Glycerol series'] = subdata_g

def calculate_ref_medians(df, N_media, N_steps):
    N_pos = np.nanmax(df['Position'])+1
    medians = np.zeros((N_pos, N_media, N_steps))
    for i in range(N_pos):
        subdata = df.query('Position == @i & Saturated == 0 & Overlap > 10')
        medians[i, 0, :] = np.nanmedian(np.array(subdata['Iodixanol series'].values.tolist()), axis=0)
        medians[i, 1, :] = np.nanmedian(np.array(subdata['Glycerol series'].values.tolist()), axis=0)
    return medians

def calculate_medians(df, N_media, N_steps):
    medians = np.zeros((N_media, N_steps))
    medians[0, :] = np.nanmedian(np.stack(df['Iodixanol series normalized'].values), axis=0)
    medians[1, :] = np.nanmedian(np.stack(df['Glycerol series normalized'].values), axis=0)
    return medians

def normalize_media_exchange(df, ref_medians, I_ref):
    df['Iodixanol series normalized'] = np.empty((len(df)), dtype=object)
    df['Glycerol series normalized'] = np.empty((len(df)), dtype=object)
    for index, particle in df.iterrows():
        j = particle['Position']
        df.at[index, 'Iodixanol series normalized'] = I_ref[:8]*df.at[index, 'Iodixanol series']/ref_medians[j, 0]
        df.at[index, 'Glycerol series normalized'] = I_ref[8:]*df.at[index, 'Glycerol series']/ref_medians[j, 1]

def calculate_scattering_fractions(df, df_ref, Ibefore, Iafter):
    df['Scattering increase all'] = np.nan
    N_pos = np.nanmax(df['Position'])+1
    for i in range(N_pos):
        subdata_snp = df_ref.query('Position == @i & Saturated == 0 & Overlap > 10')
        idx = df.query('Position == @i').index.values
        Inorm = df.loc[idx, f'{Iafter}'] / np.nanmedian(subdata_snp[f'{Iafter}'])
        Inorm0 = df.loc[idx, f'{Ibefore}'] / np.nanmedian(subdata_snp[f'{Ibefore}'])
        df.loc[idx, 'Scattering increase all'] = (Inorm - Inorm0) / Inorm0

def filter_data(data, s, lims, overlap, vlist):
    fdata = data.query(f"`{vlist[0]}` > {s[0]} or `{vlist[1]}` < {s[1]}")
    idx = []
    for i in range(lims.shape[0]):
        subdata = fdata.query('Position == @i')
        idxi = subdata[(fdata['Coordinates'].str[0] > lims[i, 0]) & (fdata['Coordinates'].str[0] < lims[i, 1]) & \
        (fdata['Coordinates'].str[1] > lims[i, 2]) & (fdata['Coordinates'].str[1] < lims[i, 3]) ].index.values
        idx = np.concatenate((idx, idxi))
    fdata = fdata.loc[idx.astype(int), :]
    for i in range(2, len(vlist)):
        fdata = fdata.query(f"`{vlist[i]}` < {s[i]}")
    fdata = fdata.query('Overlap > @overlap and Saturated == 0')
    
    return fdata

def calculate_errs(df, overlap, mult, p):     
    if len(mult) != len(p):
        raise ValueError('Size of m must match the ptype')

    s = [0]*len(p)
    N = 100

    for i, pval in enumerate(p):
        data = df[(df[pval].notnull()) & (df[pval] > 0) & \
                  (df['Overlap'] > overlap) & (df['Saturated'] == 0)][pval].values
        if pval == p[0]:
            data = 1 - data
        valmax = np.nanmedian(data)*5

        bins = np.arange(0, valmax, valmax/N)

        hist, bin_edges = np.histogram(data, bins)

        x = bin_edges[1:]
        param, _ = curve_fit(lambda x, A, m, sh: log_normal_fit(x, A, m, sh), x, hist,\
                            p0 = [1, 1, 1]) 

        mval = np.exp(param[1])
        sval = (np.exp(param[2]**2) - 1)*np.exp(2*param[1]+param[2]**2)
        sval = np.sqrt(sval)
        s[i] = mval+mult[i]*sval
    s[0] = 1-s[0]
    return s

def convert_string_to_array(string):
    replace_dict = {'\n': '', '[': '', ']': ''}
    string = string.translate(str.maketrans(replace_dict))
    string = re.split("\s+", string)
    string = [i for i in string if i]
    return np.array([float(x) for x in string])

def check_for_tuple(x):
    if type(x) is not tuple and x is not np.nan:
        return ast.literal_eval(x)
    else:
        return x

def binned_nta(x, y, wbins, xmax, normalized):
    bins = np.arange(wbins, xmax, wbins)
    idx = np.digitize(x, bins=bins)
    ybins = np.empty(len(bins))
    for i in range(len(bins)):
        ybins[i] = np.nansum(y[idx == i])
    xbins = np.empty(len(bins))
    for i in range(len(xbins)):
        if i == 0:
            xbins[i] = bins[i]/2
        else:
            xbins[i] = (bins[i]+bins[i-1])/2
    if normalized:
        return xbins, ybins/np.nansum(y)
    else:
        return xbins, ybins

def calculate_nta_median(data_folder, files):
    df_p1 = pd.read_csv(f'{data_folder}/{files[0]}.csv', sep = ',')
    df_p2 = pd.read_csv(f'{data_folder}/{files[0]}.csv', sep = ',')
    df_p3 = pd.read_csv(f'{data_folder}/{files[0]}.csv', sep = ',')
    p1 = np.nanmedian(df_p1[df_p1['Included in distribution?'] == True]['Size/nm'].values)
    p2 = np.nanmedian(df_p2[df_p2['Included in distribution?'] == True]['Size/nm'].values)
    p3 = np.nanmedian(df_p3[df_p3['Included in distribution?'] == True]['Size/nm'].values)
    return np.nanmean([p1, p2, p3])

### Template matching

def template_matching(img, template, crop):
    img_8U = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    sz = img.shape
    res = cv2.matchTemplate(img_8U, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    tform = np.float32([[1,0,crop-max_loc[0]],[0,1,crop-max_loc[1]]])
    img_trans = cv2.warpAffine(img, tform, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
    return img_trans

def make_template(frames, template_frame, crop):
    template = frames[template_frame]
    sz = template.shape
    template = template[crop:sz[0]-crop, crop:sz[1]-crop]
    template = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return template

###

def crop_by_half(image, side):
    if image.ndim == 3: 
        Nx = image.shape[2]
        if side == 'r':
            return image[:, :, Nx//2:Nx]
        elif side == 'l':
            return image[:, :, 0:Nx//2]
    elif image.ndim == 2:
        Nx = image.shape[1]
        if side == 'r':
            return image[:, Nx//2:Nx]
        elif side == 'l': 
            return image[:, 0:Nx//2]

def crop_edges(image, pixels_x, pixels_y):
    if image.ndim == 3:
        Ny = image.shape[1]
        Nx = image.shape[2]
        return image[:, pixels_y:Ny-pixels_y, pixels_x:Nx-pixels_x]
    if image.ndim == 2:
        Ny = image.shape[0]
        Nx = image.shape[1]
        return image[pixels_y:Ny-pixels_y, pixels_x:Nx-pixels_x]

def calculate_intensities(data, L_new, L_new_bg_er):
    
    # Measure the regions in the background
    s_bg = measure.regionprops(L_new_bg_er, data)
    s_fg = measure.regionprops(L_new, data)

    Np = len(s_fg)

    # Calculate the mean intensity of each background region and check for saturated pixels
    mean_bg = np.zeros(Np)
    saturated = np.zeros(Np).astype(int)

    for i, region in enumerate(s_bg):
        coords = np.transpose(region.coords)
        mean_bg[i] = np.mean(data[coords[0], coords[1]])
        sat_pixels = np.where(data[coords[0], coords[1]] == 2**16-1)[0]
        saturated[i] = len(sat_pixels)
        
    # Calculate mean intensity and coordinates for foreground regions
    mean_fg = np.zeros(Np)
    coord_list = np.zeros(Np, dtype='object')
    for i, region in enumerate(s_fg):
        coords = np.transpose(region.coords)
        mean_fg[i] = np.nansum(data[coords[0], coords[1]] - mean_bg[i])
        coord_list[i] = region.centroid
        sat_pixels = np.where(data[coords[0], coords[1]] == 2**16-1)[0]
        if len(sat_pixels) > saturated[i]:
            saturated[i] = len(sat_pixels)
    return mean_fg, coord_list, saturated


### Helper functions

def IntensityInner(n_m, r, t, n_s, n_i, l):
    y = n_effective(r, t, n_s, n_i)
    y = (y**2 - n_m**2) / (y**2 + 2*n_m**2)
    y = y**2*f_ev(r, t, evn(n_m), 's')
    return y

def log_normal_fit(x, A, m, sh):
    return A*np.exp(-0.5*(np.log(x)-m)**2/sh**2)

def v_frac(r, t):
    return ((r-t)**3)/r**3

def L(n, n_m):
    return (n**2 - n_m**2)/(n**2 + 2*n_m**2)

def n_effective(r, t, n_s, n_i, n_m):
    a = (1-v_frac(r, t))*L(n_s, n_m) + v_frac(r, t)*L(n_i, n_m)
    return n_m*np.sqrt((2*a+1)/(1-a))

def f_ev(r, delta):
    u = r/(2*delta)
    f_ev = ((1/(2*u))*(1 - np.exp(-2*u)))
    return f_ev**2

def evn(n_m):
    evn = 95.83509 + 7.2258*np.exp((n_m - 1.33)/0.01853)
    return evn

def plot_loghist(data, bins):
    x = data.copy()
    x[x <= 0] = np.nan
    x = x[~np.isnan(x)]
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    return x, logbins

def colorize(img, color, limit):
    height = img.shape[0]
    width = img.shape[1]
    out = np.zeros((height, width, 3))
    if color == 'g':
        out[:, :, 1] = img/np.nanmax(img)
    elif color == 'm':
        out[:, :, 0] = img/np.nanmax(img)
        out[:, :, 2] = img/np.nanmax(img)

    out[out > limit] = 1.0
    out = (out*255.0).astype(np.uint8)
    return out

def normalize_peaks(I, L):
    avPeak = np.nanmean(np.nanmax(I) - np.nanmin(I))
    s = measure.regionprops(L, I)
    Np = len(s)
    mask = L.copy()
    mask[mask != 0] = 1
    img = mask*I

    for i in range(Np):
        idx = np.transpose(s[i].coords)
        subdata = img[idx[0], idx[1]]
        mi = np.nanmin(subdata)
        ma = np.nanmax(subdata)
        img[idx[0], idx[1]] = avPeak*(subdata-mi)/(ma-mi)

    return img

def calculate_mape(y, y_model):
    N = len(y)
    mape = np.nansum(np.abs((y-y_model)/y))/N
    return mape

### Fitting functions

def rate_equation(t, A0, A, tau):
    return A0 + A*(1-np.exp(-t/tau))

def rate_equation_delay(t, A0, A, t0, tau):
    idx = np.nanargmin(np.abs(t-t0))
    y = np.zeros(len(t))
    y[:idx] = A0
    y[idx:] = A0 + A*(1-np.exp(-(t[idx:]-t0)/tau))
    return y

def exp(x, Amp1, Amp2, a, b):
    return Amp2 + (Amp1-Amp2)/(1+np.exp((x-b)/a))

def IntensityFit(x, tau, delay, r, t, n_s, n_i0, n_i, n_m, l):
    return IntensityInner(n_m, r, t, n_s, exp(x, n_i0, n_i, tau, delay), l)

def IntensitySphereSimple(n_m, r, t, n_s, n_i, l):
    if r-t > 0 and r > 0 and r >= r-t:
        I = mie_coated(r, t, n_s, n_i, n_m, l)*f_ev(r, evn(n_m))
    else:
        I = np.nan
    return I

def IntensitySphere1(n_m, r, t, n_s, n_i, l):
    I = np.empty((len(n_m)))
    for i in range(len(n_m)):
        if r-t > 0 and r > 0 and r >= r-t:
            I[i] = mie_coated(r, t, n_s, n_i, n_m[i], l)*f_ev(r, evn(n_m[i]))
        else:
            I[i] = np.nan
    return I

def IntensitySphere2(n_m1, n_m2, r, t, n_s, n_i, l):
    I1 = np.empty((len(n_m1)))
    I2 = np.empty((len(n_m2)))

    for i in range(len(n_m1)):
        if r-t > 0 and r > 0 and r >= r-t:
            I1[i] = mie_coated(r, t, n_s, n_i, n_m1[i], l)*f_ev(r, evn(n_m1[i]))
            I2[i] = mie_coated(r, t, n_s, n_i, n_m2[i], l)*f_ev(r, evn(n_m2[i]))
        else:
            I1[i], I2[i] = np.nan, np.nan
  
    return np.concatenate((I1, I2))

def IntensityVolume(n_m1, n_m2, r, t, n_s, fp, n_i1, l):
    I1 = np.empty((len(n_m1)))
    I2 = np.empty((len(n_m2)))

    term = L(n_i1, n_m2) + (1-fp)*(L(n_m2, n_m2) - L(n_m1[0], n_m2))
    n_i2 = np.sqrt((1+2*term)/(1-term))*n_m2

    for i in range(len(n_m1)):
        I1[i] = mie_coated(r, t, n_s, n_i1, n_m1[i], l)*f_ev(r, evn(n_m1[i]))
        I2[i] = mie_coated(r, t, n_s, n_i2[i], n_m2[i], l)*f_ev(r, evn(n_m2[i]))
  
    return np.concatenate((I1, I2))


def IntensityVolumeSphere(n_m1, n_m2, r, fp, n1, l):
    I1 = np.empty((len(n_m1)))
    I2 = np.empty((len(n_m2)))

    term = L(n1, n_m2) + (1-fp)*(L(n_m2, n_m2) - L(n_m1[0], n_m2))
    n2 = np.sqrt((1+2*term)/(1-term))*n_m2

    for i in range(len(n_m1)):
        I1[i] = mie_coated(r, 0, n1, n1, n_m1[i], l)*f_ev(r, evn(n_m1[i]))
        I2[i] = mie_coated(r, 0, n2[i], n2[i], n_m2[i], l)*f_ev(r, evn(n_m2[i]))
  
    return np.concatenate((I1, I2))

def IntensityVesicle(n_m1, n_m2, r, t, n_s, l):
    I1 = np.empty((len(n_m1)))
    I2 = np.empty((len(n_m2)))

    for i in range(len(n_m1)):
        if r-t > 0 and r > 0 and r >= r-t:
            I1[i] = mie_coated(r, t, n_s, n_m1[0], n_m1[i], l)*f_ev(r, evn(n_m1[i]))
            I2[i] = mie_coated(r, t, n_s, n_m2[i], n_m2[i], l)*f_ev(r, evn(n_m2[i]))
        else:
            I1[i], I2[i] = np.nan, np.nan
  
    return np.concatenate((I1, I2))

### Get data

def return_filtered_data(date, file, fnames, overlap, w):
    df = open_file(date, '', file)
    limsp = np.load(f'Results/{date}/lims.npy', allow_pickle=True)
    s = calculate_errs(df, overlap, w, fnames)
    filtered_data = filter_data(df, [s[0], s[1], s[2], s[3], s[4]], limsp, overlap, fnames)
    return filtered_data, s

def compose_data(df, fnames):
    flist = np.empty(len(fnames), dtype='object')
    for i, fname in enumerate(fnames):
        if fname.split(' ')[0] == 'Radius':
            flist[i] = 2*df[fname].values
        else:
            flist[i] = df[fname].values
    return flist