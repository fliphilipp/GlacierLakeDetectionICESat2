import os
import h5py
import datetime
import traceback
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
import matplotlib.pylab as plt
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from icelakes.utilities import convert_time_to_string
pd.set_option('mode.chained_assignment', 'raise')


##########################################################################################
def read_atl03(filename, geoid_h=True):
    """
    Read in an ATL03 granule. 

    Parameters
    ----------
    filename : string
        the file path of the granule to be read in
    geoid_h : boolean
        whether to include the ATL03-supplied geoid correction for photon heights

    Returns
    -------
    dfs : dict of pandas dataframes
          photon-rate data with keys ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
          each dataframe contains the following variables
          lat : float64, latitude of the photon, degrees
          lon : float64, longitude of the photon, degrees
          h : float64, elevation of the photon (geoid correction applied if geoid_h=True), meters
          dt : float64, delta time of the photon, seconds from the ATLAS SDP GPS Epoch
          mframe : uint32, the ICESat-2 major frame that the photon belongs to
          qual : int8, quality flag 0=nominal,1=possible_afterpulse,2=possible_impulse_response_effect,3=possible_tep
          xatc : float64, along-track distance of the photon, meters
          geoid : float64, geoid correction that was applied to photon elevation (supplied if geoid_h=True), meters
    dfs_bckgrd : dict of pandas dataframes
                 photon-rate data with keys ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
                 each dataframe contains the following variables
                 pce_mframe_cnt : int64, the major frame that the data belongs to
                 bckgrd_counts : int32, number of background photons
                 bckgrd_int_height : float32, height of the background window, meters
                 delta_time : float64, Time at the start of ATLAS 50-shot sum, seconds from the ATLAS SDP GPS Epoch
    ancillary : dictionary with the following keys:
                granule_id : string, the producer granule id, extracted from filename
                atlas_sdp_gps_epoch : float64, reference GPS time for ATLAS in seconds [1198800018.0]
                rgt : int16, the reference ground track number
                cycle_number : int8, the ICESat-2 cycle number of the granule
                sc_orient : the spacecraft orientation (usually 'forward' or 'backward')
                gtx_beam_dict : dictionary of the ground track / beam number configuration 
                                example: {'gt1l': 6, 'gt1r': 5, 'gt2l': 4, 'gt2r': 3, 'gt3l': 2, 'gt3r': 1}
                gtx_strength_dict': dictionary of the ground track / beam strength configuration
                                    example: {'gt1l': 'weak','gt1r': 'strong','gt2l': 'weak', ... }
                                    
    Examples
    --------
    >>> read_atl03(filename='processed_ATL03_20210715182907_03381203_005_01.h5', geoid_h=True)
    """
    
    print('  reading in', filename)
    granule_id = filename[filename.find('ATL03_'):(filename.find('.h5')+3)]
    
    # open file
    f = h5py.File(filename, 'r')
    
    # make dictionaries for beam data to be stored in
    dfs = {}
    dfs_bckgrd = {}
    beamlist = [x for x in list(f.keys()) if 'gt' in x]
    
    conf_landice = 3 # index for the land ice confidence
    
    orient = f['orbit_info']['sc_orient'][0]
    def orient_string(sc_orient):
        if sc_orient == 0:
            return 'backward'
        elif sc_orient == 1:
            return 'forward'
        elif sc_orient == 2:
            return 'transition'
        else:
            return 'error'
        
    orient_str = orient_string(orient)
    gtl = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    beam_strength_dict = {k:['weak','strong'][k%2] for k in np.arange(1,7,1)}
    if orient_str == 'forward':
        bl = np.arange(6,0,-1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    elif orient_str == 'backward':
        bl = np.arange(1,7,1)
        gtx_beam_dict = {k:v for (k,v) in zip(gtl,bl)}
        gtx_strength_dict = {k:beam_strength_dict[gtx_beam_dict[k]] for k in gtl}
    else:
        gtx_beam_dict = {k:'undefined' for k in gtl}
        gtx_strength_dict = {k:'undefined' for k in gtl}
        

    ancillary = {'granule_id': granule_id,
                 'atlas_sdp_gps_epoch': f['ancillary_data']['atlas_sdp_gps_epoch'][0],
                 'rgt': f['orbit_info']['rgt'][0],
                 'cycle_number': f['orbit_info']['cycle_number'][0],
                 'sc_orient': orient_str,
                 'gtx_beam_dict': gtx_beam_dict,
                 'gtx_strength_dict': gtx_strength_dict}
    
    # loop through all beams
    print('  reading in beam:', end=' ')
    for beam in beamlist:
        print(beam, end=' ')
        try:
            #### get photon-level data
            df = pd.DataFrame({'lat': np.array(f[beam]['heights']['lat_ph']),
                               'lon': np.array(f[beam]['heights']['lon_ph']),
                               'h': np.array(f[beam]['heights']['h_ph']),
                               'dt': np.array(f[beam]['heights']['delta_time']),
                               # 'conf': np.array(f[beam]['heights']['signal_conf_ph'][:,conf_landice]),
                               # not using ATL03 confidences here
                               'mframe': np.array(f[beam]['heights']['pce_mframe_cnt']),
                               'qual': np.array(f[beam]['heights']['quality_ph'])}) 
                               # 0=nominal,1=afterpulse,2=impulse_response_effect,3=tep

            df_bckgrd = pd.DataFrame({'pce_mframe_cnt': np.array(f[beam]['bckgrd_atlas']['pce_mframe_cnt']),
                                      'bckgrd_counts': np.array(f[beam]['bckgrd_atlas']['bckgrd_counts']),
                                      'bckgrd_int_height': np.array(f[beam]['bckgrd_atlas']['bckgrd_int_height']),
                                      'delta_time': np.array(f[beam]['bckgrd_atlas']['delta_time'])})

            #### calculate along-track distances [meters from the equator crossing] from segment-level data
            df['xatc'] = np.full_like(df.lat, fill_value=np.nan)
            ph_index_beg = np.int32(f[beam]['geolocation']['ph_index_beg']) - 1
            segment_dist_x = np.array(f[beam]['geolocation']['segment_dist_x'])
            segment_length = np.array(f[beam]['geolocation']['segment_length'])
            valid = ph_index_beg>=0 # need to delete values where there's no photons in the segment (-1 value)

            df.loc[ph_index_beg[valid], 'xatc'] = segment_dist_x[valid]
            df.xatc.fillna(method='ffill',inplace=True)
            df.xatc += np.array(f[beam]['heights']['dist_ph_along'])

            #### now we can filter out TEP (we don't do IRF / afterpulses because it seems to not be very good...)
            df.query('qual < 3',inplace=True) 
            # df.drop(columns=['qual'], inplace=True)

            #### sort by along-track distance (for interpolation to work smoothly)
            df.sort_values(by='xatc',inplace=True)
            df.reset_index(inplace=True, drop=True)

            if geoid_h:
                #### interpolate geoid to photon level using along-track distance, and add to elevation
                geophys_geoid = np.array(f[beam]['geophys_corr']['geoid'])
                geophys_geoid_x = segment_dist_x+0.5*segment_length
                valid_geoid = geophys_geoid<1e10 # filter out INVALID_R4B fill values
                geophys_geoid = geophys_geoid[valid_geoid]
                geophys_geoid_x = geophys_geoid_x[valid_geoid]
                # hacky fix for no weird stuff happening if geoid is undefined everywhere
                if len(geophys_geoid>5):
                    geoid = np.interp(np.array(df.xatc), geophys_geoid_x, geophys_geoid)
                    df['h'] = df.h - geoid
                    df['geoid'] = geoid
                else:
                    df['geoid'] = 0.0

            #### save to list of dataframes
            dfs[beam] = df
            dfs_bckgrd[beam] = df_bckgrd
        
        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            traceback.print_exc()
    print(' --> done.')
    return dfs, dfs_bckgrd, ancillary

##########################################################################################
def make_mframe_df(df):
    mframe_group = df.groupby('mframe')
    df_mframe = mframe_group[['lat','lon', 'xatc', 'dt']].mean()
    df_mframe.drop(df_mframe.head(1).index,inplace=True)
    df_mframe.drop(df_mframe.tail(1).index,inplace=True)
    df_mframe['time'] = df_mframe['dt'].map(convert_time_to_string)
    df_mframe['xatc_min'] = mframe_group['xatc'].min()
    df_mframe['xatc_max'] = mframe_group['xatc'].max()
    df_mframe['n_phot'] = mframe_group['h'].count()
    df_mframe['lake_qual_pass'] = False
    df_mframe['has_densities'] = False
    df_mframe['ratio_2nd_returns'] = 0.0
    df_mframe['quality_summary'] = 0.0
    empty_list = []
    df_mframe['xatc_2nd_returns'] = df_mframe.apply(lambda _: empty_list.copy(), axis=1)
    df_mframe['proms_2nd_returns'] = df_mframe.apply(lambda _: empty_list.copy(), axis=1)
    df_mframe['h_2nd_returns'] = df_mframe.apply(lambda _: empty_list.copy(), axis=1)
    
    return df_mframe


##########################################################################################
def find_flat_lake_surfaces(df_mframe, df, bin_height_coarse=1.0, bin_height_fine=0.02, smoothing_histogram=0.2, buffer=4.0,
                            width_surf=0.1, width_buff=0.3, rel_dens_upper_thresh=8, rel_dens_lower_thresh=4,
                            min_phot=100, min_snr_surface=200):
    
    peak_locs = np.full(len(df_mframe), np.nan, dtype=np.double)
    is_flat = np.full_like(peak_locs, np.nan, dtype=np.bool_)
    
    for i, mframe in enumerate(df_mframe.index):
        
        # select the photons in the major frame
        selector_segment = (df.mframe == mframe)
        dfseg = df[selector_segment]
        
        # check if there are enough photons in the segment
        if len(dfseg) < min_phot:
            is_flat[i] = False
        
        # find peaks
        else:
            # find main broad peak
            bins_coarse1 = np.arange(start=dfseg.h.min(), stop=dfseg.h.max(), step=bin_height_coarse)
            hist_mid1 = bins_coarse1[:-1] + 0.5 * bin_height_coarse
            peak_loc1 = hist_mid1[np.argmax(np.histogram(dfseg.h, bins=bins_coarse1)[0])]

            # decrease bin width and find finer peak
            bins_coarse2 = np.arange(start=peak_loc1-buffer, stop=peak_loc1+buffer, step=bin_height_fine)
            hist_mid2 = bins_coarse2[:-1] + 0.5 * bin_height_fine
            hist = np.histogram(dfseg.h, bins=bins_coarse2)
            window_size = int(smoothing_histogram/bin_height_fine)
            hist_vals = hist[0] / np.max(hist[0])
            hist_vals_smoothed = np.array(pd.Series(hist_vals).rolling(window_size,center=True,min_periods=1).mean())
            peak_loc2 = hist_mid2[np.argmax(hist_vals_smoothed)]
            peak_locs[i] = peak_loc2

            # calculate relative photon densities
            peak_upper = peak_loc2 + width_surf
            peak_lower = peak_loc2 - width_surf
            above_upper = peak_upper + width_buff
            below_lower = peak_lower - width_buff
            sum_peak = np.sum((dfseg.h > peak_lower) & (dfseg.h < peak_upper))
            sum_above = np.sum((dfseg.h > peak_upper) & (dfseg.h < above_upper))
            sum_below = np.sum((dfseg.h > below_lower) & (dfseg.h < peak_lower))
            signal_rate = sum_peak / (width_surf*2)
            rel_dens_upper = 1000 if sum_above==0 else signal_rate / (sum_above / width_buff)
            rel_dens_lower = 1000 if sum_below==0 else signal_rate / (sum_below / width_buff)
            noise_rate = (dfseg.h.count() - sum_peak) / (dfseg.h.max() - dfseg.h.min() - width_surf*2)
            snr_surface = signal_rate / noise_rate

            # check for flat surface, if found calculate SNR and look for bottom return
            is_flat_like_lake = (rel_dens_upper > rel_dens_upper_thresh) \
                                & (rel_dens_lower > rel_dens_lower_thresh) \
                                & (snr_surface > min_snr_surface)
            is_flat[i] = is_flat_like_lake
            
            # print('%3i, %5s, %4i, %4i, %4i' % (i, is_flat[i], snr_surface, rel_dens_lower, rel_dens_upper))

    return peak_locs, is_flat


##########################################################################################
def get_densities_and_2nd_peaks(df, df_mframe, df_selected, aspect=30, K_phot=10, dh_signal=0.2, n_subsegs=7,
                                bin_height_snr=0.1, smoothing_length=1.0, buffer=4.0, print_results=False):
    
    # somehow got duplicate indices (mframe values in index) in here
    # this shouldn't be the case because index was created using groupby on mframe 
    # below is a temporary fix ---> check out more what's wrong here
    df_mframe_selected = df_selected.copy()
    df_mframe_selected.drop_duplicates(subset=['xatc_min','xatc_max'], keep='first', inplace=True)
    
    for mframe in df_mframe_selected.index:
        
        try:
            selector_segment = (df.mframe == mframe)
            dfseg = df[selector_segment].copy()

            xmin = df_mframe_selected.loc[mframe, 'xatc_min']
            xmax = df_mframe_selected.loc[mframe, 'xatc_max']
            nphot = df_mframe_selected.loc[mframe, 'n_phot']
            peak_loc2 = df_mframe_selected.loc[mframe, 'peak']

            isabovesurface = dfseg.h > (peak_loc2+dh_signal)
            isbelowsurface = dfseg.h < (peak_loc2-dh_signal)

            # the radius in which to look for neighbors
            dfseg_nosurface = dfseg[isabovesurface | isbelowsurface]
            nphot_bckgrd = len(dfseg_nosurface.h)
            
            # radius of a circle in which we expect to find one non-lake-surface signal photon
            telem_h = dfseg_nosurface.h.max()-dfseg_nosurface.h.min()
            flat_surf_signal_h = 2*dh_signal
            h_noise = telem_h-flat_surf_signal_h
            wid_noise = (xmax-xmin)/aspect
            area = h_noise*wid_noise/nphot_bckgrd
            wid = np.sqrt(area/np.pi)

            # buffer segment for density calculation
            selector_buffer = (df.xatc >= (dfseg.xatc.min()-aspect*wid)) & (df.xatc <= (dfseg.xatc.max()+aspect*wid))
            dfseg_buffer = df[selector_buffer].copy()
            dfseg_buffer.xatc += np.random.uniform(low=-0.35, high=0.35, size=len(dfseg_buffer.xatc))

            # normalize xatc to be regularly spaced and scaled by the aspect parameter
            xmin_buff = dfseg_buffer.xatc.min()
            xmax_buff = dfseg_buffer.xatc.max()
            nphot_buff = len(dfseg_buffer.xatc)
            xnorm = np.linspace(xmin_buff, xmax_buff, nphot_buff) / aspect

            # KD tree query distances
            Xn = np.array(np.transpose(np.vstack((xnorm, dfseg_buffer['h']))))
            kdt = KDTree(Xn, leaf_size=40)
            idx, dist = kdt.query_radius(Xn, r=wid, count_only=False, return_distance=True,sort_results=True)
            density = (np.array([np.sum(1-np.abs(x/wid)) if (len(x)<(K_phot+1)) 
                       else np.sum(1-np.abs(x[:K_phot+1]/wid))
                       for x in dist]) - 1) / K_phot

            #print(' density calculated')
            densities = np.array(density[dfseg_buffer.mframe == mframe])
            densities /= np.max(densities)

            # add SNR to dataframes
            dfseg['snr'] = densities
            df.loc[selector_segment, 'snr'] = densities
            df_mframe.loc[mframe, 'has_densities'] = True

            # subdivide into segments again to check for second return
            subsegs = np.linspace(xmin, xmax, n_subsegs+1) 
            subsegwidth = subsegs[1] - subsegs[0]

            n_2nd_returns = 0
            prominences = []
            elev_2ndpeaks = []
            subpeaks_xatc = []
            for subsegstart in subsegs[:-1]:

                subsegend = subsegstart + subsegwidth
                selector_subseg = ((dfseg.xatc > subsegstart) & (dfseg.xatc < subsegend))
                dfsubseg = dfseg[selector_subseg]

                # avoid looking for peaks when there's no / very little data
                if len(dfsubseg > 20):

                    # get the median of the snr values in each bin
                    bins_subseg_snr = np.arange(start=np.max((dfsubseg.h.min(),peak_loc2-70)), stop=peak_loc2+2*buffer, step=bin_height_snr)
                    mid_subseg_snr = bins_subseg_snr[:-1] + 0.5 * bin_height_snr
                    snrstats = binned_statistic(dfsubseg.h, dfsubseg.snr, statistic='median', bins=bins_subseg_snr)
                    snr_median = snrstats[0]
                    snr_median[np.isnan(snr_median)] = 0
                    window_size_sub = int(smoothing_length/bin_height_snr)
                    snr_vals_smoothed = np.array(pd.Series(snr_median).rolling(window_size_sub,center=True,min_periods=1).mean())
                    snr_vals_smoothed /= np.max(snr_vals_smoothed)

                    # take histogram binning values into account, but clip surface peak to second highest peak height
                    subhist, subhist_edges = np.histogram(dfsubseg.h, bins=bins_subseg_snr)
                    subhist_nosurface = subhist.copy()
                    subhist_nosurface[(mid_subseg_snr < (peak_loc2+2*dh_signal)) & (mid_subseg_snr > (peak_loc2-2*dh_signal))] = 0
                    subhist_nosurface_smoothed = np.array(pd.Series(subhist_nosurface).rolling(window_size_sub,center=True,min_periods=1).mean())
                    subhist_max = subhist_nosurface_smoothed.max()
                    subhist_smoothed = np.array(pd.Series(subhist).rolling(window_size_sub,center=True,min_periods=1).mean())
                    subhist_smoothed = np.clip(subhist_smoothed, 0, subhist_max)
                    subhist_smoothed /= np.max(subhist_smoothed)

                    # combine histogram and snr values to find peaks
                    snr_hist_smoothed = subhist_smoothed * snr_vals_smoothed
                    peaks, peak_props = find_peaks(snr_hist_smoothed, height=0.1, distance=int(0.5/bin_height_snr), prominence=0.1)

                    if len(peaks) >= 2: 
                        idx_surfpeak = np.argmin(peak_loc2 - mid_subseg_snr[peaks])
                        peak_props['prominences'][idx_surfpeak] = 0

                        # classify as second peak only if prominence is larger 0.2
                        prominence_secondpeak = np.max(peak_props['prominences'])
                        prominence_threshold = 0.3
                        if prominence_secondpeak > prominence_threshold:

                            idx_2ndreturn = np.argmax(peak_props['prominences'])
                            secondpeak_h = mid_subseg_snr[peaks[idx_2ndreturn]]

                            # classify as second peak only if elevation is 1m lower than main peak (surface) and higher than 50m below surface
                            if (secondpeak_h < (peak_loc2-1.0)) & (secondpeak_h > (peak_loc2-50.0)):
                                secondpeak_xtac = subsegstart + subsegwidth/2
                                n_2nd_returns += 1
                                prominences.append(prominence_secondpeak)
                                elev_2ndpeaks.append(secondpeak_h)
                                subpeaks_xatc.append(secondpeak_xtac)

            ratio_2nd_returns = n_2nd_returns/n_subsegs
            df_mframe.loc[mframe, 'ratio_2nd_returns'] = ratio_2nd_returns
            quality_secondreturns = np.sum(prominences) / n_subsegs

            min_quality = (0.2 + (ratio_2nd_returns-0.2)*(0.1/0.8))
            quality_summary = 0.0
            quality_pass = 'No'
            if (ratio_2nd_returns >= 0.2) & (quality_secondreturns > min_quality):
                quality_pass = 'Yes'
                quality_summary = ratio_2nd_returns*(quality_secondreturns-min_quality)/(ratio_2nd_returns-min_quality)
                df_mframe.loc[mframe, 'lake_qual_pass'] = True
                df_mframe.loc[mframe, 'quality_summary'] = quality_summary

                for i in range(len(elev_2ndpeaks)):
                    df_mframe.loc[mframe, 'h_2nd_returns'].append(elev_2ndpeaks[i])
                    df_mframe.loc[mframe, 'xatc_2nd_returns'].append(subpeaks_xatc[i])
                    df_mframe.loc[mframe, 'proms_2nd_returns'].append(prominences[i])

            # if (percent_2d_returns >= 30) & (quality_secondreturns > 0.4):
            flatstring = 'Yes' if df_mframe['is_flat'].loc[mframe] else 'No'

            if print_results:
                print('   mframe %03i: h=%7.2fm. flat=%3s. 2nds=%3d%%. qual=%4.2f. pass=%3s.' % \
                    (mframe%1000, peak_loc2, flatstring, np.round(ratio_2nd_returns*100), quality_summary, quality_pass))
        
        except: 
            print('Something went wrong getting densities and peaks for mframe %i ...' % mframe)
            traceback.print_exc()

            
##########################################################################################
# check for densities and possible second returns two mframes around where passing quality lakes were detected
def check_additional_segments(df, df_mframe, print_results=False):
    num_additional = 1
    count = 0
    while (num_additional > 0) & (count<20):
        count+=1
        selected = df_mframe[df_mframe['lake_qual_pass']].index 
        lst1 = np.unique(list(selected-2) + list(selected-1) + list(selected+1) + list(selected+2))
        lst2 = df_mframe.index
        inter = list(set(lst1) & set(lst2))
        df_include_surrounding = df_mframe.loc[inter]
        # df_include_surrounding = df_mframe.loc[np.unique(list(selected-2) + list(selected-1) + list(selected+1) + list(selected+2))]
        df_additional = df_include_surrounding[~df_include_surrounding['has_densities']]
        num_additional = len(df_additional)
        if num_additional > 0:
            get_densities_and_2nd_peaks(df, df_mframe, df_additional, print_results=print_results)

            
##########################################################################################
# merge detected lake segments iteratively
def merge_lakes(df_mframe, max_dist_mframes=7, max_dist_elev=0.1, print_progress=False, debug=False):

    try:
        df_mframe.sort_index(inplace=True)
        start_mframe = list(df_mframe.index[df_mframe['lake_qual_pass']])
        stop_mframe = list(df_mframe.index[df_mframe['lake_qual_pass']])
        surf_elevs = list(df_mframe['peak'][df_mframe['lake_qual_pass']])
        n_lakes = len(surf_elevs)
        if n_lakes == 0:
            print('   ---> nothing to merge.')
            return

        any_merges = True
        iteration = 0

        # keep going until there is no change
        while any_merges:

            print('   --> iteration %3d, number of lakes: %4d' % (iteration, n_lakes))
            start_mframe_old = start_mframe
            stop_mframe_old = stop_mframe
            surf_elevs_old = surf_elevs
            n_lakes_old = n_lakes
            start_mframe = []
            stop_mframe = []
            surf_elevs = []
            any_merges = False

            for i in range(0,n_lakes-1,2):

                is_closeby = ((start_mframe_old[i + 1] - stop_mframe_old[i]) <= max_dist_mframes)
                is_at_same_elevation = (np.abs(surf_elevs_old[i + 1] - surf_elevs_old[i]) < max_dist_elev)

                if debug: 
                    print('      %3i-%3i <> %3i-%3i | xdiff: %4d, close: %5s | %7.2f > %7.2f, hdiff: %7.2f, same: %5s' % \
                         (start_mframe_old[i]%1000, stop_mframe_old[i]%1000, 
                          start_mframe_old[i + 1]%1000, stop_mframe_old[i + 1]%1000,
                          start_mframe_old[i + 1] - stop_mframe_old[i],
                          is_closeby, surf_elevs_old[i], surf_elevs_old[i + 1], np.abs(surf_elevs_old[i + 1] - surf_elevs_old[i]),
                          is_at_same_elevation), end=' ')

                # if merging two lakes, just append the combined stats as a single lake 
                if (is_closeby & is_at_same_elevation):
                    start_mframe.append(start_mframe_old[i])
                    stop_mframe.append(stop_mframe_old[i + 1])
                    surf_elevs.append((surf_elevs_old[i] + surf_elevs_old[i+1]) / 2)
                    if debug: print('--> merge')
                    any_merges = True

                # if keeping two lakes separate, add them both
                else:
                    start_mframe += start_mframe_old[i:i+2]
                    stop_mframe += stop_mframe_old[i:i+2]
                    surf_elevs += surf_elevs_old[i:i+2]
                    if debug: print('--> keep separate')

            if n_lakes%2 == 1:
                start_mframe.append(start_mframe_old[-1])
                stop_mframe.append(stop_mframe_old[-1])
                surf_elevs.append(surf_elevs_old[-1])

            # try a second time, now starting at index 1
            if not any_merges:
                start_mframe = []
                stop_mframe = []
                surf_elevs = []
                start_mframe.append(start_mframe_old[0])
                stop_mframe.append(stop_mframe_old[0])
                surf_elevs.append(surf_elevs_old[0])

                for i in range(1,n_lakes-1,2):

                    is_closeby = ((start_mframe_old[i + 1] - stop_mframe_old[i]) <= max_dist_mframes)
                    is_at_same_elevation = (np.abs(surf_elevs_old[i + 1] - surf_elevs_old[i]) < max_dist_elev)

                    if debug: 
                        print('      %3i-%3i <> %3i-%3i | xdiff: %4d, close: %5s | %7.2f > %7.2f, hdiff: %7.2f, same: %5s' % \
                             (start_mframe_old[i]%1000, stop_mframe_old[i]%1000, 
                              start_mframe_old[i + 1]%1000, stop_mframe_old[i + 1]%1000,
                              start_mframe_old[i + 1] - stop_mframe_old[i],
                              is_closeby, surf_elevs_old[i], surf_elevs_old[i + 1], np.abs(surf_elevs_old[i + 1] - surf_elevs_old[i]),
                              is_at_same_elevation), end=' ')

                    # if merging two lakes, just append the combined stats as a single lake 
                    if (is_closeby & is_at_same_elevation):
                        start_mframe.append(start_mframe_old[i])
                        stop_mframe.append(stop_mframe_old[i + 1])
                        surf_elevs.append((surf_elevs_old[i] + surf_elevs_old[i+1]) / 2)
                        if debug: print('--> merge')
                        any_merges = True

                    # if keeping two lakes separate, add them both
                    else:
                        start_mframe += start_mframe_old[i:i+2]
                        stop_mframe += stop_mframe_old[i:i+2]
                        surf_elevs += surf_elevs_old[i:i+2]
                        if debug: print('--> keep separate')

                if n_lakes%2 == 0:
                    start_mframe.append(start_mframe_old[-1])
                    stop_mframe.append(stop_mframe_old[-1])
                    surf_elevs.append(surf_elevs_old[-1])

            n_lakes = len(surf_elevs)
            iteration += 1
        
        # compile dataframe for lakes found
        dflakes = pd.DataFrame({'mframe_start': start_mframe, 'mframe_end': stop_mframe, 'surf_elev': surf_elevs})
        
        # filter out the ones that are only a single major frame long
        df_extracted_lakes = dflakes[(dflakes.mframe_end-dflakes.mframe_start) > 0].copy()# .reset_index(inplace=True)
        
    except: 
        print('Something went wrong getting densities and peaks for mframe %i ...' % mframe)
        traceback.print_exc()
        df_extracted_lakes = pd.DataFrame({'mframe_start': [], 'mframe_end': [], 'surf_elev': []})
    
    return df_extracted_lakes


##########################################################################################
# check surroundings around lakes to extend them if needed
def check_lake_surroundings(df_mframe, df_extracted_lakes, n_check=3, elev_tol=0.2): 
    
    print('extending lake', end=' ')
    for i in range(len(df_extracted_lakes)):
    
        print(' %i:'%i, end='')
        thislake = df_extracted_lakes.iloc[i]
        thiselev = thislake['surf_elev']

        # check for extending before
        extent_before = thislake['mframe_start']
        check_before = extent_before - 1
        left_to_check = n_check
        while left_to_check > 0:
            # print('check!')
            if np.abs(df_mframe.loc[check_before, 'peak'] - thiselev) < elev_tol:
                extent_before = check_before
                left_to_check = n_check
                print('<',end='')
            else:
                left_to_check -= 1

            check_before -= 1

        df_extracted_lakes.loc[i, 'mframe_start'] = extent_before

        # check for extending after
        extent_after = thislake['mframe_end']
        check_after = extent_after + 1
        left_to_check = n_check
        while left_to_check > 0:
            # print('check!')
            if np.abs(df_mframe['peak'].loc[check_after] - thiselev) < elev_tol:
                extent_after = check_after
                left_to_check = n_check
                print('>',end='')
            else:
                left_to_check -= 1

            check_after += 1

        df_extracted_lakes.loc[i, 'mframe_end'] = extent_after

    # expand each lake by two major frames
    print(' ')
    df_extracted_lakes['mframe_start'] -= 2
    df_extracted_lakes['mframe_end'] += 2
    
    
##########################################################################################
def calculate_remaining_densities(df, df_mframe, df_extracted_lakes):
    
    dfs_to_calculate_densities = []
    for i in range(len(df_extracted_lakes)):

        thislake = df_extracted_lakes.iloc[i]
        extent_start = thislake['mframe_start']
        extent_end = thislake['mframe_end']

        dfs_to_calculate_densities.append(df_mframe[(df_mframe.index >= extent_start) & (df_mframe.index <= extent_end)])

    df_to_calculate_densities = pd.concat(dfs_to_calculate_densities)
    df_to_calculate_densities = df_to_calculate_densities[~df_to_calculate_densities['has_densities']]

    get_densities_and_2nd_peaks(df, df_mframe, df_to_calculate_densities, print_results=False)
    

##########################################################################################
def plot_found_lakes(df, df_mframe, df_extracted_lakes, ancillary, gtx, polygon, fig_dir='figs/', verbose=False,
                     min_width=100, min_depth=2.4):

    plt.close('all')

    for i in range(len(df_extracted_lakes)):
        
        thislake = df_extracted_lakes.iloc[i]
        thiselev = thislake['surf_elev']
        extent_start = thislake['mframe_start']
        extent_end = thislake['mframe_end']

        # subset the dataframes to the current lake extent
        df_lake = df[(df['mframe'] >= extent_start) & (df['mframe'] <= extent_end)]
        df_mframe_lake = df_mframe[(df_mframe.index >= extent_start) & (df_mframe.index <= extent_end)]
        h_2nds = [v for l in list(df_mframe_lake['h_2nd_returns']) for v in l]
        xatc_2nds = [v for l in list(df_mframe_lake['xatc_2nd_returns']) for v in l]
        prom_2nds = [v for l in list(df_mframe_lake['proms_2nd_returns']) for v in l]

        # get statistics
        # average mframe quality summary excluding the two mframes for buffer on each side
        lake_poly = polygon[polygon.rfind('/')+1 : polygon.find('.geojson')]
        lake_quality = np.sum(df_mframe_lake['quality_summary']) / (len(df_mframe_lake) - 4)
        lake_time = convert_time_to_string(df_mframe_lake['dt'].mean())
        lake_lat = df_mframe_lake['lat'].mean()
        lake_lat_min = df_mframe_lake['lat'].min()
        lake_lat_max = df_mframe_lake['lat'].max()
        lake_lat_str = '%.5f°N'%(lake_lat) if lake_lat>=0 else '%.5f°S'%(-lake_lat)
        lake_lon = df_mframe_lake['lon'].mean()
        lake_lon_min = df_mframe_lake['lon'].min()
        lake_lon_max = df_mframe_lake['lon'].max()
        lake_lon_str = '%.5f°E'%(lake_lon) if lake_lon>=0 else '%.5f°W'%(-lake_lon)
        lake_gtx = gtx
        lake_track = ancillary['rgt']
        lake_cycle = ancillary['cycle_number']
        lake_beam_strength = ancillary['gtx_strength_dict'][lake_gtx]
        lake_beam_nr = ancillary['gtx_beam_dict'][lake_gtx]
        lake_minh = np.min((df_mframe_lake['peak'].min(), np.min(h_2nds)))
        h_range = thiselev - lake_minh
        lake_max_depth = thiselev - np.min(h_2nds)
        lake_segment_length = np.max(xatc_2nds) - np.min(xatc_2nds)
        lake_maxh = np.min((df_mframe_lake['peak'].max(), thiselev+5*h_range))
        buffer_top = np.max((0.2*h_range, 1.0))
        buffer_bottom = np.max((0.3*h_range, 2.0))
        lake_minh_plot = lake_minh - buffer_bottom
        lake_maxh_plot = lake_maxh + buffer_top
        mptyp = 'arctic' if lake_lat>=0 else 'antarctic'
        lake_oa_url = 'https://openaltimetry.org/data/icesat2/elevation?product=ATL03&zoom_level=7&tab=photon&'
        lake_oa_url += 'date={date}&minx={minx}&miny={miny}&maxx={maxx}&maxy={maxy}&tracks={track}&mapType={mptyp}&beams={beam_nr}'.format(
                date=lake_time[:10], minx=lake_lon_min, miny=lake_lat_min, maxx=lake_lon_max, maxy=lake_lat_max,
                track=lake_track, mptyp=mptyp, beam_nr=lake_beam_nr)
        
        # plot only if criteria are fulfilled
        if (lake_max_depth > min_depth) & (lake_segment_length > min_width):
            fig, ax = plt.subplots(figsize=[9, 5], dpi=100)

            ax.scatter(df_lake.xatc-df_lake.xatc.min(), df_lake.h, s=6, c='k', alpha=0.05, edgecolors='none')
            scatt = ax.scatter(df_lake.xatc-df_lake.xatc.min(), df_lake.h, s=3, c=df_lake.snr, alpha=1, edgecolors='none',
                               cmap=cmc.lajolla,vmin=0,vmax=1)

            # plot surface elevation
            xmin, xmax = ax.get_xlim()
            ax.plot([xmin, xmax], [thiselev, thiselev], 'g-', lw=0.5)

            # plot mframe bounds
            ymin, ymax = ax.get_ylim()
            mframe_bounds_xatc = list(df_mframe_lake['xatc_min']) + [df_mframe_lake['xatc_max'].iloc[-1]]
            for xmframe in mframe_bounds_xatc:
                ax.plot([xmframe-df_lake.xatc.min(), xmframe-df_lake.xatc.min()], [ymin, ymax], 'k-', lw=0.5)

            dfpass = df_mframe_lake[df_mframe_lake['lake_qual_pass']]
            dfnopass = df_mframe_lake[~df_mframe_lake['lake_qual_pass']]
            ax.plot(dfpass.xatc-df_lake.xatc.min(), dfpass.peak, marker='o', mfc='g', mec='g', linestyle = 'None', ms=5)
            ax.plot(dfnopass.xatc-df_lake.xatc.min(), dfnopass.peak, marker='o', mfc='none', mec='r', linestyle = 'None', ms=3)

            for j, prom in enumerate(prom_2nds):
                ax.plot(xatc_2nds[j]-df_lake.xatc.min(), h_2nds[j], marker='o', mfc='none', mec='b', linestyle = 'None', ms=prom*4)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.05)
            fig.colorbar(scatt, cax=cax, orientation='vertical')

            ax.set_ylim((lake_minh_plot, lake_maxh_plot))
            ax.set_xlim((0.0, df_mframe_lake['xatc_max'].iloc[-1] - df_lake.xatc.min()))

            ax.set_title('Lake at (%s, %s) on %s\nICESat-2 track %d %s (%s), cycle %d [lake quality: %.2f]' % \
                         (lake_lat_str, lake_lon_str, lake_time, lake_track, lake_gtx,lake_beam_strength, lake_cycle, lake_quality))
            ax.set_ylabel('elevation above geoid [m]')
            ax.set_xlabel('along-track distance [m]')

            # save figure
            if not os.path.exists(fig_dir): os.makedirs(fig_dir)
            epoch = df_mframe_lake['dt'].mean() + datetime.datetime.timestamp(datetime.datetime(2018,1,1))
            dateid = datetime.datetime.fromtimestamp(epoch).strftime("%Y%m%d-%H%M%S")
            granid = ancillary['granule_id'][:-3]
            latid = '%dN'%(int(np.round(lake_lat*1e5))) if lake_lat>=0 else '%dS'%(-int(np.round(lake_lat*1e5)))
            lonid = '%dE'%(int(np.round(lake_lon*1e5))) if lake_lon>=0 else '%dS'%(-int(np.round(lake_lon*1e5)))
            figname = fig_dir + 'lake_%s_%s_%s_%s-%s.jpg' % (lake_poly, granid, lake_gtx, latid, lonid)
            fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        
        if verbose:
            print('\nLAKE %i:' % i)
            print('    - %s' % lake_time)
            print('    - %s, %s' % (lake_lat_str, lake_lon_str))
            print('    - segment length: %.1f km' % (lake_segment_length/1000))
            print('    - max depth: %.1f m' % lake_max_depth)
            print('    - quality: %.2f' % lake_quality)
            print('    - quick look: %s' % lake_oa_url)
            
            
##########################################################################################
def prnt(df_lakes):
    print('results:')
    try:
        if len(df_lakes) == 0: print('<<<   NO LAKES :(   >>>')
        else:
            for i in df_lakes.index: 
                print('  lake %i: %2i-%2i (h=%7.2f)' % (i, df_lakes.loc[i, 'mframe_start']%100, 
                       df_lakes.loc[i, 'mframe_end']%100, df_lakes.loc[i, 'surf_elev']))
    except:
        print('Something went wrong here...' % mframe)
        traceback.print_exc()

            
##########################################################################################
def detect_lakes(photon_data, gtx, ancillary, polygon, verbose=False):

    # get the data frame for the gtx and aggregate info at major frame level
    df = photon_data[gtx]
    df_mframe = make_mframe_df(df)
    
    # get all the flat segments and select
    print('\n-----------------------------------------------------------------------------\n')
    print('PROCESSING GROUND TRACK: %s (%s)' % (gtx, ancillary['gtx_strength_dict'][gtx]))
    print('---> finding flat surfaces', end=' ')
    df_mframe['peak'], df_mframe['is_flat'] = find_flat_lake_surfaces(df_mframe, df)
    print('(%i / %i were flat)' % (df_mframe.is_flat.sum(), df_mframe.is_flat.count()))
    df_selected = df_mframe[df_mframe.is_flat]
    
    # calculate densities and find second peaks (where surface is flat)
    print('---> calculating densities and looking for second peaks')
    get_densities_and_2nd_peaks(df, df_mframe, df_selected, print_results=verbose)
    print('(%i / %i pass lake quality test)' % (df_mframe.lake_qual_pass.sum(), df_mframe.lake_qual_pass.count()))
    
    # calculate densities and find second peaks (two major frames around where passing quality lakes were detected)
    print('---> calculating densities for surrounding major frames')
    check_additional_segments(df, df_mframe, print_results=verbose)
    print('(%i / %i pass lake quality test)' % (df_mframe.lake_qual_pass.sum(), df_mframe.lake_qual_pass.count()))
    
    print('---> merging lake segments iteratively')
    df_lakes = merge_lakes(df_mframe, print_progress=verbose, debug=verbose)
    if df_lakes is None:
        return
    prnt(df_lakes)
    
    print('---> checking lake edges and extending lakes if they match')
    check_lake_surroundings(df_mframe, df_lakes)
    prnt(df_lakes)
    
    print('---> calculating remaining photon densities')
    calculate_remaining_densities(df, df_mframe, df_lakes)
    
    print('---> making plots of lakes found')
    plot_found_lakes(df, df_mframe, df_lakes, ancillary, gtx, polygon, fig_dir='figs/', verbose=verbose)
    
    return df_lakes, df_mframe, df