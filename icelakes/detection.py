import os
import h5py
import math
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
    df_mframe['alignment_penalty'] = 0.0
    df_mframe['range_penalty'] = 0.0
    df_mframe['length_penalty'] = 0.0
    df_mframe['quality_secondreturns'] = 0.0
    df_mframe['quality_summary'] = 0.0
    empty_list = []
    df_mframe['xatc_2nd_returns'] = df_mframe.apply(lambda _: empty_list.copy(), axis=1)
    df_mframe['proms_2nd_returns'] = df_mframe.apply(lambda _: empty_list.copy(), axis=1)
    df_mframe['h_2nd_returns'] = df_mframe.apply(lambda _: empty_list.copy(), axis=1)
    
    return df_mframe


##########################################################################################
def find_flat_lake_surfaces(df_mframe, df, bin_height_coarse=1.0, bin_height_fine=0.02, smoothing_histogram=0.2, buffer=4.0,
                            width_surf=0.1, width_buff=0.3, rel_dens_upper_thresh=6, rel_dens_lower_thresh=3,
                            min_phot=100, min_snr_surface=100):
    
    # initialize arrays for major-frame-level photon stats
    peak_locs = np.full(len(df_mframe), np.nan, dtype=np.double)
    is_flat = np.full_like(peak_locs, np.nan, dtype=np.bool_)
    surf_snr = np.full_like(peak_locs, np.nan, dtype=np.double)
    upper_snr = np.full_like(peak_locs, np.nan, dtype=np.double)
    lower_snr = np.full_like(peak_locs, np.nan, dtype=np.double)
    
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
            
            surf_snr[i] = snr_surface
            upper_snr[i] = rel_dens_upper
            lower_snr[i] = rel_dens_lower

            # check for flat surface, if found calculate SNR and look for bottom return
            is_flat_like_lake = (rel_dens_upper > rel_dens_upper_thresh) \
                                & (rel_dens_lower > rel_dens_lower_thresh) \
                                & (snr_surface > min_snr_surface)
            is_flat[i] = is_flat_like_lake
            
            # print('%3i, %5s, %4i, %4i, %4i' % (i, is_flat[i], snr_surface, rel_dens_lower, rel_dens_upper))
    
    df_mframe['peak'] = peak_locs
    df_mframe['is_flat'] = is_flat
    df_mframe['snr_surf'] = surf_snr
    df_mframe['snr_upper'] = upper_snr
    df_mframe['snr_lower'] = lower_snr

    return df_mframe


##########################################################################################
def get_densities_and_2nd_peaks(df, df_mframe, df_selected, gtx, ancillary, aspect=30, K_phot=10, dh_signal=0.2, n_subsegs=10,
                                bin_height_snr=0.1, smoothing_length=1.0, buffer=4.0, print_results=False):
    
    # somehow got duplicate indices (mframe values in index) in here
    # this shouldn't be the case because index was created using groupby on mframe 
    # below is a temporary fix ---> check out more what's wrong here
    df_mframe_selected = df_selected.copy()
    df_mframe_selected.drop_duplicates(subset=['xatc_min','xatc_max'], keep='first', inplace=True)
    df['specular'] = False
    
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
                dfsubseg = dfseg[selector_subseg].copy()

                # avoid looking for peaks when there's no / very little data
                if len(dfsubseg > 20):
                    
                    beam_strength = ancillary['gtx_strength_dict'][gtx]
                    if beam_strength == 'weak':
                        # _____________________________________________________________
                        # check for specular returns at 0.94 & 1.46 m below main peak (roughly???)
                        bin_h_spec = 0.01
                        smoothing_spec = 0.15
                        spec1 = 0.94
                        spec2 = 1.46
                        spec_tol = 0.1
                        rm_h = 0.1
                        bins_spec = np.arange(start=peak_loc2-10.0, stop=peak_loc2+5.0, step=bin_h_spec)
                        mid_spec = bins_spec[:-1] + 0.5 * bin_h_spec
                        hist_spec = np.histogram(dfsubseg.h, bins=bins_spec)
                        window_size = int(smoothing_spec/bin_h_spec)
                        hist_vals = hist_spec[0] / np.max(hist_spec[0])
                        hist_vals_smoothed = np.array(pd.Series(hist_vals).rolling(window_size,center=True,min_periods=1).mean())
                        hist_vals_smoothed /= np.max(hist_vals_smoothed)
                        peaks, peak_props = find_peaks(hist_vals_smoothed, height=0.1, distance=int(0.4/bin_h_spec), prominence=0.1)
                        if len(peaks) > 2:
                            peak_hs = mid_spec[peaks]
                            peak_proms = peak_props['prominences']
                            idx_3highest = np.flip(np.argsort(peak_proms))[:3]
                            prms = peak_proms[idx_3highest]
                            pks_h = np.sort(peak_hs[idx_3highest])
                            has_main_peak = np.abs(pks_h[2]-peak_loc2) < 0.3
                            has_1st_specular_return = np.abs(pks_h[2]-pks_h[1]-spec1) < spec_tol
                            has_2nd_specular_return = np.abs(pks_h[2]-pks_h[0]-spec2) < spec_tol
                            if has_main_peak & has_1st_specular_return & has_2nd_specular_return:
                                # print('specular return. pk-uppr=%5.2f, 1st=%5.2f, 2nd%5.2f, proms: %.2f, %.2f, %.2f' % \
                                #       (np.abs(pks_h[2]-peak_loc2), pks_h[2]-pks_h[1], pks_h[2]-pks_h[0], 
                                #        prms[0], prms[1], prms[2]))
                                correct_specular = True
                                is_upper_spec = (dfsubseg.h < (pks_h[1] + rm_h)) & (dfsubseg.h > (pks_h[1] - rm_h))
                                is_lower_spec = (dfsubseg.h < (pks_h[0] + rm_h)) & (dfsubseg.h > (pks_h[0] - rm_h))
                                is_specular = is_upper_spec | is_lower_spec
                                dfsubseg['specular'] = is_specular
                                dfsubseg.loc[dfsubseg['specular'], 'snr'] = 0.0
                                dfseg.loc[selector_subseg,'specular'] = is_specular
                        #______________________________________________________________

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
                    peaks, peak_props = find_peaks(snr_hist_smoothed, height=0.05, distance=int(0.5/bin_height_snr), prominence=0.05)
                    
                    if len(peaks) >= 2: 
                        has_surf_peak = np.min(np.abs(peak_loc2 - mid_subseg_snr[peaks])) < 0.4
                        if has_surf_peak: 
                            idx_surfpeak = np.argmin(np.abs(peak_loc2 - mid_subseg_snr[peaks]))
                            peak_props['prominences'][idx_surfpeak] = 0

                            # classify as second peak only if prominence is larger than $(prominence_threshold)
                            prominence_secondpeak = np.max(peak_props['prominences'])
                            prominence_threshold = 0.1
                            if prominence_secondpeak > prominence_threshold:

                                idx_2ndreturn = np.argmax(peak_props['prominences'])
                                secondpeak_h = mid_subseg_snr[peaks[idx_2ndreturn]]

                                # classify as second peak only if elevation is 1.1m lower than main peak (surface) 
                                # and higher than 50m below surface
                                if (secondpeak_h < (peak_loc2-1.1)) & (secondpeak_h > (peak_loc2-50.0)):
                                    secondpeak_xtac = subsegstart + subsegwidth/2
                                    n_2nd_returns += 1
                                    prominences.append(prominence_secondpeak)
                                    elev_2ndpeaks.append(secondpeak_h)
                                    subpeaks_xatc.append(secondpeak_xtac)

            df.loc[selector_segment,'specular'] = dfseg['specular']
            
            # keep only second returns that are 5 m or closer to the next one on either side 
            # (helps filter out random noise, but might in rare cases suppress a signal)
            maxdiff = 5.0
            if len(elev_2ndpeaks) > 0:
                if len(elev_2ndpeaks) > 2: # if there's at least 3 second returns, compare elevations and remove two-sided outliers
                    diffs = np.abs(np.diff(np.array(elev_2ndpeaks)))
                    right_diffs = np.array(list(diffs) + [np.abs(elev_2ndpeaks[-3]-elev_2ndpeaks[-1])])
                    left_diffs = np.array([np.abs(elev_2ndpeaks[2]-elev_2ndpeaks[0])] + list(diffs))
                    to_keep = (right_diffs < maxdiff) | (left_diffs < maxdiff)

                # just consider elevation difference if there's only two, remove if only one (shouldn't be the case...)
                elif len(elev_2ndpeaks) == 2:
                    to_keep = [True, True] if np.abs(elev_2ndpeaks[1] - elev_2ndpeaks[0]) < maxdiff else [False, False]
                elif len(elev_2ndpeaks) == 1:
                    to_keep = [False]

                n_2nd_returns = np.sum(to_keep)
                elev_2ndpeaks = np.array(elev_2ndpeaks)[to_keep]
                prominences = np.array(prominences)[to_keep]
                subpeaks_xatc = np.array(subpeaks_xatc)[to_keep]
            
            # get the second return qualities
            minqual = 0.05
            min_ratio_2nd_returns = 0.35
            quality_summary = 0.0
            range_penalty = 0.0
            alignment_penalty = 0.0
            length_penalty = 0.0
            quality_secondreturns = 0.0
            quality_pass = 'No'
            
            ratio_2nd_returns = len(elev_2ndpeaks) / n_subsegs
            # ________________________________________________________ 
            if (len(elev_2ndpeaks) > 2) & (ratio_2nd_returns > min_ratio_2nd_returns):
                h_range = np.max(elev_2ndpeaks) - np.min(elev_2ndpeaks)
                diffs = np.diff(elev_2ndpeaks)
                dirchange = np.abs(np.diff(np.sign(diffs))) > 1
                total_distance = 0.0
                for i,changed in enumerate(dirchange):
                    if changed: total_distance += (np.abs(diffs)[i] + np.abs(diffs)[i+1])/2
                alignment_penalty = 1.0 if total_distance==0 else\
                                    np.clip(np.clip(h_range, 0.5, None) / (total_distance + np.clip(h_range, 0.5, None)), 0, 1)
                range_penalty = np.clip(1/math.log(np.clip(h_range,1.1,None),5), 0, 1)
                length_penalty = (len(elev_2ndpeaks) / n_subsegs)**1.5
                quality_secondreturns = np.clip(np.mean(prominences) * ((np.clip(2*len(elev_2ndpeaks)/n_subsegs, 1, None)-1)*2+1), 0, 1)
                quality_summary = alignment_penalty * length_penalty * range_penalty * quality_secondreturns
    
            # ________________________________________________________
            
            df_mframe.loc[mframe, 'ratio_2nd_returns'] = ratio_2nd_returns
            df_mframe.loc[mframe, 'alignment_penalty'] = alignment_penalty
            df_mframe.loc[mframe, 'range_penalty'] = range_penalty
            df_mframe.loc[mframe, 'length_penalty'] = length_penalty
            df_mframe.loc[mframe, 'quality_secondreturns'] = quality_secondreturns
            df_mframe.loc[mframe, 'quality_summary'] = quality_summary
            
            if quality_summary > minqual: #& (yspread < max_yspread):
                quality_pass = 'Yes'
                df_mframe.loc[mframe, 'lake_qual_pass'] = True

            for i in range(len(elev_2ndpeaks)):
                df_mframe.loc[mframe, 'h_2nd_returns'].append(elev_2ndpeaks[i])
                df_mframe.loc[mframe, 'xatc_2nd_returns'].append(subpeaks_xatc[i])
                df_mframe.loc[mframe, 'proms_2nd_returns'].append(prominences[i])

            # if (percent_2d_returns >= 30) & (quality_secondreturns > 0.4):
            flatstring = 'Yes' if df_mframe['is_flat'].loc[mframe] else 'No'

            if print_results:
                txt  = '  mframe %03i: ' % (mframe%1000)
                txt += 'h=%7.2fm | ' % peak_loc2
                txt += 'flat=%3s | ' % flatstring
                txt += 'snrs=%4i,%4i,%4i | ' % (df_mframe.loc[mframe,'snr_surf'],df_mframe.loc[mframe,'snr_upper'],df_mframe.loc[mframe, 'snr_lower'])
                txt += '2nds=%3d%% | ' % np.round(length_penalty*100)
                txt += 'range=%4.2f ' % range_penalty
                txt += 'align=%4.2f ' % alignment_penalty
                txt += 'strength=%4.2f --> ' % quality_secondreturns
                txt += 'qual=%4.2f | ' % quality_summary
                txt += 'pass=%3s' % quality_pass
                print(txt)
            
            # adjust SNR values for specular returns
            df.loc[df['specular'], 'snr'] = 0.0
        
        except: 
            print('Something went wrong getting densities and peaks for mframe %i ...' % mframe)
            traceback.print_exc()
            
            
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

        # keep going until there is no change (i.e. no more segments can be merged further)
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
            
            # first, check non-overlapping pairs of segments: {0,1}, {2,3}, {4,5} ...
            # if n_lakes is uneven, this ignores the very last one
            for i in range(0,n_lakes-1,2):
                
                # merge lakes if they are close-by (in terms of mframe distance), and if elevations are similar
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
            
            # if n_lakes is uneven, we don't consider the very last lake for merging. so we need to keep it
            if n_lakes%2 == 1:
                start_mframe.append(start_mframe_old[-1])
                stop_mframe.append(stop_mframe_old[-1])
                surf_elevs.append(surf_elevs_old[-1])

            # if no success merging any lakes, now start comparing pairs with one index offset 
            # i.e.: compare non-overlapping pairs of segments : {1,2}, {3,4}, {5,6} ...
            if not any_merges:
                start_mframe = []
                stop_mframe = []
                surf_elevs = []
                
                # need to add lake 0, because we're not considering it for any merging
                start_mframe.append(start_mframe_old[0])
                stop_mframe.append(stop_mframe_old[0])
                surf_elevs.append(surf_elevs_old[0])
                
                # compare non-overlapping pairs of segments : {1,2}, {3,4}, {5,6} ...
                # this does not compare lake 0 to any others, if n_lakes is even it also ignores the very last one
                for i in range(1,n_lakes-1,2):
                    
                    # merge lakes if they are close-by (in terms of mframe distance), and if elevations are similar
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
                            
                # if n_lakes is even, we don't consider the very last lake for merging. so we need to keep it
                if n_lakes%2 == 0:
                    start_mframe.append(start_mframe_old[-1])
                    stop_mframe.append(stop_mframe_old[-1])
                    surf_elevs.append(surf_elevs_old[-1])

            n_lakes = len(surf_elevs)
            iteration += 1
        
        # compile dataframe for lakes found 
        df_extracted_lakes  = pd.DataFrame({'mframe_start': np.array(start_mframe), 
                                            'mframe_end': np.array(stop_mframe), 
                                            'surf_elev': np.array(surf_elevs)})
        
    except: 
        print('Something went wrong getting densities and peaks for mframe %i ...' % mframe)
        traceback.print_exc()
        df_extracted_lakes = pd.DataFrame({'mframe_start': [], 'mframe_end': [], 'surf_elev': []})
    
    return df_extracted_lakes


##########################################################################################
# check surroundings around lakes to extend them if needed (based on matching peak in surface elevation)
def check_lake_surroundings(df_mframe, df_extracted_lakes, n_check=3, elev_tol=0.2): 
    
    print('extending lake', end=' ')
    for i in range(len(df_extracted_lakes)):
        try:
            print(' %i:'%i, end='')
            thislake = df_extracted_lakes.iloc[i]
            thiselev = thislake['surf_elev']

            # check for extending before
            extent_before = int(thislake['mframe_start'])
            check_before = int(extent_before - 1)
            left_to_check = n_check
            while (left_to_check > 0) & (check_before in df_mframe.index):
                
                # if the peak of the adjacent major frame in within the tolerance threshold
                if np.abs(df_mframe.loc[int(check_before), 'peak'] - thiselev) <= elev_tol:
                    extent_before = check_before # add this major frame to the lake
                    left_to_check = n_check # reset the number of segments left to check back to the starting value
                    print('<',end='')
                else:
                    left_to_check -= 1
                check_before -= 1 # check the next major frame before (lower value) in the next iteration
            
            # set the starting value of the lake to the lowest number value that was found belonging to the lake
            df_extracted_lakes.loc[i, 'mframe_start'] = extent_before

            # check for extending after
            extent_after = int(thislake['mframe_end'])
            check_after = int(extent_after + 1)
            left_to_check = n_check
            while (left_to_check > 0) & (check_after in df_mframe.index):
                
                # if the peak of the adjacent major frame in within the tolerance threshold
                if np.abs(df_mframe.loc[int(check_after), 'peak'] - thiselev) < elev_tol:
                    extent_after = check_after # add this major frame to the lake
                    left_to_check = n_check # reset the number of segments left to check back to the starting value
                    print('>',end='')
                else:
                    left_to_check -= 1
                check_after += 1 # check the next major frame after (higher value) in the next iteration
            
            # set the end value of the lake to the highest number value that was found belonging to the lake
            df_extracted_lakes.loc[i, 'mframe_end'] = extent_after
            
        except:
            print('Something went wrong extending this lake %i ...' % i)
            traceback.print_exc()
            
    # limit to lakes longer than just one major frame
    longer_than1 = (df_extracted_lakes.mframe_end - df_extracted_lakes.mframe_start) > 0
    df_extracted_lakes = df_extracted_lakes[longer_than1].copy()
    df_extracted_lakes.reset_index(inplace=True)

    # expand each lake by two major frames (if these major frames exist)
    print(' ')
    istart = df_extracted_lakes.columns.get_loc('mframe_start')
    iend = df_extracted_lakes.columns.get_loc('mframe_end')
    
    for i in range(len(df_extracted_lakes)):
        thislake = df_extracted_lakes.iloc[i]
        
        # expand by two mframes to the left (if these mframes exist in data set)
        if int(thislake.mframe_start-2) in df_mframe.index:
            df_extracted_lakes.iloc[i, istart] -= 2
        elif int(thislake.mframe_start-1) in df_mframe.index:
            df_extracted_lakes.iloc[i, istart] -= 1
        
        # expand by two mframes to the right (if these mframes exist in data set)
        if int(thislake.mframe_end+2) in df_mframe.index:
            df_extracted_lakes.iloc[i, iend] += 2
        elif int(thislake.mframe_end+1) in df_mframe.index:
            df_extracted_lakes.iloc[i, iend] += 1
    
    return df_extracted_lakes
    
    
##########################################################################################
def calculate_remaining_densities(df, df_mframe, df_extracted_lakes, gtx, ancillary):
    
    dfs_to_calculate_densities = []
    for i in range(len(df_extracted_lakes)):

        thislake = df_extracted_lakes.iloc[i]
        extent_start = thislake['mframe_start']
        extent_end = thislake['mframe_end']

        dfs_to_calculate_densities.append(df_mframe[(df_mframe.index >= extent_start) & (df_mframe.index <= extent_end)])

    df_to_calculate_densities = pd.concat(dfs_to_calculate_densities)
    df_to_calculate_densities = df_to_calculate_densities[~df_to_calculate_densities['has_densities']]

    get_densities_and_2nd_peaks(df, df_mframe, df_to_calculate_densities, gtx, ancillary, print_results=False)
            
            
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
    df_mframe = find_flat_lake_surfaces(df_mframe, df)
    print('(%i / %i were flat)' % (df_mframe.is_flat.sum(), df_mframe.is_flat.count()))
    df_selected = df_mframe[df_mframe.is_flat]
    
    # calculate densities and find second peaks (where surface is flat)
    print('---> calculating densities and looking for second peaks')
    get_densities_and_2nd_peaks(df, df_mframe, df_selected, gtx, ancillary, print_results=verbose)
    print('(%i / %i pass lake quality test)' % (df_mframe.lake_qual_pass.sum(), df_mframe.lake_qual_pass.count()))
    
    print('---> merging lake segments iteratively')
    df_lakes = merge_lakes(df_mframe, print_progress=verbose, debug=verbose)
    if df_lakes is None:
        return df_lakes, df_mframe, df
    prnt(df_lakes)
    
    print('---> checking lake edges and extending lakes if they match')
    df_lakes = check_lake_surroundings(df_mframe, df_lakes)
    prnt(df_lakes)
    
    print('---> calculating remaining photon densities')
    calculate_remaining_densities(df, df_mframe, df_lakes, gtx, ancillary)
    
    # print('---> making plots of lakes found')
    # plot_found_lakes(df, df_mframe, df_lakes, ancillary, gtx, polygon, fig_dir='figs/', verbose=verbose)
    
    thelakes = []
    if df_lakes is not None:
        for i in range(len(df_lakes)):
            lakedata = df_lakes.iloc[i]
            thislake = melt_lake(lakedata.mframe_start, lakedata.mframe_end, lakedata.surf_elev)
            thislake.add_data(df, df_mframe, gtx, ancillary, polygon)
            thelakes.append(thislake)
    
    
    return thelakes


##########################################################################################
class melt_lake:
    def __init__(self, mframe_start, mframe_end, main_peak):
        self.mframe_start = int(mframe_start)
        self.mframe_end = int(mframe_end)
        self.main_peak = main_peak

    
    #-------------------------------------------------------------------------------------
    def add_data(self, df, df_mframe, gtx, ancillary, polygon):
        
        # useful metadata
        self.granule_id = ancillary['granule_id']
        self.rgt = ancillary['rgt']
        self.gtx = gtx
        self.polygon_filename = polygon
        self.polygon_name = polygon[polygon.rfind('/')+1 : polygon.find('.geojson')]
        self.beam_number = ancillary['gtx_beam_dict'][self.gtx]
        self.beam_strength = ancillary['gtx_strength_dict'][self.gtx]
        self.cycle_number = ancillary['cycle_number']
        self.sc_orient = ancillary['sc_orient']
        
        # add the data frames at the photon level and at the major frame level
        self.photon_data = df[(df['mframe'] >= self.mframe_start) & (df['mframe'] <= self.mframe_end)].copy()
        self.mframe_data = df_mframe[(df_mframe.index >= self.mframe_start) & (df_mframe.index <= self.mframe_end)].copy()
        self.date_time = convert_time_to_string(self.mframe_data['dt'].mean())
        self.photon_data.reset_index(inplace=True)
        
        # compile the second returns in simple arrays
        h_2nds = np.array([v for l in list(self.mframe_data['h_2nd_returns'])[2:-2] for v in l])
        xatc_2nds = np.array([v for l in list(self.mframe_data['xatc_2nd_returns'])[2:-2] for v in l])
        prom_2nds = np.array([v for l in list(self.mframe_data['proms_2nd_returns'])[2:-2] for v in l])
        self.detection_2nd_returns = {'h':h_2nds, 'xatc':xatc_2nds, 'prom':prom_2nds}
        
        # add general lat/lon info for the whole lake
        # self.detection_quality = np.sum(self.mframe_data['quality_summary']) / (len(self.mframe_data) - 4)
        self.lat = self.mframe_data['lat'].mean()
        self.lat_min = self.mframe_data['lat'].min()
        self.lat_max = self.mframe_data['lat'].max()
        self.lat_str = '%.5f°N'%(self.lat) if self.lat>=0 else '%.5f°S'%(-self.lat)
        self.lon = self.mframe_data['lon'].mean()
        self.lon_min = self.mframe_data['lon'].min()
        self.lon_max = self.mframe_data['lon'].max()
        self.lon_str = '%.5f°E'%(self.lon) if self.lon>=0 else '%.5f°W'%(-self.lon)
        
        # get the ice sheet and the melt season
        self.ice_sheet = 'GrIS' if self.lat>=0 else 'AIS'
        meltseason = 'XX'
        if self.ice_sheet=='GrIS':
            meltseason = self.date_time[:4]
        elif self.ice_sheet=='AIS':
            thismonth = int(self.date_time[5:7])
            thisyear = int(self.date_time[:4])
            if thismonth > 6:
                meltseason = str(thisyear) + '-' + str((thisyear+1)%100)
            elif thismonth <= 6:
                meltseason = str(thisyear-1) + '-' + str(thisyear%100)
        self.melt_season = meltseason
        
        # quick-look link to OpenAltimetry
        mptyp = 'arctic' if self.lat>=0 else 'antarctic'
        lake_oa_url = 'https://openaltimetry.org/data/icesat2/elevation?product=ATL03&zoom_level=7&tab=photon&'
        lake_oa_url += 'date={date}&minx={minx}&miny={miny}&maxx={maxx}&maxy={maxy}&tracks={track}&mapType={mptyp}&beams={beam_nr}'.format(
                date=self.date_time[:10], minx=self.lon_min, miny=self.lat_min, maxx=self.lon_max, maxy=self.lat_max,
                track=self.rgt, mptyp=mptyp, beam_nr=self.beam_number)
        self.oaurl = lake_oa_url