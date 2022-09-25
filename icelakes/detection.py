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
def read_atl03(filename, geoid_h=True, gtxs_to_read='all'):
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
    beams_available = [x for x in list(f.keys()) if 'gt' in x]
    if gtxs_to_read=='all':
        beamlist = beams_available
    elif gtxs_to_read=='none':
        beamlist = []
    else:
        if type(gtxs_to_read)==list: beamlist = gtxs_to_read
        elif type(gtxs_to_read)==str: beamlist = [gtxs_to_read]
        else: beamlist = beams_available
    
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
                 'gtx_strength_dict': gtx_strength_dict,
                 'gtx_dead_time_dict': {}}

    # loop through all beams
    print('  reading in beam:', end=' ')
    for beam in beamlist:
        
        print(beam, end=' ')
        try:
            
            if gtx_strength_dict[beam]=='strong':
                ancillary['gtx_dead_time_dict'][beam] = np.mean(np.array(f['ancillary_data']['calibrations']['dead_time'][beam]['dead_time'])[:16])
            else:
                ancillary['gtx_dead_time_dict'][beam] = np.mean(np.array(f['ancillary_data']['calibrations']['dead_time'][beam]['dead_time'])[16:])
               
            #### get photon-level data
            # if "/%s/heights/" not in f: break; # 
             
            df = pd.DataFrame({'lat': np.array(f[beam]['heights']['lat_ph']),
                               'lon': np.array(f[beam]['heights']['lon_ph']),
                               'h': np.array(f[beam]['heights']['h_ph']),
                               'dt': np.array(f[beam]['heights']['delta_time']),
                               # 'conf': np.array(f[beam]['heights']['signal_conf_ph'][:,conf_landice]),
                               # not using ATL03 confidences here
                               'mframe': np.array(f[beam]['heights']['pce_mframe_cnt']),
                               'ph_id_pulse': np.array(f[beam]['heights']['ph_id_pulse']),
                               'qual': np.array(f[beam]['heights']['quality_ph'])}) 
                               # 0=nominal,1=afterpulse,2=impulse_response_effect,3=tep

#             df_bckgrd = pd.DataFrame({'pce_mframe_cnt': np.array(f[beam]['bckgrd_atlas']['pce_mframe_cnt']),
#                                       'bckgrd_counts': np.array(f[beam]['bckgrd_atlas']['bckgrd_counts']),
#                                       'bckgrd_int_height': np.array(f[beam]['bckgrd_atlas']['bckgrd_int_height']),
#                                       'delta_time': np.array(f[beam]['bckgrd_atlas']['delta_time'])})

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
            #Mdfs_bckgrd[beam] = df_bckgrd
        
        except:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam))
            traceback.print_exc()
            
    f.close()
    print(' --> done.')
    if len(beamlist)==0:
        return beams_available, ancillary
    else:
        return beams_available, ancillary, dfs

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
    df_mframe['peak'] = np.nan
    df_mframe['is_flat'] = False
    df_mframe['snr_surf'] = 0.0
    df_mframe['snr_upper'] = 0.0
    df_mframe['snr_lower'] = 0.0
    df_mframe['snr_allabove'] = 0.0
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
def find_flat_lake_surfaces(df_mframe, df, bin_height_coarse=0.2, bin_height_fine=0.01, smoothing_histogram=0.1, buffer=2.0,
                            width_surf=0.1, width_buff=0.35, rel_dens_upper_thresh=5, rel_dens_lower_thresh=2,
                            min_phot=30, min_snr_surface=10, min_snr_vs_all_above=100):
    
    print('---> finding flat surfaces in photon data', end=' ')
    
    # initialize arrays for major-frame-level photon stats
    peak_locs = np.full(len(df_mframe), np.nan, dtype=np.double)
    is_flat = np.full_like(peak_locs, False, dtype=np.bool_)
    surf_snr = np.full_like(peak_locs, 0.0, dtype=np.double)
    upper_snr = np.full_like(peak_locs, 0.0, dtype=np.double)
    lower_snr = np.full_like(peak_locs, 0.0, dtype=np.double)
    all_above_snr = np.full_like(peak_locs, 0.0, dtype=np.double)
    
    for i, mframe in enumerate(df_mframe.index):
        
        try:
        
            # select the photons in the major frame
            selector_segment = (df.mframe == mframe)
            dfseg = df[selector_segment]

            # check if there are enough photons in the segment
            if len(dfseg) < min_phot:
                is_flat[i] = False

            # find peaks
            else:
                # find main broad peak
                ##############################################################################################
                #             ******************working version******************
                #             bins_coarse1 = np.arange(start=dfseg.h.min(), stop=dfseg.h.max(), step=bin_height_coarse)
                #             hist_mid1 = bins_coarse1[:-1] + 0.5 * bin_height_coarse
                #             peak_loc1 = hist_mid1[np.argmax(np.histogram(dfseg.h, bins=bins_coarse1)[0])]
                ##############################################################################################
                promininece_threshold = 0.1
                bins_coarse1 = np.arange(start=dfseg.h.min(), stop=dfseg.h.max(), step=bin_height_coarse)
                hist_mid1 = bins_coarse1[:-1] + 0.5 * bin_height_coarse
                broad_hist = np.array(pd.Series(np.histogram(dfseg.h, bins=bins_coarse1)[0]).rolling(3,center=True,min_periods=1).mean())
                broad_hist /= np.max(broad_hist)
                peaks, peak_props = find_peaks(broad_hist, height=promininece_threshold, distance=1.0, prominence=promininece_threshold)
                peak_hs = hist_mid1[peaks]
                if len(peaks) > 1:
                    peak_proms = peak_props['prominences']
                    idx_2highest = np.flip(np.argsort(peak_proms))[:2]
                    pks_h = np.sort(peak_hs[idx_2highest])
                    peak_loc1 = np.max(pks_h)
                else:
                    peak_loc1 = hist_mid1[np.argmax(broad_hist)]

                ##############################################################################################           

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
                sum_all_above = np.sum(dfseg.h > peak_upper)
                h_range_all_above = dfseg.h.max() - peak_upper
                noise_rate_all_above = sum_all_above / h_range_all_above
                signal_rate = sum_peak / (width_surf*2)
                rel_dens_upper = 1000 if sum_above==0 else signal_rate / (sum_above / width_buff)
                rel_dens_lower = 1000 if sum_below==0 else signal_rate / (sum_below / width_buff)
                noise_rate = (dfseg.h.count() - sum_peak) / (dfseg.h.max() - dfseg.h.min() - width_surf*2)
                snr_surface = signal_rate / noise_rate
                snr_allabove = signal_rate / noise_rate_all_above

                surf_snr[i] = snr_surface
                upper_snr[i] = rel_dens_upper
                lower_snr[i] = rel_dens_lower
                all_above_snr[i] = snr_allabove

                # check for flat surface, if found calculate SNR and look for bottom return
                is_flat_like_lake = (rel_dens_upper > rel_dens_upper_thresh) \
                                    & (rel_dens_lower > rel_dens_lower_thresh) \
                                    & (snr_surface > min_snr_surface) \
                                    & (snr_allabove > min_snr_vs_all_above)
                is_flat[i] = is_flat_like_lake

                # print('%4i, %5s, %4i, %4i, %4i' % (mframe, is_flat[i], snr_surface, rel_dens_lower, rel_dens_upper))
                
        except: 
            print('Something went wrong with checking if mframe %i is flat...' % mframe)
            traceback.print_exc()

    df_mframe['peak'] = peak_locs
    df_mframe['is_flat'] = is_flat
    df_mframe['snr_surf'] = surf_snr
    df_mframe['snr_upper'] = upper_snr
    df_mframe['snr_lower'] = lower_snr
    df_mframe['snr_allabove'] = all_above_snr
    
    print('(%i / %i were flat)' % (df_mframe.is_flat.sum(), df_mframe.is_flat.count()))

    return df_mframe


##########################################################################################
def get_densities_and_2nd_peaks(df, df_mframe, df_selected, gtx, ancillary, aspect=30, K_phot=10, dh_signal=0.2, n_subsegs=10,
                                bin_height_snr=0.1, smoothing_length=1.0, buffer=4.0, print_results=False):
    
    print('---> calculating photon densities & looking for second density peaks below the surface')
    
    # somehow got duplicate indices (mframe values in index) in here
    # this shouldn't be the case because index was created using groupby on mframe 
    # below is a temporary fix ---> check out more what's wrong here
    df_mframe_selected = df_selected.copy()
    df_mframe_selected.drop_duplicates(subset=['xatc_min','xatc_max'], keep='first', inplace=True)
    df['specular'] = False
    
    for mframe in df_mframe_selected.index:
        if np.sum(df.mframe == mframe) > 50:
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
                #kdt = KDTree(Xn, leaf_size=40)
                kdt = KDTree(Xn)
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
                    if len(dfsubseg > 5):

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
                            if len(hist_spec[0]) < 1:
                                break
                            if np.max(hist_spec[0]) == 0:
                                break
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
                        try:
                            snrstats = binned_statistic(dfsubseg.h, dfsubseg.snr, statistic='median', bins=bins_subseg_snr)
                        except ValueError:  #raised if empty
                            pass
                        snr_median = snrstats[0]
                        snr_median[np.isnan(snr_median)] = 0
                        window_size_sub = int(smoothing_length/bin_height_snr)
                        snr_vals_smoothed = np.array(pd.Series(snr_median).rolling(window_size_sub,center=True,min_periods=1).mean())
                        if len(snr_vals_smoothed) < 1:
                            break
                        if np.max(snr_vals_smoothed) == 0:
                            break
                            
                        snr_vals_smoothed /= np.nanmax(snr_vals_smoothed)

                        # take histogram binning values into account, but clip surface peak to second highest peak height
                        subhist, subhist_edges = np.histogram(dfsubseg.h, bins=bins_subseg_snr)
                        subhist_nosurface = subhist.copy()
                        subhist_nosurface[(mid_subseg_snr < (peak_loc2+2*dh_signal)) & (mid_subseg_snr > (peak_loc2-2*dh_signal))] = 0
                        subhist_nosurface_smoothed = np.array(pd.Series(subhist_nosurface).rolling(window_size_sub,center=True,min_periods=1).mean())
                        if len(subhist_nosurface_smoothed) < 1:
                            break
                        if np.max(subhist_nosurface_smoothed) == 0:
                            break
                        subhist_max = subhist_nosurface_smoothed.max()
                        subhist_smoothed = np.array(pd.Series(subhist).rolling(window_size_sub,center=True,min_periods=1).mean())
                        subhist_smoothed = np.clip(subhist_smoothed, 0, subhist_max)
                        if np.max(subhist_smoothed) == 0:
                            break
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
                maxdiff = 3.0
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
                minqual = 0.1
                min_ratio_2nd_returns = 0.25
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
                
    print('(%i / %i pass lake quality test)' % (df_mframe.lake_qual_pass.sum(), df_mframe.lake_qual_pass.count()))
            
            
##########################################################################################
# merge detected lake segments iteratively
def merge_lakes(df_mframe, max_dist_mframes=7, max_dist_elev=0.1, print_progress=False, debug=False):
    
    print('---> merging major frame segments that possibly represent lakes iteratively')

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
    
    print('---> checking lake edges and extending them if the surface elevation matches')
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
    
    print('---> calculating remaining photon densities')
    dfs_to_calculate_densities = []
    for i in range(len(df_extracted_lakes)):

        thislake = df_extracted_lakes.iloc[i]
        extent_start = thislake['mframe_start']
        extent_end = thislake['mframe_end']

        dfs_to_calculate_densities.append(df_mframe[(df_mframe.index >= extent_start) & (df_mframe.index <= extent_end)])
    
    if len(dfs_to_calculate_densities) > 0:
        df_to_calculate_densities = pd.concat(dfs_to_calculate_densities)
        df_to_calculate_densities = df_to_calculate_densities[~df_to_calculate_densities['has_densities']]

        get_densities_and_2nd_peaks(df, df_mframe, df_to_calculate_densities, gtx, ancillary, print_results=False)
            
            
##########################################################################################
def print_results(lake_list, gtx):
    print(('results for : %s' % gtx).upper())
    try:
        if len(lake_list) == 0: print('<<<   SAD. NO LAKES :(   >>>')
        else:
            for i,lake in enumerate(lake_list): 
                print('  lake %4i (%11s,%11s) length: %4.1f km, surface elevation: %7.2f m, quality: %.5f)' % (i, 
                                                   lake.lat_str, lake.lon_str, lake.length_extent/1000, 
                                                   lake.surface_elevation, lake.detection_quality))
    except:
        print('Something went wrong here... You may want to check that out.')
        traceback.print_exc()

            
##########################################################################################
def remove_duplicate_lakes(list_of_lakes, df, df_mframe, gtx, ancillary, polygon, nsubsegs, verbose=False):
    
    def ranges_overlap(range1, range2):
        range1, range2 = np.sort(range1), np.sort(range2) 
        return not ((range1[1] < range2[0]) | (range1[0] > range2[1]))

    # go backwards through list
    for i in np.arange(len(list_of_lakes)-1,0,-1):
        if i < len(list_of_lakes):
        
            lk2 = list_of_lakes[i]
            lk1 = list_of_lakes[i-1]

            if i==len(list_of_lakes):
                break;

            # just delete if no surface extent
            if (len(lk1.surface_extent_detection)==0) | (len(lk2.surface_extent_detection)==0):
                if verbose: print('found lake with no continuous surface extent --> tossing this one out.')
                if len(lk2.surface_extent_detection)==0: del list_of_lakes[i]
                if len(lk1.surface_extent_detection)==0: 
                    list_of_lakes[i-1] = lk2
                    del list_of_lakes[i]
                
            else:  
                # if they fully overlap
                if ranges_overlap(lk1.full_lat_extent_detection, lk2.full_lat_extent_detection):

                    # merge if  surface elevation within 0.5 m
                    if np.abs(lk1.surface_elevation - lk2.surface_elevation) < 0.3:
                        if verbose:
                            print('merging two lakes with overlapping surfaces...')
                        mframe_start = int(np.min((lk1.mframe_start, lk2.mframe_start)))
                        mframe_end = int(np.max((lk1.mframe_end, lk2.mframe_end)))
                        surf_elev = lk1.surface_elevation
                        newlake = melt_lake(mframe_start, mframe_end, surf_elev, nsubsegs)
                        newlake.add_data(df, df_mframe, gtx, ancillary, polygon)
                        newlake.get_surface_elevation()
                        newlake.get_surface_extent()
                        newlake.calc_quality_lake()
                        list_of_lakes[i-1] = newlake
                        del list_of_lakes[i]

                    # else delete the one with lower quality 
                    else:
                        if verbose:
                            print('found overlapping lakes that can\'t be merged. something\'s off here...')
                        if lk2.detection_quality > lk1.detection_quality:
                            list_of_lakes[i-1] = lk2
                        del list_of_lakes[i]

                # if the the lake surfaces don't overlap, but the attached lake segments do...
                else: 
                    if verbose:
                            print('lakes overlapping, but only in side lobes ...trimming extra photon data')
                    # if the data segments overlap, trim them 
                    lk1_datarange = [lk1.lat_min, lk1.lat_max]
                    lk2_datarange = [lk2.lat_min, lk2.lat_max]
                    if ranges_overlap(lk1_datarange, lk2_datarange):
                        lk_lowlat, lk_highlat = np.array([lk1, lk2])[np.argsort([lk1.lat, lk2.lat])]
                        if lk1.lat < lk2.lat: lk_lowlat, lk_highlat = lk1, lk2
                        if lk2.lat < lk1.lat: lk_lowlat, lk_highlat = lk2, lk1    
                        lk_lowlat.photon_data = lk_lowlat.photon_data[lk_lowlat.photon_data.lat < np.max(lk_highlat.full_lat_extent_detection)]
                        lk_highlat.photon_data = lk_highlat.photon_data[lk_highlat.photon_data.lat > np.min(lk_lowlat.full_lat_extent_detection)]
                        if lk1.lat < lk2.lat: list_of_lakes[i-1], list_of_lakes[i] = lk_lowlat, lk_highlat
                        if lk2.lat < lk1.lat: list_of_lakes[i], list_of_lakes[i-1] = lk_lowlat, lk_highlat

    return list_of_lakes


##########################################################################################
def get_gtx_stats(df_ph, lake_list):
    n_photons_total = df_ph.h.count()
    length_total = df_ph.xatc.max() - df_ph.xatc.min()
    n_photons_lakes = 0.0
    length_lakes = 0.0
    for lake in lake_list:
        length_lakes += lake.length_water_surfaces
        n_photons_lakes += lake.n_photons_where_water
    gtx_stats = {'length_total': length_total, 'length_lakes': length_lakes, 
                 'n_photons_total': n_photons_total, 'n_photons_lakes': n_photons_lakes}
    return gtx_stats


##########################################################################################
def detect_lakes(input_filename, gtx, polygon, verbose=False):
    
    gtx_list, ancillary, photon_data = read_atl03(input_filename, geoid_h=True, gtxs_to_read=gtx)
    
    print('\n-----------------------------------------------------------------------------\n')
    print('PROCESSING GROUND TRACK: %s (%s)' % (gtx, ancillary['gtx_strength_dict'][gtx]))

    # get the data frame for the gtx and aggregate info at major frame level
    df = photon_data[gtx]
    df_mframe = make_mframe_df(df)
    
    # get all the flat segments and select
    df_mframe = find_flat_lake_surfaces(df_mframe, df)
    df_selected = df_mframe[df_mframe.is_flat]
    
    # calculate densities and find second peaks (where surface is flat)
    nsubsegs = 10
    get_densities_and_2nd_peaks(df, df_mframe, df_selected, gtx, ancillary, n_subsegs=nsubsegs, print_results=verbose)
    
    # iteratively merge the detected segments into lakes 
    df_lakes = merge_lakes(df_mframe, print_progress=verbose, debug=verbose)
    if df_lakes is None: return [], {}
    df_lakes = check_lake_surroundings(df_mframe, df_lakes)
    calculate_remaining_densities(df, df_mframe, df_lakes, gtx, ancillary)
    
    # create a list of lake object, and calculate some stats for each
    thelakes = []
    if df_lakes is not None:
        for i in range(len(df_lakes)):
            lakedata = df_lakes.iloc[i]
            thislake = melt_lake(lakedata.mframe_start, lakedata.mframe_end, lakedata.surf_elev, nsubsegs)
            thislake.add_data(df, df_mframe, gtx, ancillary, polygon)
            thislake.get_surface_elevation()
            thislake.get_surface_extent()
            thislake.calc_quality_lake()
            thelakes.append(thislake)
    
    # remove any duplicates and make sure data segments don't overlap into other lakes' water surfaces
    thelakes = remove_duplicate_lakes(thelakes, df, df_mframe, gtx, ancillary, polygon, nsubsegs, verbose=verbose)          
    print_results(thelakes, gtx)
    
    # get gtx stats
    gtx_stats = get_gtx_stats(df, thelakes)
    
    return thelakes, gtx_stats


##########################################################################################
class melt_lake:
    def __init__(self, mframe_start, mframe_end, main_peak, nsubsegs):
        self.mframe_start = int(mframe_start)
        self.mframe_end = int(mframe_end)
        self.main_peak = main_peak
        self.n_subsegs_per_mframe = nsubsegs

    
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
        self.dead_time = ancillary['gtx_dead_time_dict'][self.gtx]
        speed_of_light = 299792458 # m s^-1
        self.dead_time_meters = speed_of_light * self.dead_time / 2.0
        
        # add the data frames at the photon level and at the major frame level
        self.photon_data = df[(df['mframe'] >= self.mframe_start) & (df['mframe'] <= self.mframe_end)].copy()
        self.mframe_data = df_mframe[(df_mframe.index >= self.mframe_start) & (df_mframe.index <= self.mframe_end)].copy()
        self.date_time = convert_time_to_string(self.mframe_data['dt'].mean())
        self.photon_data.reset_index(inplace=True, drop=True)
        
        # reset the xatc values to start at zero
        min_xatc = self.photon_data.xatc.min()
        self.photon_data['xatc'] -= min_xatc
        self.mframe_data['xatc_min'] -= min_xatc
        self.mframe_data['xatc_max'] -= min_xatc
        self.mframe_data['xatc'] = (self.mframe_data['xatc_min'] + self.mframe_data['xatc_max']) / 2
        
        # compile the second returns in simple arrays
        h_2nds = np.array([v for l in list(self.mframe_data['h_2nd_returns']) for v in l])
        xatc_2nds = np.array([v for l in list(self.mframe_data['xatc_2nd_returns']) for v in l])
        prom_2nds = np.array([v for l in list(self.mframe_data['proms_2nd_returns']) for v in l])
        temp_dict = {'h':h_2nds, 'xatc':xatc_2nds - min_xatc, 'prom':prom_2nds}
        self.detection_2nd_returns = pd.DataFrame(temp_dict).sort_values(by='xatc').to_dict(orient='list')
        
        self.len_subsegs = np.mean(np.abs(np.diff(self.mframe_data.xatc_min[2:-1]))) / self.n_subsegs_per_mframe
        
        to_remove = ['has_densities', 'xatc_2nd_returns', 'proms_2nd_returns', 'h_2nd_returns']
        self.mframe_data.drop(columns=to_remove, inplace=True)
        
        # add general lat/lon info for the whole lake
        self.lat_min = self.photon_data['lat'].min()
        self.lat_max = self.photon_data['lat'].max()
        self.lat = (self.mframe_data['lat'].min() + self.mframe_data['lat'].max()) / 2
        self.lat_str = '%.5f°N'%(self.lat) if self.lat>=0 else '%.5f°S'%(-self.lat)
        self.lon_min = self.photon_data['lon'].min()
        self.lon_max = self.photon_data['lon'].max()
        self.lon = (self.mframe_data['lon'].min() + self.mframe_data['lon'].max()) / 2
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

        
    #-------------------------------------------------------------------------------------
    def get_surface_elevation(self, search_width=1.0, bin_h=0.005, smoothing=0.1):
        selector = (self.photon_data.h < (self.main_peak+search_width)) & (self.photon_data.h > (self.main_peak-search_width))
        heights = self.photon_data.h[selector]
        bins = np.arange(start=self.main_peak-search_width, stop=self.main_peak+search_width, step=bin_h)
        mid = bins[:-1] + 0.5 * bin_h
        hist = np.histogram(heights, bins=bins)
        window_size = int(smoothing/bin_h)
        hist_vals_smoothed = np.array(pd.Series(hist[0]).rolling(window_size,center=True,min_periods=1).mean())
        self.surface_elevation = mid[np.argmax(hist_vals_smoothed)]

        
    #-------------------------------------------------------------------------------------
    def get_surface_extent(self, surf_width=0.45, abov_width=2.0, bin_width=1.0, smooth=31, max_ratio=0.1, min_length=100.0):
        if smooth%2 == 1: smooth += 1
        surf_selector = (self.photon_data.h > (self.surface_elevation-surf_width/2)) & (self.photon_data.h < (self.surface_elevation+surf_width/2))
        abov_selector = (self.photon_data.h > (self.surface_elevation+surf_width/2)) & (self.photon_data.h < (self.surface_elevation+surf_width/2+abov_width))
        totl_selector = (self.photon_data.h < (self.surface_elevation-surf_width/2)) | (self.photon_data.h > (self.surface_elevation+surf_width/2))
        df_surf = self.photon_data[surf_selector]
        df_abov = self.photon_data[abov_selector]
        df_totl = self.photon_data[totl_selector]
        bins = np.arange(start=self.photon_data.xatc.min(), stop=self.photon_data.xatc.max(), step=bin_width)
        mids = bins[:-1] + 0.5 * bin_width
        hist_surf = np.histogram(df_surf.xatc, bins=bins)
        hist_abov = np.histogram(df_abov.xatc, bins=bins)
        hist_totl = np.histogram(df_totl.xatc, bins=bins)
        max_all = binned_statistic(self.photon_data.xatc, self.photon_data.h, statistic='max', bins=bins)
        min_all = binned_statistic(self.photon_data.xatc, self.photon_data.h, statistic='min', bins=bins)
        surf_smooth = np.array(pd.Series(hist_surf[0]).rolling(smooth,center=True,min_periods=1).mean())
        abov_smooth = np.array(pd.Series(hist_abov[0]).rolling(smooth,center=True,min_periods=1).mean())
        totl_smooth = np.array(pd.Series(hist_totl[0]).rolling(smooth,center=True,min_periods=1).mean())
        maxs_smooth = np.array(pd.Series(max_all[0]).rolling(smooth,center=True,min_periods=1).max())
        mins_smooth = np.array(pd.Series(min_all[0]).rolling(smooth,center=True,min_periods=1).min())
        dens_surf = surf_smooth / (surf_width*bin_width)
        dens_abov = abov_smooth / (abov_width*bin_width)
        dens_totl = totl_smooth / ((maxs_smooth-mins_smooth-surf_width)*bin_width)
        dens_surf[dens_surf == 0] = 1e-20
        dens_ratio_abov = dens_abov / dens_surf
        dens_ratio_totl = dens_totl / dens_surf
        dens_ratio_abov[dens_ratio_abov>1] = 1
        dens_ratio_totl[dens_ratio_totl>1] = 1
        dens_eval = np.max(np.vstack((dens_ratio_abov,dens_ratio_totl)), axis=0)
        surf_possible = dens_eval < max_ratio
        surf_possible[(mids<250) | (mids>(np.max(mids)-250))] = False # because we added two extra major frames on each side

        # get surface segments that are continuous for longer than x meters
        current_list = []
        surface_segs = []
        i = 0
        while i < len(surf_possible):
            if surf_possible[i]:
                current_list.append(i)
            else:
                if (len(current_list) * bin_width) > min_length:
                    surface_segs.append([mids[current_list[0]], mids[current_list[-1]]])
                current_list = []
            i += 1
        self.surface_extent_detection = surface_segs
        
        # get length extent of water surface, and of the entire lake
        length_water_surfaces = 0.0
        n_photons_where_water = 0
        for xt in surface_segs:
            xtmin = np.min(xt)
            xtmax = np.max(xt)
            length_water_surfaces += (xtmax - xtmin)
            n_photons_where_water += np.sum((self.photon_data.xatc > xtmin) & (self.photon_data.xatc < xtmax))
        self.length_water_surfaces = length_water_surfaces
        self.n_photons_where_water = n_photons_where_water
        self.length_extent = 0.0 if len(surface_segs)<1 else np.abs(surface_segs[0][0]-surface_segs[-1][-1])
        
        # get surface extent in lats also
        if len(surface_segs) > 0:
            dfinterp = self.mframe_data[['xatc','lat']].sort_values(by='xatc')
            latextsinterp = np.interp(self.surface_extent_detection, dfinterp.xatc, dfinterp.lat)
            if latextsinterp[0][0] < latextsinterp[-1][-1]:
                latexts = [list(x) for x in latextsinterp]
            else:
                latexts = [list(x)[::-1] for x in list(latextsinterp)[::-1]]
            self.lat_surface_extent_detection = latexts
            self.full_lat_extent_detection = [latexts[0][0], latexts[-1][-1]]
        else:
            self.lat_surface_extent_detection = []
            self.full_lat_extent_detection = []

        # remove second returns that don't fall under the extent of the surface
        h = self.detection_2nd_returns['h']
        xatc = self.detection_2nd_returns['xatc']
        prom = self.detection_2nd_returns['prom']
        is_in_extent = np.full_like(h,False,dtype=np.bool_)
        for ext in self.surface_extent_detection:
            under_ext = (self.detection_2nd_returns['xatc'] >= ext[0]) & (self.detection_2nd_returns['xatc'] <= ext[1])
            is_in_extent[under_ext] = True
        for k in self.detection_2nd_returns.keys():
            self.detection_2nd_returns[k] = list(np.array(self.detection_2nd_returns[k])[is_in_extent])
        
        
    #-------------------------------------------------------------------------------------
    def calc_quality_lake(self, min_2nd_returns=5, len_qual_limit=1000.0, depth_qual_limit=7.0, h_range_limit=2.0,
                      depth_determination_percentile=70, verbose=False):

        total_quality = 0.0
        quality_props = {'strength_2nd_returns': 0.0,
                         'h_range_2nd_returns': 0.0,
                         'lake_length': 0.0,
                         'lake_depth': 0.0,
                         'qual_alignment': 0.0
                        }
        hs = self.detection_2nd_returns['h']
        n_second_returns = len(hs)
        if n_second_returns >= min_2nd_returns:
            
            # range (penalty if all the values cluster at same height, as often seen with saturated pulses that don't actually 
            # have a second reflective surface, but an afterpulse)
            hrange = np.percentile(hs,85) - np.percentile(hs,15)
            qual_hrange = ((hrange-0.2) / h_range_limit)**0.5
            if qual_hrange == np.nan: qual_hrange = 0.0
            quality_props['h_range_2nd_returns'] = np.clip(qual_hrange, 0, 1)

            # length (cap at a kilometer)
            length_extent = 0
            for ext in self.surface_extent_detection:
                length_extent += (np.max(ext) - np.min(ext))
            qual_lake_length = 0.1 + 0.9 * (length_extent/len_qual_limit)**0.5
            quality_props['lake_length'] = np.clip(qual_lake_length, 0, 1)
            if verbose: print(qual_lake_length)

            # depth (cap at 10)
            depth = np.percentile(np.abs(hs - self.surface_elevation), depth_determination_percentile)
            qual_lake_depth = 0.3 + 0.7 * ((depth-1.0)/(depth_qual_limit-1.0))**0.5
            quality_props['lake_depth'] = np.clip(qual_lake_depth, 0, 1)
            if verbose: print(qual_lake_depth)

            # average of prominences of second peaks in extent (absence counted a zero)
            qual_strength_2nd_returns = np.sum(self.detection_2nd_returns['prom']) / (length_extent / self.len_subsegs)
            quality_props['strength_2nd_returns'] = np.clip(qual_strength_2nd_returns, 0, 1)**0.5
            if verbose: print(qual_strength_2nd_returns, 'total:', self.len_subsegs)

            # alignment: penalize randomness , need to find a good way to normalize it
            def get_alignment_qual(h_vals, change_tolerance=0.1):
                diffs = np.diff(h_vals)
                std = np.std(h_vals)
                dirchange = np.abs(np.diff(np.sign(diffs))) > 1
                total_distance = 0.0
                for i,changed in enumerate(dirchange):
                    change1 = np.clip(np.abs(diffs)[i  ] - change_tolerance, 0.0, None)
                    change2 = np.clip(np.abs(diffs)[i+1] - change_tolerance, 0.0, None)
                    if changed: total_distance += (change1 + change2)/2
                fraction_of_totally_random = 1 - (2.0 * total_distance / (std * (n_second_returns-2)))
                qual_alignment = np.clip(fraction_of_totally_random, 0, 1)**2
                return qual_alignment
            qual_alignment = get_alignment_qual(hs)
            quality_props['qual_alignment'] = np.clip(qual_alignment, 0, 1)
            if verbose: print(qual_alignment)

            total_quality = quality_props['lake_length'] * quality_props['lake_depth'] * \
                            quality_props['strength_2nd_returns'] * quality_props['qual_alignment'] * \
                            quality_props['h_range_2nd_returns']
            
            if verbose: 
                print('          lake_depth: %5.2f m -->  %4.2f' % (depth, qual_lake_depth))
                print('         lake_length: %5i m -->  %4.2f' % (length_extent, qual_lake_length))
                print('strength_2nd_returns:             %5.2f' % qual_strength_2nd_returns)
                print('      qual_alignment:             %5.2f' % qual_alignment)
                print('         qual_hrange:             %5.2f' % qual_hrange)
                print('       TOTAL QUALITY:             %5.2f' % total_quality)
                print(' ')
        else:
            if verbose: print('Not enough second returns to estimate quality... Setting to zero.')
        
        if np.isnan(total_quality): total_quality = 0.0
        for k in quality_props.keys():
            if np.isnan(quality_props[k]): quality_props[k] = 0.0
        self.detection_quality_info = quality_props
        self.detection_quality = total_quality

        return self.detection_quality, self.detection_quality_info
    

    #-------------------------------------------------------------------------------------
    def plot_detected(self, fig_dir='figs', verbose=False, min_width=0.0, min_depth=0.0, print_mframe_info=True):

        import matplotlib
        from matplotlib.patches import Rectangle

        if len(self.detection_2nd_returns['h'])>0:
            lake_minh = np.min(self.detection_2nd_returns['h'])
        else: return
        lake_max_depth = np.abs(self.main_peak - np.min(self.detection_2nd_returns['h']))
        lake_segment_length = np.abs(np.max(self.detection_2nd_returns['xatc']) - np.min(self.detection_2nd_returns['xatc']))
        lake_maxh = np.min((self.mframe_data['peak'].max(), self.main_peak+0.5*lake_max_depth))
        buffer_bottom = np.max((0.5*lake_max_depth, 2.0))
        lake_minh_plot = lake_minh - buffer_bottom
        buffer_top = (lake_maxh - lake_minh_plot) * 0.5
        lake_maxh_plot = lake_maxh + buffer_top
        ylms = (lake_minh_plot, lake_maxh_plot)
        xlms = (0.0, self.mframe_data.xatc_max.max())

        if (lake_max_depth > min_depth) & (lake_segment_length > min_width):
            fig, ax = plt.subplots(figsize=[9, 5], dpi=100)

            # plot the ATL03 photon data
            scatt = ax.scatter(self.photon_data.xatc, self.photon_data.h,s=5, c=self.photon_data.snr, alpha=1, 
                               edgecolors='none', cmap=cmc.lajolla, vmin=0, vmax=1)
            p_scatt = ax.scatter([-9999]*4, [-9999]*4, s=15, c=[0.0,0.25,0.75,1.0], alpha=1, edgecolors='none', cmap=cmc.lajolla, 
                                 vmin=0, vmax=1, label='ATL03 photons')

            # plot surface elevation
            for xtent in self.surface_extent_detection:
                ax.plot(xtent, [self.surface_elevation, self.surface_elevation], 'g-', lw=3)
            p_surf_elev, = ax.plot([-9999]*2, [-9999]*2, 'g-', lw=3, label='lake surface')

            # plot the second returns from detection
            for j, prom in enumerate(self.detection_2nd_returns['prom']):
                ax.plot(self.detection_2nd_returns['xatc'][j], self.detection_2nd_returns['h'][j], 
                                        marker='o', mfc='none', mec='b', linestyle = 'None', ms=prom*8)
            p_2nd_return, = ax.plot(-9999, -9999, marker='o', mfc='none', mec='b', ls='None', ms=3, label='second returns')

            # plot mframe bounds
            ymin, ymax = ax.get_ylim()
            mframe_bounds_xatc = list(self.mframe_data['xatc_min']) + [self.mframe_data['xatc_max'].iloc[-1]]
            for xmframe in mframe_bounds_xatc:
                ax.plot([xmframe, xmframe], [ymin, ymax], 'k-', lw=0.5)

            # visualize which segments initially passed
            for i, passing in enumerate(self.mframe_data['lake_qual_pass']):
                mf = self.mframe_data.iloc[i]
                if passing:
                    xy = (mf.xatc_min, ylms[0])
                    width = mf.xatc_max - mf.xatc_min
                    height = ylms[1] - ylms[0]
                    rct = Rectangle(xy, width, height, ec=(1,1,1,0), fc=(0,0,1,0.1), zorder=-1000, label='major frame passed lake check')
                    p_passed = ax.add_patch(rct)
                p_mfpeak, = ax.plot((mf.xatc_min,mf.xatc_max), (mf.peak,mf.peak),'k-',lw=0.5, label='major frame peak')

            # add a legend
            hdls = [p_scatt, p_surf_elev, p_2nd_return, p_mfpeak, p_passed]
            ax.legend(handles=hdls, loc='lower left', fontsize=7, scatterpoints=4)

            # add the colorbar 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='4%', pad=0.05)
            cbar = fig.colorbar(scatt, cax=cax, orientation='vertical')
            cbar.ax.get_yaxis().set_ticks([])
            for j, lab in enumerate([0.2, 0.4, 0.6, 0.8]):
                cbar.ax.text(.5, lab, '%.1f'%lab, ha='center', va='center', fontweight='black')
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.set_ylabel('photon density', rotation=270, fontsize=8)

            # add labels and description in title
            txt  = 'ICESat-2 Lake Detection: %s, ' % ('Greenland Ice Sheet' if self.lat>=0 else 'Antarctic Ice Sheet')
            txt += '%s Melt Season' % self.melt_season
            fig.suptitle(txt, y=0.95, fontsize=14)

            txt  = 'location: %s, %s (area: %s) | ' % (self.lat_str, self.lon_str, self.polygon_name)
            txt += 'time: %s UTC | surface elevation: %.2f m\n' % (self.date_time, self.surface_elevation)
            txt += 'RGT %s %s cycle %i | ' % (self.rgt, self.gtx.upper(), self.cycle_number)
            txt += 'beam %i (%s, %s spacecraft orientation) | ' % (self.beam_number, self.beam_strength, self.sc_orient)
            txt += 'granule ID: %s' % self.granule_id
            ax.set_title(txt, fontsize=8)

            ax.set_ylabel('elevation above geoid [m]',fontsize=8)
            ax.tick_params(axis='x', which='major', labelsize=7)
            ax.tick_params(axis='y', which='major', labelsize=6)
            # set limits
            ax.set_ylim(ylms)
            ax.set_xlim(xlms)

            # add latitude
            #_________________________________________________________
            lx = self.photon_data.sort_values(by='lat').iloc[[0,-1]][['lat','xatc']].reset_index(drop=True)
            lat = np.array(lx.lat)
            xatc = np.array(lx.xatc)
            def forward(x):
                return lat[0] + x * (lat[1] - lat[0]) / (xatc[1] - xatc[0])
            def inverse(l):
                return xatc[0] + l * (xatc[1] - xatc[0]) / (lat[1] - lat[0])
            secax = ax.secondary_xaxis(-0.065, functions=(forward, inverse))
            secax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            secax.set_xlabel('latitude / along-track distance',fontsize=8,labelpad=0)
            secax.tick_params(axis='both', which='major', labelsize=7)
            secax.ticklabel_format(useOffset=False) # show actual readable latitude values

            # rename x ticks
            xticklabs = ['%g km' % (xt/1000) for xt in list(ax.get_xticks())]
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels(xticklabs)

            # add mframe info text
            if print_mframe_info:
                txt  = 'mframe:\n' % (mf.name%1000)
                txt += 'photons:\n' % mf.n_phot
                txt += 'peak:\n'
                txt += 'flat:\n'
                txt += 'SNR surf all:\n'
                txt += 'SNR surf above:\n'
                txt += 'SNR up:\n'
                txt += 'SNR low:\n'
                txt += '2nds:\n'
                txt += '2nds strength:\n'
                txt += '2nds number:\n'
                txt += '2nds spread:\n'
                txt += '2nds align:\n'
                txt += '2nds quality:\n'
                txt += 'pass:'
                # trans = ax.get_xaxis_transform()
                bbox = {'fc':(1,1,1,0.75), 'ec':(1,1,1,0), 'pad':1}
                ax.text(-0.005, 0.98, txt, transform=ax.transAxes, fontsize=4, ha='right', va='top', bbox=bbox)
                for i,loc in enumerate(self.mframe_data['xatc']):
                    mf = self.mframe_data.iloc[i]
                    txt  = '%i\n' % (mf.name%1000)
                    txt += '%i\n' % mf.n_phot
                    txt += '%.2f\n' % mf.peak
                    txt += '%s\n' % ('Yes.' if mf.is_flat else 'No.')
                    txt += '%i\n' % np.round(mf.snr_surf)
                    txt += '%i\n' % np.round(mf.snr_allabove)
                    txt += '%i\n' % np.round(mf.snr_upper)
                    txt += '%i\n' % np.round(mf.snr_lower)
                    txt += '%i%%\n' % np.round(mf.ratio_2nd_returns*100)
                    txt += '%.2f\n' % mf.quality_secondreturns
                    txt += '%.2f\n' % mf.length_penalty
                    txt += '%.2f\n' % mf.range_penalty
                    txt += '%.2f\n' % mf.alignment_penalty
                    txt += '%.2f\n' % mf.quality_summary
                    txt += '%s' % ('Yes.' if mf.lake_qual_pass else 'No.')
                    trans = ax.get_xaxis_transform()
                    bbox = {'fc':(1,1,1,0.75), 'ec':(1,1,1,0), 'pad':1}
                    ax.text(loc, 0.98, txt, transform=trans, fontsize=4,ha='center', va='top', bbox=bbox)

            # add detection quality description
            txt  = 'LAKE QUALITY: %6.4f'%self.detection_quality
            txt += '\n---------------------------\n'
            txt += '2nd returns: %6.4f\n' % self.detection_quality_info['strength_2nd_returns']
            txt += 'alignment: %6.4f\n' % self.detection_quality_info['qual_alignment']
            txt += 'depth: %6.4f\n' % self.detection_quality_info['lake_depth']
            txt += 'length: %6.4f\n' % self.detection_quality_info['lake_length']
            txt += 'depth range: %6.4f' % self.detection_quality_info['h_range_2nd_returns']
            bbox = {'fc':(1,1,1,0.75), 'ec':(1,1,1,0), 'pad':1}
            ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',fontsize=6, weight='bold', bbox=bbox)

            fig.patch.set_facecolor('white')
            fig.tight_layout()
            ax.set_ylim(ylms)
            ax.set_xlim(xlms)

            plt.close(fig)

            return fig