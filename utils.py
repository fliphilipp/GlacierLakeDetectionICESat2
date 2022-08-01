#########################################################################################
# Utility functions for supraglacial melt lakes detection & depth retrieval on OSG      # 
# Author: Philipp S. Arndt, Scripps Polar Center, UCSD                                  #
#########################################################################################

def download_granule_nsidc(granule_id, shapefile, granule_output_path, uid, pwd): 
    """
    Download a single ICESat-2 ATL03 granule based on its producer ID,
    subsets it to a given shapefile, and puts it into the specified
    output directory as a .h5 file. A NASA earthdata user id (uid), and
    the associated password are required. 

    Parameters
    ----------
    granule_id : string
        the producer_granule_id for CMR search
    shapefile : string
        filepath to the shapefile used for spatial subsetting
    granule_output_path : string
        folder in which to save the subsetted granule
    uid : string
        earthdata user id
    pwd : string
        the associated password

    Returns
    -------
    nothing

    Examples
    --------
    >>> download_granule_nsidc(granule_id='ATL03_20210715182907_03381203_005_01.h5', 
                               shapefile='/shapefiles/jakobshavn.shp', 
                               granule_output_path='/IS2data', 
                               uid='myuserid', 
                               pwd='mypasword')
    """
    
    import requests
    import getpass
    import socket 
    import json
    import zipfile
    import io
    import math
    import os
    import shutil
    import pprint
    import re
    import time
    import geopandas as gpd
    import fiona
    from shapely.geometry import Polygon, mapping
    from shapely.geometry.polygon import orient
    from requests.auth import HTTPBasicAuth
    from xml.etree import ElementTree as ET
    import sys
    import numpy as np
    
    short_name = 'ATL03'
    version = granule_id[30:33]
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    capability_url = f'https://n5eil02u.ecs.nsidc.org/egi/capabilities/{short_name}.{version}.xml'
    base_url = 'https://n5eil02u.ecs.nsidc.org/egi/request'
    
    shapefile_filepath = str(os.getcwd() + shapefile)
    
    # set the variables for subsetting
    vars_sub = ['/ancillary_data/atlas_sdp_gps_epoch',
                '/orbit_info/rgt',
                '/orbit_info/cycle_number',
                '/orbit_info/sc_orient',
                '/gtx/geolocation/ph_index_beg',
                '/gtx/geolocation/segment_dist_x',
                '/gtx/geolocation/segment_length',
                '/gtx/geophys_corr/dem_h',
                '/gtx/geophys_corr/geoid',
                '/gtx/bckgrd_atlas/pce_mframe_cnt',
                '/gtx/bckgrd_atlas/bckgrd_counts',
                '/gtx/bckgrd_atlas/bckgrd_int_height',
                '/gtx/bckgrd_atlas/delta_time',
                '/gtx/heights/lat_ph',
                '/gtx/heights/lon_ph',
                '/gtx/heights/h_ph',
                '/gtx/heights/dist_ph_along',
                '/gtx/heights/delta_time',
                '/gtx/heights/pce_mframe_cnt',
                '/gtx/heights/quality_ph']
    beam_list = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
    var_list = sum([[v.replace('/gtx','/'+bm) for bm in beam_list] if '/gtx' in v else [v] for v in vars_sub],[])
    
    # search for the given granule
    search_params = {
        'short_name': short_name,
        'page_size': 100,
        'page_num': 1,
        'producer_granule_id': granule_id}

    granules = []
    headers={'Accept': 'application/json'}
    while True:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)

        if len(results['feed']['entry']) == 0:
            # Out of results, so break out of loop
            break

        # Collect results and increment page_num
        granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    print('\nDownloading ICESat-2 data. Found granules:')
    for result in granules:
        print('  '+result['producer_granule_id'], f', {float(result["granule_size"]):.2f} MB',sep='')
        
    # Use geopandas to read in polygon file as GeoDataFrame object 
    # Note: a KML or geojson, or almost any other vector-based spatial data format could be substituted here.
    gdf = gpd.read_file(shapefile_filepath)
    
    # Simplify polygon for complex shapes in order to pass a reasonable request length to CMR. 
    # The larger the tolerance value, the more simplified the polygon.
    # Orient counter-clockwise: CMR polygon points need to be provided in counter-clockwise order. 
    # The last point should match the first point to close the polygon.
    poly = orient(gdf.simplify(0.05, preserve_topology=False).loc[0],sign=1.0)

    geojson = gpd.GeoSeries(poly).to_json() # Convert to geojson
    geojson = geojson.replace(' ', '') #remove spaces for API call
    
    #Format dictionary to polygon coordinate pairs for CMR polygon filtering
    polygon = ','.join([str(c) for xy in zip(*poly.exterior.coords.xy) for c in xy])
    
    print('\nInput shapefile:', shapefile)
    print('Simplified polygon coordinates based on shapefile input:', polygon)
    
    # Create session to store cookie and pass credentials to capabilities url
    session = requests.session()
    s = session.get(capability_url)
    response = session.get(s.url,auth=(uid,pwd))

    root = ET.fromstring(response.content)

    #collect lists with each service option
    subagent = [subset_agent.attrib for subset_agent in root.iter('SubsetAgent')]
    
    # this is for getting possible variable values from the granule search
    if len(subagent) > 0 :
        # variable subsetting
        variables = [SubsetVariable.attrib for SubsetVariable in root.iter('SubsetVariable')]  
        variables_raw = [variables[i]['value'] for i in range(len(variables))]
        variables_join = [''.join(('/',v)) if v.startswith('/') == False else v for v in variables_raw] 
        variable_vals = [v.replace(':', '/') for v in variables_join]
    
    # make sure to only request the variables that are available
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    var_list_subsetting = intersection(variable_vals,var_list)
    
    if len(subagent) < 1 :
        print('No services exist for', short_name, 'version', latest_version)
        agent = 'NO'
        coverage,Boundingshape = '',''
    else:
        agent = ''
        subdict = subagent[0]
        if subdict['spatialSubsettingShapefile'] == 'true':
            Boundingshape = geojson
        else:
            Boundingshape = ''
        coverage = ','.join(var_list_subsetting)
        
    page_size = 100
    request_mode = 'stream'
    page_num = math.ceil(len(granules)/page_size)

    param_dict = {'short_name': short_name, 
                  'producer_granule_id': granule_id,
                  'version': version,  
                  'polygon': polygon,
                  'Boundingshape': Boundingshape,  
                  'Coverage': coverage, 
                  'page_size': page_size, 
                  'request_mode': request_mode, 
                  'agent': agent, 
                  'email': 'yes'}

    #Remove blank key-value-pairs
    param_dict = {k: v for k, v in param_dict.items() if v != ''}

    #Convert to string
    param_string = '&'.join("{!s}={!r}".format(k,v) for (k,v) in param_dict.items())
    param_string = param_string.replace("'","")

    #Print API base URL + request parameters
    endpoint_list = [] 
    for i in range(page_num):
        page_val = i + 1
        API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
        endpoint_list.append(API_request)

    print('\nAPI request URL:')
    print(*endpoint_list, sep = "\n") 
    
    # Create an output folder if the folder does not already exist.
    path = str(os.getcwd() + granule_output_path)
    if not os.path.exists(path):
        os.mkdir(path)

    # Different access methods depending on request mode:
    for i in range(page_num):
        page_val = i + 1
        print('\nOrder: ', page_val)
        print('Requesting...')
        request = session.get(base_url, params=param_dict)
        print('HTTP response from order response URL: ', request.status_code)
        request.raise_for_status()
        d = request.headers['content-disposition']
        fname = re.findall('filename=(.+)', d)
        dirname = os.path.join(path,fname[0].strip('\"'))
        print('Downloading...')
        open(dirname, 'wb').write(request.content)
        print('Data request', page_val, 'is complete.')

    # Unzip outputs
    for z in os.listdir(path): 
        if z.endswith('.zip'): 
            zip_name = path + "/" + z 
            zip_ref = zipfile.ZipFile(zip_name) 
            zip_ref.extractall(path) 
            zip_ref.close() 
            os.remove(zip_name) 

    # Clean up Outputs folder by removing individual granule folders 
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), path)
            except OSError:
                pass
        for name in dirs:
            os.rmdir(os.path.join(root, name))
            
    print('\nUnzipped files and cleaned up directory.')
    print('Output data saved in:', granule_output_path)
            
    return

class edc:
    u = 'arndtp'
    p = 'My5ecretPw0rd!'

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
    >>> read_atl03(filename=processed_ATL03_20210715182907_03381203_005_01.h5, geoid_h=True)
    """
    
    import h5py
    import pandas as pd
    import numpy as np
    
    print('  reading in', filename)
    
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
        

    ancillary = {'atlas_sdp_gps_epoch': f['ancillary_data']['atlas_sdp_gps_epoch'][0],
                 'rgt': f['orbit_info']['rgt'][0],
                 'cycle_number': f['orbit_info']['cycle_number'][0],
                 'sc_orient': orient_str,
                 'gtx_beam_dict': gtx_beam_dict,
                 'gtx_strength_dict': gtx_strength_dict}
    
    # loop through all beams
    print('  reading in beam:', end=' ')
    for beam in beamlist:
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
                geoid = np.interp(np.array(df.xatc), segment_dist_x+0.5*segment_length, geophys_geoid)
                df['h'] = df.h - geoid
                df['geoid'] = geoid

            #### save to list of dataframes
            dfs[beam] = df
            dfs_bckgrd[beam] = df_bckgrd
            print(beam, end=' ')
        except Exception as e:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam), e)
    print(' --> done.')
    return dfs, dfs_bckgrd, ancillary