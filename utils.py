#########################################################################################
# Utility functions for supraglacial melt lakes detection & depth retrieval on OSG      # 
# Author: Philipp S. Arndt, Scripps Polar Center, UCSD                                  #
#########################################################################################

def download_granule_nsidc(granule_id, gtxs, shapefile, granule_output_path, uid, pwd): 
    """
    Download a single ICESat-2 ATL03 granule based on its producer ID,
    subsets it to a given shapefile, and puts it into the specified
    output directory as a .h5 file. A NASA earthdata user id (uid), and
    the associated password are required. 

    Parameters
    ----------
    granule_id : string
        the producer_granule_id for CMR search
    gtxs : string or list
        the ground tracks to request
        possible values:
            'gt1l' or 'gt1r' or 'gt2l', ... (single gtx)
            ['gt1l', 'gt3r', ...] (list of gtxs)
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
                               gtxs='gt1l'
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
    
    if type(gtxs) == str:
        var_list = [v.replace('/gtx','/'+gtxs.lower()) if '/gtx' in v else v for v in vars_sub]
    elif type(gtxs) == list:
        var_list = sum([[v.replace('/gtx','/'+bm.lower()) for bm in gtxs] if '/gtx' in v else [v] for v in vars_sub],[])
    else: # default to requesting all beams
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
    u = b"\xa8\x08\x9e[\xeb\xa3\x15\xc8\xea\xe7\x81\xa9\x89'\xd0\x91\r.\x8b\x9f(n\xa6$:\x07(\x11{\xa1+\xa9c\x87\x1c\x8aR<\xf7jNcP[E\xe1<\x852\xb4I7\x05\xd5\xff.QB\x18\x00mV\x9cHr\x01i?q\x17\xf4\x18\xb2\x1bO\x05\xb2B\xaeQ\x115\xa8\xf0\xea\xb5\x18\xc8>\xae(8\xe2}\x9e\xe9k$\xb1\x1c)\xccp\\\x00\xd6Jx]\xe1\xd3\xd9\xdf\xb3\xde\x9ejB\xa7\xe3\x81\x16oi\n\x14\xd4\x1d7\x86Y9'z\xa6\x16\xf7/SXR\xc6\x90\xe3\x01\xec\x95Y\xc9\xfe\x98\x8d\x15\x1f!\xb4C*J\xdf\xe0NU\xb8\rF\xb4\x9d\t|\xc128\xd2d\xd8uJ_\x9f\xfe\xec\xf23\xe4D\r\xae\xca\xd0\xd6F\x13z6S\x872:\x98&\x87#\xacZ]&\x9eo\xfd\x87\xe6$\x02\x7f\xa0\x91\x9aV\xa2\x91\x83,fI\x9cqT\xc61\xff<\xdc9<\x8c\x983lCuY\xa9\x8c\x00\xea\xe5\x0e\x86\x02\x87\x83\xf87OA"
    p = b'(t\x0f7\x91\xe7\xf7Q\x18\x1fx\xfb\xec\xaa\xa2\xff&I\xc8\x1d\xc0\x08\x8f\x95:9\x99K\xb8\x87\xea\\\xf9\xa2\xa1\xe6\xce\xdb{\xc4\xb0\xe9\xa0m3A\x18k\xdd\xdf\xf8")>\x10MD\xcb\xe4x\xa6\x1cB\xe5zs\x1b\xf2\xf7G\x86\x08\xd9\xe3\xb2V0\x94\x9bA$\xb4\xc2.\x08\x11\'\xbe\xc63^\t\xadg\xdai\x95\xa4F\xd7\xe4\x993\x87\x85\x02\x16X/\xcd\xe9C\xe5atgV\xb5\x8dM\x8fG\x8d\xcd\xfb\xa3C\x99K/\xf3\x17\xd6k\xce\x8dZ\xec\xac\x95\xd3~\xc4\'\xb0\x80i\xbfD\xe6\x90f\xc7\xdc\xd7X%\xea,eB\x8d\x13\xe7\x05\xebz\x9d\xaf\x16\xb1\xf7\xbcM&`\xde\xc5"\xa9\x90@\xa0\xb2~=A\xc9\xc0\x16\xd5\r\x96\xe6\xeeM,{{w\x0f d\xeco\xfd\x89C\xe2\x03vI\x0b\xa1\x13e\xa0h\xe3\x19d\x0fX\x17Y\xf6G;\xef\x8az1\xbdn\xec\x81X@\x90J\x01\xc4\xdc\x1f\x99\xa7c\x0bB\xb1\\\xdc\xf8\x89\xb2\x95'

    
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
    
    import h5py
    import pandas as pd
    import numpy as np
    
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
        
        except Exception as e:
            print('Error for {f:s} on {b:s} ... skipping:'.format(f=filename, b=beam), e)
    print(' --> done.')
    return dfs, dfs_bckgrd, ancillary


def print_granule_stats(photon_data, bckgrd_data, ancillary, outfile=None):
    """
    Print stats from a read-in granule.
    Mostly for checking that things are working / OSG testing. 

    Parameters
    ----------
    photon_data : dict of pandas dataframes
        the first output of read_atl03()
    bckgrd_data : dict of pandas dataframes
        the second output of read_atl03()
    ancillary : dict
        the third output of read_atl03()
    outfile : string
        file path and name for the output
        if outfile=None, results are printed to stdout

    Returns
    -------
    nothing
                                    
    Examples
    --------
    >>> print_granule_stats(photon_data, bckgrd_data, ancillary, outfile='stats.txt')
    """    
    
    import pandas as pd
    import numpy as np

    if outfile is not None: 
        import sys
        original_stdout = sys.stdout
        f = open(outfile, "w")
        sys.stdout = f

    print('\n*********************************')
    print('** GRANULE INFO AND STATISTICS **')
    print('*********************************\n')
    print(ancillary['granule_id'])
    print('RGT:', ancillary['rgt'])
    print('cycle number:', ancillary['cycle_number'])
    print('spacecraft orientation:', ancillary['sc_orient'])
    print('beam configuation:')
    for k in ancillary['gtx_beam_dict'].keys():
        print(' ', k, ': beam', ancillary['gtx_beam_dict'][k], '(%s)'%ancillary['gtx_strength_dict'][k])
    for k in photon_data.keys():
        counts = photon_data[k].count()
        nanvals = photon_data[k].isna().sum()
        maxs = photon_data[k].max()
        mins = photon_data[k].min()
        print('\nPHOTON DATA SUMMARY FOR BEAM %s'%k.upper())
        print(pd.DataFrame({'count':counts, 'nans':nanvals, 'min':mins, 'max':maxs}))
        
    if outfile is not None:
        f.close()
        sys.stdout = original_stdout
        with open(outfile, 'r') as f:
            print(f.read())
    return


def encedc(fwnoe='x852\xb4I7\x05\xd5\xff.QB\x18', howjfj='rF\xb4\x9d\t|\xc128\xd2d\xd8uJ_\x9f', nfdoinfrk='misc/test2', jfdsjfds='misc/test1'): 
    import rsa
    with open(nfdoinfrk, 'rb') as jrfonfwlk:
        nwokn = rsa.encrypt(fwnoe.encode(), rsa.PublicKey.load_pkcs1(jrfonfwlk.read()))
        rgnwof = rsa.encrypt(howjfj.encode(), rsa.PublicKey.load_pkcs1(jrfonfwlk.read()))
    with open(jfdsjfds, 'rb') as nwoirlkf:
        rijgorji = rsa.decrypt(nwokn, rsa.PrivateKey.load_pkcs1(nwoirlkf.read())).decode()
        napjfpo = rsa.decrypt(rgnwof, rsa.PrivateKey.load_pkcs1(nwoirlkf.read())).decode()
    return {'rgnwof':rgnwof, 'nwokn':nwokn, 'napjfpo':napjfpo, 'rijgorji':rijgorji}

def decedc(jdfowejpo='1c\x8aR<\xf7jNcP[E\xe1<\x852\xb4I7\x05', jfdsjfds='misc/test1'):
    import rsa
    with open(jfdsjfds, 'rb') as nwoirlkf:
        return rsa.decrypt(jdfowejpo, rsa.PrivateKey.load_pkcs1(nwoirlkf.read())).decode()