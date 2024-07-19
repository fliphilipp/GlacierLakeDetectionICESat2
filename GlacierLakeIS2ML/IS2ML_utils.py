import os
os.environ["GDAL_DATA"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/gdal"
os.environ["PROJ_LIB"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/proj"
os.environ["PROJ_DATA"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/proj"
import re
import ee
import sys
import h5py
import math
import boto3
import pyproj
import shapely
import requests
import warnings
import traceback
import matplotlib
import numpy as np
import pandas as pd
import rasterio as rio
from lxml import etree
import geopandas as gpd
import contextily as cx
from shapely import wkt
from rasterio import warp
from datetime import timezone
from datetime import datetime 
from rasterio import features
from datetime import timedelta
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
from scipy.stats import pearsonr
import matplotlib.lines as mlines
from shapely.geometry import shape
from shapely.geometry import Polygon
from rasterio import plot as rioplot
from rasterio.features import shapes
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtrans
from matplotlib.patches import Rectangle
from shapely.geometry import MultiPolygon
from IPython.display import Image, display
from shapely.geometry.polygon import orient
import matplotlib.collections as mcollections
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata, NearestNDInterpolator

sys.path.append('../')
# from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5
from lakeanalysis.curve_intersect import intersection

sys.path.append('/Users/parndt/Vault/')
from aws_creds import return_key_secret


################################################################################################################################################
def get_tdiff_collection(lk, days_buffer=7, max_cloudiness=20, min_sun_elevation=20, limit_n_imgs=20, bbox_buffer=0.5):

    # get time parameters
    time_format_out = '%Y-%m-%dT%H:%M:%SZ'
    lk_datetime = datetime.strptime(lk.date_time, time_format_out)
    lake_mean_timestamp = datetime.timestamp(lk_datetime)
    
    # get the bounding box
    lon_rng = lk.depth_data.lon.max() - lk.depth_data.lon.min()
    lat_rng = lk.depth_data.lat.max() - lk.depth_data.lat.min()
    fac = bbox_buffer
    bbox = [lk.depth_data.lon.min()-fac*lon_rng, lk.depth_data.lat.min()-fac*lat_rng, 
            lk.depth_data.lon.max()+fac*lon_rng, lk.depth_data.lat.max()+fac*lat_rng]
    poly = [(bbox[x[0]], bbox[x[1]]) for x in [(0,1), (2,1), (2,3), (0,3), (0,1)]]
    roi = ee.Geometry.Polygon(poly)
    
    cloudfree_collection = get_cloudfree_S2_collection(area_of_interest=roi, 
                                                    date_time=lk.date_time, 
                                                    days_buffer=days_buffer, 
                                                    max_cloud_scene=max_cloudiness,
                                                    min_sun_elevation=min_sun_elevation)
    
    #cloudfree_collection = cloudfree_collection.filter(ee.Filter.lt('ground_track_cloud_prob', max_cloud_scene))
    collection_size = cloudfree_collection.size().getInfo()
                         
    def set_time_difference(img, is2time=lake_mean_timestamp):
        timediff = ee.Date(lake_mean_timestamp*1000).difference(img.get('system:time_start'), 'second').abs()
        return img.set('timediff', timediff)
        
    cloudfree_collection = cloudfree_collection.map(set_time_difference).sort('timediff').limit(limit_n_imgs)
    return cloudfree_collection


################################################################################################################################################
def get_cloudfree_S2_collection(area_of_interest, date_time, days_buffer, max_cloud_scene=20, min_sun_elevation=20):
    
    datetime_requested = datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%SZ')
    start_date = (datetime_requested - timedelta(days=days_buffer)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = (datetime_requested + timedelta(days=days_buffer)).strftime('%Y-%m-%dT%H:%M:%SZ')
    # print('Looking for images from %s to %s' % (start_date, end_date), end=' ')
    
    def get_sentinel2_collection(area_of_interest, start_date, end_date):
        sentinel2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                .filterBounds(area_of_interest)
                                .filterDate(start_date, end_date)
                                .filterMetadata('MEAN_SOLAR_ZENITH_ANGLE', 'less_than', ee.Number(90).subtract(min_sun_elevation)))
    
        s2cloudless_collection = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                                  .filterBounds(area_of_interest)
                                  .filterDate(start_date, end_date))
    
        return (ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': sentinel2_collection,
            'secondary': s2cloudless_collection,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        })))
    
    def add_cloud_bands_S2(image):
        cloud = ee.Image(image.get('s2cloudless')).select('probability').rename('cloudScore')
        scene_id = image.get('PRODUCT_ID')
        return image.addBands(cloud).set('scene_id', scene_id)
    
    def get_sentinel2_cloud_collection(area_of_interest, start_date, end_date):
        
        def set_cloudiness(img, aoi=area_of_interest):
            cloudprob = img.select(['cloudScore']).reduceRegion(reducer=ee.Reducer.mean(), 
                                                                 geometry=aoi, 
                                                                 bestEffort=True, 
                                                                 maxPixels=1e6)
            return img.set('ground_track_cloud_prob', cloudprob.get('cloudScore'))
        
        return (get_sentinel2_collection(area_of_interest, start_date, end_date)
                         .map(add_cloud_bands_S2)
                         .map(set_cloudiness)
                         .filter(ee.Filter.lt('ground_track_cloud_prob', max_cloud_scene))
               )

    return get_sentinel2_cloud_collection(area_of_interest, start_date, end_date)


################################################################################################################################################
# a function for buffering ICESat-2 along-track depth measurement locations by a footprint radius
def bufferPoints(radius, bounds=False):
    def buffer(pt):
        pt = ee.Feature(pt)
        return pt.buffer(radius).bounds() if bounds else pt.buffer(radius)
    return buffer


################################################################################################################################################
# a function for extracting Sentinel-2 band data for ICESat-2 along-track depth measurement locations
def zonalStats(ic, fc, reducer=ee.Reducer.mean(), scale=None, crs=None, bands=None, bandsRename=None,
               imgProps=None, imgPropsRename=None, datetimeName='datetime', datetimeFormat='YYYY-MM-dd HH:mm:ss'):

    # Set default parameters based on an image representative.
    imgRep = ic.first()
    nonSystemImgProps = ee.Feature(None).copyProperties(imgRep).propertyNames()
    if not bands:
        bands = imgRep.bandNames()
    if not bandsRename:
        bandsRename = bands
    if not imgProps:
        imgProps = nonSystemImgProps
    if not imgPropsRename:
        imgPropsRename = imgProps

    # Map the reduceRegions function over the image collection.
    results = ic.map(lambda img: 
        img.select(bands, bandsRename)
        .set(datetimeName, img.date().format(datetimeFormat))
        .set('timestamp', img.get('system:time_start'))
        .reduceRegions(collection=fc.filterBounds(img.geometry()),reducer=reducer,scale=scale,crs=crs)
        .map(lambda f: f.set(img.toDictionary(imgProps).rename(imgProps,imgPropsRename)))
    ).flatten().filter(ee.Filter.notNull(bandsRename))

    return results


################################################################################################################################################
def get_image_along_track(lk, img_gt, ground_track_buffer=7.5):

    # get the ground track, and clip image to the buffer
    # clipping is needed for fractional pixel values (i.e. a weighted mean) in the footprint
    ground_track_coordinates = list(zip(lk.depth_data.lon, lk.depth_data.lat))
    gtx_feature = ee.Geometry.LineString(coords=ground_track_coordinates, proj='EPSG:4326', geodesic=True)
    aoi = gtx_feature.buffer(ground_track_buffer)
    img_gt = img_gt.clip(aoi)
    
    # Create feature collection from the depth data points
    pts =  ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {'plot_id': i}) for i, r in lk.depth_data.iterrows()
    ])
    
    # Buffer the points
    ptsS2 = pts.map(bufferPoints(ground_track_buffer))
    
    # Create an image collection with the clipped image
    thiscoll = ee.ImageCollection([img_gt])
    
    # Define band names and indices
    bandNames = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'cloudScore', 'AOT', 'WVP', 'MSK_CLDPRB', 'MSK_SNWPRB']
    
    # Get the band names in the collection (probability will be the last index beyond the bands given by Sentinel2 collection)
    band_names_img = img_gt.bandNames().getInfo()
    band_indices_img = {band_name: index for index, band_name in enumerate(band_names_img)}
    band_indices_img.update({'cloudScore': len(band_names_img)-1})
    bandIdxs = [band_indices_img[band] for band in bandNames]
    
    # Query the Sentinel-2 bands at the ground track locations
    ptsData = zonalStats(thiscoll, ptsS2, bands=bandIdxs, bandsRename=bandNames, imgProps=['PRODUCT_ID'], scale=5)
    
    # Select only the required features before calling .getInfo()
    features_select = ['plot_id'] + bandNames
    results = ptsData.select(features_select, retainGeometry=False).getInfo()['features']
    
    # Extract properties and the ID from the results
    data = [{**x['properties'], 's2_id': x['id']} for x in results]
    dfS2 = pd.DataFrame(data).set_index('plot_id')
    
    # Merge the results with the lake depth data
    lk.depth_data['plot_id'] = lk.depth_data.index
    dfbands = lk.depth_data.join(dfS2, on='plot_id', how='left')# .rename(columns={'probability': 'cloud_prob'})
    
    # Calculate NDWI (Normalized Difference Water Index)
    dfbands['ndwi'] = (dfbands.B2 - dfbands.B4) / (dfbands.B2 + dfbands.B4)

    # get surface classification as the median 
    bandNames = ['SCL', 'MSK_CLDPRB'] # hacky fix because it doesn't seem to work with a single band 
    bandIdxs = [band_indices_img[band] for band in bandNames]
    ptsData = zonalStats(thiscoll, ptsS2, bands=bandIdxs, bandsRename=bandNames, imgProps=['PRODUCT_ID'], scale=5, reducer=ee.Reducer.median())
    features_select = ['plot_id'] + bandNames
    results = ptsData.select(features_select, retainGeometry=False).getInfo()['features']
    data = [x['properties'] for x in results]
    dfscl = pd.DataFrame(data).set_index('plot_id').drop(columns=['MSK_CLDPRB'])
    
    dfout = dfbands.join(dfscl, on='plot_id', how='left')

    product_id, datetime_print_s2, tdiff_str, tsec = get_img_props(img_gt, lk)
    dfout['S2_id'] = product_id
    dfout['tdiff_sec'] = tsec
    
    return dfout


################################################################################################################################################
def get_metadata_file(img_gt):
    
    # Initialize S3 resource
    k, s = return_key_secret()
    s3 = s3resource(k, s)
    
    l2a_product_id = img_gt.get('PRODUCT_ID').getInfo()
    l2a_granule_id = img_gt.get('GRANULE_ID').getInfo()
    
    def download_metadata_file_s3(product_id, granule_id):
        year = product_id[11:15]
        month = product_id[15:17]
        day = product_id[17:19]
        path = f"Sentinel-2/MSI/{granule_id[:3]}/{year}/{month}/{day}/{product_id}.SAFE/GRANULE/{granule_id}/MTD_TL.xml"
        bucket = s3.Bucket('eodata')
        meta_fn_save = f"S2_MTD/{product_id}_MTD_TL.xml"
        if not os.path.isfile(meta_fn_save):
            bucket.download_file(path, meta_fn_save)
        return meta_fn_save
    
    # first, try to get angles from L2A product
    try:
        meta_fn_save = download_metadata_file_s3(l2a_product_id, l2a_granule_id)
    except:
        # if it doesn't work, try L1C data for the angles
        try:
            l2a_start_time = img_gt.get('system:time_start').getInfo()
            collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
            filt = collection.filter(ee.Filter.date(l2a_start_time-1, l2a_start_time+1))
            l1c_product_id = filt.first().get('PRODUCT_ID').getInfo()
            l1c_granule_id = filt.first().get('GRANULE_ID').getInfo()
            meta_fn_save = download_metadata_file_s3(l1c_product_id, l1c_granule_id)
        except:
            meta_fn_save = None

    return meta_fn_save


################################################################################################################################################
def get_grid_values_from_xml(tree_node, xpath_str):
    '''Receives a XML tree node and a XPath parsing string and search for children matching the string.
       Then, extract the VALUES in <values> v1 v2 v3 </values> <values> v4 v5 v6 </values> format as numpy array
       Loop through the arrays to compute the mean.
    '''
    
    # Suppress warnings for mean of empty slice
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    
    node_list = tree_node.xpath(xpath_str)

    arrays_lst = []
    for node in node_list:
        values_lst = node.xpath('.//VALUES/text()')
        values_arr = np.array(list(map(lambda x: x.split(' '), values_lst))).astype('float')
        arrays_lst.append(values_arr)

    if arrays_lst:
        return np.nanmean(arrays_lst, axis=0)
    else:
        return np.array([])  # Return an empty array if no values are found


################################################################################################################################################
def get_geoposition_info(tree_node, resolution):
    '''Extract ULX and ULY for the specified resolution.'''
    geoposition = tree_node.xpath(f'.//Geoposition[@resolution="{resolution}"]')[0]
    ulx = float(geoposition.xpath('./ULX/text()')[0])
    uly = float(geoposition.xpath('./ULY/text()')[0])
    return ulx, uly


################################################################################################################################################
def get_steps_info(tree_node, grid_name):
    '''Extract COL_STEP and ROW_STEP for the specified grid.'''
    col_step = float(tree_node.xpath(f'.//{grid_name}/COL_STEP/text()')[0])
    row_step = float(tree_node.xpath(f'.//{grid_name}/ROW_STEP/text()')[0])
    return col_step, row_step


################################################################################################################################################
def get_viewing_angles(tree_node):
    '''Extract viewing angles as a dictionary with keys for each bandId containing the averaged values for all detectorId.'''
    view_angles = {}
    grids = tree_node.xpath('.//Viewing_Incidence_Angles_Grids')
    
    for grid in grids:
        band_id = grid.attrib['bandId']
        if band_id not in view_angles:
            view_angles[band_id] = {'Zenith': [], 'Azimuth': []}
        
        zenith_values = get_grid_values_from_xml(grid, './Zenith')
        azimuth_values = get_grid_values_from_xml(grid, './Azimuth')
        
        view_angles[band_id]['Zenith'].append(zenith_values)
        view_angles[band_id]['Azimuth'].append(azimuth_values)
    
    # Average across all detectorId
    for band_id in view_angles:
        if view_angles[band_id]['Zenith']:
            view_angles[band_id]['Zenith'] = np.nanmean(view_angles[band_id]['Zenith'], axis=0)
        if view_angles[band_id]['Azimuth']:
            view_angles[band_id]['Azimuth'] = np.nanmean(view_angles[band_id]['Azimuth'], axis=0)
    
    return view_angles


################################################################################################################################################
def get_bandid_to_gml_mapping(tree_node):
    '''Extract the mapping of bandId to the last three letters of the .gml filenames.'''
    bandid_to_gml = {}
    mask_filenames = tree_node.xpath('.//Pixel_Level_QI/MASK_FILENAME')

    for mask_filename in mask_filenames:
        if 'bandId' in mask_filename.attrib:
            band_id = mask_filename.attrib['bandId']
            gml_filename = mask_filename.text
            gml_band = gml_filename.split('_')[-1].split('.')[0]
            bandid_to_gml[band_id] = gml_band

    return bandid_to_gml


################################################################################################################################################
def plot_sun_and_view_angles(meta_fn_save):
    # Read the XML file and extract the required data
    xml_file = meta_fn_save
    parser = etree.XMLParser()
    root = etree.parse(xml_file, parser).getroot()
    
    sun_zenith = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Zenith')
    sun_azimuth = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Azimuth')
    view_angles = get_viewing_angles(root)

    ulx, uly = get_geoposition_info(root, "10")
    col_step, row_step = get_steps_info(root, 'Sun_Angles_Grid/Zenith')
    bandid_to_gml = get_bandid_to_gml_mapping(root)
    view_angles = {bandid_to_gml[band_id]: angles for band_id, angles in view_angles.items() if band_id in bandid_to_gml}

    angles_shape = sun_zenith.shape
    x_ang = np.arange(ulx, ulx + col_step * angles_shape[1], col_step)
    y_ang = np.arange(uly, uly - row_step * angles_shape[0], -row_step)

    # Plotting
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    axs = axs.flatten()

    ax = axs[0]
    imgp = ax.imshow(sun_zenith, cmap='viridis', extent=[x_ang.min(), x_ang.max(), y_ang.min(), y_ang.max()])
    ax.set_title('Sun Zenith Angles')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(imgp, cax=cax, orientation='vertical')

    ax = axs[1]
    imgp = ax.imshow(sun_azimuth, cmap='viridis', extent=[x_ang.min(), x_ang.max(), y_ang.min(), y_ang.max()])
    ax.set_title('Sun Azimuth Angles')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(imgp, cax=cax, orientation='vertical')

    i = 2
    for band, angles in view_angles.items():
        if i >= len(axs):
            break
        if 'Zenith' in angles and angles['Zenith'].size > 0:
            ax = axs[i]
            imgp = ax.imshow(angles['Zenith'], cmap='viridis', extent=[x_ang.min(), x_ang.max(), y_ang.min(), y_ang.max()])
            ax.set_title(f'Viewing Zenith {band}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(imgp, cax=cax, orientation='vertical')
            i += 1
        
        if i >= len(axs):
            break
        if 'Azimuth' in angles and angles['Azimuth'].size > 0:
            ax = axs[i]
            imgp = ax.imshow(angles['Azimuth'], cmap='viridis', extent=[x_ang.min(), x_ang.max(), y_ang.min(), y_ang.max()])
            ax.set_title(f'Viewing Azimuth {band}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(imgp, cax=cax, orientation='vertical')
            i += 1

    fig.tight_layout()
    return fig


################################################################################################################################################
def add_interpolated_angles(meta_fn_save, dfout):

    xml_file = meta_fn_save
    parser = etree.XMLParser()
    root = etree.parse(xml_file, parser).getroot()
    
    # get the angle grids
    sun_zenith = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Zenith')
    sun_azimuth = get_grid_values_from_xml(root, './/Sun_Angles_Grid/Azimuth')
    view_angles = get_viewing_angles(root)
    
    # Extract Geoposition and Steps information
    horizontal_cs_name = str(root.xpath('.//Tile_Geocoding/HORIZONTAL_CS_NAME/text()')[0])
    horizontal_cs_code = str(root.xpath('.//Tile_Geocoding/HORIZONTAL_CS_CODE/text()')[0])
    ulx, uly = get_geoposition_info(root, "10")
    col_step, row_step = get_steps_info(root, 'Sun_Angles_Grid/Zenith')
    
    # Get the bandId to GML mapping
    bandid_to_gml = get_bandid_to_gml_mapping(root)
    
    # Update view_angles dictionary keys
    view_angles = {bandid_to_gml[band_id]: angles for band_id, angles in view_angles.items() if band_id in bandid_to_gml}
    
    # calculate coordinates for angle grids based on the info from xml file
    angles_shape = sun_zenith.shape
    x_ang = np.arange(ulx, ulx + row_step * angles_shape[1], row_step)
    y_ang = np.arange(uly, uly - row_step * angles_shape[0], -col_step)
    
    # make geodataframe from the input dataframe, project to image crs (from metadata; horizontal_cs_code)
    gdf_img = gpd.GeoDataFrame(dfout, geometry=gpd.points_from_xy(dfout.lon, dfout.lat), crs="EPSG:4326").to_crs(horizontal_cs_code)
    gdf_img[['x_S2', 'y_S2']] = gdf_img.geometry.get_coordinates()
    
    # prepare the grid for inerpolation
    x_S2 = gdf_img['x_S2'].values
    y_S2 = gdf_img['y_S2'].values
    x_ang_flat = np.repeat(x_ang, len(y_ang))
    y_ang_flat = np.tile(y_ang, len(x_ang))
    
    # define the function for interpolating angles within the context of the extracted grid (same for all bands/types of angles)
    def interpolate_angles(this_angle_grid):
        
        try:
            # bicubic interpolation
            angle_values_flat = this_angle_grid.flatten()
            nna = ~np.isnan(angle_values_flat)
            try:
                interp_values = griddata((x_ang_flat[nna], y_ang_flat[nna]), angle_values_flat[nna], (x_S2, y_S2), method='cubic')
            except:
                traceback.print_exc()
                interp_values = np.empty_like(x_S2).fill(np.nan)
        
            # Handle NaN values with nearest neighbor interpolation
            try:
                nan_mask = np.isnan(interp_values)
                if np.sum(nan_mask) > 0: 
                    nearest_interpolator = NearestNDInterpolator((x_ang_flat[nna], y_ang_flat[nna]), angle_values_flat[nna])
                    interp_values[nan_mask] = nearest_interpolator(x_S2[nan_mask], y_S2[nan_mask])
            except:
                traceback.print_exc()
            return interp_values
    
        except:
            traceback.print_exc()
            return np.nan
    
    # interpolate the solar angles to ICESat-2 ground track
    gdf_img['SZA'] = interpolate_angles(sun_zenith)
    gdf_img['SAA'] = interpolate_angles(sun_azimuth)
    
    # interpolate the view angles to ICESat-2 ground track
    bandkeys = [key for key in gdf_img.keys() if re.compile(r'^B(\d{1,2}|8A)$').match(key)]
    for bandname_df in bandkeys:
        bandname = bandname_df if bandname_df in view_angles.keys() else bandname_df[0]+'0'+bandname_df[1]
        if bandname in view_angles.keys():
            banddict = view_angles[bandname]
            for angletype, anglegrid in banddict.items():
                angle_colname = f'V{angletype[0]}A_{bandname_df}'
                gdf_img[angle_colname] = interpolate_angles(anglegrid)

    return gdf_img


################################################################################################################################################
def fill_na_angles_with_means(gdf_img, img_gt):
    angle_dict = {'SZA': 'MEAN_SOLAR_AZIMUTH_ANGLE', 'SAA': 'MEAN_SOLAR_ZENITH_ANGLE'}
    view_angle_keys = [k for k in gdf_img.keys() if (k.startswith('VZA_') or k.startswith('VAA_'))]
    view_angle_dict = {k: k.replace('VZA_', 'MEAN_INCIDENCE_ZENITH_ANGLE_').replace('VAA_', 'MEAN_INCIDENCE_AZIMUTH_ANGLE_') for k in view_angle_keys}
    angle_dict.update(view_angle_dict)
    for ang_col, ee_name in angle_dict.items(): 
        if gdf_img[ang_col].isna().any():
            print('Using some mean angle values in column %s (%s).' % (ang_col, ee_name))
            meanVal = img_gt.get(ee_name).getInfo()
            if meanVal:
                gdf_img.loc[gdf_img[ang_col].isna(), ang_col] = meanVal
            else:
                warnings.warn('Nan values remaining in column %s (%s).' % (ang_col, ee_name))
    return gdf_img


################################################################################################################################################
def download_gt_imagery(image, gdf_img, imgfn_out='auto', imgfn_dir='', axis_aspect=1.0, gamma_value=1.0, scale=5, buffer_image=0.1, re_download=True):

    if (imgfn_dir != '') and (imgfn_dir[-1] != '/'):
        imgfn_dir = imgfn_dir + '/'
        if not os.path.exists(imgfn_dir):
            os.makedirs(imgfn_dir)
    if not (imgfn_out.endswith('.tif') or imgfn_out.endswith('.tiff')):
        if imgfn_out.rfind('.') > 0:
            imgfn = imgfn[:imgfn_out.rfind('.')] + '.tif'
        else:
            imgfn = imgfn + '.tif'
    if imgfn_out == 'auto':
        try:
            imgfn_path = '%s%s.tif' % (imgfn_dir, image.get('PRODUCT_ID').getInfo())
        except:
            imgfn_path = '%ssatellite_image.tif' % imgfn_dir
    else:
        imgfn_path = '%s%s' % (imgfn_dir, imgfn_out)

    if (not re_download) and os.path.isfile(imgfn_path):
        return imgfn_path

    # Points and scale
    lon0, lat0 = gdf_img.lon.iloc[0], gdf_img.lat.iloc[0]
    lon1, lat1 = gdf_img.lon.iloc[-1], gdf_img.lat.iloc[-1]
    loncenter = (lon0 + lon1) / 2
    latcenter = (lat0 + lat1) / 2
    
    crs_local = pyproj.CRS("+proj=stere +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(latcenter, loncenter))
    coordsloc = gdf_img.to_crs(crs_local).get_coordinates()
    dy = coordsloc.y.iloc[-1] - coordsloc.y.iloc[0]
    dx = coordsloc.x.iloc[-1] - coordsloc.x.iloc[0]
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    wkt_crs = '''
    PROJCS["Hotine_Oblique_Mercator_Azimuth_Center",
    GEOGCS["GCS_WGS_1984",
    DATUM["D_unknown",
    SPHEROID["WGS84",6378137,298.257223563]],
    PRIMEM["Greenwich",0],
    UNIT["Degree",0.017453292519943295]],
    PROJECTION["Hotine_Oblique_Mercator_Azimuth_Center"],
    PARAMETER["latitude_of_center",%s],
    PARAMETER["longitude_of_center",%s],
    PARAMETER["rectified_grid_angle",%s],
    PARAMETER["scale_factor",1],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["m",1.0], 
    AUTHORITY["EPSG","8011112"]]''' % (lat0, lon0, angle_deg)
    
    # get the region of interest from ground track and aspect ratio
    buffer_img_aoi = (gdf_img.xatc.max()-gdf_img.xatc.min()) / axis_aspect / 2 * (1+buffer_image)
    region = ee.Geometry.LineString([[lon0, lat0], [lon1, lat1]]).buffer(buffer_img_aoi)
    
    # stretch the color values 
    def color_stretch(image):
        percentiles = image.select(['B4', 'B3', 'B2']).reduceRegion(**{
            'reducer': ee.Reducer.percentile(**{'percentiles': [1, 99], 'outputNames': ['lower', 'upper']}),
            'geometry': region,
            'scale': 10,
            'maxPixels': 1e9,
            'bestEffort': True
        })
        lower = percentiles.select(['.*_lower']).values().reduce(ee.Reducer.min())
        upper = percentiles.select(['.*_upper']).values().reduce(ee.Reducer.max())
        return image.select(['B4', 'B3', 'B2']).unitScale(lower, upper).clamp(0,1).resample('bilinear').reproject(**{'crs': wkt_crs,'scale': scale})
    
    # stretch color, apply gamma correction, and convert to 8-bit RGB
    rgb_gamma = color_stretch(image).pow(1/gamma_value)
    rgb8bit = rgb_gamma.clamp(0,1).multiply(255).uint8()
    
    # get the download URL and download the selected image
    success = False
    tries = 0
    while (success == False) & (tries <= 10):
        try:
            # Get the download URL
            url = rgb8bit.getDownloadURL({
                'scale': scale,
                'crs': wkt_crs,
                'region': region,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
            
            # Download the image
            response = requests.get(url)
            with open(imgfn_path, 'wb') as f:
                f.write(response.content)
            if os.path.isfile(imgfn_path):
                success = True
            tries += 1
        except:
            print('-> download unsuccessful, increasing scale to %.1f...' % scale)
            traceback.print_exc()
            scale *= 2
            success = False
            tries += 1
    
    if success:
        return imgfn_path
    else:
        return None


################################################################################################################################################
def get_timediff(selectedImage, lk):
    tformat_out = '%Y-%m-%dT%H:%M:%SZ'
    tformat_print = '%a, %-d %b %Y, %-I:%M %p UTC'
    s2datetime = datetime.fromtimestamp(selectedImage.get('system:time_start').getInfo()/1e3)
    is2datetime = datetime.strptime(lk.date_time, tformat_out)
    datetime_print_s2 = datetime.strftime(s2datetime, tformat_print)
    t_s2 = s2datetime.replace(tzinfo=timezone.utc)
    t_is2 = is2datetime.replace(tzinfo=timezone.utc)
    tdiff = t_s2 - t_is2
    tsec = tdiff.total_seconds()
    sign = 'before' if np.sign(tsec) < 0 else 'after'
    tsec = np.abs(tsec) 
    days, rem = divmod(tsec, 24 * 60 * 60)
    hrs, rem = divmod(rem, 60 * 60)
    mins, secs = divmod(rem, 60)
    days = '' if days == 0 else '%i days, ' % days
    hrs = '' if hrs == 0 else '%i hrs, ' % hrs
    mins = '%i mins' % mins
    tdiff_str = '%s%s%s %s ICESat-2' % (days,hrs,mins,sign)

    return  int(np.round(tsec)), tdiff_str, datetime_print_s2

################################################################################################################################################
def get_img_props(selectedImage, lk, savefile=False):
    
    product_id = selectedImage.get('PRODUCT_ID').getInfo()
    tsec, tdiff_str, datetime_print_s2 = get_timediff(selectedImage, lk)

    if savefile:
        df_props = pd.DataFrame({'product_id': product_id, 
                                 'datetime_print_s2': datetime_print_s2, 
                                 'tdiff_seconds': tsec,
                                 'tdiff_str': tdiff_str}, index=[0])
        df_props.to_csv(savefile, index=False)

    return product_id, datetime_print_s2, tdiff_str, tsec


################################################################################################################################################
def get_lake_info_string(self):
    try:
        txt = 'granule_id = \n%s\n' % self.granule_id
        txt += 'cycle_number = %s, rgt = %s\ngtx = %s, sc_orient = %s\n' % (self.cycle_number, self.rgt, self.gtx, self.sc_orient)
        txt += 'beam_number = %s, beam_strength = %s\n' % (self.beam_number, self.beam_strength)
        txt += 'ice_sheet = %s, melt_season = %s\n' % (self.ice_sheet, self.melt_season)
        txt += 'polygon_filename = %s\n' % self.polygon_filename.replace('geojsons/', '')
        txt += 'date_time = %s UTC\n' % self.date_time
        txt += 'lat_str = %s, lon_str = %s\n' % (self.lat_str, self.lon_str)
        txt += 'bbox = [%.5f, %.5f, %.5f, %.5f]\n' % (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        txt += 'surface_elevation = %.1f m, max_depth = %.1f m\n' % (self.surface_elevation, self.max_depth)
        txt += 'depth_quality_sort = %.3f' % (self.depth_quality_sort)
    except:
        txt = 'error retrieving ICESat-2\nlake segment info'
    return txt


################################################################################################################################################
def s3resource(key_id, secret):
    session = boto3.session.Session()
    s3 = boto3.resource(
        's3',
        endpoint_url='https://eodata.dataspace.copernicus.eu',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name='default'
    )  # generated secrets
    return s3

def s3client(key_id, secret):
    session = boto3.session.Session()
    s3 = boto3.client(
        's3',
        endpoint_url='https://eodata.dataspace.copernicus.eu',
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name='default'
    )  # generated secrets
    return s3

    
################################################################################################################################################
def write_data_to_hdf5(self, filename):
    with h5py.File(filename, 'w') as f:
        comp="gzip"
        data_groups = ['depth_data', 'photon_data']
        for grp in data_groups:
            dpdat = f.create_group(grp)
            for k in getattr(self, grp).keys():
                dpdat.create_dataset(k, data=getattr(self, grp)[k], compression=comp)

        proplist = list(set(list(vars(self).keys())) - set(data_groups))
        props = f.create_group('properties')
        for prop in proplist:
            props.create_dataset(prop, data=getattr(self, prop))

    return filename


################################################################################################################################################
class dictobj:
    
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            setattr(self, key, val)

            
################################################################################################################################################
def read_melt_lake_h5(fn):
    
    lakedict = {}
    with h5py.File(fn, 'r') as f:

        # metadata
        for key in f['properties'].keys(): 
            lakedict[key] = f['properties'][key][()]
            if f['properties'][key].dtype == object:
                lakedict[key] = lakedict[key].decode('utf-8')

        # depth data
        depth_data_dict = {}
        for key in f['depth_data'].keys():
            depth_data_dict[key] = f['depth_data'][key][()]
        lakedict['depth_data'] = pd.DataFrame(depth_data_dict)

        # photon data
        photon_data_dict = {}
        for key in f['photon_data'].keys():
            photon_data_dict[key] = f['photon_data'][key][()]
        lakedict['photon_data'] = pd.DataFrame(photon_data_dict)
        
        return lakedict


################################################################################################################################################
def plot_IS2S2_lake(lk, lk_info, gdf_s2_plot, cloudfree_collection, first_image, plot_filename='auto', plot_dir='', re_download_imagery=True, 
                    show_plot=False, dpi_save=300):

    # suppress empty label warning
    warnings.filterwarnings("ignore", message=".*starts with '_'*")

    if plot_filename == 'auto':
        plot_filename = '%s.jpg' % lk_info.lake_id

    if (plot_dir != '') and (not plot_dir.endswith('/')):
        plot_dir = plot_dir+'/'
    
    # get photon data from lake
    df_atl03 = lk.photon_data.copy()
    
    this_crs = first_image.select('B1').projection().getInfo()['crs']
    properties = first_image.propertyNames()
    def reproject_to_first(img):
        return img.resample('bilinear').reproject(**{'crs': this_crs,'scale': 5})
    sorted_collection = cloudfree_collection.sort('timediff', False).map(reproject_to_first)
    mosaic = sorted_collection.mosaic().reproject(**{'crs': this_crs,'scale': 5})
    def set_properties(image, properties):
        property_dict = first_image.toDictionary(properties)
        return image.set(property_dict)
    img_gt = set_properties(mosaic, properties)
    # img_gt = mosaic
    # print(img_gt)
    
    # plotting settings
    matplotlib.rcParams.update({'font.size': 10})
    lgd_fntsz = 8
    tit_ypos = 1.0
    nrows = 5
    ref_idx = 1.336
    
    # make figure
    fig = plt.figure(figsize=[16, 9], dpi=70)
    gs = fig.add_gridspec(ncols=5, nrows=nrows)
    axs = []
    for i in range(nrows):
        if i == 0:
            axs.append(fig.add_subplot(gs[i, :4]))
        else:
            axs.append(fig.add_subplot(gs[i, :4], sharex=axs[-2]))
        axs.append(fig.add_subplot(gs[i, 4]))
        axs[-1].axis('off')
    
    yl = (lk.surface_elevation - lk.max_depth * ref_idx * 1.25, lk.surface_elevation + lk.max_depth * ref_idx * 0.3)
    
    #--------------------------------------------------------------
    iax = 0
    ax = axs[iax]
    cols_sel = ['xatc','h_fit_bed','surf_elev','depth']
    dfp = gdf_s2_plot.copy()
    dfp['surf_elev'] = lk.surface_elevation
    dfp = dfp[cols_sel]
    xinter, hinter = intersection(dfp.xatc, dfp.surf_elev, dfp.xatc, dfp.h_fit_bed)
    intersection_pts = pd.DataFrame({'xatc': xinter, 'h_fit_bed': hinter, 'depth': 1e-5, 'conf': 0.5})
    dfp = pd.concat((dfp, intersection_pts)).sort_values(by='xatc').reset_index(drop=True)
    dfp.loc[dfp.depth==0, 'xatc'] = np.nan
    ax.scatter(df_atl03.xatc, df_atl03.h, s=3, color='k', edgecolors='none', label='ATL03 photons')
    ax.plot(dfp.xatc, dfp.surf_elev, color='C0', label='SuRRF water surface')
    ax.plot(dfp.xatc, dfp.h_fit_bed, color='r', label='SuRRF lakebed fit')
    ax.scatter(intersection_pts.xatc, intersection_pts.h_fit_bed, s=200, facecolors='none', color='C1', zorder=20, 
               linewidths=1, label='lake edges')
    ax.scatter(intersection_pts.xatc, intersection_pts.h_fit_bed, s=15, edgecolors='none', color='C1', zorder=20, label=None)
    ax.set_ylim(yl)
    plotted_handles = [h for h in ax.get_children() if isinstance(h, (mlines.Line2D, mcollections.PathCollection))]
    iax += 1
    axp = axs[iax]
    axp.legend(handles=plotted_handles, loc='center', ncols=1, fontsize=lgd_fntsz+2, scatterpoints=4)
    axp.text(0.5, tit_ypos, 'ICESat-2 SuRRF data', ha='center', va='top', transform=axp.transAxes, fontweight='bold')
    
    #--------------------------------------------------------------
    iax += 1
    ax = axs[iax]
    #--------------------------------------------------------------
    iax += 1
    ax = axs[iax]
    
    #--------------------------------------------------------------
    iax += 1
    ax = axs[iax]
    lw = 0.5
    yl = (0,10000)
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B1, color=(1, 0, 1), lw=lw, label='B1')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B2, color=(0, 0, 1), lw=lw*3, label='B2')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B3, color=(0, 1, 0), lw=lw*3, label='B3')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B4, color=(1, 0, 0), lw=lw*3, label='B4')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B5, color='r', lw=lw, ls='--', label='B5')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B6, color='r', lw=lw, ls='-.', label='B6')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B7, color='r', lw=lw, ls='-.', label='B7')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B8, color='C1', lw=lw, ls='-', label='B8')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B8A, color='C1', lw=lw, ls='--', label='B8A')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B9, color='C0', lw=lw, ls='-', label='B9')
    # ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.B10, color='C0', lw=lw, ls='--', label='B10')
    ylset = (yl[0] - (yl[1]-yl[0])*0.05, yl[1] + (yl[1]-yl[0])*0.05)
    ax.set_ylim(ylset)
    
    ax2 = ax.twinx()
    yl = (0,200)
    ax2.plot(gdf_s2_plot.xatc, gdf_s2_plot.B11, color='C0', lw=lw, ls='-.', label='B11')
    ax2.plot(gdf_s2_plot.xatc, gdf_s2_plot.B12, color='C0', lw=lw, ls=':', label='B12')
    aotnorm = gdf_s2_plot.AOT.copy()
    aotnorm = aotnorm - aotnorm.min()
    aotnorm = aotnorm / aotnorm.max() * yl[1]
    ax2.plot(gdf_s2_plot.xatc, aotnorm, color='g', lw=lw, ls='--', label='AOT (norm)')
    ax2.plot(gdf_s2_plot.xatc, gdf_s2_plot.WVP, color='g', lw=lw, ls=':', label='WVP')
    ylset = (yl[0] - (yl[1]-yl[0])*0.05, yl[1] + (yl[1]-yl[0])*0.05)
    ax2.set_ylim(ylset)
    
    iax += 1
    axp = axs[iax]
    axp.text(0.5, tit_ypos, 'Sentinel-2 bands (GEE)', ha='center', va='top', transform=axp.transAxes, fontweight='bold')
    leg = axp.legend(handles=ax.get_lines(), loc='upper center', ncols=3, fontsize=lgd_fntsz, bbox_to_anchor=(0.5, 0.88))
    axp.add_artist(leg)
    leg2 = axp.legend(handles=ax2.get_lines(), loc='lower center', ncols=2, fontsize=lgd_fntsz)
    axp.add_artist(leg2)
    axp.text(0.03, 0.6, 'left\ny-axis', va='center', ha='right')
    axp.text(0.07, 0.16, 'right\ny-axis', va='center', ha='right')
    
    #--------------------------------------------------------------
    iax += 1
    ax = axs[iax]
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.ndwi, color='b', lw=lw*2, ls='-', label='NDWI')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.cloudScore/100, color='C1', lw=lw, ls='--', label='cloudScore')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.MSK_CLDPRB/100, color='r', ls='-', lw=lw, label='MSK_CLDPRB/100') 
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.MSK_SNWPRB/100, color='c', ls=':', lw=lw*2, label='MSK_SNWPRB/100') 
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.SZA/90, color='y', ls='-', label='SZA/90')
    ax.plot(gdf_s2_plot.xatc, gdf_s2_plot.SAA/360, color='y', ls='--', label='SAA/360')
    
    ax2 = ax.twinx()
    yl = (0,11)
    ax2.scatter(gdf_s2_plot.xatc, gdf_s2_plot.SCL, s=3, color='g', label='SCL class (right axis)')
    ylset = (yl[0] - (yl[1]-yl[0])*0.05, yl[1] + (yl[1]-yl[0])*0.05)
    ax2.set_ylim(ylset)
    ytlabs = ['no_data', 'sat_px', 'top_shdw', 'cld_shdw', 'veg', 'no_veg', 'water', 'n/a', 'cld_med', 'cld_high', 'cirr', 'snow_ice']
    ax2.set_yticks(ticks=np.arange(0, 12, 1), labels=ytlabs)
    ax2.grid(which='major', axis='y', color='gray', lw=0.1)
    
    ph1 = [h for h in ax.get_children() if isinstance(h, (mlines.Line2D, mcollections.PathCollection))]
    ph2 = [h for h in ax2.get_children() if isinstance(h, (mlines.Line2D, mcollections.PathCollection))]
    plotted_handles = ph1 + ph2
    iax += 1
    axp = axs[iax]
    axp.text(0.5, tit_ypos, 'Sentinel-2 other', ha='center', va='top', transform=axp.transAxes, fontweight='bold')
    axp.legend(handles=plotted_handles, loc='lower center', ncols=1, fontsize=lgd_fntsz, scatterpoints=4)
    
    #--------------------------------------------------------------
    iax += 1
    ax = axs[iax]
    ax.set_ylabel('zenith')
    ax2 = ax.twinx()
    ax2.set_ylabel('azimuth')
    bandcolors = {
        'B1': (1, 0, 1),
        'B2': (0, 0, 1),
        'B3': (0, 1, 0),
        'B4': (1, 0, 0),
        'B5': 'C0',
        'B6': 'C1',
        'B7': 'C2',
        'B8': 'C3',
        'B8A': 'C4',
        'B9': 'C5',
        'B11': 'C6',
        'B12': 'C7'
    }
    for k in gdf_s2_plot.keys():
        if 'VZA' in k:
            bandid = k[4:]
            linewidth = lw*2 if bandid in ['B2', 'B3', 'B4'] else lw
            ax.plot(gdf_s2_plot.xatc, gdf_s2_plot[k], color=bandcolors[bandid], lw=linewidth, ls='-', label=bandid)
        if 'VAA' in k:
            bandid = k[4:]
            linewidth = lw*2 if bandid in ['B2', 'B3', 'B4'] else lw
            ax2.plot(gdf_s2_plot.xatc, gdf_s2_plot[k], color=bandcolors[bandid], lw=linewidth, ls='--')
    
    iax += 1
    axp = axs[iax]
    leg = axp.legend(handles=ax.get_lines(), loc='lower center', ncols=3, fontsize=lgd_fntsz)
    axp.add_artist(leg)
    hsolid = mlines.Line2D([],[],color='k', ls='-', label='zenith')
    hdash = mlines.Line2D([],[],color='k', ls='--', label='azimuth')
    axp.legend(handles=[hsolid, hdash], loc='upper center', ncols=2, fontsize=lgd_fntsz, bbox_to_anchor=(0.5, 0.84))
    axp.text(0.5, tit_ypos, 'Sentinel-2 angles (AWS xml)', ha='center', va='top', transform=axp.transAxes, fontweight='bold')
    
    ax.set_xlim((gdf_s2_plot.xatc.min(),gdf_s2_plot.xatc.max()))
    
    for iax in range(0,len(axs),2):
        axs[iax].grid(which='major', axis='x', color='gray', lw=0.2)
        if iax < len(axs)-2:
            axs[iax].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    for iax in [0, 4, 6, 8]:
        ax = axs[iax]
        for xpt in intersection_pts.xatc:
            ax.axvline(xpt, color='C1', ls=':', lw=0.8)
    
    fig.tight_layout()
    
    ax = axs[2]
    image = cloudfree_collection.first()
    imgfn_out = '%s_sentinel2_rgb.tif' % lk.lake_id
    imgfn_dir = 'imagery_gt'
    axbbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axis_aspect = axbbox.width/axbbox.height
    print('--> downloading imagery for plot...')
    image_path = download_gt_imagery(img_gt, gdf_s2_plot, imgfn_out=imgfn_out, imgfn_dir=imgfn_dir, axis_aspect=axis_aspect, 
                                     gamma_value=1.0, scale=5, buffer_image=0.1, re_download=re_download_imagery)

    print('    image_path:', image_path)
    
    if image_path: 
        # product_id, datetime_print_s2, tdiff_str, tdiff_secs = get_img_props(img_gt, lk)
        with rio.open(image_path) as gtImage:
            rioplot.show(gtImage, ax=ax)
            crs_img = gtImage.crs
        saveprops_path = image_path.replace('.tif','_props.csv')
        if re_download_imagery or (not os.path.isfile(saveprops_path)):
            product_id, datetime_print_s2, tdiff_str, tdiff_secs = get_img_props(img_gt, lk, savefile=saveprops_path)
        else:
            df_props = pd.read_csv(saveprops_path).iloc[0]
            product_id = df_props.product_id
            datetime_print_s2 = df_props.datetime_print_s2
            tdiff_str = df_props.tdiff_str
        txt = 'closest imagery: Sentinel-2 on %s (%s), time difference: %s ' % (datetime_print_s2, product_id, tdiff_str)
        if lk_info.n_scenes > 1:
            txt += '[all: %.1f - %.1f hrs]' % (lk_info.tdiff_sec_min/3600, lk_info.tdiff_sec_max/3600)
        tbx = ax.text(0.001, 0.01, txt, transform=ax.transAxes, ha='left', va='bottom',fontsize=7)
        tbbox = dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5)
        tbx.set_bbox(tbbox)
    
    ax.plot([0, 5*(len(gdf_s2_plot)-1)], [0, 0], 'k-')
    ax.plot([intersection_pts.xatc.min(), intersection_pts.xatc.max()], [0, 0], 'r-')
    ax.axis('off')
    ax.set_xlim((gdf_s2_plot.xatc.min(),gdf_s2_plot.xatc.max()))
    yl_img_pm = gdf_s2_plot.xatc.max() / axis_aspect / 2
    ax.set_ylim((-yl_img_pm, yl_img_pm))
                     
    txt = 'NDWI match max: %.1f%% (%.1f%% original, %.1f%% normalized; >%g)' % (lk_info.ndwi_match_perc_max, lk_info.ndwi_match_perc, lk_info.ndwi_match_perc_norm, lk_info.ndwi_thresh)
    txt += '   |   R^2 max: %.3f (%.3f everywhere, %.3f matching NDWI)' % (lk_info.Rsquared_max, lk_info.Rsquared_max_full, lk_info.Rsquared_max_match)
    tbx = ax.text(0.5, 0.98, txt, transform=ax.transAxes, ha='center', va='top',fontsize=10)
    tbbox = dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5)
    tbx.set_bbox(tbbox)
    
    ax = axs[3]
    txt = get_lake_info_string(lk)
    ax.text(0.4, 0.5, txt, ha='center', va='center', fontsize=7)
    
    fig.tight_layout()
    plt.close(fig)
    figpath = plot_dir + plot_filename
    fig.savefig(figpath, dpi=dpi_save)
    if show_plot:
        display(fig)

    return figpath


################################################################################################################################################                   
# the column names for the tabular data
is2keys = ['lat', 'lon', 'xatc', 'depth', 'conf', 'h_fit_surf', 'h_fit_bed']
s2keys = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 
          'AOT', 'WVP', 'SCL', 'MSK_CLDPRB', 'MSK_SNWPRB', 'cloudScore', 'ndwi',
          'SZA', 'SAA',
          'VZA_B1', 'VZA_B2', 'VZA_B3', 'VZA_B4', 'VZA_B5', 'VZA_B6', 'VZA_B7', 'VZA_B8', 'VZA_B8A', 'VZA_B9', 'VZA_B11', 'VZA_B12',
          'VAA_B1', 'VAA_B2', 'VAA_B3', 'VAA_B4', 'VAA_B5', 'VAA_B6', 'VAA_B7', 'VAA_B8', 'VAA_B8A', 'VAA_B9', 'VAA_B11', 'VAA_B12',
          'tdiff_sec', 'plot_id', 'S2_id']

def get_data_and_plot(lake_id, FLUID_SuRRF_info, base_dir=None, ground_track_buffer = 7.5, max_cloudiness = 20, days_buffer = 7, min_sun_elevation = 20, 
                     limit_n_imgs = 20, ndwi_thresh = 0.15, data_dir = 'detection_out_data/', re_calculate_if_existing=True):

    if not base_dir:
        base_dir = os.getcwd()

    # get lake info and load the FLUID-SuRRF ICESat-2 data
    lk_info = FLUID_SuRRF_info[FLUID_SuRRF_info.lake_id==lake_id].iloc[0]
    lk_fn = '%s%s%s/%s.h5' % (base_dir, data_dir, lk_info.label, lk_info.lake_id)
    lk = dictobj(read_melt_lake_h5(lk_fn))
    print(lk_info.lake_id)
    new_id = '_'.join([pt for ipt, pt in enumerate(lk_info.lake_id.split('_')) if ipt not in [0,1,3,4,5,6,8]])
    fn_check = 'training_data_CSVs/%s.jpg' % new_id
    if os.path.isfile(fn_check) and (not re_calculate_if_existing):
        print('--> already processed, skipping...')
        return fn_check, None, None, None
        
    else:
        
        # get earth engine Sentinel-2 Level-2A collection for given parameters, sorted by time difference from ICESat-2
        print('--> getting Sentinel-2 Level-2A collection')
        cloudfree_collection = get_tdiff_collection(lk, days_buffer=days_buffer, max_cloudiness=max_cloudiness, 
                                                    min_sun_elevation=min_sun_elevation, limit_n_imgs=limit_n_imgs)
        # info = cloudfree_collection.getInfo()
        # scenes_info = [{'id': f['id'].split('/')[-1], 'tdiff_hrs': f['properties']['timediff']/3600, 'cloud_prob_perc': f['properties']['ground_track_cloud_prob']} for f in info['features']]
        # scenes_info_df = pd.DataFrame(scenes_info)
        # display(scenes_info_df)
        
        image_list = cloudfree_collection.toList(cloudfree_collection.size())
        n_images = cloudfree_collection.size().getInfo()
        print('--> there are %i images for the given parameters ' % n_images, end=' ')
        print('[days_buffer=%.0f, min_sun_elevation=%.0f, max_cloudiness=%.0f, limit_n_imgs=%0.f]' % (days_buffer, min_sun_elevation, max_cloudiness, limit_n_imgs))
        
        is_missing_data = True
        ith_image = 0
        made_query = False
        closest_scene = 'none'
        gdf_updated = None
        
        # iterate through all scenes while still missing data
        while is_missing_data and (ith_image < n_images):
        
            # get the image
            thisImage = ee.Image(image_list.get(ith_image))
        
            # increment image idx
            ith_image += 1
            print('    image %2i:' % ith_image, end=' ')
        
            # quick check if footprint matches the ICESat-2 ground track
            footprint_ee = thisImage.get('system:footprint').getInfo()
            #footprint_gdf = gpd.GeoDataFrame(geometry=[Polygon(footprint_ee['coordinates'])], crs="EPSG:4326")
            footprint_polygon = Polygon(footprint_ee['coordinates'])
        
            # always query Sentinel-2 for the first image
            df_ground_track_missing_data = lk.depth_data if (not made_query) else gdf_img[missing_idxs_now]
            ground_track = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df_ground_track_missing_data.lon, df_ground_track_missing_data.lat), crs="EPSG:4326")
            within_idxs = ground_track.within(footprint_polygon)
            in_footprint = within_idxs.any()
            all_in_footprint = within_idxs.all()
        
            if not in_footprint:
                print('--> not in footprint', end=' ')
        
            if in_footprint:
        
                if all_in_footprint:
                    print('--> fully in footprint', end=' ')
                else:
                    print('--> partially in footprint', end=' ')
                    
                # if this is not the first image:
                # copy over the missing data column positions from the previous iteration
                if made_query:
                    missing_idxs_previous = missing_idxs_now.copy()
        
                try:
                    # get the along-track values from google earth engine
                    print('--> get EE data', end=' ')
                    df_ee = get_image_along_track(lk, thisImage, ground_track_buffer=ground_track_buffer)
            
                    # save the metadata file with the sun/view angle grids from AWS S3
                    print('--> get metadata from S3', end=' ')
                    meta_fn_save = get_metadata_file(thisImage)
            
                    # interpolate angle grids to ICESat-2 ground track
                    print('--> interpolate angles', end=' ')
                    gdf_img = add_interpolated_angles(meta_fn_save, df_ee)
            
                    # fill any missing values with the ee-provided mean angles
                    gdf_img = fill_na_angles_with_means(gdf_img, thisImage)
            
                    # check if any data is still missing 
                    missing_idxs_thisImage = gdf_img[s2keys].isna().any(axis=1)
                    
                    # if it's the first image create gdf_updated
                    if (not made_query):
                        print('--> got first S2 data', end=' ')
                        gdf_updated = gdf_img.copy()
                        
                    # if it's not the first image, update the values that were missing in the previous iteration
                    else:
                        # get positions where there was data missing previously, but now there is new data
                        print('--> get missing data', end=' ')
                        has_newdata_idxs = missing_idxs_previous & (~missing_idxs_thisImage)
            
                        # if there is any newly updated data
                        if has_newdata_idxs.any():
                            print('--> got more data', end=' ')
                            # update in gdf_updated, where we collect all the results
                            gdf_updated.loc[has_newdata_idxs,:] = gdf_img[has_newdata_idxs]
            
                    # update is_missing_data to tell the while loop to stop once we have data for all ground track locations
                    missing_idxs_now = gdf_updated[s2keys].isna().any(axis=1)
                    is_missing_data = missing_idxs_now.any()
                    nmissing = missing_idxs_now.sum()
                    ntotal = len(missing_idxs_now)
                    ndone = '(%i/%i)' % (ntotal-nmissing, ntotal)
        
                    # set queried flag to True once getting data from the first image
                    if (~missing_idxs_now).any():
                        made_query = True
                        closest_scene = thisImage.get('scene_id').getInfo()
                        first_image = thisImage
        
                    print(('--> still missing data %s...' % ndone) if is_missing_data else ('--> complete. (%i data points)' % ntotal))
        
                except:
                    traceback.print_exc()
        
        if lk_info.lake_id.startswith('lake_'):
            new_id = '_'.join([pt for ipt, pt in enumerate(lk_info.lake_id.split('_')) if ipt not in [0,1,3,4,5,6,8]])
            fluidsurrf_file_id = lk_info.lake_id
            lk_info.lake_id = new_id
            lk.lake_id = new_id
        else:
            new_id = lk_info.lake_id
    
        if gdf_updated is not None:
    
            # prep the final data
            gdf_final = gdf_updated[is2keys + s2keys].copy()
            gdf_final['IS2_id'] = lk_info.lake_id
            gdf_final = gdf_final[~gdf_final[is2keys].isna().any(axis=1)]
            
            # remove the x-axis (xatc) offset from zero from the data
            xoff = gdf_final.xatc.min()
            lk.photon_data.xatc -= xoff
            gdf_final.xatc -= xoff
            
            training_data_csv_dir = 'training_data_CSVs/'
            if not os.path.exists(training_data_csv_dir):
                os.makedirs(training_data_csv_dir)
            gdf_final_fn = '%s%s.csv' % (training_data_csv_dir, lk_info.lake_id)
            gdf_final.to_csv(gdf_final_fn, index=False)
            
            # get extra lake info and save to info file 
            def norma(x):
                xn = x - np.nanmin(x)
                return xn / np.nanmax(xn)

            ndwi_match = (gdf_updated.depth > 0) == (gdf_updated.ndwi > ndwi_thresh)
            ndwi_match_norm = (gdf_updated.depth > 0) == (norma(gdf_updated.ndwi) > ndwi_thresh)
            ndwi_match_perc = ndwi_match.mean() * 100
            ndwi_match_perc_norm = ndwi_match_norm.mean() * 100
            lk_info['ndwi_match_perc'] = ndwi_match_perc
            lk_info['ndwi_match_perc_norm'] = ndwi_match_perc_norm
            lk_info['ndwi_match_perc_max'] = lk_info[['ndwi_match_perc', 'ndwi_match_perc_norm']].max()
            lk_info['ndwi_thresh'] = ndwi_thresh
                
            def check_finite(arr1, arr2):
                return ~(np.isinf(arr1) | np.isnan(arr1) | np.isinf(arr2) | np.isnan(arr2))
            min_fin = 10
            bg_avg = norma(np.mean(np.vstack((norma(np.log(gdf_updated.B2)), norma(np.log(gdf_updated.B3)))), axis=0))
            fin = check_finite(gdf_updated.depth, bg_avg)
            mat = ndwi_match & fin
            lk_info['Rsquared_log_B2B3'] = 0.0 if (np.sum(fin) < min_fin) else pearsonr(-gdf_updated.depth[fin], bg_avg[fin]).statistic
            lk_info['Rsquared_log_B2B3_match'] = 0.0 if (np.sum(mat) < min_fin) else pearsonr(-gdf_updated.depth[mat], bg_avg[mat]).statistic
            fin = check_finite(gdf_updated.depth, np.log(gdf_updated.B2))
            mat = ndwi_match & fin
            lk_info['Rsquared_log_B2'] = 0.0 if (np.sum(fin) < min_fin) else pearsonr(-gdf_updated.depth[fin], np.log(gdf_updated.B2)[fin]).statistic
            lk_info['Rsquared_log_B2_match'] = 0.0 if (np.sum(mat) < min_fin) else pearsonr(-gdf_updated.depth[mat], np.log(gdf_updated.B2)[mat]).statistic
            fin = check_finite(gdf_updated.depth, np.log(gdf_updated.B3))
            mat = ndwi_match & fin
            lk_info['Rsquared_log_B3'] = 0.0 if (np.sum(fin) < min_fin) else pearsonr(-gdf_updated.depth[fin], np.log(gdf_updated.B3)[fin]).statistic
            lk_info['Rsquared_log_B3_match'] = 0.0 if (np.sum(mat) < min_fin) else pearsonr(-gdf_updated.depth[mat], np.log(gdf_updated.B3)[mat]).statistic
            fin = check_finite(gdf_updated.depth, np.log(gdf_updated.B4))
            mat = ndwi_match & fin
            lk_info['Rsquared_log_B4'] = 0.0 if (np.sum(fin) < min_fin) else pearsonr(-gdf_updated.depth[fin], np.log(gdf_updated.B4)[fin]).statistic
            lk_info['Rsquared_log_B4_match'] = 0.0 if (np.sum(mat) < min_fin) else pearsonr(-gdf_updated.depth[mat], np.log(gdf_updated.B4)[mat]).statistic
            lk_info['Rsquared_max_full'] = lk_info[['Rsquared_log_B2', 'Rsquared_log_B3', 'Rsquared_log_B4', 'Rsquared_log_B2B3']].max()
            lk_info['Rsquared_max_match'] = lk_info[['Rsquared_log_B2_match', 'Rsquared_log_B3_match', 'Rsquared_log_B4_match', 'Rsquared_log_B2B3_match']].max()
            lk_info['Rsquared_max'] = lk_info[['Rsquared_max_full', 'Rsquared_max_match']].max()
            
            product_id, datetime_print_s2, tdiff_str, tdiff_secs = get_img_props(first_image, lk)
            lk_info['datetime_S2'] = datetime_print_s2
            lk_info['tdiff_str'] = tdiff_str
            lk_info['tdiff_sec_mean'] = gdf_final.tdiff_sec.mean()
            lk_info['tdiff_sec_min'] = gdf_final.tdiff_sec.min()
            lk_info['tdiff_sec_max'] = gdf_final.tdiff_sec.max()
            lk_info['n_scenes'] = len(np.unique(gdf_final.S2_id))
            lk_info['S2_id_first'] = closest_scene
            lk_info['fluidsurrf_file_id'] = fluidsurrf_file_id
            lk_info_fn = '%s%s_lakeinfo.csv' % (training_data_csv_dir, lk_info.lake_id)
            lk_info.to_frame().T.to_csv(lk_info_fn, index=False)
            
            # update photon data, lake props in lk, and write to hdf5 in atl03_segments/
            lk.depth_data = gdf_final
            prop_keys = ['beam_number', 'beam_strength', 'cycle_number', 'date_time', 'dead_time', 'dead_time_meters', 'depth_quality_sort', 
                         'granule_id', 'gtx', 'ice_sheet', 'lake_id', 'lat', 'lat_max', 'lat_min', 'lat_str', 'len_subsegs', 'lon', 'lon_max', 
                         'lon_min', 'lon_str', 'max_depth', 'melt_season', 'mframe_end', 'mframe_start', 'n_subsegs_per_mframe', 'polygon_filename', 
                         'polygon_name', 'rgt', 'sc_orient', 'surface_elevation', 'depth_data', 'photon_data']
            for prop in list(vars(lk).keys()):
                if not (prop in prop_keys):
                    delattr(lk, prop)
            for k in lk_info.keys():
                setattr(lk, k, lk_info[k])
            lk.depth_data['S2_id'] = lk.depth_data['S2_id'].astype('|S100')
            lk.depth_data['IS2_id'] = lk.depth_data['IS2_id'].astype('|S100')
            delattr(lk, 'geometry')
            atl03_dir = 'atl03_segments/'
            if not os.path.exists(atl03_dir):
                os.makedirs(atl03_dir)
            h5fn = write_data_to_hdf5(lk, '%s%s.h5' % (atl03_dir, lk_info.lake_id))
            
            re_download_imagery = True
            lk = dictobj(read_melt_lake_h5(h5fn))
            lk_info = pd.read_csv(lk_info_fn).iloc[0].T
            df = pd.read_csv(gdf_final_fn)
            gdf_s2_plot = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
            plot_dir = 'training_data_CSVs'
            plot_filename = '%s.jpg' % lk_info.lake_id
            
            fig_path = plot_IS2S2_lake(lk, lk_info, gdf_s2_plot, cloudfree_collection, first_image, plot_filename=plot_filename, plot_dir=plot_dir, 
                                       re_download_imagery=re_download_imagery, show_plot=True)
            return fig_path, lk, gdf_final, lk_info
        else:
            warnings.warn('\n\nCould not get data for this lake.\n')
            # create dummy figure to indicate missing data
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'NO DATA for this lake.', ha='center', va='center', fontsize=16)
            fig_path = 'training_data_CSVs/%s.jpg' % new_id
            fig.savefig(fig_path)
            plt.close(fig)
            print('')
            traceback.print_exc()
            return fig_path, None, None, None