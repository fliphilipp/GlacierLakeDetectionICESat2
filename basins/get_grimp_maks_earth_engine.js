// Step 1: Load the GeoTIFF
var assetPath = 'projects/ee-philipparndt/assets/GimpIceMask_90m_2015_v1-2'; // GRIMP ice mask, imported into EE as an asset
var geotiff = ee.Image(assetPath);

var min_area_filter = 1000000;
var thisscale = 90;

var geotiffViz = {min: 0, max: 1, palette: ['00FFFF', '0000FF']};
// Map.addLayer(geotiff, geotiffViz, 'input geotiff', true);

// Step 2: Mask out the areas with value 0
var mask = geotiff.select('b1').eq(1).selfMask();

var min_area = 0.07; // km^2  (about 30 pixels)
var reduce_scale = ee.Number(min_area).multiply(1e6).sqrt().multiply(0.8).round();

// Step 3: Convert the masked image to vectors (polygons)
var vectors = mask.reduceToVectors({
  //scale: reduce_scale,
  scale: thisscale,
  geometryType: 'polygon',
  eightConnected: false,
  maxPixels: 1e13,
  crs: 'EPSG:3413',
  tileScale: 16, // Reduces aggregation tile size, higher number reduces risk of memory error
  // geometry: geometry,
});
var err = ee.ErrorMargin(reduce_scale);
var addArea = function(feature) {
  var polygonArea = ee.Number(feature.geometry(err).area(err).divide(1e6));
  return feature.set({areaKm2: polygonArea});
};

vectors = vectors.map(addArea);
// print('total lake patches',water_vectors.size());

var vectors_filtered = vectors.filter(ee.Filter.gt('areaKm2', min_area_filter));

var empty = ee.Image().byte();
var filledOutlines = empty.paint(vectors_filtered).paint(vectors_filtered, 0, 1);

var roi_viz = {color: 'black', fillColor: '00000000'};
Map.addLayer(mask, {palette: ['red']}, 'filtered ice mask');
// Map.addLayer(filledOutlines, {palette: ['blue']}, 'filtered ice vectors');

Export.table.toDrive({
  collection: vectors_filtered,
  description: 'GimpIceMask_90m_2015_v1-2_vector_polygons_scale90',
  folder:"Continent_wide_shapefiles", // If wanting to save GeoJson directly into this folder
  fileFormat: 'SHP'
});
