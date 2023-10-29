//* Function to mask clouds using the Sentinel-2 QA band
//* QA60 is the cloud mask band
//* @param {ee.Image} image Sentinel-2 image
//* @return {ee.Image} cloud masked Sentinel-2 image

// Do not change this part of the code. 'QA60' is the cloud band. 
function maskS2clouds(image) {
var qa = image.select('QA60');

// Bits 10 and 11 are clouds and cirrus, respectively.
var cloudBitMask = 1 << 10;
var cirrusBitMask = 1 << 11;

// Both flags should be set to zero, indicating clear conditions.
var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
                  .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

return image.updateMask(mask).divide(10000);}

// Collect your S2 imagery.
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

/////               Sentinel-2 Imagery                     //////// 

var S2_2019 = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate('2019-07-01', '2019-09-30')
                  .filterBounds(roi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
                  .map(maskS2clouds);

var Optical_Select = S2_2019.select('B1','B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12')

var Optical_Median = Optical_Select.median();
var Optical_Data = Optical_Median.clip(roi);
// print('All metadata:', Optical_Data)

// Create parameters for visualizing the log transformed imagery.
var visualization_log = {
                  min: -4,
                  max: -1,
                  bands: ['B4', 'B3', 'B2'],
                  };

// Log transform the optical imagery. 
var Optical_Data_Log = Optical_Data.log();

// Add the log transformed optical imagery into GEE for viewing.
 Map.addLayer(Optical_Data_Log, visualization_log, 'Cloud_Free_S2');

// Create ratios.

var B2_B3 = Optical_Data.normalizedDifference(['B2', 'B3']).rename('B2_B3');
var B2_B4 = Optical_Data.normalizedDifference(['B2', 'B4']).rename('B2_B4');

//var B1_B2L = B1_B2.log();

var Optical_Data_Comp = ee.Image.cat(Optical_Data_Log, B2_B3, B2_B4);
 print('All metadata:', Optical_Data_Comp);

var Bands =   ['B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'B2_B3', 'B2_B4']

var band_viz = {
  min: -1,
  max: 1,
  palette: ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
};

Map.addLayer(B2_B3, band_viz, 'B2_B3');

// Prepare model training data. 
var Training = Optical_Data_Comp.select(Bands).sampleRegions({
                collection: cal_data,
                properties: ['Depth'],
                scale: 10,
                tileScale: 16
                });

var Lin_Training = Optical_Data_Comp.select(Bands).sampleRegions({
  collection: cal_data,
  properties: ['Depth'],
  scale: 10
})
// Add a constant property to each feature to be used as an independent variable.
.map(function(feature) {
  return feature.set('constant', 1);
});
                
// Filter out the null property values and try again.
var Training_Null = Training.filter(
  ee.Filter.notNull(Training.first().propertyNames())
);

// Filter out the null property values and try again.
var Lin_Training_Null = Lin_Training.filter(
  ee.Filter.notNull(Lin_Training.first().propertyNames())
);

// Determine correlation between in situ data and satellite imagery.
// Call reducer.
 var cor = ee.Reducer.pearsonsCorrelation();

// Then choose two properties to see the correlation:
 var reduced = Training_Null.reduceColumns(cor, ['Depth', 'B4'])
 print(reduced)

// Determine correlation between satellite image bands.
// First sample the bands.
// var sample = Optical_Data_Comp.sample({'region':roi, 
                            //  'scale': 10, 
                            //'projection': 'EPSG:4326',
                            //  'numPixels':1000});
// Call reducer correctly:
// var cor = ee.Reducer.pearsonsCorrelation();
// Then choose two properties to see the correlation:
// var reduced9 = sample.reduceColumns(cor, ['B2_B3', 'B2_B4'])
// print(reduced9)

// Linear regression

var linearRegression = ee.Dictionary(Lin_Training_Null.reduceColumns({
  reducer: ee.Reducer.linearRegression({
    numX: 14,
    numY: 1
  }),
  selectors: ['constant', 'B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'B2_B3', 'B2_B4', 'Depth']
}));

//print('Linear regression results:', linearRegression);

// Extract the coefficients as a list.
var coefficients = ee.Array(linearRegression.get('coefficients'))
    .project([0])
    .toList();

// print('Coefficients', coefficients);

// Create the predicted tree cover based on linear regression.
var MLR_Regression = ee.Image(1)
    .addBands(Optical_Data_Comp.select(Bands))
    .multiply(ee.Image.constant(coefficients))
    .reduce(ee.Reducer.sum())
    .rename('Multiple_LR')
    .clip(roi);

//Map.addLayer(Multiple_LR, {
    //min: 0,
    //max: 100
//}, 'Multiple_LR');

// Machine learning models.
// Create RF model.
var RF_Model = ee.Classifier.smileRandomForest(100)//, null, 1, 0.5, null, 0)
  .setOutputMode('REGRESSION')
  .train({
    features: Training_Null,
    classProperty: 'Depth',
    inputProperties: Bands
    });

var RF_Regression = Optical_Data_Comp.select(Bands).classify(RF_Model, 'predicted');

// Create SVM model.
var SVM_Model = ee.Classifier.libsvm({svmType: 'EPSILON_SVR', kernelType: 'RBF', gamma: 0.5, cost:10})
  .setOutputMode('REGRESSION')
  .train({
    features: Training_Null,
    classProperty: 'Depth',
    inputProperties: Bands
    });

var SVM_Regression = Optical_Data_Comp.select(Bands).classify(SVM_Model, 'predicted');

// Create CART model.
var CART_Model = ee.Classifier.smileCart()
  .setOutputMode('REGRESSION')
  .train({
    features: Training_Null,
    classProperty: 'Depth',
    inputProperties: Bands
    });

var CART_Regression = Optical_Data_Comp.select(Bands).classify(CART_Model);

//Map.addLayer(RF_Regression, {
    //min: 0,
    //max: 50
//}, 'predicted');

// Export the selected image to your Google Drive
Export.image.toDrive({
  image: RF_Regression,
    description: "RF_Regression",
    scale: 10,
    folder: 'GEE_Outputs',
    region: roi,
    maxPixels: 10000000000000,
    fileFormat: 'GeoTIFF'
}); 