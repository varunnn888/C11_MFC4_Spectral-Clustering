DATASET DESCRIPTION

About Dataset
Context
This dataset is being used for classifying the use of land in geospatial images. The end goal for the classification is that the top 2 uses of land in an image are given as output to the user.

Content
This dataset contains images belonging to the EuroSat dataset. There are 2 folders, namely,

EuroSAT → Contains RGB images collected from the Sentinel Dataset.
EuroSATallBands → Contains .tif files which have all the bands of the spectrum as collected from the Sentinel-2 satellite.
Each image is 64x64 pixels with a Ground Sampling Distance of 10m. They were all collected from the Sentinel-2 satellite

The 2 directories containing the following class folders :

AnnualCrop
Forest
HerbaceousVegatation
Highway
Industrial
Pasture
PermanentCrop
Residential
River
SeaLake
Note: Drop Index columns [the first column in all CSV files] while reading the training, testing and validation CSV files.
