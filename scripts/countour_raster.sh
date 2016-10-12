#Bash script to contour DEMs
in_tifs=$(find C:/workspace/GLCM/output/ | egrep "*.tif$")
array1=($in_tifs)
outdir="C:/workspace/Merged_SS/raster/2015_04/"
ext='.shp'
sep='/'
for i in "${!array1[@]}"; do
tif_file=${array1[$i]}
echo $tif_file
DIR=$(dirname "${tif_file}}")
base="$(echo ${tif_file##*/})"
base2=${base%.tif}
oName=$DIR$sep$base2$ext
proj='_nad83'
newName=$DIR$sep$base2$proj$ext
echo $oName
(cd C:/Program\ Files/GDAL; gdal_contour -b 1 -a std -snodata -99 -fl 2.3 2.5 3.5  -nln layer -lco layer $tif_file $oName)
(cd C:/Program\ Files/GDAL; ogr2ogr -s_srs EPSG:2762 -t_srs EPSG:26949 -f "ESRI Shapefile" $newName $oName)
python "C:/workspace/GLCM/scripts/delete_shapefile.py" $oName
done

