from __future__ import division
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic


def read_raster(in_raster):
    in_raster=in_raster
    ds = gdal.Open(in_raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    data[data==-99] = np.nan
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    del ds
    # create a grid of xy coordinates in the original projection
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    return data, xx, yy, gt

def rescale(dat,mn,mx):
    """
     rescales an input dat between mn and mx
    """
    m = np.min(dat.flatten())
    M = np.max(dat.flatten())
    return (mx-mn)*(dat-m)/(M-m)+mn

#def crop_toseg(mask, im):
#    true_points = np.argwhere(mask)
#    top_left = true_points.min(axis=0)
#    # take the largest points and use them as the bottom right of your crop
#    bottom_right = true_points.max(axis=0)
#    cmask = mask[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
#    cmask = cmask.astype(bool)
#    nrows, ncols = np.shape(cmask)
#    
#    maxarea = (0, [])
#    
#    arr = np.asarray(cmask, dtype='int')
#    wid = np.zeros(dtype=int, shape=arr.shape)
#    hght = np.zeros(dtype=int, shape=arr.shape)
#    
#    for row in xrange(nrows):
#        for col in xrange(ncols):
#            if arr[row][col] == 0:
#                continue
#    
#            if row == 0:
#                hght[row][col] = 0
#            else:
#                hght[row][col] = hght[row-1][col]+1
#    
#            if col == 0:
#                wid[row][col] = 0
#            else:
#                wid[row][col] = wid[row][col-1]+1
#            min_w = wid[row][col]
#    
#            for delta_hght in xrange(hght[row][col]):
#                min_w = np.min([min_w, wid[row-delta_hght][col]])
#                areanow = (delta_hght+1)*min_w
#                if areanow > maxarea[0]:
#                    maxarea = (areanow, [(row-delta_hght, col-min_w+1, row, col)])
#    
#    t = np.squeeze(maxarea[1])
#    
#    min_row = t[0]; max_row = t[2]
#    min_col = t[1]; max_col = t[3]
#         
#    cim = im[min_row:max_row, min_col:max_col]
#       
#    return cmask,cim  
def crop_toseg(mask, im):
   true_points = np.argwhere(mask)
   top_left = true_points.min(axis=0)
   # take the largest points and use them as the bottom right of your crop
   bottom_right = true_points.max(axis=0)
   return mask[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1], im[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]  
   
if __name__ == '__main__':
    
        ss_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\10_percent_shift\raster\ss_50_rasterclipped.tif",
                'R01765':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif",
                'R01767':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"}
        
        for (k1,v1) in ss_dict.items()[0:1]:
            im = read_raster(v1)[0]
            im[np.isnan(im)] = 0
            im = rescale(im,0,1)
            segments_slic = slic(im, n_segments=900, compactness=.1)
            
            im = read_raster(v1)[0]
            im[np.isnan(im)] = 0
            for k in np.unique(segments_slic):
                mask = np.zeros(im.shape[:2], dtype = "uint8")
                mask[segments_slic == k] = 255
                cmask, cim = crop_toseg(mask, im)
                test = cmask.astype(bool)
                
                nrows, ncols = np.shape(test)

                maxarea = (0, [])
                
                arr = np.asarray(test, dtype='int')
                wid = np.zeros(dtype=int, shape=arr.shape)
                hght = np.zeros(dtype=int, shape=arr.shape)
                
                for row in xrange(nrows):
                    for col in xrange(ncols):
                        if arr[row][col] == 0:
                            continue
                
                        if row == 0:
                            hght[row][col] = 0
                        else:
                            hght[row][col] = hght[row-1][col]+1
                
                        if col == 0:
                            wid[row][col] = 0
                        else:
                            wid[row][col] = wid[row][col-1]+1
                        min_w = wid[row][col]
                
                        for delta_hght in xrange(hght[row][col]):
                            min_w = np.min([min_w, wid[row-delta_hght][col]])
                            areanow = (delta_hght+1)*min_w
                            if areanow > maxarea[0]:
                                maxarea = (areanow, [(row-delta_hght, col-min_w+1, row, col)])
                
                t = np.squeeze(maxarea[1])
                
                min_row = t[0]; max_row = t[2]
                min_col = t[1]; max_col = t[3]
                
                plt.imshow(cim)
                plt.plot([min_row, min_row, max_row, max_row, min_row], [min_col, max_col, max_col, min_col, min_col], 'w-o')
                
                crop_im = cim[min_row:max_row, min_col:max_col]
                

                
                
                
                # create a binary blob (circle)
X,Y = np.meshgrid(np.arange(-200,200), np.arange(-200,200))
image = (X**2 + Y**2)<(180**2)

nrows, ncols = np.shape(image)

maxarea = (0, [])

arr = np.asarray(image, dtype='int')
wid = np.zeros(dtype=int, shape=arr.shape)
hght = np.zeros(dtype=int, shape=arr.shape)

for row in xrange(nrows):
    for col in xrange(ncols):
        if arr[row][col] == 0:
            continue

        if row == 0:
            hght[row][col] = 0
        else:
            hght[row][col] = hght[row-1][col]+1

        if col == 0:
            wid[row][col] = 0
        else:
            wid[row][col] = wid[row][col-1]+1
        min_w = wid[row][col]

        for delta_hght in xrange(hght[row][col]):
            min_w = np.min([min_w, wid[row-delta_hght][col]])
            areanow = (delta_hght+1)*min_w
            if areanow > maxarea[0]:
                maxarea = (areanow, [(row-delta_hght, col-min_w+1, row, col)])

t = np.squeeze(maxarea[1])

min_row = t[0]; max_row = t[2]
min_col = t[1]; max_col = t[3]

plt.imshow(image)
plt.plot([min_row, min_row, max_row, max_row, min_row], [min_col, max_col, max_col, min_col, min_col], 'w-o')

crop_im = image[min_row:max_row, min_col:max_col]




