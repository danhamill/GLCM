# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:06:10 2016

@author: dan
"""

import sys
import ogr

fn = sys.argv[1]
import os
driver = ogr.GetDriverByName("ESRI Shapefile")
if os.path.exists(fn):
     driver.DeleteDataSource(fn)

