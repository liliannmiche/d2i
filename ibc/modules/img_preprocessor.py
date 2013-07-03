# -*- coding: utf-8 -*-
"""Processes an image and returns all data (properties + descriptors).

If something goes wrong, returns string describing the error.
@author: akusoka1
"""

from ibc_config import IBCConfig as cf
from tables import openFile
import numpy as np
import Image
import hashlib
import cPickle
import os


def _get_img_list():
    """Get a list of valid images, with old and new paths.
    """    
    # load websites description
    wsd = []
    for line in open(cf._ws_descr, 'r'):
        # use [:-1] to remove "newline" character
        wsd.append(line[:-1].split(";"))
    imglist = []

    # checking that img_dir exists
    if not os.path.isdir(cf._img_dir):
        os.mkdir(cf._img_dir)

    # creating list of old/new images, creating new unique website folders
    dirs = []
    for _,dirs,_ in os.walk(cf._img_dir):
        break
    Nws = len(dirs)  # current amount of websites

    for ws in wsd:
        # creating another website folder
        Nws += 1
        newdir = os.path.join(cf._img_dir, "ws%06d"%Nws)
        os.mkdir(newdir)

        # iterating over existing files
        Nimg = 0  # image number in that website
        for root,dirs,files in os.walk(os.path.join(cf._raw_dir, ws[0])):
            for f in files:
                img_raw = os.path.join(root, f)
                
                # check image size
                if os.stat(img_raw).st_size < cf._min_size:
                    print "Image %s too small" % img_raw.split("/")[-2:]
                    continue
                
                # check if an image can be opened (= is a valid image file)
                try:
                    _ = Image.open(img_raw).convert('RGB')  # drop the result
                except:
                    print "Not a valid image: %s" % img_raw
                    continue
                
                Nimg += 1
                img_new = os.path.join(newdir, "img%06d.jpg"%Nimg)
                imglist.append((img_raw, img_new, ws))
    return imglist    


def _normalize_images():
    """Normalize images, no statistics kept.
    """
    imglist = []
    for img_raw, img_new, ws in _get_img_list():
        img_obj = Image.open(img_raw).convert('RGB')

        # check image dimensions, resize if needed
        maxs = cf._max_dim
        (x, y) = img_obj.size
        if (x > maxs) and (x >= y):
            y = int(y * (float(maxs) / x))
            x = maxs
            img_obj = img_obj.resize((x, y), Image.ANTIALIAS)
        elif y > maxs:
            x = int(x * (float(maxs) / y))
            y = maxs
            img_obj = img_obj.resize((x, y), Image.ANTIALIAS)        

        # saving processed image
        img_obj.save(img_new, 'JPEG', quality=cf._jpeg_quality)
        imglist.append((img_new, ws[1]))
    # saving image-website mapping
    cPickle.dump(imglist, open(cf._img_data, "wb"), -1)
        
        
def _normalize_images_hdf5():
    """Normalize images, write image records with statistics to HDF5 file.
    """    
    imglist = []  # new list with all image attributes, to fill HDF5
    for img_raw, img_new, ws in _get_img_list():
        img_obj = Image.open(img_raw).convert('RGB')
        data = {} # all image information

        data["true_size"] = img_obj.size
        data["true_name"] = img_raw
        data["orig_sha1"] = hashlib.sha1(open(img_raw, 'r').read()).hexdigest()
        data["classN"] = int(ws[2])
        data["ws_index"] = ws[1]  # just an URL here, find website index later
        
        # check image dimensions, resize if needed
        maxs = cf._max_dim
        (x, y) = img_obj.size
        if (x > maxs) and (x >= y):
            y = int(y * (float(maxs) / x))
            x = maxs
            img_obj = img_obj.resize((x, y), Image.ANTIALIAS)
        elif y > maxs:
            x = int(x * (float(maxs) / y))
            y = maxs
            img_obj = img_obj.resize((x, y), Image.ANTIALIAS)        
        data["new_size"] = img_obj.size

        # saving processed image
        data["file_name"] = img_new
        img_obj.save(img_new, 'JPEG', quality=cf._jpeg_quality)
        
        # adding image information
        imglist.append(data)

    # finding website indices
    db = openFile(cf._hdf5, "a")
    Websites = db.root.Websites
    nr_in_site = {}
    for data in imglist:
        url = data["ws_index"]
        idx = Websites.readWhere('url=="%s"'%url, field="index")[0]
        data["ws_index"] = idx
        # calculating Images.nr_in_site, and Websites.img_present
        if not idx in nr_in_site:  # add another website      :
            nr_in_site[idx] = 0
            data["nr_in_site"] = 0
        else:
            nr_in_site[idx] += 1
            data["nr_in_site"] = nr_in_site[idx]
    
    # writing image table records
    Images = db.root.Images
    img_last = Images.attrs.last_index
    nr_in_class = Images.attrs.nr_in_class

    for data in imglist:
        # adding image
        row = Images.row
        row["index"] = img_last + 1
        classN = data["classN"]
        row["classN"] = classN
        row["site_index"] = data["ws_index"]
        row["nr_in_class"] = nr_in_class[classN]
        row["nr_in_site"] = data["nr_in_site"]
        row["orig_sha1"] = data["orig_sha1"]
        row["true_size"] = data["true_size"]
        row["new_size"] = data["new_size"]
        row["true_name"] = data["true_name"]
        row["file_name"] = data["file_name"]
        row.append()                
        img_last += 1
        nr_in_class[classN] += 1
    
    # writing Websites.img_present attribute of websites
    for idx in nr_in_site:
        Websites.modify_column(idx, idx+1, colname="img_present",
                               column=nr_in_site[idx])

    Images.attrs.last_index = img_last
    Images.attrs.nr_in_class = nr_in_class
    Images.flush()
    Websites.flush()
    db.close()


def normalize_images():
    """Chooses either single or batch version.
    """
    if cf._mode == "hdf5":
        _normalize_images_hdf5()
    else:
        _normalize_images()













