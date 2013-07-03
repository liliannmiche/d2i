"""Setup file for the WIC project.

DO NOT modify table descriptions unnecessarily! Changing them will require
changing all filler files, and re-computing or re-structuring database files.
Generally, additional fields in tables do not lead to noticable degradation
in performance of increase of storage space.

"""
__author__ = "Anton Akusok"
__license__ = "PSFL"
__version__ = "1.0.0"

from ibc_config import IBCConfig as cf
from tables import Int64Col, Float64Col, UInt8Col, Int8Col,\
                   StringCol, IsDescription, openFile
import numpy as np

class WebsitesRecord(IsDescription):
    """Table type which describes websites.

    This and following table types are used to create database tables
    with 'pyTables' (alias 'tables') library. pyTables provides a tabular
    extension to HDF5 storage mechanism, which normally uses array data.
    Tables allow storing and searching attributes in tables efficiently.
    Also, tables are useful for storing variable amounts of local features
    per image - using attributes as pointers to the image owning each feature,
    and utilizing extremely fast sequential I/O with a sorter table (compared
    to common relational databases).

    If needed, data from a table can be red directrly as a traditional array.
    Though table structure is fixed, their content depends on the exact method
    used, so several tables corresponding to particular method are stored
    together in a database file for convenience.
    
    Attributes:
        index: Unique index of a website.
        classN: True class of a website.
        img_count: Total number of images on that web page.
        img_present: Number of database images in that website.
        predict: Means and confidences for website classes predictions.
        folder: Folder containing images from that website.
        url: Web page of that website, if known.
        class_text: Textual notation of the class
    """
    index           = Int64Col(dflt=-1, pos=0)
    classN          = Int8Col( dflt=-1, pos=1)
    img_count       = Int64Col(dflt=-1, pos=2)
    img_present     = Int64Col(dflt=-1, pos=3)
    predict         = Float64Col(shape=(cf._maxc, 2), dflt=-1, pos=4)
    folder          = StringCol(itemsize=1024)
    url             = StringCol(itemsize=1024)
    class_text      = StringCol(itemsize=1024)

class ImagesRecord(IsDescription):
    """Table type which describes images.

    Attributes:
        index: Unique image index.
        classN: True class of image's website, true class for the image
            is unknown.
        site_index: Index of website this image is taken from.
        reg_first: Index of the first local image region in Regions table.
        reg_count: Number of regions that image has.
        nr_in_site: Number of image in website, can be used to restrict an
            amount of images processed from one website.
        nr_in_class: Number of image in class, can be used to restrict an
            amount of images processed from each class.
        predict:  Means and confidences for image classes predictions.
        orig_sha1: Sha1 of an original image, used to find duplicates.
        true_size: Dimensionality (x,y) of an original image.
        new_size: Dimensionality (x,y) of a resized image.
        true_name: File name of an original downloaded image.
        file_name: File name of a preprocessed image.
    """
    index           = Int64Col(dflt=-1, pos=0)
    classN          = Int8Col( dflt=-1, pos=1)
    site_index      = Int64Col(dflt=-1, pos=2)
    reg_first       = Int64Col(dflt=-1, pos=3)
    reg_count       = Int64Col(dflt=-1, pos=4)
    nr_in_site      = Int64Col(dflt=-1, pos=5)
    nr_in_class     = Int64Col(dflt=-1, pos=6)
    img_repr        = Float64Col(shape=(2*cf._maxc), dflt=-1)
    predict         = Float64Col(shape=(cf._maxc, 2), dflt=-1, pos=7)
    orig_sha1       = StringCol(itemsize=40, dflt="", pos=8)  # for finding duplicates
    true_size       = Int64Col(shape=(2), dflt=-1, pos=9)  # (x,y) before resize
    new_size        = Int64Col(shape=(2), dflt=-1, pos=9)  # (x,y) after resize
    true_name       = StringCol(itemsize=255, pos=10)  # name of a downloaded file
    file_name       = StringCol(itemsize=255, pos=11)  # name of a preprocessed image
    

class RegionsRecord(IsDescription):
    """Table type what describes image local samples.

    Attributes:
        index: Unique index of local sample. Samples from one image assume to
            have consecutive indices, so they can be red sequentially.
        img_class: "Class" of image (class of website that image belongs to).
            Assumed to be the class of that local sample, although true
            dependence is very weak. Used for sample-based algorithms (finding
            best centroids for kNN, GRLVQ, etc.)
        img_site: Website index of parental image. Allows website-based
            selection of samples.
        img_index: Index of parental image. Allows image-based selection of
            samples, although use of Image's 'reg_first' and 'reg_nr'
            attributes is faster if they are available.
        pyramid: Used in spatial pyramid approach.
        center: Center coordinates (x,y) of that local sample.
        radius: Radius of that local sample. Knowing center and radius,
            it is possible to read pixel representation of sample from image.
        cornerness: Some value returned by sample detection algorithm.
            May be useful, don't take much space anyway.
        predict:  Means and confidences for region classes predictions.
                  Can also store nearest neighbour indices here.

    For practical purposes, all regions are detected and stored alongside images
    in a database. Descriptors of images are stored in a separate table, but 
    in cells with corresponding addresses (row indices).
    """
    index           = Int64Col(dflt=-1, pos=0)
    img_class       = Int8Col( dflt=-1, pos=1)
    img_site        = Int64Col(dflt=-1, pos=2)
    img_index       = Int64Col(dflt=-1, pos=3)
    pyramid         = Int8Col( dflt=-1)
    center          = Int64Col(shape=(2), dflt=-1)
    radius          = Int64Col(dflt=-1) 
    cornerness      = Float64Col(dflt=0)
    predict         = Float64Col(shape=(cf._maxc, 2), dflt=-1)
    neighbours      = Float64Col(shape=(cf._nn_count, 2), dflt=-1)


class DColorSIFTRecord(IsDescription):
    """Table type for storing local desriptors.

    Attributes:
        index: Same as index of the corresponding sample (region). Allows
            read descriptors in the same manner as local samples, or find
            information about descriptor in Regions table.
        classN: Assumed true class of corresponding region. Stored here for
            faster access.
        data: The descriptor itself. Current descriptor type is some color
            version of SIFT, with the dimensionality 3*128 = 384.

    Description table is put together with the corresponding Websites, Images
    and Regions table. Database file name should point to exact methods used,
    for instance 'csift_ds.h5' for C-SIFT descriptors of dense sampled
    features.

    Compression x4 achieved using UInt8Col instead of IntCol. Further
    compression possible, but even the fastest one slows down I/O 10 times,
    because reads from here require index searching to get descriptors of
    a specific image, and searching a compressed data is bad idea.
    """
    index           = Int64Col(dflt=-1, pos=0)
    classN          = Int8Col(pos=1)
    data            = UInt8Col(shape=(384))
































        
