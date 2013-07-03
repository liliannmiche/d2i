# Image Based Classifier (IBC)

Two main scripts are: "ibc.py" and "ibc_config.py". The first one runs the toolbox (function names are same as on the diagram). The second one sets the parameters to run.

To start the demo, you need the following inputs:
(a)  A folder with subfolders with images. Each subfolder represents one website; it may contain arbitrary folders and files including non-image ones (which won't be processed).
(b)  A website description file, "_ws_descr" configuration parameter. It has a line for each website, in a format "subfolder_name;unique_url;true_class_id" (put -1 for unknown id's).
(c)  Centroids file ("_C_file") and ELM parameters file ("_elm_param"); they can be found in the attached toy set.

Parameters of the "ibc_config.py" file:
* "_dir" : dataset and outputs directory
* "_ibc" : source code directory
* "_maxc" : max number of classes; set to 12 because you will need new centroids and elm parameters if change these
* "_raw_dir" : input directory from (a)
* "_ws_descr" : file from (b)
* "_mode" : set this to "demo", the "hdf5" mode creates big database file and stores everything there, and not needed for getting the results
* "_hdf5" : path to HDF5 database, irrelevant to "demo" mode
* "_img_data" : path to temporary file with image data
* "_img_dir" : folder where to store preprocessed images; you can delete these after getting classification results
* "_min_size" : minimum image size in bytes; heuristically found to remove auxiliary images like buttons
* "_max_dim" : maximum dimension of image in either x or y; if exceeded, the image will be downscaled
* "_jpeg_quality" : quality of saving preprocessed images
* "_temp_dir" : directory for temporary files; preferably in RAM
* "_descr_extractor" : type of local features and descriptors; currently the only one implemented
* "_max_reg" : limit of local regions per image; safe value as typically it 100-500 per image
* "_cD_bin" : path to binary "colorDescriptor" software; change it for a Windows version; website mentioned in previous letters
* "_C_file" : file containing parameters for centroids and their labels, attached in "toyset.zip"
* "_nn_count" : number of nearest neighbours to extract, practically no difference in run time with 1 nn
* "_nn_batch" : amount of descriptors to process at once

Next 5 parameters irrelevant, as you cannot train ELM without full processed SPIIRAS database (about 170GB)
* "_f_out" : file to save predictions in text format, like "url;predicted_class;[classifier_output_array]"
* "_n_wrk" : number of parallel workers, set to number of cores for faster results
* ...
* "_port" : port to run a server. Use different ports in you want to start several "ibc.py" scripts in parallel.
* ...
* "_show_progress" : shows estimated time remaining but spams a lot, set to "False" for less output


