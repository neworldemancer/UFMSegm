1. Initial Setup for processing
_ Intro _

There are 2 options for the set up: A. when the CNN inference is performed on a (linux) multi-GPU CUDA-enabled server, and the rest of the processing - locally on a windows machine with one CUDA-enabled GPU; B. when all processing is on the windows machine with CUDA-enabled GPU. The CNN inference and the processing pipleline is managed from the Jupyter notebook. The frame alignment, segmntation based on the predicted cell masks etc - is run on the windows machine using the bat-scripts executing VivoFollow/LTS based utilities (only binaries supplied here - feel free to contact developers). Intercation between notebook and the processing on the windows machine is via the `run_remote_bat.bat` script, waiting for apperence of new `remote.bat` jobs in same folder.

Remote operation via LAN shared folder is provided for demonstration purpose. It should be in general avoided for performance reasons, and if used, should be performed ONLY on closed private networks for security reason.

Here we describe set up for both approaches.
In the 1-computer (B) case, the the directory structure should be the following:
UFMTrack\
 - datasets_seg\
 - trained models\
 - UFMSegm\
     - dataset inference
	 - model evaluation
	 - model training
	 
`datasets_seg` - is the directory where the datasets are placed, numbered as 000, 001,...

The priocessing notebooks, including the template notebook `Inference.ipynb` is in the `dataset inference` directory.

`model evaluation` - contains evaluation of trained model for reference.

`model training` contains scripts and notebooks used for data preparation, model training and inference.

In the case of 2-comuter setup (A) this whole data structure should be put on the multi-GPU server. The directory with datasets (`datasets_seg`) is then shared with smb server and mounted on the Windows computer as network drive (here for example drive `Q:\` is used).

_ 0. Common setup for A&B _
Copy VivoFollow to C:\VivoFollow on the Windows machine. Requires cudart64_XX.dll in bin folder, a CUDA-enabled GPU, and vcredist 2013. Win64 binaries only provided.

Inference machine should have installed py3.6+, jupyter, TensorFlow 1.13, Pillow, matplotlib, etc.

In the template inference notebook, section "Run device" set list of GPU ids to be used. for 8 GPU: `dev_ids = [0, 1, 2, 3, 4, 5, 6, 7]`

_ A. Processing on 2 machines_ 
Set the directory where the datasets are located:
 - in the template inference notebook (section "Datasets to be processed") 
 - in the C:\VivoFollow\bin\*.bat and *.cfg files to actual (currently set to Q:/)

Copy VivoFollow\bin\run_remote_bat.bat to the dataset directory (Q:/)

_ B. Processing on one machine _
Since all files are on same computer, the dataset directory would be located in, e.g., `d:\UFMTrack\datasets_seg`
correspondingly, 
	1. the template notebook section "Datasets to be processed" has to be modified:
		datasets_path_proc = 'Q:\\'
		->
		datasets_path_proc = 'D:\\UFMTrack\\datasets_seg\\'
	2. the processing bat scripts `proc_iv.bat` and `proc_ds_flr_n.bat`:
		set datadir=Q:\%1\
		->
		set datadir=D:\UFMTrack\datasets_seg\%1\
		
2. Dataset processing
_ 1. prepare data _
Datasets to be processed should be put in the `datasets_seg` folder, numbered as 000, 001, ... (3 digits, padded with zeroes). In case of multi-tile datasets, one folder per tile should be created.
Datasets can be generated from the czi files with `Split_tiled_datasets_2.ipynb` notebook (see github for updates, pending atm due to server issues).
The output structure per dataset:
000\
  - hr_ds_name_tile1\
      - hr_ds_name_tile1_t001.tif
	  - ...
	  - hr_ds_name_tile1_t192.tif
  - block_info.txt
  - info.txt
  
`info.txt` - containes the huma-readable dataset name. The raw 8-bit tiff data folder and the files inside are named accordingly to it. if it is part of tiles dataset - the suffix `_tileX` should be present. Tile idx starts from 1.

`block_info.txt` - contains the clocks of different actiosition regimes, formated as:
`0 31|31 192`
Each of the blocks will be histogram-normalized independently. Theese correspond to low accumulation flow rate and the physiological flow rate. Change of flow causes change in brightness.

_ 2. prepare notebook _
Start run_remote_bat.bat from Q:/ to wait for jobs (alignment and segmentation)

Make a copy of the template notebook and name it according to your experiment, e.g. Inference__Exp_YYYY.MM.DD__User_A__Cond_BCD.ipynb

Set the range of dataset ids to be processed. They should be last in the datasets folder, since meged datasets from tiles will have sequential id after the last tile ds is - prcessed together. In the notebook section "Datasets to be processed" set `datasets_ids` variable to list of individual datasets to be processed. tiles will be autiomatically detected and merged. E.g:
`datasets_ids = list(range(0, 8)) # for datasets 0, 1, 2, 3, 4, 5, 6, 7`

Run all notebook cells. If the set up was carried out correctly - all should work.