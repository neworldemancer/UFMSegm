Copy VivoFollow to C:\VivoFollow
Set the directory where the datasets are located in the inference notebook and *.bat files / *.cfg to actual (currently set to Q:/)
Copy run_remote_bat.bat to the dataset directory (Q:/)

Requires cudart64_XX.dll in bin folder, a CUDA-enabled GPU, and vcredist 2013
Win64 binaries only provided.

Remote operation via LAN shared folder should be performed only on closed private networks for security reason.
SolverServer_64.exe has to be running
Start run_remote_bat.bat from Q:/ to wait for jobs (alignment, segmentation, tracking)