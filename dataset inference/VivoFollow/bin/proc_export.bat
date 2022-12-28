@echo off


set IVFCA_ID=%1
set tmp_path=C:\VivoFollow\bin\batchoutput

pushd C:\VivoFollow\bin
C:\VivoFollow\bin\CellConvertor_64.exe -cfg:C:\VivoFollow\bin\CellConvertor_batch.cfg

copy %tmp_path%\tracks.dat q:\deep\BBB_Home\jpnb\BBB_data_proc\%IVFCA_ID%\segmentation\cells\

IF exist %tmp_path%\tr_cells_tmp.dat (
  copy %tmp_path%\tr_cells_tmp.dat q:\deep\BBB_Home\jpnb\BBB_data_proc\%IVFCA_ID%\segmentation\cells\
)

popd