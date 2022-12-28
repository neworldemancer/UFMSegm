@echo off
pushd %~dp0 

set workdir=%CD%\

set data_dir=Q:\%1\
set inp=%data_dir%\tiles_info

set out=%data_dir%\tiles_merging

set outp=-save_dir:%out%
set cfg=-cfg:%workdir%\TAligner.cfg

set inp0=-input_list_0:%inp%\raw.tl  -time_start_idx_0:0 -times_num_0:%2 -out_path_0:%data_dir%\imgs_aligned_all\raw\%%03d.png -merge_type_0:crop
set inp1=-input_list_1:%inp%\cell.tl -time_start_idx_1:0 -times_num_1:%2 -out_path_1:%data_dir%\pred_cdc\cell\img_%%03d.png -merge_type_1:max
set inp2=-input_list_2:%inp%\diap.tl -time_start_idx_2:0 -times_num_2:%2 -out_path_2:%data_dir%\pred_cdc\diap\img_%%03d.png -merge_type_2:max
set inp3=-input_list_3:%inp%\cntc.tl -time_start_idx_3:0 -times_num_3:%2 -out_path_3:%data_dir%\pred_cdc\cntC\img_%%03d.png -merge_type_3:max

set inp4=-input_list_4:%inp%\flr1.tl -time_start_idx_4:0 -times_num_4:%2 -out_path_4:%data_dir%\imgs_aligned_all\flr1\%%03d.png -merge_type_4:overwrite
set inp5=-input_list_5:%inp%\flr2.tl -time_start_idx_5:0 -times_num_5:%2 -out_path_5:%data_dir%\imgs_aligned_all\flr2\%%03d.png -merge_type_5:overwrite

if [%3] equ [1] (
	set inp_flr=%inp4%
) else (
	if [%3] equ [2] (
		set inp_flr=%inp4% %inp5%
	) else (
		set inp_flr=
	)
)
set align_in=-tiles_list:%inp%\raw.tl  -tiles_map:%inp%\map.tm -time_start_idx:0 -times_num:%2 -time_stride:10

C:\VivoFollow\bin\TilesAligner_64.exe %cfg% %outp% %align_in% %inp0% %inp1% %inp2% %inp3% %inp_flr%
popd 