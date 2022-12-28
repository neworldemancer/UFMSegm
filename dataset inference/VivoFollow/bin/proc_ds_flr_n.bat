@echo off

set workdir=%CD%\

set datadir=Q:\%1\

set stack_tmpl=%datadir%\imgs_aligned_all\raw\%%03d.png
set flr1_tmpl=%datadir%\imgs_aligned_all\flr1\%%03d.png
set flr2_tmpl=%datadir%\imgs_aligned_all\flr2\%%03d.png

set cell_mask_tmpl=%datadir%\pred_cdc\cell\img_%%03d.png
set diap_mask_tmpl=%datadir%\pred_cdc\diap\img_%%03d.png
set cntr_mask_tmpl=%datadir%\pred_cdc\cntC\img_%%03d.png


set tgt_root_dir=%datadir%\segmentation\
mkdir %tgt_root_dir%

set tgt_dir_cells=%tgt_root_dir%\cells\
set tgt_dir_centr=%tgt_root_dir%\centr\


set prog=C:\VivoFollow\bin\CellSegmenter_64.exe

set cfg=%workdir%CellSegmenter_proc_CDC.cfg
set params=-cfg:%cfg% -start_ofset:0 -num_imgs:%2 -end_ofset:0 

if [%3] equ [1] (
	set aux_par_flr=-aux_0:%flr1_tmpl% -aux_name_0:flr1
) else (
	if [%3] equ [2] (
		set aux_par_flr=-aux_0:%flr1_tmpl% -aux_name_0:flr1 -aux_1:%flr2_tmpl% -aux_name_1:flr2
	) else (
		set aux_par_flr=
	)
)


%prog% %params% -out_dir:%tgt_dir_cells% -stack:%stack_tmpl% -mask:%cell_mask_tmpl% -mask_hi:%cntr_mask_tmpl% -diap:%diap_mask_tmpl% %aux_par_flr%
