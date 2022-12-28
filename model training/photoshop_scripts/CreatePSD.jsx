// save all open documents;  
// 2011, use it at your own risk;  
#target photoshop;

psdOpts = new PhotoshopSaveOptions();  
psdOpts.embedColorProfile = true;  
psdOpts.alphaChannels = false;  
psdOpts.layers = true;  
psdOpts.spotColors = true;

folder = new Folder("q:/deep/BBB_Home/jpnb/data/data_prep/0/");  
f = folder.selectDlg("wot");

nmax = 500;

res = f!=null;
if(res){
	var w = new Window("dialog", "Merge data & outputs to PSD");
	w.add("statictext", undefined, "Fluorescent Labels:");

	var ch1 = w.add("checkbox", undefined, "Ch1");
	var ch2 = w.add("checkbox", undefined, "Ch2");
	ch1.value = true;
	ch2.value = true;
	
	w.add("button", undefined, "OK");
	w.add("button", undefined, "Cancel");
	res = w.show()==1;

	if(res){
		var d_path = f+"/"; //"q:/deep/BBB_Home/jpnb/data/data_prep/0/";
		var save_dir = d_path + "psd/";
		var in_dir = d_path + "raw_and_mask/";

		var save_folder = Folder(save_dir);
		//Check if it exist, if not create it.
		if(!save_folder.exists) save_folder.create();
		
		
		if(ch1.value || ch2.value){
			labels_prfx="";
			if(ch1.value){
				if(ch2.value){
					labels_prfx = "l"
				}else{
					labels_prfx = "l1"
				}
			}else{
				labels_prfx = "l2"
			}
				
			for(i=0; i<nmax; ++i){
				idx_str = ("000" + i).substr(-3,3)
				
				var f_raw = new File(in_dir + "r_"+idx_str+".png");
				var f_lbl = new File(in_dir + labels_prfx+"_"+idx_str+".png");
				if(!f_raw.exists || !f_lbl.exists)
					break;

				app.open( f_raw );
				var raw_doc = app.activeDocument;
				raw_doc.activeLayer.name = "data"
				raw_doc.activeLayer.allLocked = true;
				
				app.open( f_lbl );
				var lbl_doc = app.activeDocument;
				lbl_doc.artLayers[0].duplicate(raw_doc);

				lbl_doc.close(SaveOptions.DONOTSAVECHANGES);
				raw_doc.activeLayer.opacity = 70
				raw_doc.activeLayer.name = "label"

				var save_path = save_dir + "r"+labels_prfx+"_"+idx_str+".psd";
				raw_doc.saveAs((new File(save_path)), psdOpts, false, Extension.LOWERCASE);
				raw_doc.close(SaveOptions.DONOTSAVECHANGES);
			}
		}else{
			for(i=0; i<nmax; ++i){
				idx_str = ("000" + i).substr(-3,3)
				
				var f_raw = new File(in_dir + "r_"+idx_str+".png");
				if(!f_raw.exists)
					break;

				app.open( f_raw );
				var raw_doc = app.activeDocument;
				raw_doc.activeLayer.name = "data"
				raw_doc.activeLayer.allLocked = true;
				raw_doc.artLayers.add();
				raw_doc.activeLayer.name = "label"

				var save_path = save_dir + "r_"+idx_str+".psd";
				raw_doc.saveAs((new File(save_path)), psdOpts, false, Extension.LOWERCASE);
				raw_doc.close(SaveOptions.DONOTSAVECHANGES);
			}
		}
	}
}