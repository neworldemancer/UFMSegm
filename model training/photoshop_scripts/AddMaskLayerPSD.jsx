// save all open documents;  
// 2011, use it at your own risk;  
#target photoshop;

psdOpts = new PhotoshopSaveOptions();  
psdOpts.embedColorProfile = true;  
psdOpts.alphaChannels = false;
psdOpts.layers = true;  
psdOpts.spotColors = true;

folder = new Folder("d:/data/BBB_2/outputs/1/");  //("q:/deep/BBB_Home/jpnb/data/data_prep/0/");  
f = folder.selectDlg("wot");

nmax = 500;

res = f!=null;
started = false
if(res){
	var d_path = f+"/"; //"q:/deep/BBB_Home/jpnb/data/data_prep/0/";
	var save_dir = d_path + "psd_labels/all_diap/";
	var in_dir = d_path + "psd_labels/all/";

	var save_folder = Folder(save_dir);
	//Check if it exist, if not create it.
	if(!save_folder.exists) save_folder.create();
	

	for(i=0; i<nmax; ++i){
		idx_str = ("000" + i).substr(-3,3)
		
		var f_raw = new File(in_dir + "rl_"+idx_str+".psd");
		
		if(!f_raw.exists){
			if(started)
				break;
			else
				continue;
		}
		
		started = true;

		app.open( f_raw );
		var raw_doc = app.activeDocument;

		for (var l = 0, il = raw_doc.layers.length; l < il; l++) { // preprocess old style docs
			var curLayer = raw_doc.layers[l];
			//alert (curLayer.name, i);
			
			if (curLayer.name == "Layer 0")
				curLayer.name = "data"
			if (curLayer.name == "Layer 1")
				curLayer.name = "label"
		}
		raw_doc.activeLayer = raw_doc.layers.getByName("data");
		raw_doc.activeLayer.visible = true;
		raw_doc.activeLayer.allLocked = true;
		
		raw_doc.activeLayer = raw_doc.layers.getByName("label");
		raw_doc.activeLayer.visible = false;
		raw_doc.activeLayer.allLocked = true;
		
		raw_doc.artLayers.add();
		raw_doc.activeLayer.name = "diap"

		var save_path = save_dir + "rl_"+idx_str+".psd";
		raw_doc.saveAs((new File(save_path)), psdOpts, false, Extension.LOWERCASE);
		raw_doc.close(SaveOptions.DONOTSAVECHANGES);
	}
}