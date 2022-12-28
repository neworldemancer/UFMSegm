// save all open documents;  
// 2011, use it at your own risk;  
#target photoshop;


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
		var d_path = f+"/";
		var in_dir = d_path + "psd/";
		var save_dir = d_path + "outputs/";

		var save_folder = Folder(save_dir);
		//Check if it exist, if not create it.
		if(!save_folder.exists) save_folder.create();
		
		labels_prfx="";
		if(ch1.value || ch2.value){
			if(ch1.value){
				if(ch2.value){
					labels_prfx = "l"
				}else{
					labels_prfx = "l1"
				}
			}else{
				labels_prfx = "l2"
			}
		}
				
		for(i=0; i<nmax; ++i){
			idx_str = ("000" + i).substr(-3,3)
			
			var f_labeled_psd = new File(in_dir + "r"+labels_prfx+"_"+idx_str+".psd");
			if(!f_labeled_psd.exists)
				break;
			
			app.open( f_labeled_psd );
			var psd_doc = app.activeDocument;
			var num_layers = psd_doc.layers.length;
			var label_layer = psd_doc.layers[0]; //seems 0 is top... ????
			psd_doc.activeLayer = label_layer;
			label_layer.opacity = 100
			
			var desc = new ActionDescriptor();  
			var ref = new ActionReference();  
			ref.putClass( charIDToTypeID( "Dcmn" ) );  
			desc.putReference( charIDToTypeID( "null" ), ref );  
			desc.putString( charIDToTypeID( "Nm  " ), "Untitled" );  
			var ref1 = new ActionReference();  
			ref1.putEnumerated( charIDToTypeID( "Lyr " ), charIDToTypeID( "Ordn" ), charIDToTypeID( "Trgt" ) );  
			desc.putReference( charIDToTypeID( "Usng" ), ref1 );  
			desc.putString( charIDToTypeID( "LyrN" ), "label" );  
			executeAction( charIDToTypeID( "Mk  " ), desc, DialogModes.NO );
			
			var lbl_doc = app.activeDocument;

			psd_doc.close(SaveOptions.DONOTSAVECHANGES);
			
			var opts, file;
			opts = new ExportOptionsSaveForWeb();
			opts.format = SaveDocumentType.PNG;
			opts.PNG8 = false;
			opts.quality = 100;
			
			var save_path = save_dir + "lbl_"+idx_str+".png";
			pngFile = new File(save_path);
			
			lbl_doc.exportDocument(pngFile, ExportType.SAVEFORWEB, opts);
			lbl_doc.close(SaveOptions.DONOTSAVECHANGES);
		}
	}
}