// save all open documents;  
// 2011, use it at your own risk;  
#target photoshop;  
if (app.documents.length > 0) {  
	var theFirst = app.activeDocument;  
	var theDocs = app.documents;  
	// psd options for unsaved files;  
	psdOpts = new PhotoshopSaveOptions();  
	psdOpts.embedColorProfile = true;  
	psdOpts.alphaChannels = false;  
	psdOpts.layers = true;  
	psdOpts.spotColors = true;  
	// go through all files;  
	for (var m = 0; m < theDocs.length; m++) {  
		var theDoc = theDocs[m];  
		app.activeDocument = theDoc;  
		// getting the name and location;  
		var raw_doc = app.activeDocument;
		
		var l = raw_doc.layers.getByName("label");
		var locked = l.allLocked
		
		if (locked)
			l.allLocked = false;
		
		l.opacity = 150 - l.opacity;
		l.allLocked = locked;
	};  
	app.activeDocument = theFirst;
};  