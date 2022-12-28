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
		var docName = theDoc.name;  
		// thanks to xbytor for the regexp;  
		if (docName.indexOf(".") != -1) {var basename = docName.match(/(.*)\.[^\.]+$/)[1]}  
		else {var basename = docName};  
		// getting the location, if unsaved save to desktop;  
		try {  
			var docPath = theDoc.path;  
			// save the file if unsaved;  
			if (theDoc.saved == false) {theDoc.save()}  
		}  
		// if file has not been saved yet save to desktop;  
		catch (e) {  
			//var docPath = "~/Desktop";  
			//theDoc.saveAs((new File(docPath+'/'+basename+".psd")), psdOpts, false, Extension.LOWERCASE);  
		};  
	};  
	app.activeDocument = theFirst;  
};  