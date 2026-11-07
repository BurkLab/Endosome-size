/*
Batch neurite ROI preprocessing for endosome analysis
Opens ND2 files via Bio-Formats. Also measures intensity of Peripherin channel. Can be ignored if only endosome size is important. 

Assumptions:
- XY images only, already MIP or single zSlice if originally Z-stack.
- Channel 3 = Peripherin/Cytoskeletal marker.
- Channel 4 = Endosome marker.
- This can be changed at the beginning of the running code.
- The cytoskeletal marker will be used to define ROIs and masks for the endosome analysis later, so make sure to include a good marker in your stainings.
- User draws multiple line ROIs per image (tracing neurites) and adds them to ROI Manager manually (press T).
- Line width is set visually by user before adding ROI. Width is NOT stored separately.

Outputs per image:
- ROI zip backup saved before processing
- per neurite: raw PRPH crop, raw Endosome crop, PRPH-derived neurite mask, QC overlay
- master neurite_metadata.csv in output folder

Important:
- The PRPH intensity measurement is made inside the threshold-derived mask AND inside the line-derived ROI shape, NOT the whole rectangular crop.

*/

requires("1.54p");

macro "Batch neurite line ROIs: PRPH mask + Endosome crops [F1]" {
    setBatchMode(false);

    inputDir = getDirectory("Choose input folder containing images");
    if (inputDir == "") exit("No input folder selected.");

    outputDir = getDirectory("Choose output folder");
    if (outputDir == "") exit("No output folder selected.");

    Dialog.create("Settings");
    Dialog.addNumber("PRPH channel", 3);
    Dialog.addNumber("Endosome channel", 4);
    Dialog.addString("File extensions, comma-separated", ".nd2,.tif,.tiff");
    Dialog.addCheckbox("Clear ROI Manager before each image", true);
    Dialog.show();

    prphChannel = Dialog.getNumber();
    endoChannel = Dialog.getNumber();
    extensions = Dialog.getString();
    clearRoiManagerEachImage = Dialog.getCheckbox();

    metadataPath = outputDir + "neurite_metadata.csv";
    logPath = outputDir + "processing_log.txt";
    if (!File.exists(metadataPath)) {
        File.append("original_image,neurite_id,prph_mean,prph_integrated,prph_area_px,prph_area_um2,prph_crop_file,endosome_crop_file,mask_file,qc_overlay_file,roi_zip\n", metadataPath);
    }
    File.append("\n=== New/resume run started ===\n", logPath);

    list = getFileList(inputDir);
    nProcessedImages = 0;

    for (i = 0; i < list.length; i++) {
        filename = list[i];
        if (File.isDirectory(inputDir + filename)) continue;
        if (!hasAllowedExtension(filename, extensions)) continue;

        fullPath = inputDir + filename;
        base = stripExtension(filename);
        safeBase = sanitize(base);
        imageOutDir = outputDir + safeBase + File.separator;
        File.makeDirectory(imageOutDir);

        openImageNoPrompt(fullPath);
        origID = getImageID();
        origTitle = getTitle();

        // Improve display for drawing ROIs: show PRPH channel and auto-adjust brightness.
        Stack.setChannel(prphChannel);
        run("Grays");
        resetMinAndMax();
        run("Enhance Contrast", "saturated=0.35");

        if (clearRoiManagerEachImage) {
            roiManager("Reset");
        }
        roiManager("Show All");

        roiZipPath = imageOutDir + safeBase + "_line_rois.zip";

        if (File.exists(roiZipPath)) {
            // Resume mode: load saved ROI ZIP instead of asking you to redraw.
            roiManager("Open", roiZipPath);
            waitForUser("Resume mode for:\n\n" + filename + "\n\nLoaded existing ROI ZIP.\nCheck ROI Manager, then click OK to continue.\nAlready completed neurites will be skipped.");
            File.append("Loaded existing ROI ZIP: " + roiZipPath + "\n", logPath);
        } else {
            waitForUser("Draw neurites for:\n\n" + filename + "\n\nUse segmented/freehand LINE ROIs.\nSet the line width visually if desired.\nAdd each ROI to ROI Manager by pressing T.\n\nWhen all neurites for this image are in ROI Manager, click OK.");

            nRois = roiManager("count");
            if (nRois == 0) {
                File.append("No ROIs, skipped image: " + filename + "\n", logPath);
                closeImageByID(origID);
                continue;
            }

            roiMetaPath = imageOutDir + safeBase + "_roi_metadata.csv";
            roiMeta = "original_image,neurite_id,roi_name\n";
            for (r = 0; r < nRois; r++) {
                neuriteID = r + 1;
                roiName = "neurite_" + pad(neuriteID, 3);
                roiManager("Select", r);
                roiManager("Rename", roiName);
                roiMeta += filename + "," + neuriteID + "," + roiName + "\n";
            }
            // Save ROI backup immediately after all ROIs are defined and named, before any processing.
            roiManager("Save", roiZipPath);
            File.saveString(roiMeta, roiMetaPath);
            File.append("Saved new ROI ZIP: " + roiZipPath + "\n", logPath);
        }

        nRois = roiManager("count");
        if (nRois == 0) {
            File.append("ROI ZIP/manager empty, skipped image: " + filename + "\n", logPath);
            closeImageByID(origID);
            continue;
        }

        // Get calibration from original image.
        selectImage(origID);
        getPixelSize(unit, pixelWidth, pixelHeight, voxelDepth);
        if (pixelWidth <= 0) pixelWidth = 1;
        if (pixelHeight <= 0) pixelHeight = 1;
        pixelArea = pixelWidth * pixelHeight;

        for (r = 0; r < nRois; r++) {
            neuriteID = r + 1;
            neuriteLabel = "neurite" + pad(neuriteID, 3);

            // Expected output paths. Resume mode skips neurites where all outputs already exist.
            prphCropPath = imageOutDir + safeBase + "_" + neuriteLabel + "_PRPH.tif";
            endoCropPath = imageOutDir + safeBase + "_" + neuriteLabel + "_Endosomes.tif";
            maskPath = imageOutDir + safeBase + "_" + neuriteLabel + "_mask.tif";
            qcPath = imageOutDir + safeBase + "_" + neuriteLabel + "_QC_overlay.tif";

            if (File.exists(prphCropPath) && File.exists(endoCropPath) && File.exists(maskPath) && File.exists(qcPath)) {
                File.append("SKIP complete: " + filename + " / " + neuriteLabel + "\n", logPath);
                continue;
            }

            File.append("PROCESS: " + filename + " / " + neuriteLabel + "\n", logPath);

            selectImage(origID);
            roiManager("Select", r);

            // Convert current line ROI to area ROI if needed.
            // If it is already an area ROI, this may fail silently or keep selection unchanged.
            run("Line to Area");

            // Save this area ROI temporarily in ROI Manager at the end, then delete it after crops.
            roiManager("Add");
            areaIndex = roiManager("count") - 1;

            // Create an ROI-shape mask crop from the line-derived area ROI.
            // This is critical: the image crops are rectangular bounding boxes, but
            // PRPH thresholding and later Python analysis must be restricted to the
            // actual line/area ROI inside that rectangle.
            selectImage(origID);
            roiManager("Select", areaIndex);
            run("Create Mask");
            roiMaskFullID = getImageID();
            rename("ROI_MASK_FULL_TMP");
            roiManager("Select", areaIndex);
            run("Duplicate...", "title=ROI_MASK_TMP duplicate");
            roiMaskID = getImageID();
            rename("ROI_MASK_TMP");
            closeImageByID(roiMaskFullID);

            // Duplicate PRPH channel crop from area ROI.
            selectImage(origID);
            Stack.setChannel(prphChannel);
            roiManager("Select", areaIndex);
            run("Duplicate...", "title=PRPH_TMP duplicate channels=" + prphChannel);
            prphID = getImageID();
            rename("PRPH_TMP");

            // Duplicate endosome channel crop from same area ROI.
            selectImage(origID);
            Stack.setChannel(endoChannel);
            roiManager("Select", areaIndex);
            run("Duplicate...", "title=ENDO_TMP duplicate channels=" + endoChannel);
            endoID = getImageID();
            rename("ENDO_TMP");

            // Remove temporary area ROI from manager, leaving original line ROIs intact.
            roiManager("Select", areaIndex);
            roiManager("Delete");

            // Save raw crops.
            selectImage(prphID);
            saveAs("Tiff", prphCropPath);
            selectImage(endoID);
            saveAs("Tiff", endoCropPath);

            // Create PRPH mask interactively from duplicate of PRPH crop.
            selectImage(prphID);
            run("Duplicate...", "title=MASK_TMP");
            maskID = getImageID();
            rename("MASK_TMP");
            selectImage(maskID);
            run("Enhance Contrast", "saturated=0.35");
            setAutoThreshold("Default dark");
            run("Threshold...");
            waitForUser("Adjust PRPH threshold for " + filename + " / " + neuriteLabel + ".\n\nWhite mask should correspond to the neurite/PRPH-positive region only.\nClick Apply in the Threshold window if needed, then click OK here.");
            // Ensure binary threshold mask.
            selectImage(maskID);
            run("Convert to Mask");
            run("Options...", "iterations=1 count=1 black do=Nothing");

            // Restrict the threshold mask to the actual line/area ROI shape.
            // Without this AND step, threshold-positive PRPH structures elsewhere
            // inside the rectangular crop would be included incorrectly.
            selectImage(roiMaskID);
            run("Convert to Mask");
            imageCalculator("AND create", maskID, roiMaskID);
            restrictedMaskID = getImageID();
            rename("MASK_RESTRICTED_TMP");
            closeImageByID(maskID);
            maskID = restrictedMaskID;

            selectImage(maskID);
            saveAs("Tiff", maskPath);

            // Measure PRPH intensity inside threshold-derived mask.
            selectImage(maskID);
            run("Create Selection");
            if (selectionType() == -1) {
                // Empty mask: record NaNs and continue, but still save crops/mask.
                prphMean = NaN;
                prphIntDen = NaN;
                prphAreaPx = 0;
            } else {
                roiManager("Reset");
                roiManager("Add");
                selectImage(prphID);
                roiManager("Select", 0);
                run("Set Measurements...", "area mean integrated redirect=None decimal=6");
                run("Clear Results");
                run("Measure");
                prphAreaPx = getResult("Area", 0);
                prphMean = getResult("Mean", 0);
                prphIntDen = getResult("IntDen", 0);
                roiManager("Reset");
            }
            prphAreaUm2 = prphAreaPx * pixelArea;

            // Create QC overlay: PRPH crop with mask outline.
            selectImage(prphID);
            run("Duplicate...", "title=QC_TMP");
            qcID = getImageID();
            rename("QC_TMP");
            selectImage(maskID);
            run("Create Selection");
            if (selectionType() != -1) {
                selectImage(qcID);
                run("RGB Color");
                setForegroundColor(0, 255, 0);
                run("Draw", "slice");
            }
            selectImage(qcID);
            saveAs("Tiff", qcPath);

            // Append metadata row.
            row = filename + "," + neuriteID + "," + prphMean + "," + prphIntDen + "," + prphAreaPx + "," + prphAreaUm2 + "," + prphCropPath + "," + endoCropPath + "," + maskPath + "," + qcPath + "," + roiZipPath + "\n";
            File.append(row, metadataPath);
            File.append("DONE: " + filename + " / " + neuriteLabel + "\n", logPath);

            // Close temporary images.
            closeImageByID(qcID);
            closeImageByID(maskID);
            closeImageByID(roiMaskID);
            closeImageByID(prphID);
            closeImageByID(endoID);

            // Reopen/restore ROI Manager original ROIs after it was reset for measurement.
            // Because measurement reset wiped the ROI Manager, reload from ZIP and continue at next ROI.
            roiManager("Open", roiZipPath);
            nRois = roiManager("count");
        }

        closeImageByID(origID);
        roiManager("Reset");
        nProcessedImages++;
    }

    File.append("=== New/resume run finished ===\n", logPath);
    showMessage("Done", "Processed/resumed " + nProcessedImages + " image(s).\n\nMetadata saved to:\n" + metadataPath + "\n\nLog saved to:\n" + logPath);
}

function openImageNoPrompt(path) {
    // Opens ND2/TIFF via Bio-Formats without showing the Bio-Formats Importer dialog.
    // Uses hyperstack view so Stack.setChannel(3/4) works.
    lower = toLowerCase(path);
    if (endsWith(lower, ".nd2")) {
        run("Bio-Formats Importer", "open=[" + path + "] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
    } else {
        // TIFFs usually open cleanly with standard ImageJ open; this avoids unnecessary Bio-Formats prompts.
        open(path);
    }
}

function closeImageByID(id) {
    if (id == 0) return;
    selectImage(id);
    close();
}

function hasAllowedExtension(filename, extensions) {
    lower = toLowerCase(filename);
    exts = split(extensions, ",");
    for (e = 0; e < exts.length; e++) {
        ext = trim(exts[e]);
        if (ext == "") continue;
        if (endsWith(lower, toLowerCase(ext))) return true;
    }
    return false;
}

function stripExtension(filename) {
    dot = lastIndexOf(filename, ".");
    if (dot < 0) return filename;
    return substring(filename, 0, dot);
}

function sanitize(s) {
    s = replace(s, " ", "_");
    s = replace(s, ":", "_");
    s = replace(s, ";", "_");
    s = replace(s, ",", "_");
    s = replace(s, "(", "_");
    s = replace(s, ")", "_");
    s = replace(s, "[", "_");
    s = replace(s, "]", "_");
    s = replace(s, "{", "_");
    s = replace(s, "}", "_");
    s = replace(s, "-", "_");
    return s;
}

function pad(n, width) {
    s = "" + n;
    while (lengthOf(s) < width) s = "0" + s;
    return s;
}
