/*
 * This script if a FIJI macro that converts all of the .lut files in the FIJI application
 * to .txt format. This is useful because the .lut file encoding is confusing and inconsistent
 * so I let FIJI handle it.
 */

// Define output directory - modify this path as needed
outputDir = "...";

// Get the LUTs directory
lutsDir = getDirectory("luts");

// Get list of all files in LUTs directory
lutList = getFileList(lutsDir);

// Process each LUT file
for (i = 0; i < lutList.length; i++) {
    // Check if file is a LUT
    if (endsWith(lutList[i], ".lut")) {
        // Get LUT name without extension
        lutName = substring(lutList[i], 0, lastIndexOf(lutList[i], "."));
        
        // Create new image to apply LUT
        newImage("Temp", "8-bit Ramp", 256, 1, 1);
        
        // Apply the LUT
        run("LUT... ", "open=[" + lutsDir + lutList[i] + "]");
        
        // Get RGB values
        rgbValues = "";
        for (j = 0; j < 256; j++) {
            v = getPixel(j, 0);
            r = (v >> 16) & 0xff;
            g = (v >> 8) & 0xff;
            b = v & 0xff;
            rgbValues = rgbValues + "\t" + r + "\t" + g + "\t" + b + "\n";
        }
        
        // Save RGB values to file
        File.saveString(rgbValues, outputDir + lutName + ".txt");
        
        // Clean up
        close();
    }
}

showMessage("LUT Export Complete", "RGB values have been exported for all LUTs");