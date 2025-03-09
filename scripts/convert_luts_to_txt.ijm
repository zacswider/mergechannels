/*
 * This script if a FIJI macro that converts all of the .lut files in the FIJI application
 * to .txt format. This is useful because the .lut file encoding is confusing and inconsistent
 * so I let FIJI handle it.
 */

// Define output directory - modify this path as needed
outputDir = "";

lutsDir = getDirectory("luts");

lutList = getFileList(lutsDir);
for (i = 0; i < lutList.length; i++) {
    if (endsWith(lutList[i], ".lut")) {
        lutName = substring(lutList[i], 0, lastIndexOf(lutList[i], "."));
        newImage("Temp", "8-bit Ramp", 256, 1, 1);
        run("LUT... ", "open=[" + lutsDir + lutList[i] + "]");
        Color.getLut(reds, greens, blues);
        rgbValues = "";
        for (j = 0; j < 256; j++) {
        	rgbValues = rgbValues + reds[j] + "\t" + greens[j] + "\t" + blues[j] + "\n";
        }
        File.saveString(rgbValues, outputDir + lutName + ".txt");
        close();
    }
}

showMessage("LUT Export Complete", "RGB values have been exported for all LUTs");