/*
 * This script is a FIJI macro that converts all of the .lut files in the FIJI application
 * to .txt format. This is useful because the .lut file encoding is confusing and inconsistent
 * so I let FIJI handle it.
 * Also loads the built-in colormaps [Fire, Grays, Ice, Spectrum, 3-3-2 RGB, Red, Green, Blue, Cyan, Magenta, Yellow, Red/Green]
 */

// Define output directory - modify this path as needed
outputDir = "";

// Process .lut files
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

// Process built-in lookup tables
builtInLuts = newArray("Fire", "Grays", "Ice", "Spectrum", "3-3-2 RGB",
                       "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Red/Green");

for (i = 0; i < builtInLuts.length; i++) {
    lutName = builtInLuts[i];
    newImage("Temp", "8-bit Ramp", 256, 1, 1);
    run(lutName);
    Color.getLut(reds, greens, blues);
    rgbValues = "";
    for (j = 0; j < 256; j++) {
        rgbValues = rgbValues + reds[j] + "\t" + greens[j] + "\t" + blues[j] + "\n";
    }
    // Replace slash with underscore for filename
    safeFileName = replace(lutName, "/", "%");
    File.saveString(rgbValues, outputDir + safeFileName + ".txt");
    close();
}

showMessage("LUT Export Complete", "RGB values have been exported for all LUTs");
