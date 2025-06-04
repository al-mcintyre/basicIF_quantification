Example command to run quantification of mean intensity per segmented region (e.g. nucleus) and expanded region.
Results are saved in "csvs" folder with columns for the centroid locations of each region and quantification of mean intensity for one or more channels. 

./quantify_intensity.py -s ${IMG}0_cp_masks.png -c ${IMG}0.png ${IMG}1.png --exp 5 --subBG

For full array of options, run
./quantify_intensity.py -h
