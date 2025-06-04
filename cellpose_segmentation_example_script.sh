DIR=$PATH2IMGS
IMG=$DIR/2025_04_29_Exp.2_control_293T_GFp_CAGESsi_siRNA_1_t0_c0.png
DIAM=130

python -m cellpose --image_path $IMG --pretrained_model nuclei --diameter $DIAM --save_png --verbose --no_npy
