# info about pixel-alignment
gdalinfo image.tif | grep -e 'Origin' -e 'Size' -e 'PROJCRS'

# transform .tif to align its grid according to a reference .tif
gdalwarp -r bilinear -tr pixel_size pixel_size -tap -te xmin ymin xmax ymax input_image.tif output_image.tif

# get xmin ymin xmax ymax from reference .tif
gdalinfo image.tif | grep "Upper Left\|Lower Right"

# train pix2pix
python train.py --dataroot ../Data/pix2pix_dataset --name test_multiespectral2 --model pix2pix --batch_size 2 --input_nc 3 --output_nc 3 --netG unet_128 --preprocess none

# test pix2pix
python test.py --dataroot /home/lucas-fontoura/Documents/Pix2Pix/Data/pix2pix_dataset --name test_multiespectral2 --model test --dataset_mode single --input_nc 3 --output_nc 3 --netG unet_128 --preprocess none --norm batch