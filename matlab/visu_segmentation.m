clear variables
close all
clc

%% Define path
image_dir = '/local/durandt/datasets/sol/images';
if ~exist(image_dir, 'dir')
    fprintf('image_dir does not exist (%s)\n', image_dir);
end
annotation_dir = '/local/durandt/datasets/sol/masks';
if ~exist(annotation_dir, 'dir')
    fprintf('annotation_dir does not exist (%s)\n', annotation_dir);
end
prediction_dir = '/local/durandt/results/pytorch/segmentation/masks/sol/supervised/preprocessing_random_crop/max_small_size=2000/crop_size=512/resnet_50_hc5_psp/upsampling=bilinear/pyramid_pooling=avg/pyramid_sizes=[1, 2, 3, 6]/dim=256/groups=1/fusion=sum/lr=0.01/lrp=0.1/dropout2d=False/dropout=0.1/val';
prediction_dir = '/local/durandt/results/pytorch/segmentation/masks/sol/supervised/preprocessing_random_crop/max_small_size=2000/crop_size=512/resnet_50_hc4_psp/upsampling=bilinear/pyramid_pooling=avg/pyramid_sizes=[1, 2, 3, 6]/dim=256/groups=1/fusion=sum/lr=0.01/lrp=0.1/dropout2d=False/dropout=0.1/val';
if ~exist(prediction_dir, 'dir')
    fprintf('prediction_dir does not exist (%s)\n', prediction_dir);
end
file_images = '/local/durandt/datasets/sol/files/val.csv';


%% read list of images
fprintf('read file %s ...', file_images);
image_filename = textread(file_images,'%s');
image_filename = image_filename(2:end); % remove header
fprintf(' %d images\n', length(image_filename))

%% define colormap
cmap = [0, 255, 0;
        218, 165, 32;
        139, 69, 19;
        255, 255, 255] / 255;

%%
for i = 1:length(image_filename)
    filename = image_filename{i};
    image_file = sprintf('%s/%s', image_dir, filename);
    im = imread(image_file);
    subplot(1,3,1), imagesc(im), axis image
    
    prediction_file = sprintf('%s/%s.png', prediction_dir, filename);
    prediction = imread(prediction_file);
    subplot(1,3,3), imagesc(prediction), axis image
    colormap(cmap)
    
    filename = filename(1:end-3);
    annotation_file = sprintf('%s/%spng', annotation_dir, filename);
    anno = imread(annotation_file);
    subplot(1,3,2), imagesc(anno), axis image
    
    pause
end