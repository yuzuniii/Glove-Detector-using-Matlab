clc;
clear;
close all;

img = imread('blueglove.jpg');

[mask, segmentedImg] = segmentGlove(img);

figure;

subplot(1,3,1)
imshow(img)
title('Original Image')

subplot(1,3,2)
imshow(mask)
title('Binary Mask')

subplot(1,3,3)
imshow(segmentedImg)
title('Segmented Glove')