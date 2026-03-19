function [mask, segmentedImg] = segmentBlueGlove(img)
% segmentBlueGlove  Segment blue nitrile glove using HSV hue gating.
%
% Blue nitrile has a distinctive hue (0.44-0.70) that is absent in
% most backgrounds. Two-pass approach captures the pale inner surface
% of inside-out gloves which has low saturation.

img    = im2uint8(img);
hsvImg = rgb2hsv(img);
H = hsvImg(:,:,1);
S = hsvImg(:,:,2);
V = hsvImg(:,:,3);

% Pass 1: capture blue pixels
blueMask    = (H >= 0.44) & (H <= 0.70) & (S >= 0.15);

% Pass 2: dilate to capture adjacent pale regions (inside-out inner surface)
blueDilated = imdilate(blueMask, strel('disk', 12));
brightMask  = V > 0.20;
rawMask     = blueMask | (blueDilated & brightMask);
rawMask     = imfill(rawMask, 'holes');

[mask, ~]    = refineMask(rawMask);
segmentedImg = gddApplyMask(img, mask);

fprintf('  segmentBlueGlove: %d px\n', nnz(mask));
end