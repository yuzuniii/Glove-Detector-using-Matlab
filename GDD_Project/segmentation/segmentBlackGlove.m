function [mask, segmentedImg] = segmentBlackGlove(img)
% segmentBlackGlove  Segment black nitrile glove on bright background.
%
% Black nitrile is significantly darker than the grey/white background.
% Uses Otsu thresholding to find the dark glove region.

img    = im2uint8(img);
hsvImg = rgb2hsv(img);
V      = hsvImg(:,:,3);

% Otsu on value channel
otsuT   = graythresh(V);

% Dark pixels = glove (threshold slightly above Otsu to capture all glove)
rawMask = V < (otsuT * 1.15);
rawMask = imfill(rawMask, 'holes');
rawMask = bwareaopen(rawMask, round(0.03 * numel(V)));

% If result too small, lower threshold
if nnz(rawMask) < 0.05 * numel(V)
    rawMask = V < (otsuT * 1.3);
    rawMask = imfill(rawMask, 'holes');
    rawMask = bwareaopen(rawMask, round(0.03 * numel(V)));
end

[mask, ~]    = refineMask(rawMask);
segmentedImg = gddApplyMask(img, mask);

fprintf('  segmentBlackGlove: %d px\n', nnz(mask));
end