function [mask, segmentedImg] = segmentGlove(img)

% --------------------------------------
% STEP 1: Convert to Grayscale
% --------------------------------------
gray = rgb2gray(img);

% --------------------------------------
% STEP 2: Noise Removal
% Median filtering from lab exercises
% --------------------------------------
clean = medfilt2(gray,[3 3]);

% --------------------------------------
% STEP 3: Adaptive Threshold Segmentation
% --------------------------------------
BW = imbinarize(clean,'adaptive','Sensitivity',0.5);

% --------------------------------------
% STEP 4: Refine Mask
% --------------------------------------
mask = refineMask(BW);

% --------------------------------------
% STEP 5: Generate Segmented Glove Image
% --------------------------------------
segmentedImg = img;

for c = 1:3
    channel = segmentedImg(:,:,c);
    channel(~mask) = 0;
    segmentedImg(:,:,c) = channel;
end

end