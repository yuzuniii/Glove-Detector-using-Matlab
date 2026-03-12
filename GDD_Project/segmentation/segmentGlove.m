function [mask, segmentedImg] = segmentGlove(img)

% Convert to HSV
hsvImg = rgb2hsv(img);

S = hsvImg(:,:,2);

% Threshold saturation to detect glove
BW = S > 0.2;

% Refine mask
mask = refineMask(BW);

% Generate segmented image testing
segmentedImg = img;

for c = 1:3
    channel = segmentedImg(:,:,c);
    channel(~mask) = 0;
    segmentedImg(:,:,c) = channel;
end

end