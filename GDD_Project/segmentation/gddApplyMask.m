function segmentedImg = gddApplyMask(img, mask)
% applyMask  Zero out background pixels outside mask.
segmentedImg = img;
for c = 1:3
    ch = segmentedImg(:,:,c);
    ch(~mask) = 0;
    segmentedImg(:,:,c) = ch;
end
end