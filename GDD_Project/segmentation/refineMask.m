function mask = refineMask(BW)

% Remove small blobs
BW = bwareaopen(BW,2000);

% Fill holes
BW = imfill(BW,'holes');

% Morphological smoothing
se = strel('disk',5);
BW = imopen(BW,se);
BW = imclose(BW,se);

% Find connected components
CC = bwconncomp(BW);

% Safety check if nothing detected
if CC.NumObjects == 0
    warning('Segmentation failed: No object detected');
    mask = BW;
    return;
end

% Keep largest component
numPixels = cellfun(@numel,CC.PixelIdxList);
[~,idx] = max(numPixels);

mask = false(size(BW));
mask(CC.PixelIdxList{idx}) = true;

end