function [mask, debugInfo] = refineMask(BW)
% refineMask  Clean up a raw binary glove mask.
%
% Keeps the largest connected component PLUS any component that is:
%   - At least 25% the size of the largest, AND
%   - Centroid within 40% of image diagonal from largest component centroid
% This handles folded/inside-out gloves where the wrinkled section
% disconnects from the main body in colour space.

debugInfo = struct();
debugInfo.initialMask = BW;

% Remove specks smaller than 1000px
cleaned = bwareaopen(BW, 1000);
debugInfo.afterAreaOpen = cleaned;

% Fill internal holes
filled = imfill(cleaned, 'holes');
debugInfo.afterFill = filled;

% Smooth boundaries
se      = strel('disk', 7);
opened  = imopen(filled,  se);
closed  = imclose(opened, se);
debugInfo.afterSmooth = closed;

% Connected components
CC = bwconncomp(closed);

if CC.NumObjects == 0
    warning('refineMask: no objects found — returning filled mask.');
    mask = filled;
    debugInfo.finalMask = mask;
    return;
end

% Sort by size
nPx       = cellfun(@numel, CC.PixelIdxList);
[~, sidx] = sort(nPx, 'descend');

% Start with largest
mask = false(size(closed));
mask(CC.PixelIdxList{sidx(1)}) = true;

% Get centroid and size of largest
p1   = regionprops(mask, 'Centroid');
cx1  = p1.Centroid(1);
cy1  = p1.Centroid(2);
big  = nPx(sidx(1));

% Image diagonal for relative distance threshold
diagLen = sqrt(size(BW,1)^2 + size(BW,2)^2);

% Merge nearby large components
for k = 2:CC.NumObjects
    idx  = sidx(k);
    sz   = nPx(idx);

    % Must be at least 25% of largest
    if sz < 0.25 * big
        break
    end

    % Get centroid
    tmp = false(size(closed));
    tmp(CC.PixelIdxList{idx}) = true;
    p2  = regionprops(tmp, 'Centroid');
    cx2 = p2.Centroid(1);
    cy2 = p2.Centroid(2);

    dist = sqrt((cx2-cx1)^2 + (cy2-cy1)^2);

    % Keep if within 40% of image diagonal
    if dist < 0.40 * diagLen
        mask = mask | tmp;
    end
end

% Final fill after merging
mask = imfill(mask, 'holes');
debugInfo.finalMask = mask;

end