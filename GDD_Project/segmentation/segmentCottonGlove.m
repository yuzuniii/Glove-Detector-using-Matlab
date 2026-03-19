function [mask, segmentedImg] = segmentCottonGlove(img)
% segmentCottonGlove  Segment cotton/fabric glove from grey background.
%
% Approach:
%   1. Sample background LAB colour from top-left safe region
%   2. Compute LAB distance map from background colour
%   3. Threshold to get background candidates
%   4. Label connected components, keep only those touching image edges
%      (= connected background) — glove never touches all edges
%   5. Morphological cleanup + largest component

imgDouble = im2double(img);
[rows, cols, ~] = size(imgDouble);

% Convert to LAB for perceptual colour distance
labImg = rgb2lab(imgDouble);
L = labImg(:,:,1);
A = labImg(:,:,2);
B = labImg(:,:,3);

% Step 1: Sample background from top-left corner (always safe background)
safeRows = max(1, round(rows * 0.05));
safeCols = max(1, round(cols * 0.05));
bgL = mean(mean(L(1:safeRows, 1:safeCols)));
bgA = mean(mean(A(1:safeRows, 1:safeCols)));
bgB = mean(mean(B(1:safeRows, 1:safeCols)));
fprintf('  segmentCottonGlove: bg LAB=[%.1f, %.1f, %.1f]\n', bgL, bgA, bgB);

% Step 2: LAB distance from background
% Weight A and B channels more (colour difference more discriminating than L)
distMap = sqrt(0.30*(L - bgL).^2 + 2.0*(A - bgA).^2 + 2.0*(B - bgB).^2);

% Step 3: Background = low distance from bg colour
bgThresh = 12;
bgRaw = distMap < bgThresh;

% Step 4: Keep only background connected to image border
% Label connected components of background
[labeled, nComp] = bwlabel(bgRaw);

% Find labels touching top, left, right edges (not bottom — glove may touch)
edgeLabels = unique([labeled(1,:), labeled(:,1)', labeled(:,end)']);
edgeLabels(edgeLabels == 0) = [];

% Connected background = components touching edges
connBg = ismember(labeled, edgeLabels);

% Glove = not connected background
gloveMask = ~connBg;

% Step 5: Morphological cleanup
gloveMask = imclose(gloveMask, strel('disk', 12));
gloveMask = imfill(gloveMask, 'holes');
minGloveArea = round(rows * cols * 0.03);
gloveMask = bwareaopen(gloveMask, minGloveArea);

% Keep largest connected component
cc = bwconncomp(gloveMask);
if cc.NumObjects == 0
    % Fallback: use raw distance threshold directly
    gloveMask = distMap > bgThresh;
    gloveMask = imclose(gloveMask, strel('disk', 15));
    gloveMask = imfill(gloveMask, 'holes');
    cc = bwconncomp(gloveMask);
end

if cc.NumObjects > 0
    sizes   = cellfun(@numel, cc.PixelIdxList);
    [~, ix] = max(sizes);
    mask    = false(rows, cols);
    mask(cc.PixelIdxList{ix}) = true;
else
    mask = gloveMask;
end

% Apply mask
segmentedImg = imgDouble;
segmentedImg(repmat(~mask, [1 1 3])) = 0;

fprintf('  segmentCottonGlove: %d px\n', nnz(mask));
end