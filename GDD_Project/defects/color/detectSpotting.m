function [isSpotting, bboxes, debug] = detectSpotting(segmentedImg, mask)
% detectSpotting  Detect spotting defects on glove surface.
%
% APPROACH:
%   Spotting appears as many SMALL DISCRETE DARK DOTS scattered across
%   the glove surface. This differs from:
%   - Stain: one large continuous dark patch
%   - Dirty: global darkening of entire surface
%   - Scratch: elongated linear dark mark
%
%   Detection strategy:
%   1. Convert to grayscale, find pixels significantly darker than
%      the local glove surface (dark anomaly map)
%   2. Use strict SIZE CONSTRAINTS — each spot is small (5-200px)
%   3. Require MULTIPLE spots (count > threshold) to confirm spotting
%      vs a single small stain
%   4. Generate individual bounding boxes per spot cluster
%
% Inputs:
%   segmentedImg - RGB, background zeroed
%   mask         - logical, true = glove pixel
%
% Outputs:
%   isSpotting   - true if spotting detected
%   bboxes       - Nx4 [x y w h] one box per spot or cluster
%   debug        - diagnostic struct

debug = struct();

if ~any(mask(:))
    isSpotting = false;
    bboxes     = [];
    return;
end

% ====================================================================
% 1. MASK EROSION
% ====================================================================
shortSide  = min(size(mask,1), size(mask,2));
diskR      = max(6, min(15, round(0.025 * shortSide)));
maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.30 * nnz(mask)
    maskEroded = imerode(mask, strel('disk', round(diskR/2)));
end
if ~any(maskEroded(:))
    maskEroded = mask;
end

debug.maskEroded = maskEroded;

% ====================================================================
% 2. DARK SPOT DETECTION
%    Spots are significantly darker than the surrounding glove surface.
%    Use local mean subtraction with a medium window (neighbourhood
%    larger than a spot but smaller than the whole glove).
%    Window: ~5% of glove diameter, clamped 21-61px
% ====================================================================
gray = im2double(rgb2gray(segmentedImg));
gray(~maskEroded) = 0;
gray = imgaussfilt(gray, 0.8);   % very light smoothing

gloveArea = nnz(mask);
winSize   = round(0.05 * sqrt(gloveArea));
winSize   = max(21, min(61, winSize));
if mod(winSize,2)==0, winSize = winSize+1; end

% Dark anomaly: how much darker than local neighbourhood
localMean  = imfilter(gray, fspecial('average', winSize), 'replicate');
darkMap    = localMean - gray;
darkMap(darkMap < 0)   = 0;
darkMap(~maskEroded)   = 0;
darkMap    = mat2gray(darkMap);

debug.darkMap  = darkMap;
debug.winSize  = winSize;

% ====================================================================
% 3. THRESHOLD
%    Otsu at 0.55x — spots are strong dark anomalies
% ====================================================================
maskedDark = darkMap(maskEroded);
if isempty(maskedDark) || max(maskedDark) < 1e-4
    isSpotting = false;
    bboxes     = [];
    debug.spotMask     = false(size(mask));
    debug.filteredMask = false(size(mask));
    debug.isSpotting   = false;
    debug.spotCount    = 0;
    debug.bboxCount    = 0;
    fprintf('=== detectSpotting: no signal ===\n');
    return;
end

otsu           = graythresh(maskedDark);
thresholdValue = otsu * 0.55;
spotMask       = (darkMap > thresholdValue) & maskEroded;

% Remove boundary band
boundaryBand = imdilate(mask & ~maskEroded, strel('disk', 3));
spotMask(boundaryBand) = 0;

% Remove very tiny noise (< 5px)
spotMask = bwareaopen(spotMask, 5);

debug.spotMask    = spotMask;
debug.threshValue = thresholdValue;

% ====================================================================
% 4. REGION FILTERING — STRICT SIZE CONSTRAINTS
%    Spots are small discrete blobs:
%    minArea = 5px  (tiny spots)
%    maxArea = max(500, 0.5% of glove) — individual spots are small
%              If a region is larger than this it is a stain, not a spot
%
%    Compactness: spots are roughly circular/compact
%    Eccentricity < 0.90 — not too elongated (that would be a scratch)
% ====================================================================
minSpotArea = 5;
maxSpotArea = max(500, 0.005 * gloveArea);

stats = regionprops(spotMask, 'Area', 'BoundingBox', ...
    'Eccentricity', 'PixelIdxList', 'Centroid');

filteredMask = false(size(spotMask));
bboxes       = [];
spotCentroids = [];
nSmall = 0; nLarge = 0; nThin = 0; nOk = 0;

for i = 1:numel(stats)
    a   = stats(i).Area;
    ecc = stats(i).Eccentricity;

    if a < minSpotArea
        nSmall = nSmall+1; continue
    end
    if a > maxSpotArea
        nLarge = nLarge+1; continue   % too large — this is a stain, not a spot
    end
    if ecc > 0.90
        nThin = nThin+1; continue    % too elongated — scratch or line artifact
    end

    filteredMask(stats(i).PixelIdxList) = true;
    bboxes        = [bboxes; stats(i).BoundingBox]; %#ok<AGROW>
    spotCentroids = [spotCentroids; stats(i).Centroid]; %#ok<AGROW>
    nOk           = nOk+1;
end

spotCount = nOk;
debug.spotCount    = spotCount;
debug.filteredMask = filteredMask;

% ====================================================================
% 5. SPOTTING DECISION
%    Require MULTIPLE spots to confirm spotting defect.
%    A single small dark region could be a tiny stain or noise.
%    Threshold: >= 3 spots = spotting confirmed.
% ====================================================================
minSpotCount = 3;
isSpotting   = (spotCount >= minSpotCount);

% ====================================================================
% 6. MERGE VERY CLOSE SPOTS INTO CLUSTER BOXES
%    Individual spot boxes are very small and numerous.
%    Merge spots within 40px of each other into cluster boxes
%    for cleaner visualisation.
% ====================================================================
if isSpotting && ~isempty(bboxes)
    bboxes = mergeBoxes(bboxes, 40);
end

% ====================================================================
% Console report
% ====================================================================
fprintf('=== detectSpotting report ===\n');
fprintf('  Glove area    : %d px\n',  gloveArea);
fprintf('  Window size   : %d px\n',  winSize);
fprintf('  Otsu          : %.4f  threshold: %.4f\n', otsu, thresholdValue);
fprintf('  Max spot area : %.0f px\n', maxSpotArea);
fprintf('  Spots found   : %d  (small=%d large=%d thin=%d ok=%d)\n', ...
        numel(stats), nSmall, nLarge, nThin, nOk);
fprintf('  Min required  : %d\n', minSpotCount);
fprintf('  Spotting      : %d\n', isSpotting);
fprintf('  Boxes output  : %d\n', size(bboxes,1));
if ~isempty(stats)
    areas = sort([stats.Area],'descend');
    fprintf('  Top areas     : ');
    fprintf('%d ', areas(1:min(8,end)));
    fprintf('\n');
end

debug.isSpotting   = isSpotting;
debug.bboxCount    = size(bboxes,1);
debug.minAreaUsed  = minSpotArea;
debug.maxAreaUsed  = maxSpotArea;
debug.otsu         = otsu;
debug.spotCentroids = spotCentroids;
gloveArea2         = max(gloveArea,1);
debug.spottingDensity = nnz(filteredMask) / gloveArea2;

end


% ------------------------------------------------------------------
function merged = mergeBoxes(bboxes, gap)
changed = true;
while changed
    changed = false;
    n = size(bboxes,1); used = false(n,1); merged = [];
    for i = 1:n
        if used(i), continue; end
        b = bboxes(i,:);
        for j = i+1:n
            if used(j), continue; end
            bi = [b(1)-gap, b(2)-gap, b(3)+2*gap, b(4)+2*gap];
            bj = [bboxes(j,1)-gap, bboxes(j,2)-gap, bboxes(j,3)+2*gap, bboxes(j,4)+2*gap];
            ox = max(bi(1),bj(1)) < min(bi(1)+bi(3),bj(1)+bj(3));
            oy = max(bi(2),bj(2)) < min(bi(2)+bi(4),bj(2)+bj(4));
            if ox && oy
                x1=min(b(1),bboxes(j,1)); y1=min(b(2),bboxes(j,2));
                x2=max(b(1)+b(3),bboxes(j,1)+bboxes(j,3));
                y2=max(b(2)+b(4),bboxes(j,2)+bboxes(j,4));
                b=[x1,y1,x2-x1,y2-y1]; used(j)=true; changed=true;
            end
        end
        merged=[merged;b]; %#ok<AGROW>
    end
    bboxes=merged;
end
merged=bboxes;
end