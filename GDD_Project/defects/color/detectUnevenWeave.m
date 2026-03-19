function [isUnevenWeave, bboxes, debug] = detectUnevenWeave(segmentedImg, mask)
% detectUnevenWeave  Detect uneven/coarse weave defect on cotton gloves.
%
% APPROACH:
%   1. Compute local texture variance (stdfilt 21x21)
%   2. Find palm outlier pixels (median + 2*std threshold)
%   3. Close with SMALL radius to bridge only immediate thread gaps
%   4. Evaluate merged patch: fill density + minimum total raw coverage

debug = struct();
bboxes = [];

if ~any(mask(:))
    isUnevenWeave = false; return;
end

[rows, cols, ~] = size(segmentedImg);
gray      = im2double(rgb2gray(segmentedImg));
gloveArea = nnz(mask);

% ====================================================================
% GLOVE GEOMETRY + PALM REGION
% ====================================================================
gloveProps = regionprops(mask, 'BoundingBox');
gloveTop = 1; gloveH = rows; gloveW = cols;
if ~isempty(gloveProps)
    gBB      = gloveProps(1).BoundingBox;
    gloveTop = round(gBB(2));
    gloveH   = round(gBB(4));
    gloveW   = round(gBB(3));
end

palmTop = max(1,    gloveTop + round(gloveH * 0.20));
palmBot = min(rows, gloveTop + round(gloveH * 0.85));
palmMask = false(rows, cols);
palmMask(palmTop:palmBot, :) = true;
palmMask = palmMask & mask;

diskR      = max(3, round(0.005 * min(gloveW, gloveH)));
maskEroded = imerode(mask, strel('disk', diskR));
if ~any(maskEroded(:)), maskEroded = mask; end
palmMaskEroded = palmMask & maskEroded;
if nnz(palmMaskEroded) < 500, palmMaskEroded = palmMask; end

fprintf('  detectUnevenWeave: gloveW=%d gloveH=%d palmPx=%d\n', ...
    gloveW, gloveH, nnz(palmMaskEroded));

% ====================================================================
% TEXTURE VARIANCE MAP
% ====================================================================
varMap = stdfilt(gray, ones(21));
varMap(~maskEroded) = 0;
debug.varMap = mat2gray(varMap);

palmVarVals = varMap(palmMaskEroded);
if isempty(palmVarVals) || max(palmVarVals) < 1e-5
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    fprintf('=== detectUnevenWeave: no signal ===\n  Uneven weave : 0\n'); return;
end

baselineMedian = median(palmVarVals);
baselineStd    = std(palmVarVals);
outlierThresh  = baselineMedian + 2.0 * baselineStd;
outlierThresh  = max(outlierThresh, baselineMedian * 1.30);

fprintf('  detectUnevenWeave: median=%.4f  std=%.4f  thresh=%.4f\n', ...
    baselineMedian, baselineStd, outlierThresh);

% Raw high-variance pixels
highVarRaw = (varMap > outlierThresh) & palmMaskEroded;
highVarRaw = bwareaopen(highVarRaw, 20);

fprintf('  detectUnevenWeave: rawHighVarPx=%d\n', nnz(highVarRaw));
debug.highVarRaw = highVarRaw;
debug.outlierThresh = outlierThresh;

% ====================================================================
% MERGE WITH SMALL RADIUS
%   Use 1% of glove width — just enough to bridge gaps between
%   adjacent thread clusters without absorbing surrounding normal knit.
%   Previous 3% was too large → absorbed normal knit → low fillDens.
% ====================================================================
mergeRadius  = max(8, round(0.010 * gloveW));
mergedMask   = imclose(highVarRaw, strel('disk', mergeRadius));
mergedMask   = imfill(mergedMask, 'holes');
mergedMask   = mergedMask & palmMaskEroded;

fprintf('  detectUnevenWeave: mergedPx=%d  mergeR=%d\n', nnz(mergedMask), mergeRadius);
debug.mergedMask = mergedMask;

if ~any(mergedMask(:))
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    fprintf('  Uneven weave : 0\n'); return;
end

% ====================================================================
% EVALUATE MERGED PATCHES
%
%   Requirements for each merged patch:
%   1. rawHighVar pixels within it >= 500px absolute
%      (the raw outlier coverage must be significant regardless of patch size)
%   2. Fill density >= 0.25 (>25% of merged area is genuinely high-variance)
%   3. Total merged area >= 0.3% of glove (visible defect)
%   4. Solidity >= 0.20
%
%   The key insight: check RAW pixel count within each merged region,
%   not just fill density. The coarse weave has ~25000 raw px total.
%   We want the merged region that captures the most raw px.
% ====================================================================
CC = bwconncomp(mergedMask);
if CC.NumObjects == 0
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    fprintf('  Uneven weave : 0\n'); return;
end

% Find the component with most raw high-var pixels inside it
rawCounts = zeros(CC.NumObjects, 1);
for ci = 1:CC.NumObjects
    compMask = false(rows,cols);
    compMask(CC.PixelIdxList{ci}) = true;
    rawCounts(ci) = nnz(highVarRaw & compMask);
end
[~, bestIdx] = max(rawCounts);

px       = CC.PixelIdxList{bestIdx};
area     = numel(px);
rawCount = rawCounts(bestIdx);
fillDens = rawCount / area;

compMask = false(rows, cols); compMask(px) = true;
props    = regionprops(compMask, 'Solidity', 'BoundingBox');
sol      = props(1).Solidity;
bb       = props(1).BoundingBox;

fprintf('  detectUnevenWeave: best patch area=%d rawPx=%d fillDens=%.3f sol=%.2f\n', ...
    area, rawCount, fillDens, sol);

% Minimum raw pixel count: 500px absolute
% This ensures a real coarse weave patch (thousands of raw px) always passes
% while tiny false positives (few dozen raw px) always fail
minRawPx    = 500;
minArea     = max(500, round(0.003 * gloveArea));
minFillDens = 0.15;

if rawCount < minRawPx
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    debug.totalFrac = 0;
    fprintf('  detectUnevenWeave: REJECTED insufficient raw px (%d < %d)\n', rawCount, minRawPx);
    fprintf('  Uneven weave : 0\n'); return;
end

if area < minArea
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    debug.totalFrac = 0;
    fprintf('  detectUnevenWeave: REJECTED area too small (%d < %d)\n', area, minArea);
    fprintf('  Uneven weave : 0\n'); return;
end

if fillDens < minFillDens
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    debug.totalFrac = 0;
    fprintf('  detectUnevenWeave: REJECTED low fill density (%.3f < %.2f)\n', fillDens, minFillDens);
    fprintf('  Uneven weave : 0\n'); return;
end

if sol < 0.20
    isUnevenWeave = false;
    debug.isUnevenWeave = false; debug.filteredMask = false(rows,cols);
    debug.totalFrac = 0;
    fprintf('  detectUnevenWeave: REJECTED low solidity (%.2f)\n', sol);
    fprintf('  Uneven weave : 0\n'); return;
end

% Accepted
% Expand the merged component bbox using only raw high-var pixels
% that are SPATIALLY NEAR the merged component (within 2x bbox size).
% This captures the full defect patch without grabbing distant noise.
expandMask = imdilate(compMask, strel('disk', max(30, round(0.02*gloveW))));
localRaw   = highVarRaw & expandMask;
if nnz(localRaw) > 10
    [rawR, rawC] = find(localRaw);
    pad = 30;
    x1 = max(1,    min(rawC)-pad); y1 = max(1,    min(rawR)-pad);
    x2 = min(cols, max(rawC)+pad); y2 = min(rows, max(rawR)+pad);
else
    pad = 30;
    x1 = max(1,    bb(1)-pad); y1 = max(1,    bb(2)-pad);
    x2 = min(cols, bb(1)+bb(3)+pad); y2 = min(rows, bb(2)+bb(4)+pad);
end
bboxes = [x1, y1, x2-x1, y2-y1];

filteredMask = compMask;
totalFrac    = area / max(gloveArea, 1);

isUnevenWeave = true;

fprintf('=== detectUnevenWeave report ===\n');
fprintf('  Best patch    : area=%d rawPx=%d fillDens=%.3f\n', area, rawCount, fillDens);
fprintf('  Coverage frac : %.4f\n', totalFrac);
fprintf('  Uneven weave  : 1\n');

debug.filteredMask  = filteredMask;
debug.isUnevenWeave = isUnevenWeave;
debug.totalFrac     = totalFrac;
debug.bboxCount     = size(bboxes,1);
end