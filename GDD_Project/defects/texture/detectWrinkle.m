function [isWrinkle, bboxes, debug] = detectWrinkle(segmentedImg, mask)
% detectWrinkle  Detect wrinkle/dent defects on a segmented glove.
%
% Changelog (this version):
%   - maxArea tightened: 10% -> 5% of glove pixels
%     On high-res images (2.3M px glove) the 10% cap = 231K px which
%     allowed very large merged blobs to pass. 5% = 115K px is a
%     tighter cap that better represents a localised wrinkle region.
%   - minArea floor raised: max(1000,...) -> min(1000, max(200,...))
%     Keeps floor reasonable on all image sizes.
%
% TWO-PATHWAY DESIGN:
%   PATH A — Gradient + local normalisation (coloured gloves)
%   PATH B — Morphological top-hat (white/low-contrast gloves)
%            Activated when mean raw gradient < 0.04

% ====================================================================
% 1. GRAYSCALE
% ====================================================================
gray = im2double(rgb2gray(segmentedImg));

% ====================================================================
% 2. ERODE MASK
% ====================================================================
shortSide  = min(size(gray,1), size(gray,2));
diskR      = max(8, min(20, round(0.035 * shortSide)));

maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.30 * nnz(mask)
    diskR      = max(5, round(diskR / 2));
    maskEroded = imerode(mask, strel('disk', diskR));
end
if ~any(maskEroded(:))
    maskEroded = mask;
end

gray(~maskEroded) = 0;

% ====================================================================
% 3. SMOOTH
% ====================================================================
gray = imgaussfilt(gray, 1.5);

% ====================================================================
% 4. TWO-SCALE GRADIENT
% ====================================================================
[Gx1,Gy1] = imgradientxy(gray, 'sobel');
g2         = imgaussfilt(gray, 3.0);
[Gx2,Gy2]  = imgradientxy(g2,  'sobel');

grad1    = sqrt(Gx1.^2 + Gy1.^2);
grad2    = sqrt(Gx2.^2 + Gy2.^2);
gradRaw  = max(grad1, grad2);
gradRaw(~maskEroded) = 0;

% ====================================================================
% 5. PATHWAY DECISION
% ====================================================================
gloveArea   = nnz(mask);
meanGradRaw = mean(gradRaw(maskEroded));
useTopHat   = (meanGradRaw < 0.04);

fprintf('  Mean raw gradient: %.4f -> using %s\n', meanGradRaw, ...
    condStr(useTopHat, 'TOP-HAT (Path B)', 'GRADIENT (Path A)'));

% ====================================================================
% 6a. PATH A: local normalisation + threshold
% ====================================================================
if ~useTopHat
    grad = mat2gray(gradRaw);
    grad(~maskEroded) = 0;

    winSize = round(0.06 * sqrt(gloveArea));
    winSize = max(31, min(101, winSize));
    if mod(winSize,2)==0, winSize = winSize+1; end

    localMeanGrad              = imfilter(grad, fspecial('average',winSize), 'replicate');
    localMeanGrad(~maskEroded) = 1;
    normGrad                   = grad ./ (localMeanGrad + 0.001);
    normGrad(~maskEroded)      = 0;
    normGrad                   = mat2gray(normGrad);

    maskedNorm     = normGrad(maskEroded);
    otsu_used      = graythresh(maskedNorm);
    thresholdValue = otsu_used * 0.6;
    binMask        = (normGrad > thresholdValue) & maskEroded;
    scoreMap       = normGrad;
    winSizeUsed    = winSize;

% ====================================================================
% 6b. PATH B: top-hat
% ====================================================================
else
    winSizeUsed = 0;
    grayRaw     = im2double(rgb2gray(segmentedImg));
    grayRaw(~maskEroded) = 0;

    th1    = imtophat(grayRaw, strel('disk', 15));
    th2    = imtophat(grayRaw, strel('disk', 25));
    topHat = mat2gray(max(th1, th2));
    topHat(~maskEroded) = 0;

    bth1   = imbothat(grayRaw, strel('disk', 15));
    bth2   = imbothat(grayRaw, strel('disk', 25));
    botHat = mat2gray(max(bth1, bth2));
    botHat(~maskEroded) = 0;

    combined = mat2gray(topHat + botHat);
    combined(~maskEroded) = 0;

    maskedComb = combined(maskEroded);
    if isempty(maskedComb) || max(maskedComb) < 1e-4
        thresholdValue = 1;
        otsu_used      = 1;
    else
        otsu_used      = graythresh(maskedComb);
        thresholdValue = otsu_used * 0.5;
    end

    binMask  = (combined > thresholdValue) & maskEroded;
    scoreMap = combined;
    grad     = mat2gray(gradRaw);
end

% ====================================================================
% 7. CLEAN BINARY MASK
% ====================================================================
binMask = bwareaopen(binMask, 10);
boundaryBand = imdilate(mask & ~maskEroded, strel('disk', 3));
binMask(boundaryBand) = 0;

% ====================================================================
% 8. MORPHOLOGICAL CLOSING
% ====================================================================
closeDisk = condVal(useTopHat, 5, 8);
closed    = imclose(binMask, strel('disk', closeDisk));
closed    = imfill(closed, 'holes');
closed    = bwareaopen(closed, 100);
closed    = closed & maskEroded;

% ====================================================================
% 9. REGION FILTERING
%    maxArea tightened: 10% -> 5% of glove pixels
%    minArea: min(1000, max(200, 0.1% of glove))
% ====================================================================
minArea   = min(1000, max(200, 0.001 * gloveArea));
maxArea   = 0.05 * gloveArea;   % tightened from 10%
minAspect = condVal(useTopHat, 0.05, 0.08);

stats = regionprops(closed, 'Area', 'BoundingBox', 'PixelIdxList');

filteredMask = false(size(closed));
bboxes       = [];
nSmall = 0; nLarge = 0; nThin = 0; nOk = 0;

for i = 1:numel(stats)
    a  = stats(i).Area;
    bb = stats(i).BoundingBox;
    ar = min(bb(3),bb(4)) / max(bb(3),bb(4));

    if a < minArea,    nSmall = nSmall+1; continue; end
    if a > maxArea,    nLarge = nLarge+1; continue; end
    if ar < minAspect, nThin  = nThin+1;  continue; end

    filteredMask(stats(i).PixelIdxList) = true;
    bboxes = [bboxes; bb]; %#ok<AGROW>
    nOk    = nOk + 1;
end

% ====================================================================
% 10. MERGE NEARBY BOXES
% ====================================================================
mergeGap = condVal(useTopHat, 15, 8);
if size(bboxes,1) > 1
    bboxes = mergeBoxes(bboxes, mergeGap);
end

isWrinkle = ~isempty(bboxes);

% ====================================================================
% Console report
% ====================================================================
fprintf('=== detectWrinkle report ===\n');
fprintf('  Glove area   : %d px\n',  gloveArea);
fprintf('  Disk radius  : %d px\n',  diskR);
fprintf('  Norm window  : %d px\n',  winSizeUsed);
fprintf('  Path         : %s\n',     condStr(useTopHat,'B (top-hat)','A (gradient)'));
fprintf('  Otsu         : %.4f  threshold: %.4f\n', otsu_used, thresholdValue);
fprintf('  minArea      : %.0f px\n', minArea);
fprintf('  maxArea      : %.0f px  (5%% of glove)\n', maxArea);
fprintf('  Components   : %d  (small=%d large=%d thin=%d ok=%d)\n', ...
        numel(stats), nSmall, nLarge, nThin, nOk);
fprintf('  Boxes output : %d\n', size(bboxes,1));
if ~isempty(stats)
    areas = sort([stats.Area],'descend');
    fprintf('  Top areas    : ');
    fprintf('%d ', areas(1:min(8,end)));
    fprintf('\n');
end

% ====================================================================
% Debug struct
% ====================================================================
debug.gradientMagnitude = mat2gray(gradRaw);
debug.normGrad          = scoreMap;
debug.localStdMap       = scoreMap;
debug.localEntropyMap   = scoreMap;
debug.textureMask       = binMask;
debug.adaptiveMask      = binMask;
debug.clusterMask       = closed;
debug.mergedMask        = filteredMask;
debug.filteredMask      = filteredMask;
debug.maskEroded        = maskEroded;
debug.boundaryBand      = mask & ~maskEroded;
debug.thresholdValue    = thresholdValue;
debug.componentCount    = numel(stats);
debug.bboxCount         = size(bboxes,1);
debug.minAreaUsed       = minArea;
debug.maxAreaAllowed    = maxArea;
debug.normWindowSize    = winSizeUsed;
debug.usedTopHat        = useTopHat;
debug.meanGradRaw       = meanGradRaw;
gloveArea2              = max(gloveArea, 1);
debug.wrinkleDensity    = nnz(filteredMask) / gloveArea2;
debug.rawAreas          = [];
debug.rawEccentricities = [];
debug.rawSolidities     = [];

end


% ------------------------------------------------------------------
function s = condStr(cond, a, b)
if cond, s = a; else, s = b; end
end

function v = condVal(cond, a, b)
if cond, v = a; else, v = b; end
end


% ------------------------------------------------------------------
function merged = mergeBoxes(bboxes, gap)
changed = true;
while changed
    changed = false;
    n    = size(bboxes,1);
    used = false(n,1);
    merged = [];
    for i = 1:n
        if used(i), continue; end
        b = bboxes(i,:);
        for j = i+1:n
            if used(j), continue; end
            bi = [b(1)-gap,         b(2)-gap, ...
                  b(3)+2*gap,        b(4)+2*gap];
            bj = [bboxes(j,1)-gap,  bboxes(j,2)-gap, ...
                  bboxes(j,3)+2*gap, bboxes(j,4)+2*gap];
            ox = max(bi(1),bj(1)) < min(bi(1)+bi(3), bj(1)+bj(3));
            oy = max(bi(2),bj(2)) < min(bi(2)+bi(4), bj(2)+bj(4));
            if ox && oy
                x1 = min(b(1),           bboxes(j,1));
                y1 = min(b(2),           bboxes(j,2));
                x2 = max(b(1)+b(3),      bboxes(j,1)+bboxes(j,3));
                y2 = max(b(2)+b(4),      bboxes(j,2)+bboxes(j,4));
                b  = [x1, y1, x2-x1, y2-y1];
                used(j)  = true;
                changed  = true;
            end
        end
        merged = [merged; b]; %#ok<AGROW>
    end
    bboxes = merged;
end
merged = bboxes;
end