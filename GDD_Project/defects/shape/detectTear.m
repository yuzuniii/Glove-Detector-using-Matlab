function [isTear, bboxes, debug] = detectTear(segmentedImg, mask, origImg)
% detectTear  Detect tear/hole defects in black nitrile gloves.
%
% VISUAL SIGNATURE (from glovetear.png):
%   A tear is a HOLE through the glove — the background is visible
%   through it. On a black glove on a grey/light background, the hole
%   appears as a BRIGHT region INSIDE the glove mask boundary.
%   The torn edges have jagged white/grey rubber fragments around the hole.
%
% APPROACH:
%   1. Find the glove mask interior (eroded to exclude boundary)
%   2. Look for bright pixels INSIDE the mask — these are holes showing
%      background through, or exposed inner surface at tear edges
%   3. Bright region must be compact (not elongated like a stain smear)
%      and must be interior (not touching the glove boundary)
%
% Why this works for black gloves:
%   Black glove: V < 0.35 everywhere on glove surface
%   Tear hole:   V > 0.45 (background/inner surface visible)
%   Contrast is large and reliable on black nitrile.

debug = struct();

if ~any(mask(:))
    isTear = false;
    bboxes = [];
    return;
end

imgDouble = im2double(segmentedImg);
hsvImg    = rgb2hsv(imgDouble);
V = hsvImg(:,:,3);

[rows, cols] = size(mask);
gloveArea    = nnz(mask);

% ====================================================================
% STEP 1: Erode mask to get interior only
%   Removes boundary pixels where background bleeds into glove edge
% ====================================================================
shortSide  = min(rows, cols);
diskR      = max(10, min(25, round(0.04 * shortSide)));
maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.25 * gloveArea
    maskEroded = imerode(mask, strel('disk', round(diskR/2)));
end
if ~any(maskEroded(:))
    maskEroded = mask;
end

debug.maskEroded = maskEroded;

% ====================================================================
% STEP 2: Find bright pixels INSIDE the glove mask
%
%   On a black nitrile glove, glove surface has V < 0.35.
%   A tear exposes background (V ≈ 0.60-0.85) or inner surface.
%   Threshold: pixels with V > glove_mean + 3*glove_std inside mask.
%
%   Also compute absolute threshold: V > 0.40 inside eroded mask.
%   Use the more restrictive of the two.
% ====================================================================
gloveV     = V(maskEroded);
gloveMeanV = mean(gloveV);
gloveStdV  = std(gloveV);

% Statistical threshold: far above glove surface brightness
statThresh = gloveMeanV + 3.0 * gloveStdV;
statThresh = max(statThresh, 0.38);   % floor

% Absolute threshold: tear hole shows background brightness
absThresh  = 0.40;

brightThresh = max(statThresh, absThresh);

fprintf('  detectTear: gloveMeanV=%.3f stdV=%.3f brightThresh=%.3f\n', ...
    gloveMeanV, gloveStdV, brightThresh);

% Bright mask = potential tear holes, inside eroded glove mask only
brightMask = (V > brightThresh) & maskEroded;
brightMask = bwareaopen(brightMask, 50);   % remove tiny noise

debug.brightMask  = brightMask;
debug.brightThresh = brightThresh;

% ====================================================================
% STEP 2b: Colour-match filter — distinguish tear from stain
%
%   A TEAR shows the background through the hole — bright region colour
%   matches the background (same grey surface, same LAB values).
%
%   A STAIN is a different material — white chalk, powder, residue —
%   its colour differs from the background (warmer, different A/B).
%
%   Method: sample background LAB from image corners.
%   For each bright region, compute mean LAB distance from background.
%   Low distance  → matches background → TEAR (hole showing through)
%   High distance → different from background → STAIN (not a tear)
%
%   Threshold: background LAB distance < 15 LAB units → tear candidate
% ====================================================================
labImg = rgb2lab(im2double(segmentedImg));
Lch = labImg(:,:,1);
Ach = labImg(:,:,2);
Bch = labImg(:,:,3);

% Sample background LAB from ORIGINAL image corners (not segmented —
% segmented image has background zeroed to black which gives wrong LAB).
if nargin >= 3 && ~isempty(origImg)
    labOrig = rgb2lab(im2double(origImg));
else
    labOrig = rgb2lab(im2double(segmentedImg));  % fallback
end
LoC = labOrig(:,:,1);
AoC = labOrig(:,:,2);
BoC = labOrig(:,:,3);

pSz = max(5, round(min(rows,cols) * 0.06));
bgL = [LoC(1:pSz,1:pSz); LoC(1:pSz,cols-pSz+1:cols);
       LoC(rows-pSz+1:rows,1:pSz); LoC(rows-pSz+1:rows,cols-pSz+1:cols)];
bgA = [AoC(1:pSz,1:pSz); AoC(1:pSz,cols-pSz+1:cols);
       AoC(rows-pSz+1:rows,1:pSz); AoC(rows-pSz+1:rows,cols-pSz+1:cols)];
bgB = [BoC(1:pSz,1:pSz); BoC(1:pSz,cols-pSz+1:cols);
       BoC(rows-pSz+1:rows,1:pSz); BoC(rows-pSz+1:rows,cols-pSz+1:cols)];

bgMeanL = mean(bgL(:));
bgMeanA = mean(bgA(:));
bgMeanB = mean(bgB(:));

fprintf('  detectTear: bg LAB=[%.1f,%.2f,%.2f]\n', bgMeanL, bgMeanA, bgMeanB);

% For each bright region, check if its LAB matches background
CC_bright = bwconncomp(brightMask);
tearCandidateMask = false(rows, cols);

for ci = 1:CC_bright.NumObjects
    px = CC_bright.PixelIdxList{ci};
    % Mean LAB of this bright region (from segmented image — glove pixels only)
    rL = mean(Lch(px));
    rA = mean(Ach(px));
    rB = mean(Bch(px));
    % LAB distance from background
    labDist = sqrt((rL-bgMeanL)^2 + (rA-bgMeanA)^2 + (rB-bgMeanB)^2);
    fprintf('  detectTear: region %d: LAB=[%.1f,%.2f,%.2f] dist=%.1f\n', ...
        ci, rL, rA, rB, labDist);
    % Tear: bright region looks like background (low LAB distance)
    % Stain: bright region is a different material (high LAB distance)
    %
    % Two conditions BOTH required:
    % 1. LAB distance < 15 (colour match to background)
    % 2. Luminance within 12 units of background (tear shows same surface)
    %    Stain is darker than background: L=62 vs bgL=79 → diff=17 → NOT tear
    lDiff = abs(rL - bgMeanL);
    if labDist < 15 && lDiff < 10
        tearCandidateMask(px) = true;
        fprintf('  detectTear: region %d ACCEPTED as tear (dist=%.1f lDiff=%.1f)\n', ci, labDist, lDiff);
    else
        fprintf('  detectTear: region %d rejected (dist=%.1f lDiff=%.1f)\n', ci, labDist, lDiff);
    end
end

% Use tear candidates only (colour-matched to background)
brightMask = tearCandidateMask;
brightMask = bwareaopen(brightMask, 50);
debug.tearCandidateMask = brightMask;

% ====================================================================
% STEP 3: Region filtering
%
%   Tear hole properties:
%     Area: min 0.1% of glove (visible hole), max 30% (catastrophic tear)
%     Solidity > 0.30: holes are somewhat compact (not thin lines)
%     Eccentricity < 0.98: not extremely elongated (that'd be a scratch)
%     NOT touching glove boundary (interior hole, not edge damage)
%       → use eroded mask already handles this
% ====================================================================
minArea = max(500, 0.001 * gloveArea);
maxArea = 0.30  * gloveArea;

stats = regionprops(brightMask, 'Area', 'BoundingBox', ...
    'Solidity', 'Eccentricity', 'PixelIdxList');

filteredMask = false(rows, cols);
bboxes       = [];
nSmall = 0; nLarge = 0; nShape = 0; nOk = 0;

for i = 1:numel(stats)
    a   = stats(i).Area;
    sol = stats(i).Solidity;
    ecc = stats(i).Eccentricity;
    bb  = stats(i).BoundingBox;

    if a < minArea,  nSmall = nSmall+1; continue; end
    if a > maxArea,  nLarge = nLarge+1; continue; end
    if sol < 0.25,   nShape = nShape+1; continue; end   % too fragmented
    if ecc > 0.98,   nShape = nShape+1; continue; end   % too elongated

    filteredMask(stats(i).PixelIdxList) = true;
    bboxes = [bboxes; bb]; %#ok<AGROW>
    nOk    = nOk + 1;
end

% Merge nearby boxes (tear may produce multiple bright fragments)
if size(bboxes, 1) > 1
    bboxes = mergeBoxes(bboxes, 20);
end

% Add padding around tear box for visibility
padding = 20;
for i = 1:size(bboxes, 1)
    bboxes(i,1) = max(1,      bboxes(i,1) - padding);
    bboxes(i,2) = max(1,      bboxes(i,2) - padding);
    bboxes(i,3) = min(cols - bboxes(i,1), bboxes(i,3) + 2*padding);
    bboxes(i,4) = min(rows - bboxes(i,2), bboxes(i,4) + 2*padding);
end

isTear = ~isempty(bboxes);

% ====================================================================
% Console report
% ====================================================================
fprintf('=== detectTear report ===\n');
fprintf('  Glove area    : %d px\n',  gloveArea);
fprintf('  Bright thresh : %.3f\n',   brightThresh);
fprintf('  minArea       : %.0f px\n', minArea);
fprintf('  maxArea       : %.0f px\n', maxArea);
fprintf('  Components    : %d  (small=%d large=%d shape=%d ok=%d)\n', ...
    numel(stats), nSmall, nLarge, nShape, nOk);
if ~isempty(stats)
    areas = sort([stats.Area], 'descend');
    fprintf('  Top areas     : ');
    fprintf('%d ', areas(1:min(6,end)));
    fprintf('\n');
end
fprintf('  Boxes output  : %d\n', size(bboxes,1));
fprintf('  Tear found    : %d\n', isTear);

debug.filteredMask   = filteredMask;
debug.isTear         = isTear;
debug.componentCount = numel(stats);
debug.bboxCount      = size(bboxes,1);
debug.minAreaUsed    = minArea;
debug.maxAreaUsed    = maxArea;

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
            ox = max(bi(1),bj(1)) < min(bi(1)+bi(3), bj(1)+bj(3));
            oy = max(bi(2),bj(2)) < min(bi(2)+bi(4), bj(2)+bj(4));
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