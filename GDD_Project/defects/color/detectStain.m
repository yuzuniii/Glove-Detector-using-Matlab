function [isStain, bboxes, debug] = detectStain(segmentedImg, mask)
% detectStain  Detect stain defects on glove surface.
%
% Works across all glove types (white latex, black nitrile, blue nitrile)
% by analysing LOCAL COLOUR ANOMALIES in LAB colour space.
%
% APPROACH:
%   A stain is a region where BOTH:
%     1. Luminance (L) differs significantly from local neighbourhood
%     2. Colour (AB channels) shifts away from the dominant glove colour
%
%   Using LAB colour space because:
%   - L channel captures brightness anomalies independent of colour
%   - AB channels capture colour shift independent of brightness
%   - Together they detect any colour/intensity anomaly on any glove type
%
%   Pipeline:
%   1. Convert to LAB colour space
%   2. Compute local anomaly map for L, A, B channels separately
%   3. Combine into a single stain score map
%   4. Threshold and filter regions by area and compactness
%      (stains are compact blobs, not elongated lines)
%
% Inputs:
%   segmentedImg - RGB, background zeroed
%   mask         - logical, true = glove pixel
%
% Outputs:
%   isStain      - true if stain detected
%   bboxes       - Nx4 [x y w h] bounding boxes
%   debug        - intermediate maps

debug = struct();

if ~any(mask(:))
    isStain = false;
    bboxes  = [];
    return;
end

% ====================================================================
% 1. CONVERT TO LAB COLOUR SPACE
%    LAB separates luminance from colour — ideal for colour anomaly
%    detection regardless of the base glove colour.
% ====================================================================
imgDouble = im2double(segmentedImg);
labImg    = rgb2lab(imgDouble);

L = labImg(:,:,1);   % Luminance     0-100
A = labImg(:,:,2);   % Green-Red    -128 to +127
B = labImg(:,:,3);   % Blue-Yellow  -128 to +127

% Zero outside mask
L(~mask) = 0;
A(~mask) = 0;
B(~mask) = 0;

debug.L = mat2gray(L);
debug.A = mat2gray(A);
debug.B = mat2gray(B);

% ====================================================================
% 2. MASK EROSION
%    Remove boundary pixels to avoid glove edge responses
% ====================================================================
shortSide  = min(size(L,1), size(L,2));
diskR      = max(8, min(20, round(0.03 * shortSide)));
maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.30 * nnz(mask)
    maskEroded = imerode(mask, strel('disk', round(diskR/2)));
end
if ~any(maskEroded(:))
    maskEroded = mask;
end

L(~maskEroded) = 0;
A(~maskEroded) = 0;
B(~maskEroded) = 0;

debug.maskEroded = maskEroded;

% ====================================================================
% 3. DIRECT LAB COLOUR THRESHOLDING
%
%   WHY ALL PREVIOUS APPROACHES FAILED:
%   - Local anomaly (small window): grip texture creates anomalies
%     everywhere → whole glove lights up
%   - Local anomaly (large window): stain becomes its own local mean
%     → near-zero anomaly → 1 giant blob = whole glove
%   - Global deviation from median: stdL=20 (stain inflates std) →
%     normalised scores too similar across glove → whole glove again
%
%   THE REAL FIX: The stain has a specific absolute colour signature.
%   Don't measure relative anomaly — just look for stain-coloured pixels directly.
%
%   In LAB space, the three regions are:
%     White glove:    L > 70,  A ≈ -2 to +2,   B ≈ 0 to +8   (bright, neutral)
%     Grey background: L ≈ 50,  A ≈  0,          B ≈  0        (excluded by mask)
%     Brown stain:    L < 65,  A > +3,           B > +8        (dark + warm)
%
%   STRATEGY:
%   Step 1 — Compute glove colour baseline from the BRIGHT pixels only
%             (top 60% brightness) — these are definitely clean glove,
%             not stain. Get their median L, A, B.
%   Step 2 — A pixel is "stain-like" if it deviates from clean baseline
%             AND is sufficiently dark (L well below clean median).
%   Step 3 — Build a score map: darkness_drop × colour_shift
%             This is high only where the pixel is BOTH dark AND coloured
%             differently from the clean glove — exactly the stain.
% ====================================================================
gloveArea = nnz(mask);
winSizeUsed = 0;   % not used in this path, kept for log compatibility

% --- Determine glove type from glove median brightness ---
Lvals = L(maskEroded);
gloveMedL = median(Lvals(Lvals > 0));   % median L of glove (ignore bg zeros)
isDarkGlove = gloveMedL < 30;           % black nitrile: medL ≈ 20-28. Blue nitrile: medL ≈ 38-55 → NOT dark

fprintf('  detectStain: gloveMedL=%.1f  isDarkGlove=%d\n', gloveMedL, isDarkGlove);

% --- Compute clean glove baseline ---
% For LIGHT gloves (white/blue): clean = top 60% brightest pixels (excludes dark stain)
% For DARK gloves (black):       clean = bottom 60% darkest pixels (excludes bright stain)
if isDarkGlove
    brightnessThresh = prctile(Lvals, 60);   % keep darker 60% = clean black surface
    cleanMask = maskEroded & (L <= brightnessThresh);
else
    brightnessThresh = prctile(Lvals, 40);   % keep brighter 60% = clean white surface
    cleanMask = maskEroded & (L >= brightnessThresh);
end

if nnz(cleanMask) < 100, cleanMask = maskEroded; end

cleanL = median(L(cleanMask));
cleanA = median(A(cleanMask));
cleanB = median(B(cleanMask));

fprintf('  detectStain: clean baseline LAB = [%.1f, %.2f, %.2f]\n', cleanL, cleanA, cleanB);

% --- Build stain score map ---
% LIGHT glove: stain is DARKER than glove → look for brightness DROP
% DARK glove:  stain is BRIGHTER than glove → look for brightness RISE
if isDarkGlove
    % Black glove: white/bright stain → pixels much BRIGHTER than clean black surface
    brightRise = max(0, L - cleanL);         % high for bright stain pixels
    brightRise(~maskEroded) = 0;
    colourShift = abs(A - cleanA) + abs(B - cleanB);  % any colour deviation
    colourShift(~maskEroded) = 0;
    riseNorm    = mat2gray(brightRise);
    colourNorm  = mat2gray(colourShift);
    % AND logic: must be brighter AND colour-different from clean black
    stainScore = 0.7 * riseNorm + 0.3 * (riseNorm .* colourNorm);
    stainScore(~maskEroded) = 0;
    debug.Ldiff = mat2gray(brightRise);
else
    % Light glove: dark stain -> pixels much DARKER than clean surface
    % Special case: blue gloves have strongly negative B (cleanB ~ -40)
    % The dark stain has B closer to 0 (less blue). So B-shift is reliable.
    % BUT colourShift = max(0, B - cleanB) would fire on ANY lighter pixel.
    % Fix: use absolute L-drop as primary signal (dark stain = lower L).
    % For blue gloves, also use absolute deviation from clean colour.
    darkDrop = max(0, cleanL - L);
    darkDrop(~maskEroded) = 0;

    % Colour deviation: absolute distance in A+B from clean baseline
    % (works for brown stains on white AND dark stains on blue)
    colourDev = abs(A - cleanA) + abs(B - cleanB);
    colourDev(~maskEroded) = 0;

    darkNorm   = mat2gray(darkDrop);
    colourNorm = mat2gray(colourDev);

    % Use darkDrop as primary — colourDev as secondary AND gate
    % Only score high when BOTH dark AND colour-different from clean surface
    stainScore = 0.55 * (darkNorm .* (colourNorm > 0.3)) + 0.45 * darkNorm;
    stainScore(~maskEroded) = 0;
    debug.Ldiff = mat2gray(darkDrop);
end

debug.Adiff      = mat2gray(abs(A - cleanA));
debug.Bdiff      = mat2gray(abs(B - cleanB));
debug.pathAScore = stainScore;
debug.pathBScore = stainScore;
debug.stainScore = stainScore;

% ====================================================================
% 5. THRESHOLD
%    Otsu on stain score. Because the score is multiplicative
%    (dark × coloured), the distribution is bimodal:
%    most pixels score near 0 (clean glove), stain pixels score high.
%    Otsu naturally finds the valley between these two modes.
% ====================================================================
maskedScore = stainScore(maskEroded);
if isempty(maskedScore) || max(maskedScore) < 1e-4
    isStain = false;
    bboxes  = [];
    debug.stainMask    = false(size(mask));
    debug.filteredMask = false(size(mask));
    debug.isScratch    = false;
    debug.threshValue  = 0;
    debug.componentCount = 0;
    debug.bboxCount    = 0;
    fprintf('=== detectStain: no signal found ===\n');
    return;
end

otsu           = graythresh(maskedScore);
thresholdValue = otsu * (isDarkGlove * 2.50 + (~isDarkGlove) * 0.65);
stainMask      = (stainScore > thresholdValue) & maskEroded;
stainMask      = bwareaopen(stainMask, 20);

% Suppress boundary band
boundaryBand = imdilate(mask & ~maskEroded, strel('disk', 3));
stainMask(boundaryBand) = 0;

debug.stainMask   = stainMask;
debug.threshValue = thresholdValue;

% ====================================================================
% 6. MORPHOLOGICAL CLOSING
%    disk(8): stains are blobs with possible small holes.
%    Larger than scratch closing because stains cover more area.
% ====================================================================
closed = imclose(stainMask, strel('disk', 8));
closed = imfill(closed, 'holes');
closed = bwareaopen(closed, 50);
closed = closed & maskEroded;

debug.closedMask = closed;

% ====================================================================
% 7. REGION FILTERING
%
%   Area:
%     min = max(200, 0.01% of glove) — stains are visible marks
%     max = 20% of glove — stains don't cover the whole glove
%
%   Eccentricity < 0.97
%     Stains are compact blobs, not thin lines.
%     This distinguishes stains from fold lines or scratches.
%     (opposite of scratch filter which requires HIGH eccentricity)
%
%   Solidity > 0.40
%     Stains are relatively solid filled regions.
%     Very low solidity = fragmented/noisy region, not a stain.
% ====================================================================
if isDarkGlove
    minArea = max(1000, 0.005 * gloveArea);   % dark glove: real stain must be large (0.5% of glove)
else
    minArea = max(200, 0.0001 * gloveArea);
end
% NO maxArea cap — a stain can be any size up to the full glove.
% The old cap (0.40 * gloveArea) was rejecting the large stain blob
% (1.6M px on a 1.78M px glove) as "too large". Removed entirely.
% Quality is controlled by eccentricity and solidity instead.
maxArea = gloveArea;   % effectively no upper limit
% Tighter cap for dark gloves — stain is localised
if isDarkGlove
    maxArea = 0.35 * gloveArea;
end

stats = regionprops(closed, 'Area', 'BoundingBox', ...
    'Eccentricity', 'Solidity', 'PixelIdxList');

filteredMask = false(size(closed));
bboxes       = [];
nSmall = 0; nLarge = 0; nThin = 0; nLowSol = 0; nOk = 0;

for i = 1:numel(stats)
    a   = stats(i).Area;
    ecc = stats(i).Eccentricity;
    sol = stats(i).Solidity;
    bb  = stats(i).BoundingBox;

    if a < minArea
        nSmall = nSmall+1; continue
    end
    % nLarge counter kept for logging but no longer filters anything
    if a > maxArea
        nLarge = nLarge+1; continue
    end
    if ecc > 0.97
        nThin = nThin+1; continue   % too elongated — not a stain blob
    end
    if sol < 0.35
        nLowSol = nLowSol+1; continue   % too fragmented
    end

    filteredMask(stats(i).PixelIdxList) = true;
    bboxes = [bboxes; bb]; %#ok<AGROW>
    nOk    = nOk+1;
end

% ====================================================================
% 8. BBOX SELECTION
%    For dark gloves: merge all accepted blobs into one encompassing box.
%    White stains on black gloves are often fragmented into several blobs
%    that together form the stain — merging gives the full stain extent.
%    For light gloves: keep only the largest blob bbox (avoids spanning box
%    from scattered small dark spots).
% ====================================================================
if size(bboxes,1) > 1
    if isDarkGlove
        % Dark glove: merge all accepted blobs into one encompassing bbox
        x1 = min(bboxes(:,1));
        y1 = min(bboxes(:,2));
        x2 = max(bboxes(:,1) + bboxes(:,3));
        y2 = max(bboxes(:,2) + bboxes(:,4));
        bboxes = [x1, y1, x2-x1, y2-y1];
    else
        % Light glove: keep only the largest blob
        bboxAreas = bboxes(:,3) .* bboxes(:,4);
        [~, sortIdx] = sort(bboxAreas, 'descend');
        bboxes = bboxes(sortIdx(1), :);
    end
end

isStain = ~isempty(bboxes);

% ====================================================================
% Console report
% ====================================================================
fprintf('=== detectStain report ===\n');
fprintf('  Glove area    : %d px\n',  gloveArea);
fprintf('  PathB window  : %d px\n',  winSizeUsed);
fprintf('  Otsu          : %.4f  threshold: %.4f (x0.70)\n', otsu, thresholdValue);
fprintf('  minArea       : %.0f px\n', minArea);
fprintf('  maxArea       : %.0f px\n', maxArea);
fprintf('  Components    : %d  (small=%d large=%d thin=%d lowSol=%d ok=%d)\n', ...
        numel(stats), nSmall, nLarge, nThin, nLowSol, nOk);
fprintf('  Boxes output  : %d\n', size(bboxes,1));
if ~isempty(stats)
    areas = sort([stats.Area],'descend');
    fprintf('  Top areas     : ');
    fprintf('%d ', areas(1:min(6,end)));
    fprintf('\n');
end
fprintf('  Stain found   : %d\n', isStain);

% ====================================================================
% Debug struct
% ====================================================================
debug.filteredMask   = filteredMask;
debug.isStain        = isStain;
debug.threshValue    = thresholdValue;
debug.componentCount = numel(stats);
debug.bboxCount      = size(bboxes,1);
debug.minAreaUsed    = minArea;
debug.maxAreaUsed    = maxArea;
debug.winSize        = winSizeUsed;
gloveArea2           = max(gloveArea,1);
debug.stainDensity   = nnz(filteredMask) / gloveArea2;

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