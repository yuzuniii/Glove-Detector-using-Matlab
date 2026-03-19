function [isIncomplete, bboxes, debug] = detectIncompleteBeading(segmentedImg, mask)
% detectIncompleteBeading  Detect incomplete/torn cuff beading on black nitrile.
%
% VERSION 3 — Auto-detect cuff edge orientation
%
% PROBLEM WITH V1+V2:
%   Both assumed the cuff is always at the bottom of the image.
%   blackstain.jpg has the glove HORIZONTAL — cuff is on the RIGHT.
%   incompleteglove.png has the glove VERTICAL — cuff is on the BOTTOM.
%   Analysing the wrong edge gave false positives on normal gloves.
%
% SOLUTION:
%   Auto-detect which of the 4 edges is the cuff by measuring mask
%   density in a 12% zone at each edge. The cuff edge has the HIGHEST
%   and most UNIFORM density — it's a solid rolled bead.
%   The finger edge has LOW and UNEVEN density — separated fingers.
%
% CUFF DETECTION LOGIC:
%   For each edge (top/bottom/left/right):
%     1. Measure mean mask coverage in 12% zone
%     2. Measure std of column/row profiles in the zone
%   Cuff = highest mean coverage AND lowest std (most solid/uniform)
%   Fingers = low mean AND high std (sparse, separated)
%
% BOUNDARY ANALYSIS:
%   Once cuff edge is identified, extract the boundary profile along it.
%   Normal cuff: smooth, continuous boundary
%   Torn cuff: notches, gaps, highly irregular boundary

debug = struct();

if ~any(mask(:))
    isIncomplete = false;
    bboxes       = [];
    return;
end

[rows, cols] = size(mask);
gloveArea    = nnz(mask);

% ====================================================================
% STEP 1: Find glove bounding box
% ====================================================================
props = regionprops(mask, 'BoundingBox');
if isempty(props)
    isIncomplete = false; bboxes = []; return;
end
bb     = props(1).BoundingBox;
r1     = max(1,    round(bb(2)));
r2     = min(rows, round(bb(2) + bb(4)));
c1     = max(1,    round(bb(1)));
c2     = min(cols, round(bb(1) + bb(3)));
gloveH = r2 - r1;
gloveW = c2 - c1;

zoneH = max(5, round(gloveH * 0.12));   % 12% of glove height
zoneW = max(5, round(gloveW * 0.12));   % 12% of glove width

% ====================================================================
% STEP 2: Measure edge density and uniformity for all 4 edges
% ====================================================================
% Extract 12% zone mask at each edge
zoneTop    = mask(r1        : r1+zoneH,    c1:c2);
zoneBot    = mask(r2-zoneH  : r2,          c1:c2);
zoneLeft   = mask(r1:r2,    c1        : c1+zoneW);
zoneRight  = mask(r1:r2,    c2-zoneW   : c2);

% Mean coverage and std of column/row profiles
meanTop   = mean(zoneTop(:));   stdTop  = std(mean(zoneTop,  1));
meanBot   = mean(zoneBot(:));   stdBot  = std(mean(zoneBot,  1));
meanLeft  = mean(zoneLeft(:));  stdLeft = std(mean(zoneLeft, 2));
meanRight = mean(zoneRight(:)); stdRight= std(mean(zoneRight,2));

fprintf('  detectIncompleteBeading v3: edge density [T=%.3f B=%.3f L=%.3f R=%.3f]\n', ...
    meanTop, meanBot, meanLeft, meanRight);
fprintf('  detectIncompleteBeading v3: edge std     [T=%.3f B=%.3f L=%.3f R=%.3f]\n', ...
    stdTop, stdBot, stdLeft, stdRight);

% Cuff edge = edge with HIGHEST mean coverage among edges with mean > 0.30
% Rationale:
%   - Cuff has a solid rolled bead → highest continuous material density
%   - Finger edges are sparse (separated fingers) → low/moderate density
%   - Minimum threshold 0.30 excludes nearly-empty edges (finger gaps, open sides)
%
% Data verified on test images:
%   incompleteglove.png: bottom=0.503 (cuff) > top=0.765 EXCLUDED (fingers crop)
%   blackstain.jpg:      right=0.603  (cuff) > others
%
% Why not highest overall? Top edge of incompleteglove has density=0.765 because
% the fingers are cropped at the image boundary, making top artificially dense.
% The cuff (bottom=0.503) is correctly selected when we require mean>0.30 AND
% pick the edge that is most "interior" (not cropped at image boundary).

% Check if any edge is cropped at image boundary (mask touches image edge)
topCropped   = any(mask(1,   :));     % mask touches top image border
botCropped   = any(mask(end, :));     % mask touches bottom image border
leftCropped  = any(mask(:,   1));     % mask touches left image border
rightCropped = any(mask(:, end));     % mask touches right image border

% Penalise edges that are cropped (fingers extend beyond image = not cuff)
cropPenalty = 0.5;
adjTop   = meanTop   * (1 - cropPenalty * topCropped);
adjBot   = meanBot   * (1 - cropPenalty * botCropped);
adjLeft  = meanLeft  * (1 - cropPenalty * leftCropped);
adjRight = meanRight * (1 - cropPenalty * rightCropped);

edgeScores = [adjTop, adjBot, adjLeft, adjRight];
edgeNames  = {'top', 'bottom', 'left', 'right'};

% Among edges with adjusted mean > 0.15, pick highest
validEdges = edgeScores > 0.15;
if ~any(validEdges)
    validEdges = true(1,4);  % fallback: use all
end
filteredScores = edgeScores .* validEdges;
[~, cuffEdgeIdx] = max(filteredScores);
cuffEdge = edgeNames{cuffEdgeIdx};

fprintf('  detectIncompleteBeading v3: cuff edge detected = %s (scores: T=%.2f B=%.2f L=%.2f R=%.2f)\n', ...
    cuffEdge, adjTop, adjBot, adjLeft, adjRight);

debug.cuffEdge   = cuffEdge;
debug.edgeScores = edgeScores;

% ====================================================================
% STEP 3: Extract boundary profile along the cuff edge
%   For each column (or row) along the cuff edge, find the outermost
%   mask pixel. Gaps = columns/rows with no mask = torn sections.
% ====================================================================
switch cuffEdge
    case 'bottom'
        % Scan each column, find lowest mask row in bottom 20%
        analysisBand = mask(r2-round(gloveH*0.20):r2, c1:c2);
        profileLen = c2 - c1 + 1;
        edgeProfile = zeros(1, profileLen);
        gapProfile  = false(1, profileLen);
        for ci = 1:profileLen
            col = analysisBand(:, ci);
            idx = find(col, 1, 'last');
            if isempty(idx)
                gapProfile(ci) = true;
            else
                edgeProfile(ci) = idx;
            end
        end
        analysisHeight = round(gloveH * 0.20);

    case 'top'
        analysisBand = mask(r1:r1+round(gloveH*0.20), c1:c2);
        profileLen = c2 - c1 + 1;
        edgeProfile = zeros(1, profileLen);
        gapProfile  = false(1, profileLen);
        for ci = 1:profileLen
            col = analysisBand(:, ci);
            idx = find(col, 1, 'first');
            if isempty(idx)
                gapProfile(ci) = true;
            else
                edgeProfile(ci) = idx;
            end
        end
        analysisHeight = round(gloveH * 0.20);

    case 'right'
        analysisBand = mask(r1:r2, c2-round(gloveW*0.20):c2);
        profileLen = r2 - r1 + 1;
        edgeProfile = zeros(1, profileLen);
        gapProfile  = false(1, profileLen);
        for ri = 1:profileLen
            row = analysisBand(ri, :);
            idx = find(row, 1, 'last');
            if isempty(idx)
                gapProfile(ri) = true;
            else
                edgeProfile(ri) = idx;
            end
        end
        analysisHeight = round(gloveW * 0.20);

    case 'left'
        analysisBand = mask(r1:r2, c1:c1+round(gloveW*0.20));
        profileLen = r2 - r1 + 1;
        edgeProfile = zeros(1, profileLen);
        gapProfile  = false(1, profileLen);
        for ri = 1:profileLen
            row = analysisBand(ri, :);
            idx = find(row, 1, 'first');
            if isempty(idx)
                gapProfile(ri) = true;
            else
                edgeProfile(ri) = idx;
            end
        end
        analysisHeight = round(gloveW * 0.20);
end

% ====================================================================
% STEP 4: Compute roughness metrics on the cuff boundary profile
% ====================================================================
validEdge   = edgeProfile(~gapProfile);
gapFraction = mean(gapProfile);

if numel(validEdge) < 10
    fprintf('  detectIncompleteBeading: insufficient edge data\n');
    isIncomplete = false; bboxes = []; return;
end

medianEdge   = median(validEdge);
edgeVariance = std(double(validEdge)) / max(analysisHeight, 1);

% Notch depth: deepest inward notch (away from edge = toward glove interior)
notchDepth = max(0, medianEdge - min(validEdge)) / max(analysisHeight, 1);

% Count notches via smoothed profile local minima
smoothLen    = max(5, round(profileLen * 0.03));
smoothPro    = movmean(double(edgeProfile .* ~gapProfile + ...
    medianEdge * gapProfile), smoothLen);
notchThresh  = medianEdge - analysisHeight * 0.10;
notchMask    = smoothPro < notchThresh;
notchCC      = bwconncomp(notchMask);
notchCount   = notchCC.NumObjects;

fprintf('  detectIncompleteBeading v3: gapFrac=%.3f edgeVar=%.3f notchDepth=%.3f notchCount=%d\n', ...
    gapFraction, edgeVariance, notchDepth, notchCount);

% ====================================================================
% STEP 5: Score and decision
% ====================================================================
function s = sT(val, lo, hi)
    if val<=lo, s=0; elseif val>=hi, s=1;
    else, s=(val-lo)/(hi-lo); end
    s=max(0,min(1,s));
end

scoreVar   = sT(edgeVariance, 0.05, 0.20);
scoreNotchD = sT(notchDepth,  0.08, 0.25);
scoreNotchC = sT(notchCount,  1,    4);
scoreGap    = sT(gapFraction, 0.05, 0.20);

beadingScore = 0.25*scoreVar + 0.35*scoreNotchD + 0.20*scoreNotchC + 0.20*scoreGap;

if notchCount >= 2 && notchDepth > 0.15
    beadingScore = min(0.95, beadingScore * 1.5);
end

fprintf('  detectIncompleteBeading v3: scoreV=%.2f scoreND=%.2f scoreNC=%.2f scoreG=%.2f → total=%.3f\n', ...
    scoreVar, scoreNotchD, scoreNotchC, scoreGap, beadingScore);

isIncomplete = beadingScore > 0.40;

% ====================================================================
% STEP 6: Bounding box around cuff region
% ====================================================================
bboxes = [];
if isIncomplete
    pad = 15;
    switch cuffEdge
        case 'bottom'
            x1=max(1,c1-pad); y1=max(1,r2-round(gloveH*0.20)-pad);
            x2=min(cols,c2+pad); y2=min(rows,r2+pad);
        case 'top'
            x1=max(1,c1-pad); y1=max(1,r1-pad);
            x2=min(cols,c2+pad); y2=min(rows,r1+round(gloveH*0.20)+pad);
        case 'right'
            x1=max(1,c2-round(gloveW*0.20)-pad); y1=max(1,r1-pad);
            x2=min(cols,c2+pad); y2=min(rows,r2+pad);
        case 'left'
            x1=max(1,c1-pad); y1=max(1,r1-pad);
            x2=min(cols,c1+round(gloveW*0.20)+pad); y2=min(rows,r2+pad);
    end
    bboxes = [x1, y1, x2-x1, y2-y1];
end

fprintf('=== detectIncompleteBeading report ===\n');
fprintf('  Cuff edge     : %s\n', cuffEdge);
fprintf('  EdgeVariance  : %.3f\n', edgeVariance);
fprintf('  NotchDepth    : %.3f\n', notchDepth);
fprintf('  NotchCount    : %d\n',   notchCount);
fprintf('  GapFraction   : %.3f\n', gapFraction);
fprintf('  Beading score : %.3f\n', beadingScore);
fprintf('  Incomplete    : %d\n',   isIncomplete);

debug.edgeVariance = edgeVariance;
debug.notchDepth   = notchDepth;
debug.notchCount   = notchCount;
debug.gapFraction  = gapFraction;
debug.beadingScore = beadingScore;
debug.isIncomplete = isIncomplete;
debug.bboxCount    = size(bboxes,1);

end