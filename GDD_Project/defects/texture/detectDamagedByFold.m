function [isFoldDamage, bboxes, debug] = detectDamagedByFold(segmentedImg, mask)
% detectDamagedByFold  Detect fold damage (hard crease lines) on gloves.
%
% Changelog (this version):
%   - Dominant line proximity threshold tightened: 50 -> 20px
%     Lines with midpoint Y more than 20px from the main fold are excluded.
%     This keeps the vertical box extent very tight around the crease.
%   - yPad reduced: 15 -> 8px for thinner bounding box
%   - Bounding box clamped to glove bounding box so it never extends
%     beyond the glove boundary into background.

debug = struct();

if ~any(mask(:))
    isFoldDamage = false;
    bboxes       = [];
    return;
end

% ====================================================================
% 1. GRAYSCALE + MASK EROSION
% ====================================================================
gray = im2double(rgb2gray(segmentedImg));

shortSide  = min(size(gray,1), size(gray,2));
diskR      = max(8, min(20, round(0.03 * shortSide)));
maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.30 * nnz(mask)
    maskEroded = imerode(mask, strel('disk', round(diskR/2)));
end
if ~any(maskEroded(:))
    maskEroded = mask;
end

gray(~maskEroded) = 0;
gray = imgaussfilt(gray, 0.8);

% ====================================================================
% 2. TOP-HAT ENHANCEMENT
% ====================================================================
th  = imtophat(gray, strel('disk', 20));
th  = mat2gray(th);
th(~maskEroded) = 0;

grayEnhanced = imadjust(gray);
grayEnhanced(~maskEroded) = 0;

debug.topHat     = th;
debug.maskEroded = maskEroded;

% ====================================================================
% 3. EDGE DETECTION
% ====================================================================
thMasked = th(maskEroded);
if isempty(thMasked) || max(thMasked) < 1e-4
    edgeHigh = 0.1;
else
    edgeHigh = prctile(thMasked, 85);
end
edgeLow = edgeHigh * 0.25;

edges1 = edge(th,           'canny', [edgeLow,       edgeHigh]);
edges2 = edge(grayEnhanced, 'canny', [edgeLow * 0.5, edgeHigh * 0.7]);
edges  = (edges1 | edges2) & maskEroded;

debug.edgeMap = edges;

% ====================================================================
% 4. GLOVE DIMENSIONS
% ====================================================================
gloveProps = regionprops(mask, 'BoundingBox');
if isempty(gloveProps)
    isFoldDamage = false;
    bboxes       = [];
    return;
end
gloveBB    = gloveProps(1).BoundingBox;
gloveWidth = gloveBB(3);

% ====================================================================
% 5. HOUGH LINE TRANSFORM
% ====================================================================
minLineLen = round(0.30 * gloveWidth);
fillGap    = 40;

[H, theta, rho] = hough(edges);
peakThresh      = ceil(0.15 * max(H(:)));
peaks           = houghpeaks(H, 30, 'Threshold', peakThresh);

fprintf('=== detectDamagedByFold report ===\n');
fprintf('  Glove width      : %.0f px\n', gloveWidth);
fprintf('  Min line length  : %.0f px\n', minLineLen);

if isempty(peaks)
    isFoldDamage = false;
    bboxes       = [];
    debug.houghLines = [];
    debug.lineMap    = repmat(mat2gray(th), [1,1,3]);
    debug.foldLines  = [];
    fprintf('  Hough lines found: 0\n');
    fprintf('  Fold damage      : 0\n');
    return;
end

lines = houghlines(edges, theta, rho, peaks, ...
    'FillGap', fillGap, 'MinLength', minLineLen);

fprintf('  Hough lines found: %d\n', length(lines));

% ====================================================================
% 6. FILTER LINES
% ====================================================================
% Fold line must span at least 50% of glove width
% A real fold crease runs across the full palm — random wrinkle/scratch 
% lines are shorter. 50% threshold eliminates most false positives.
foldLines = filterLines(lines, mask, maskEroded, gloveWidth, 0.50, gloveBB);
if isempty(foldLines)
    foldLines = filterLines(lines, mask, maskEroded, gloveWidth, 0.35, gloveBB);
end

fprintf('  Fold lines (raw) : %d\n', length(foldLines));

% ====================================================================
% 7. DOMINANT LINE SELECTION
%    Find longest line (main fold crease).
%    Keep only lines within 20px vertically — very tight proximity.
% ====================================================================
if ~isempty(foldLines)
    lineLengths = arrayfun(@(l) norm(l.point2 - l.point1), foldLines);
    [~, maxIdx] = max(lineLengths);
    mainLine    = foldLines(maxIdx);
    mainMidY    = (mainLine.point1(2) + mainLine.point2(2)) / 2;

    proximityThresh = 20;   % tightened from 50px
    nearLines = [];
    for i = 1:length(foldLines)
        midY = (foldLines(i).point1(2) + foldLines(i).point2(2)) / 2;
        if abs(midY - mainMidY) <= proximityThresh
            nearLines = [nearLines, foldLines(i)]; %#ok<AGROW>
        end
    end
    foldLines = nearLines;
end

fprintf('  Fold lines kept  : %d\n', length(foldLines));

% ====================================================================
% 8. TIGHT BOUNDING BOXES
%    xPad = 8px, yPad = 8px — both small to keep box thin and tight
%    Then clamp to glove bounding box so box never exits glove region
% ====================================================================
yPad   = 8;
bboxes = [];

for i = 1:length(foldLines)
    p1 = foldLines(i).point1;
    p2 = foldLines(i).point2;

    % For fold crease: horizontal line spans full glove width at that row
    % Use glove mask extent at the crease Y position for tight horizontal fit
    creaseY = round((p1(2) + p2(2)) / 2);
    creaseY = max(1, min(size(mask,1), creaseY));
    gloveRowPixels = find(mask(creaseY, :));
    if ~isempty(gloveRowPixels)
        x1 = max(1,            gloveRowPixels(1));
        x2 = min(size(mask,2), gloveRowPixels(end));
    else
        x1 = max(1,            min(p1(1),p2(1)));
        x2 = min(size(mask,2), max(p1(1),p2(1)));
    end
    % Vertical: scan rows near the detected line to find the darkest row
    % That row IS the fold crease — use it as the true centre
    gloveHeight  = gloveBB(4);
    searchPad    = round(0.04 * gloveHeight);  % search ±4% of glove height
    searchTop    = max(1,            creaseY - searchPad);
    searchBot    = min(size(mask,1), creaseY + searchPad);
    rowMeans     = zeros(searchBot - searchTop + 1, 1);
    for ri = searchTop:searchBot
        rowPx = gray(ri, mask(ri,:));
        if ~isempty(rowPx)
            rowMeans(ri - searchTop + 1) = mean(rowPx);
        else
            rowMeans(ri - searchTop + 1) = 1;
        end
    end
    [~, minIdx] = min(rowMeans);
    trueCentreY = searchTop + minIdx - 1;
    yPadAuto    = max(5, round(0.020 * gloveHeight));
    y1 = max(1,            trueCentreY - yPadAuto);
    y2 = min(size(mask,1), trueCentreY + yPadAuto);

    bboxes = [bboxes; x1, y1, x2-x1, y2-y1]; %#ok<AGROW>
end

% Merge only directly touching fragments
if size(bboxes,1) > 1
    bboxes = mergeBoxes(bboxes, 5);
end

% ====================================================================
% 9. CLAMP BOXES TO ACTUAL MASK WIDTH AT FOLD LINE Y ROW
%    The glove bounding box includes outstretched fingers which are
%    wider than the palm where the fold line sits. Instead of clamping
%    to the full glove bbox, find exactly which columns the mask covers
%    at the fold line's Y position and clamp to those columns.
%    This prevents the box extending into the finger region or background.
% ====================================================================
if ~isempty(bboxes)
    for i = 1:size(bboxes,1)
        % Get the Y centre of this box
        foldY = round(bboxes(i,2) + bboxes(i,4)/2);
        foldY = max(1, min(size(mask,1), foldY));

        % Find which columns the ERODED mask covers at this Y row.
        % Using eroded mask (not full mask) excludes thin finger boundary
        % pixels that extend the row width beyond the palm area.
        maskRow = maskEroded(foldY, :);
        cols    = find(maskRow);

        if ~isempty(cols)
            rowX1 = cols(1);
            rowX2 = cols(end);

            % Clamp box X to mask extent at this row
            bx1 = max(bboxes(i,1), rowX1);
            bx2 = min(bboxes(i,1) + bboxes(i,3), rowX2);

            % Clamp box Y to glove bounding box
            by1 = max(bboxes(i,2), gloveBB(2));
            by2 = min(bboxes(i,2) + bboxes(i,4), gloveBB(2) + gloveBB(4));

            bboxes(i,:) = [bx1, by1, bx2-bx1, by2-by1];
        end
    end

    % Remove degenerate boxes
    valid  = bboxes(:,3) > 5 & bboxes(:,4) > 5;
    bboxes = bboxes(valid,:);
end

isFoldDamage = ~isempty(bboxes);

fprintf('  Bboxes output    : %d\n', size(bboxes,1));
fprintf('  Fold damage      : %d\n', isFoldDamage);

% ====================================================================
% 10. DEBUG LINE MAP
% ====================================================================
lineMap = repmat(mat2gray(th), [1,1,3]);
for i = 1:length(foldLines)
    p1   = foldLines(i).point1;
    p2   = foldLines(i).point2;
    nPts = max(abs(p2-p1)) + 1;
    xs   = max(1, min(size(mask,2), round(linspace(p1(1),p2(1),nPts))));
    ys   = max(1, min(size(mask,1), round(linspace(p1(2),p2(2),nPts))));
    idx  = sub2ind(size(th), ys, xs);
    r = lineMap(:,:,1); r(idx) = 1;
    g = lineMap(:,:,2); g(idx) = 0;
    b = lineMap(:,:,3); b(idx) = 0;
    lineMap = cat(3, r, g, b);
end

debug.lineMap      = lineMap;
debug.foldLines    = foldLines;
debug.houghLines   = lines;
debug.isFoldDamage = isFoldDamage;

end


% ------------------------------------------------------------------
function foldLines = filterLines(lines, mask, maskEroded, gloveWidth, minFrac, gloveBB)
minLen    = minFrac * gloveWidth;
foldLines = [];

% Palm region: fold crease is in middle 35-80% of glove height
% Finger boundaries are in top 35%, cuff is in bottom 20%
palmTop = 0; palmBot = size(mask,1);
if nargin >= 6 && ~isempty(gloveBB)
    palmTop = round(gloveBB(2) + gloveBB(4) * 0.35);
    palmBot = round(gloveBB(2) + gloveBB(4) * 0.80);
end

for i = 1:length(lines)
    p1  = lines(i).point1;
    p2  = lines(i).point2;
    len = norm(p2 - p1);
    if len < minLen, continue; end

    dx       = p2(1) - p1(1);
    dy       = p2(2) - p1(2);
    angleDeg = abs(atan2d(dy, dx));
    if angleDeg > 20 && angleDeg < 160, continue; end

    % Line midpoint must be in palm region (not finger or cuff area)
    midY = (p1(2) + p2(2)) / 2;
    if midY < palmTop || midY > palmBot, continue; end

    nSamples   = 10;
    xs = round(linspace(p1(1), p2(1), nSamples));
    ys = round(linspace(p1(2), p2(2), nSamples));
    xs = max(1, min(size(mask,2), xs));
    ys = max(1, min(size(mask,1), ys));
    inMaskFrac = mean(maskEroded(sub2ind(size(mask), ys, xs)));
    if inMaskFrac < 0.60, continue; end

    foldLines = [foldLines, lines(i)]; %#ok<AGROW>
end
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
            bi = [b(1)-gap, b(2)-gap, b(3)+2*gap, b(4)+2*gap];
            bj = [bboxes(j,1)-gap, bboxes(j,2)-gap, ...
                  bboxes(j,3)+2*gap, bboxes(j,4)+2*gap];
            ox = max(bi(1),bj(1)) < min(bi(1)+bi(3), bj(1)+bj(3));
            oy = max(bi(2),bj(2)) < min(bi(2)+bi(4), bj(2)+bj(4));
            if ox && oy
                x1 = min(b(1), bboxes(j,1));
                y1 = min(b(2), bboxes(j,2));
                x2 = max(b(1)+b(3), bboxes(j,1)+bboxes(j,3));
                y2 = max(b(2)+b(4), bboxes(j,2)+bboxes(j,4));
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