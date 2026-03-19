function [isScratch, bboxes, debug] = detectScratch(segmentedImg, mask)
% detectScratch  Detect scratch marks on blue nitrile gloves.
%
% APPROACH v7 — DIRECTIONAL MORPHOLOGICAL FILTERING:
%
%   CORE INSIGHT:
%   The grip texture creates ISOLATED dark dots/diamonds (~20-30px each).
%   The scratch creates a CONTINUOUS dark LINE across multiple texture cells.
%
%   A linear structuring element (SE) at the scratch angle will:
%   - PRESERVE the scratch (continuous line fits inside linear SE)
%   - SUPPRESS texture dots (isolated, don't fit inside linear SE)
%
%   Algorithm:
%   1. Compute blackhat(gray, disk(25)) — highlights dark features
%   2. Apply morphological opening with line SE at 8 angles (0-160 deg)
%   3. Take max response across all angles — this is the "linear" map
%   4. Threshold and run Hough on this cleaned map
%
%   The directional filter makes the scratch the DOMINANT feature
%   regardless of whether it is lighter or darker than texture.

debug = struct();

if ~any(mask(:))
    isScratch = false; bboxes = []; return;
end

gray = im2double(rgb2gray(segmentedImg));
[rows, cols] = size(mask);
gloveArea = nnz(mask);

% ====================================================================
% 1. MASK EROSION + PALM REGION
% ====================================================================
shortSide  = min(rows, cols);
diskR      = max(8, min(20, round(0.03 * shortSide)));
maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.30 * nnz(mask)
    maskEroded = imerode(mask, strel('disk', round(diskR/2)));
end
if ~any(maskEroded(:)), maskEroded = mask; end
gloveArea = nnz(mask);

gloveProps = regionprops(mask, 'BoundingBox');
palmMask   = true(rows, cols);
gloveWidth = cols;
if ~isempty(gloveProps)
    gBB        = gloveProps(1).BoundingBox;
    gloveWidth = gBB(3);
    % Vertical: middle 40-80% of glove height (palm, not fingers or cuff)
    palmTop    = max(1,    round(gBB(2) + gBB(4) * 0.40));
    palmBot    = min(rows, round(gBB(2) + gBB(4) * 0.80));
    % Horizontal: centre 20-80% of glove width (palm, not thumb side edges)
    palmLeft   = max(1,    round(gBB(1) + gBB(3) * 0.20));
    palmRight  = min(cols, round(gBB(1) + gBB(3) * 0.80));
    palmMask   = false(rows, cols);
    palmMask(palmTop:palmBot, palmLeft:palmRight) = true;
end

% Fill outside glove with mean to avoid boundary artifacts
grayFilled = gray;
grayFilled(~maskEroded) = mean(gray(maskEroded));

% ====================================================================
% 2. BLACKHAT TRANSFORM
%    Highlights features DARKER than local background
%    disk(25): background estimated over 50px neighbourhood
%    Texture dots (~20px): partially suppressed
%    Scratch line (~5px wide, 300px long): highlighted
% ====================================================================
diskSize = max(15, round(0.015 * sqrt(gloveArea)));
bth = imbothat(grayFilled, strel('disk', diskSize));
bth(~maskEroded) = 0;

debug.botHatMap = mat2gray(bth);

% ====================================================================
% 3. DIRECTIONAL LINEAR OPENING
%    Apply opening with line SE at 8 angles (0,22.5,45,67.5,90,112.5,135,157.5)
%    Opening with line SE: PRESERVES features aligned with SE, removes others
%    Take MAX across all angles = "most linear" map
%    Scratch is preserved at its angle; texture dots are removed at all angles
% ====================================================================
lineLen   = max(30, round(0.03 * gloveWidth));  % line length ~ 3% of glove width
angles    = [0, 22, 45, 67, 90, 112, 135, 157];
dirMap    = zeros(rows, cols);

for ang = angles
    se       = strel('line', lineLen, ang);
    opened   = imopen(bth, se);
    dirMap   = max(dirMap, opened);
end

dirMap(~maskEroded) = 0;
dirMap(~palmMask)   = 0;

debug.normBotHat = mat2gray(dirMap);
debug.gradMap    = mat2gray(dirMap);
debug.darkMap    = mat2gray(dirMap);

% ====================================================================
% 4. THRESHOLD + CANNY EDGES ON DIRECTIONAL MAP
% ====================================================================
palmPixels = dirMap(maskEroded & palmMask);
if isempty(palmPixels) || max(palmPixels) < 1e-6
    isScratch = false; bboxes = [];
    debug = fillDebug(debug, rows, cols, gloveArea);
    fprintf('=== detectScratch: no signal ===\n  Scratch found : 0\n');
    return;
end

% Threshold: keep top 5% of directional response
thresh = prctile(palmPixels(palmPixels > 0), 95);
scratchBin = (dirMap > thresh) & maskEroded & palmMask;
scratchBin = bwareaopen(scratchBin, 10);

debug.scratchMask = scratchBin;
debug.threshValue = thresh;

% ====================================================================
% 5. HOUGH WITH LARGE FILLGAP — find longest continuous scratch line
%
%   KEY INSIGHT: The scratch is ONE continuous mark but the directional
%   map fragments it at grip-dot locations. Using a large FillGap (50px)
%   bridges these fragments into one long line.
%   Grip texture: fragments stay short even with large FillGap because
%   they are spatially scattered, not collinear.
%   Scratch: collinear fragments merge into one long line >> grip lines.
% ====================================================================
minLineLen = max(80, round(0.08 * gloveWidth));  % raise min to 8% of glove width
[H, theta, rho] = hough(scratchBin);
peakThresh = ceil(0.10 * max(H(:)));
peaks = houghpeaks(H, 10, 'Threshold', peakThresh);

fprintf('=== detectScratch report ===\n');
fprintf('  Line SE length: %d px\n', lineLen);
fprintf('  Min line len  : %d px\n', minLineLen);

if isempty(peaks)
    isScratch = false; bboxes = [];
    debug = fillDebug(debug, rows, cols, gloveArea);
    fprintf('  No Hough peaks\n  Scratch found : 0\n');
    return;
end

% Large FillGap bridges scratch fragments separated by grip dots
lines = houghlines(scratchBin, theta, rho, peaks, ...
    'FillGap', 50, 'MinLength', minLineLen);
fprintf('  Hough lines   : %d\n', length(lines));

% ====================================================================
% 6. FILTER: diagonal + inside mask
%    Then keep ONLY the single longest line.
%    A real scratch = one dominant long line.
%    Grip texture = many short similar lines (none long enough after
%    raising minLineLen to 8% of glove width).
% ====================================================================
scratchLines = [];
for i = 1:length(lines)
    p1 = lines(i).point1; p2 = lines(i).point2;
    dx = p2(1)-p1(1); dy = p2(2)-p1(2);
    ang = abs(atan2d(dy, dx));
    if ~((ang>=25 && ang<=80)||(ang>=100 && ang<=155))
        continue;
    end
    nS=8;
    xs=max(1,min(cols,round(linspace(p1(1),p2(1),nS))));
    ys=max(1,min(rows,round(linspace(p1(2),p2(2),nS))));
    if mean(maskEroded(sub2ind([rows,cols],ys,xs))) < 0.60, continue; end
    scratchLines = [scratchLines, lines(i)]; %#ok<AGROW>
end
fprintf('  Diagonal lines: %d\n', length(scratchLines));

if isempty(scratchLines)
    isScratch = false; bboxes = [];
    debug = fillDebug(debug, rows, cols, gloveArea);
    fprintf('  No diagonal lines found\n  Scratch found : 0\n');
    return;
end

% Keep only the single longest line
lineLengths = arrayfun(@(l) norm(double(l.point2)-double(l.point1)), scratchLines);
[maxLen, maxIdx] = max(lineLengths);
fprintf('  Longest line  : %.0f px\n', maxLen);
scratchLines = scratchLines(maxIdx);

% ====================================================================
% 7. TIGHT BBOX from Canny edge pixels near the line
% ====================================================================
bboxes = [];
scratchMaskOut = false(rows, cols);

for i = 1:length(scratchLines)
    p1 = scratchLines(i).point1; p2 = scratchLines(i).point2;
    lineVec = double(p2-p1); lineLen2 = max(norm(lineVec),1);
    lineDir = lineVec/lineLen2; anchor = double(p1);

    % Build bbox from line endpoints with generous padding
    % Extra padding on each side to ensure full scratch is covered
    padSmall = 30;   % tight side padding
    padLarge = 60;   % generous padding on extent sides
    x1 = max(1,    min(p1(1),p2(1)) - padLarge);
    x2 = min(cols, max(p1(1),p2(1)) + padLarge);
    y1 = max(1,    min(p1(2),p2(2)) - padSmall);
    y2 = min(rows, max(p1(2),p2(2)) + padSmall);
    bboxes=[bboxes; x1,y1,x2-x1,y2-y1]; %#ok<AGROW>

    nPts=max(abs(p2-p1))+1;
    xs=max(1,min(cols,round(linspace(p1(1),p2(1),nPts))));
    ys=max(1,min(rows,round(linspace(p1(2),p2(2),nPts))));
    scratchMaskOut(sub2ind([rows,cols],ys,xs))=true;
end

if size(bboxes,1)>1, bboxes=mergeBoxes(bboxes,15); end

isScratch = ~isempty(bboxes);
fprintf('  Boxes output  : %d\n', size(bboxes,1));
fprintf('  Scratch found : %d\n', isScratch);
if isScratch && ~isempty(scratchLines)
    p1=scratchLines(1).point1; p2=scratchLines(1).point2;
    fprintf('  Line: [%.0f,%.0f]->[%.0f,%.0f] angle=%.0f\n',...
        p1(1),p1(2),p2(1),p2(2),abs(atan2d(p2(2)-p1(2),p2(1)-p1(1))));
end

debug.clusterMask  = scratchMaskOut;
debug.filteredMask = scratchMaskOut;
debug.mergedMask   = scratchMaskOut;
debug.isScratch    = isScratch;
debug.otsu         = thresh;
debug.componentCount = length(lines);
debug.bboxCount    = size(bboxes,1);
debug.minAreaUsed  = minLineLen;
debug.maxAreaUsed  = gloveArea;
debug.scratchDensity = nnz(scratchMaskOut)/max(gloveArea,1);
end


function debug = fillDebug(debug, rows, cols, gloveArea)
debug.scratchMask  = false(rows,cols);
debug.clusterMask  = false(rows,cols);
debug.filteredMask = false(rows,cols);
debug.mergedMask   = false(rows,cols);
debug.isScratch    = false;
debug.otsu         = 0; debug.threshValue = 0;
debug.componentCount=0; debug.bboxCount=0;
debug.minAreaUsed=0; debug.maxAreaUsed=gloveArea;
debug.scratchDensity=0;
end


function merged = mergeBoxes(bboxes, gap)
changed=true;
while changed
    changed=false; n=size(bboxes,1); used=false(n,1); merged=[];
    for i=1:n
        if used(i),continue;end; b=bboxes(i,:);
        for j=i+1:n
            if used(j),continue;end
            bi=[b(1)-gap,b(2)-gap,b(3)+2*gap,b(4)+2*gap];
            bj=[bboxes(j,1)-gap,bboxes(j,2)-gap,bboxes(j,3)+2*gap,bboxes(j,4)+2*gap];
            if max(bi(1),bj(1))<min(bi(1)+bi(3),bj(1)+bj(3)) && ...
               max(bi(2),bj(2))<min(bi(2)+bi(4),bj(2)+bj(4))
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