function [isDirty, bboxes, debug] = detectDirty(segmentedImg, mask)
% detectDirty  Detect dirty/contaminated gloves using global analysis.
%
% A dirty glove has significantly lower mean brightness than a clean glove.
% Dirt distributed across the surface causes global darkening.
%
% Key fix: dark pixel fraction now calculated on ABSOLUTE threshold
% not relative to mean, so it works even when gloveArea is small
% due to poor segmentation.

debug = struct();

if ~any(mask(:))
    isDirty = false;
    bboxes  = [];
    return;
end

imgDouble = im2double(segmentedImg);
hsvImg    = rgb2hsv(imgDouble);
V = hsvImg(:,:,3);
S = hsvImg(:,:,2);

gloveV = V(mask);
gloveS = S(mask);

meanV  = mean(gloveV);
stdV   = std(gloveV);
meanS  = mean(gloveS);

% Dark pixel fraction — absolute threshold
% Pixels below 0.50 brightness are definitively dark on a white glove
absThresh     = 0.50;
relThresh     = max(meanV - 1.5*stdV, 0.40);
darkThresh    = min(absThresh, relThresh);
darkPixelFrac = mean(gloveV < darkThresh);

debug.valueMap      = V;
debug.satMap        = S;
debug.meanV         = meanV;
debug.stdV          = stdV;
debug.meanS         = meanS;
debug.darkThresh    = darkThresh;
debug.darkPixelFrac = darkPixelFrac;

% Texture variance
gray    = im2double(rgb2gray(segmentedImg));
gray(~mask) = 0;
varMap  = stdfilt(gray, ones(9));
meanVar = mean(varMap(mask));
debug.varMap  = varMap;
debug.meanVar = meanVar;

% Scoring
% Score A: global brightness drop
scoreA = scoreRange(1 - meanV, 0.05, 0.20, 0.45, 0.65);

% Score B: dark pixel fraction (absolute)
scoreB = scoreThresh(darkPixelFrac, 0.05, 0.30);

% Score C: texture variance
scoreC = scoreThresh(meanVar, 0.02, 0.08);

dirtScore = 0.40*scoreA + 0.40*scoreB + 0.20*scoreC;

% Hard gate
if meanV > 0.72 || darkPixelFrac < 0.05
    dirtScore = dirtScore * 0.2;
end

debug.scoreA    = scoreA;
debug.scoreB    = scoreB;
debug.scoreC    = scoreC;
debug.dirtScore = dirtScore;

isDirty = dirtScore > 0.40;

% Bounding boxes around dark regions
bboxes = [];
if isDirty
    Vclean            = V; Vclean(~mask) = 1;
    darkMask          = (Vclean < darkThresh) & mask;
    darkMask          = bwareaopen(darkMask, 50);
    darkMask          = imclose(darkMask, strel('disk', 15));
    darkMask          = imfill(darkMask, 'holes');
    darkMask          = bwareaopen(darkMask, 200);
    gloveArea         = nnz(mask);
    stats             = regionprops(darkMask, 'Area', 'BoundingBox', 'PixelIdxList');
    filteredMask      = false(size(darkMask));
    for i = 1:numel(stats)
        a = stats(i).Area;
        if a < max(200, 0.001*gloveArea) || a > 0.70*gloveArea, continue; end
        filteredMask(stats(i).PixelIdxList) = true;
        bboxes = [bboxes; stats(i).BoundingBox]; %#ok<AGROW>
    end
    if size(bboxes,1) > 1, bboxes = mergeBoxes(bboxes, 40); end
    debug.darkMask     = darkMask;
    debug.filteredMask = filteredMask;
else
    debug.darkMask     = false(size(mask));
    debug.filteredMask = false(size(mask));
end

fprintf('=== detectDirty report ===\n');
fprintf('  Mean brightness  : %.3f\n', meanV);
fprintf('  Dark threshold   : %.3f\n', darkThresh);
fprintf('  Dark pixel frac  : %.3f\n', darkPixelFrac);
fprintf('  Mean variance    : %.4f\n', meanVar);
fprintf('  Score A/B/C      : %.2f / %.2f / %.2f\n', scoreA, scoreB, scoreC);
fprintf('  Dirt score       : %.3f\n', dirtScore);
fprintf('  Dirty detected   : %d\n',  isDirty);
fprintf('  Boxes output     : %d\n',  size(bboxes,1));

debug.isDirty  = isDirty;
debug.dirtScore = dirtScore;
debug.bboxCount = size(bboxes,1);

end

function s = scoreRange(val, minVal, lo, hi, maxVal)
if val < minVal || val > maxVal, s = 0;
elseif val >= lo && val <= hi,   s = 1;
elseif val < lo, s = (val-minVal)/(lo-minVal+0.001);
else,            s = 1-(val-hi)/(maxVal-hi+0.001);
end
s = max(0,min(1,s));
end

function s = scoreThresh(val, softLow, softHigh)
if val <= softLow,       s = 0;
elseif val >= softHigh,  s = 1;
else, s = (val-softLow)/(softHigh-softLow);
end
s = max(0,min(1,s));
end

function merged = mergeBoxes(bboxes, gap)
changed = true;
while changed
    changed = false; n = size(bboxes,1); used = false(n,1); merged = [];
    for i = 1:n
        if used(i), continue; end
        b = bboxes(i,:);
        for j = i+1:n
            if used(j), continue; end
            bi=[b(1)-gap,b(2)-gap,b(3)+2*gap,b(4)+2*gap];
            bj=[bboxes(j,1)-gap,bboxes(j,2)-gap,bboxes(j,3)+2*gap,bboxes(j,4)+2*gap];
            ox=max(bi(1),bj(1))<min(bi(1)+bi(3),bj(1)+bj(3));
            oy=max(bi(2),bj(2))<min(bi(2)+bi(4),bj(2)+bj(4));
            if ox&&oy
                x1=min(b(1),bboxes(j,1));y1=min(b(2),bboxes(j,2));
                x2=max(b(1)+b(3),bboxes(j,1)+bboxes(j,3));
                y2=max(b(2)+b(4),bboxes(j,2)+bboxes(j,4));
                b=[x1,y1,x2-x1,y2-y1];used(j)=true;changed=true;
            end
        end
        merged=[merged;b]; %#ok<AGROW>
    end
    bboxes=merged;
end
merged=bboxes;
end