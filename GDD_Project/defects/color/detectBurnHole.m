function [isBurnHole, bboxes, debug] = detectBurnHole(segmentedImg, mask, origImg)
% detectBurnHole  Detect burn holes on cotton gloves.
%
% APPROACH — FOUR-GATE SYSTEM:
%   Gate 0 — SINGLE VOID SIZE: largest contiguous raw bg-matched region
%             must be >= 1% of glove area. Burn hole = one large void.
%             Weave holes = many tiny voids, none large individually.
%   Gate 1 — CIRCULARITY (after morphological closing)
%   Gate 2 — FILL DENSITY
%   Gate 3 — DARK RING

debug = struct();
bboxes = [];

if ~any(mask(:))
    isBurnHole = false; return;
end

[rows, cols, ~] = size(segmentedImg);
imgDouble = im2double(segmentedImg);
gloveArea = nnz(mask);

gloveProps = regionprops(mask, 'BoundingBox');
gloveW = cols; gloveH = rows;
if ~isempty(gloveProps)
    gBB = gloveProps(1).BoundingBox;
    gloveW = round(gBB(3)); gloveH = round(gBB(4));
end

diskR      = max(10, min(30, round(0.03 * min(gloveW, gloveH))));
maskEroded = imerode(mask, strel('disk', diskR));
if nnz(maskEroded) < 0.20*gloveArea
    maskEroded = imerode(mask, strel('disk', round(diskR/2)));
end
if ~any(maskEroded(:)), maskEroded = mask; end

if nargin >= 3 && ~isempty(origImg)
    labOrig = rgb2lab(im2double(origImg));
else
    labOrig = rgb2lab(imgDouble);
end

pSz = max(8, round(min(rows,cols)*0.06));
Lo = labOrig(:,:,1); Ao = labOrig(:,:,2); Bo = labOrig(:,:,3);
bgL = mean([reshape(Lo(1:pSz,1:pSz),[],1); reshape(Lo(1:pSz,cols-pSz+1:cols),[],1); ...
            reshape(Lo(rows-pSz+1:rows,1:pSz),[],1); reshape(Lo(rows-pSz+1:rows,cols-pSz+1:cols),[],1)]);
bgA = mean([reshape(Ao(1:pSz,1:pSz),[],1); reshape(Ao(1:pSz,cols-pSz+1:cols),[],1); ...
            reshape(Ao(rows-pSz+1:rows,1:pSz),[],1); reshape(Ao(rows-pSz+1:rows,cols-pSz+1:cols),[],1)]);
bgB = mean([reshape(Bo(1:pSz,1:pSz),[],1); reshape(Bo(1:pSz,cols-pSz+1:cols),[],1); ...
            reshape(Bo(rows-pSz+1:rows,1:pSz),[],1); reshape(Bo(rows-pSz+1:rows,cols-pSz+1:cols),[],1)]);

fprintf('  detectBurnHole: bg LAB=[%.1f, %.2f, %.2f]\n', bgL, bgA, bgB);

labImg = rgb2lab(imgDouble);
L = labImg(:,:,1); A = labImg(:,:,2); B = labImg(:,:,3);

bgDist = sqrt(0.5*(L-bgL).^2 + 1.5*(A-bgA).^2 + 1.5*(B-bgB).^2);
bgDist(~maskEroded) = Inf;

holeThresh  = 15;
holeMaskRaw = (bgDist < holeThresh) & maskEroded;
holeMaskRaw = bwareaopen(holeMaskRaw, 30);

% ====================================================================
% GATE 0 — MINIMUM SINGLE CONTIGUOUS VOID
%   Burn hole: one large contiguous opening >= 1% of glove area
%   Weave holes: each tiny opening << 1% of glove area
%   1% of a 2M px glove = 20,000px — well above any weave hole
% ====================================================================
minSingleVoid = max(2000, round(0.010 * gloveArea));
rawCC    = bwconncomp(holeMaskRaw);
if rawCC.NumObjects == 0
    isBurnHole = false;
    debug.isBurnHole = false; debug.filteredMask = false(rows,cols);
    fprintf('  detectBurnHole: no raw void pixels\n  Burn hole : 0\n'); return;
end
rawSizes   = cellfun(@numel, rawCC.PixelIdxList);
maxRawSize = max(rawSizes);

fprintf('  detectBurnHole: rawPx=%d  maxSingleVoid=%d  minRequired=%d\n', ...
    nnz(holeMaskRaw), maxRawSize, minSingleVoid);

if maxRawSize < minSingleVoid
    isBurnHole = false;
    debug.isBurnHole = false; debug.filteredMask = false(rows,cols);
    debug.maxRawSize = maxRawSize;
    fprintf('=== detectBurnHole: no large single void — not a burn hole ===\n');
    fprintf('  Burn hole : 0\n'); return;
end

closeRadius    = max(15, round(0.025*gloveW));
holeMaskClosed = imclose(holeMaskRaw, strel('disk', closeRadius));
holeMaskClosed = imfill(holeMaskClosed, 'holes');
holeMaskClosed = holeMaskClosed & maskEroded;
holeMaskClosed = bwareaopen(holeMaskClosed, 300);

fprintf('  detectBurnHole: closedPx=%d  closeR=%d\n', nnz(holeMaskClosed), closeRadius);
debug.holeMaskRaw = holeMaskRaw; debug.holeMaskClosed = holeMaskClosed;

if ~any(holeMaskClosed(:))
    isBurnHole = false;
    debug.isBurnHole = false; debug.filteredMask = false(rows,cols);
    fprintf('  Burn hole : 0\n'); return;
end

CC = bwconncomp(holeMaskClosed);
filteredMask = false(rows,cols); bboxes = [];
minHoleArea = max(300, round(0.003*gloveArea));
maxHoleArea = 0.15*gloveArea;
gloveMedianL = median(L(maskEroded));
nSmall=0; nLarge=0; nNotCirc=0; nLowFill=0; nNoRing=0; nOk=0;

for ci = 1:CC.NumObjects
    px = CC.PixelIdxList{ci}; area = numel(px);
    if area < minHoleArea, nSmall=nSmall+1; continue; end
    if area > maxHoleArea, nLarge=nLarge+1; continue; end
    compMask = false(rows,cols); compMask(px) = true;
    perim = regionprops(compMask,'Perimeter');
    if isempty(perim)||perim(1).Perimeter<1, nNotCirc=nNotCirc+1; continue; end
    circ = 4*pi*area/(perim(1).Perimeter^2);
    if circ < 0.35
        nNotCirc=nNotCirc+1;
        fprintf('  detectBurnHole: region %d area=%d circ=%.3f REJECTED\n',ci,area,circ); continue;
    end
    fillD = nnz(holeMaskRaw & compMask)/area;
    if fillD < 0.20
        nLowFill=nLowFill+1;
        fprintf('  detectBurnHole: region %d REJECTED low fill %.3f\n',ci,fillD); continue;
    end
    ringR    = max(20, round(sqrt(area)*0.25));
    ringMask = imdilate(compMask,strel('disk',ringR)) & ~compMask & maskEroded;
    if nnz(ringMask)<20, nNoRing=nNoRing+1; continue; end
    darkDrop = gloveMedianL - mean(L(ringMask));
    fprintf('  detectBurnHole: region %d area=%d circ=%.3f fill=%.3f darkDrop=%.1f\n',...
        ci,area,circ,fillD,darkDrop);
    if darkDrop < 8.0
        nNoRing=nNoRing+1;
        fprintf('  detectBurnHole: region %d REJECTED weak ring\n',ci); continue;
    end
    props = regionprops(compMask,'BoundingBox');
    if isempty(props), continue; end
    bb = props(1).BoundingBox; pad = max(25,round(sqrt(area)*0.5));
    x1=max(1,bb(1)-pad); y1=max(1,bb(2)-pad);
    x2=min(cols,bb(1)+bb(3)+pad); y2=min(rows,bb(2)+bb(4)+pad);
    filteredMask(px)=true; bboxes=[bboxes;x1,y1,x2-x1,y2-y1]; %#ok<AGROW>
    nOk=nOk+1;
end

if size(bboxes,1)>1, [~,ix]=max(bboxes(:,3).*bboxes(:,4)); bboxes=bboxes(ix,:); end
isBurnHole = ~isempty(bboxes);

fprintf('=== detectBurnHole report ===\n');
fprintf('  Components: %d (small=%d large=%d notCirc=%d lowFill=%d noRing=%d ok=%d)\n',...
    CC.NumObjects,nSmall,nLarge,nNotCirc,nLowFill,nNoRing,nOk);
fprintf('  Burn hole : %d\n', isBurnHole);

debug.filteredMask=filteredMask; debug.isBurnHole=isBurnHole; debug.bboxCount=size(bboxes,1);
end