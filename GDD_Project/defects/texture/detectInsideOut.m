function [isInsideOut, confidence, debug] = detectInsideOut(segmentedImg, mask)
% detectInsideOut  Detect inside-out gloves using multi-feature analysis.
%
% Changelog:
%   - innerFrac scoring changed to scoreRange with cap at 0.95:
%     innerFrac=1.00 now scores LOW because it means either segmentation
%     grabbed non-glove pixels or the outer surface is completely absent.
%     Real inside-out gloves show inner surface at 0.20-0.75 of glove area
%     with blue outer still visible at edges.
%
%   - bimodalIdx threshold raised: softLow 0.20->0.45, softHigh 0.45->0.70
%     Previous thresholds let normal gloves with moderate bimodality
%     (e.g. 0.698) score 1.00. Now requires stronger bimodality.
%
%   - satCOV threshold raised: softLow 0.40->0.50, softHigh 0.65->0.80
%     Same reason — tighter discrimination between normal and inside-out.

debug = struct();

if ~any(mask(:))
    isInsideOut = false;
    confidence  = 0;
    return;
end

% ====================================================================
% 1. HSV
% ====================================================================
imgDouble = im2double(segmentedImg);
hsvImg    = rgb2hsv(imgDouble);
S = hsvImg(:,:,2);
V = hsvImg(:,:,3);
H = hsvImg(:,:,1);

debug.satMap = S;
debug.valMap = V;
debug.hueMap = H;

% ====================================================================
% 2. SATURATION ANALYSIS
% ====================================================================
S_vals    = S(mask);
meanSat   = mean(S_vals);
stdSat    = std(S_vals);
medianSat = median(S_vals);

lowSatThresh = max(0.15, medianSat * 0.5);
lowSatMask   = mask & (S < lowSatThresh);
lowSatFrac   = sum(lowSatMask(:)) / sum(mask(:));

debug.lowSatMask   = lowSatMask;
debug.lowSatThresh = lowSatThresh;
debug.lowSatFrac   = lowSatFrac;
debug.meanSat      = meanSat;
debug.stdSat       = stdSat;

% ====================================================================
% 3. BIMODALITY
% ====================================================================
satCOV = stdSat / (meanSat + 0.001);

[satHist, satEdges]  = histcounts(S_vals, 20, 'Normalization', 'probability');
satCentres           = (satEdges(1:end-1) + satEdges(2:end)) / 2;
satHistSmooth        = conv(satHist, ones(1,3)/3, 'same');
midIdx               = round(length(satHistSmooth) / 2);
peakVal              = max(satHistSmooth);
midVal               = satHistSmooth(midIdx);
bimodalIdx           = 1 - (midVal / (peakVal + 0.001));

debug.satCOV     = satCOV;
debug.bimodalIdx = bimodalIdx;
debug.satHist    = satHist;
debug.satCentres = satCentres;

% ====================================================================
% 4. INNER REGION SPATIAL COHERENCE
% ====================================================================
lowSatClean = bwareaopen(lowSatMask, 500);
lowSatClean = imfill(lowSatClean, 'holes');

CC = bwconncomp(lowSatClean);
if CC.NumObjects > 0
    compSizes   = cellfun(@numel, CC.PixelIdxList);
    largestComp = max(compSizes);
    innerFrac   = largestComp / sum(mask(:));
else
    innerFrac   = 0;
end

debug.lowSatClean = lowSatClean;
debug.innerFrac   = innerFrac;

% ====================================================================
% 5. TEXTURE DIFFERENCE
% ====================================================================
gray      = im2double(rgb2gray(segmentedImg));
varMap    = stdfilt(gray, ones(7));
outerMask = mask & ~lowSatClean;

if sum(lowSatClean(:)) > 100 && sum(outerMask(:)) > 100
    innerVar = mean(varMap(lowSatClean));
    outerVar = mean(varMap(outerMask));
    varRatio = innerVar / (outerVar + 0.001);
else
    innerVar = 0; outerVar = 0; varRatio = 1;
end

debug.varMap   = varMap;
debug.innerVar = innerVar;
debug.outerVar = outerVar;
debug.varRatio = varRatio;

% ====================================================================
% 6. BRIGHTNESS DIFFERENCE
% ====================================================================
if sum(lowSatClean(:)) > 100 && sum(outerMask(:)) > 100
    innerBrightness = mean(V(lowSatClean));
    outerBrightness = mean(V(outerMask));
    brightDiff      = innerBrightness - outerBrightness;
else
    innerBrightness = 0; outerBrightness = 0; brightDiff = 0;
end

debug.innerBrightness = innerBrightness;
debug.outerBrightness = outerBrightness;
debug.brightDiff      = brightDiff;

% ====================================================================
% 7. CONFIDENCE SCORING
%
%   a) lowSatFrac  [0.25] — peaks at 0.20-0.70, falls off toward 0.95
%      A real inside-out glove shows inner surface at 20-70% of area.
%
%   b) satCOV      [0.30] — threshold raised: 0.50-0.80
%      High COV requires stronger saturation spread to score.
%
%   c) bimodalIdx  [0.25] — threshold raised: 0.45-0.70
%      Requires a more pronounced bimodal distribution to score.
%      Normal gloves with mild bimodality (0.4-0.6) now score lower.
%
%   d) innerFrac   [0.15] — scoreRange peaks at 0.20-0.75, falls at 0.95
%      innerFrac=1.00 now scores 0 — means outer region is absent,
%      which indicates segmentation error not inside-out glove.
%
%   e) varRatio    [0.05] — unchanged, low weight
% ====================================================================
scoreA = scoreRange(lowSatFrac,  0.10, 0.20, 0.95, 1.00);  % extended: full inside-out glove is almost all low-sat
scoreB = scoreThresh(satCOV,     0.50, 0.80);
scoreC = scoreThresh(bimodalIdx, 0.45, 0.70);
scoreD = scoreRange(innerFrac,   0.10, 0.20, 0.95, 1.00);  % extended upper limit
scoreE = scoreThresh(varRatio-1, 0.10, 0.30);

weights    = [0.25, 0.30, 0.25, 0.15, 0.05];
scores     = [scoreA, scoreB, scoreC, scoreD, scoreE];
confidence = sum(weights .* scores);

% Hard gate: both bimodality and satCOV must show meaningful separation
if satCOV < 0.45 && bimodalIdx < 0.40
    confidence = confidence * 0.3;
end

isInsideOut = confidence > 0.50;

% ====================================================================
% Console report
% ====================================================================
fprintf('=== detectInsideOut report ===\n');
fprintf('  Low sat fraction : %.3f  (score=%.2f)\n', lowSatFrac,   scoreA);
fprintf('  Sat COV          : %.3f  (score=%.2f)\n', satCOV,       scoreB);
fprintf('  Bimodality idx   : %.3f  (score=%.2f)\n', bimodalIdx,   scoreC);
fprintf('  Inner region frac: %.3f  (score=%.2f)\n', innerFrac,    scoreD);
fprintf('  Var ratio        : %.3f  (score=%.2f)\n', varRatio,     scoreE);
fprintf('  Confidence       : %.3f  -> %s\n', confidence, ...
    yn(isInsideOut, 'INSIDE-OUT DETECTED', 'normal glove'));

debug.scores      = scores;
debug.weights     = weights;
debug.confidence  = confidence;
debug.isInsideOut = isInsideOut;

end


% ------------------------------------------------------------------
function s = scoreRange(val, minVal, lo, hi, maxVal)
if val < minVal || val > maxVal
    s = 0;
elseif val >= lo && val <= hi
    s = 1;
elseif val < lo
    s = (val - minVal) / (lo - minVal + 0.001);
else
    s = 1 - (val - hi) / (maxVal - hi + 0.001);
end
s = max(0, min(1, s));
end


% ------------------------------------------------------------------
function s = scoreThresh(val, softLow, softHigh)
if val <= softLow,     s = 0;
elseif val >= softHigh, s = 1;
else, s = (val - softLow) / (softHigh - softLow);
end
s = max(0, min(1, s));
end


% ------------------------------------------------------------------
function str = yn(cond, a, b)
if cond, str = a; else, str = b; end
end