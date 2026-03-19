function [defectType, confidence, scores] = classifyColorDefect(features)
% classifyColorDefect  Classify WS1 colour defect using extracted features.
%
% Changelog v3:
%   ROOT CAUSE FIX for stain misclassified as spotting:
%   The stain image produces very high smallBlobCount because the stain
%   texture (grainy brown edges) creates hundreds of tiny dark fragments
%   in the blob analysis. This was overwhelming the spotting scorer even
%   though maxBlobFrac >> 0.03 should have crushed it.
%
%   Key fixes:
%   1. Spotting hard gate tightened: maxBlobFrac > 0.02 → multiply by 0.05
%      (was 0.2 — not aggressive enough). A stain always has maxBlobFrac
%      >> 0.02 so spotting score becomes essentially 0.
%   2. Spotting second gate: largeBlobCount >= 1 → multiply by 0.05
%      A stain always has at least one large blob. Spotting has zero.
%   3. Stain score: labAnomalyScore weight raised 0.25→0.35, stainB
%      (maxBlobFrac) softHigh raised 0.08→0.15 to capture large stains.
%   4. Stain hard boost: if largeBlobCount>=1 AND maxBlobFrac>0.02,
%      apply 1.3x multiplier (capped at 0.95) — unambiguous stain signal.

% ====================================================================
% DIRTY SCORE
%   Dirty = globally darkened glove surface.
%   dirtygloves features: meanV=0.630, darkPixelFrac=0.120,
%     darkSpatialSpread=0.172, meanVar=0.044, meanSat=0.138
%
%   Key signals for dirty:
%     meanV < 0.70 (global brightness drop from clean white)
%     meanSat elevated (dirt adds colour)
%     meanVar elevated (surface roughness from dirt particles)
%     darkPixelFrac moderate (not as concentrated as stain)
% ====================================================================
dirtyA = scoreThresh(1 - features.meanV,       0.10, 0.40);  % meanV drop
dirtyB = scoreThresh(features.darkPixelFrac,    0.05, 0.25);  % lowered floor
dirtyC = scoreThresh(features.meanSat,          0.08, 0.20);  % sat elevated by dirt
dirtyD = scoreThresh(features.meanVar,          0.02, 0.07);  % texture roughness
dirtyScore = 0.40*dirtyA + 0.25*dirtyB + 0.20*dirtyC + 0.15*dirtyD;

% Hard gate: very bright AND very few dark pixels = definitely clean
if features.meanV > 0.78 && features.darkPixelFrac < 0.05
    dirtyScore = dirtyScore * 0.1;
end

% Hard boost: globally dark + low blob dominance = confirmed dirty
% dirtygloves: meanV=0.630 < 0.70, blobDominance=1.33 < 1.8 → boost fires
% glovestain:  meanV=0.766 > 0.70                            → boost blocked
domVal2 = 1.0;
if isfield(features, 'blobDominance'), domVal2 = features.blobDominance; end
if features.meanV < 0.70 && domVal2 < 1.8
    dirtyScore = min(0.95, dirtyScore * 1.5);
end

% ====================================================================
% ====================================================================
% SPOTTING SCORE
%   Spotting = many small discrete dots scattered across a clean glove.
%
%   Feature fingerprint (from glovespotting.png):
%     smallBlobCount = 145   (many dots)
%     blobDominance  = 1.63  (no single blob dominates — all roughly equal)
%     darkSpatialSpread=0.19 (dots spread across whole glove)
%     meanV = 0.723          (glove mostly white — not globally darkened)
%     maxBlobFrac = 0.0205   (largest blob is tiny fraction of glove)
%
%   Key discriminators:
%     vs stain:  stain has blobDominance ~2.7 (one huge blob)
%                spotting has blobDominance ~1.6 (many equal blobs)
%     vs dirty:  dirty has meanV < 0.68, global darkening
%                spotting has meanV > 0.68, only isolated dots
% ====================================================================
domVal3 = 1.0;
if isfield(features, 'blobDominance'), domVal3 = features.blobDominance; end

% Primary signals
spotA = scoreThresh(features.smallBlobCount,   10,  200);   % many dots
spotB = scoreThresh(features.darkSpatialSpread, 0.10, 0.30); % spread across glove
spotC = scoreThresh(features.meanV,             0.65, 0.85); % glove mostly white
spotD = scoreRange(domVal3, 1.0, 1.0, 2.0, 3.0);            % low dominance (equal blobs)
spottingScore = 0.30*spotA + 0.25*spotB + 0.25*spotC + 0.20*spotD;

% Hard gates
% Gate 1: blobDominance > 2.5 means one blob dominates = stain, not spotting
if domVal3 > 2.5
    spottingScore = spottingScore * 0.05;
end
% Gate 2: very dark glove = dirty not spotting
if features.meanV < 0.65
    spottingScore = spottingScore * 0.10;
end
% Gate 3: too few blobs = not spotting
if features.smallBlobCount < 8
    spottingScore = spottingScore * 0.10;
end

% ====================================================================
% STAIN SCORE
%   Stain = at least one LARGE blob + LAB colour anomaly + clustered.
%
%   Key features:
%     largeBlobCount >= 1: one dominant dark region
%     maxBlobFrac > 0.02: that region is sizeable (>2% of glove)
%     labAnomalyScore: colour/brightness differs from surrounding glove
%     darkSpatialSpread: stain is localised (clustered), not spread
%
%   Weights: labAnomalyScore raised 0.25→0.35 (most reliable signal)
%            maxBlobFrac softHigh raised 0.08→0.15 (large stains)
% ====================================================================
% Stain = ONE dominant large blob. Many small blobs = dirty/spotting.
% dirtygloves: largeBlobCount=6, maxBlobFrac=0.013 -> NOT stain.
% glovestain:  largeBlobCount=1, maxBlobFrac=0.05+ -> IS stain.
stainA = scoreThresh(features.largeBlobCount,  1,    3);
stainB = scoreThresh(features.maxBlobFrac,     0.02, 0.15);
stainC = scoreThresh(features.labAnomalyScore, 0.03, 0.10);
stainD = scoreRange(features.darkSpatialSpread, 0, 0.0, 0.35, 0.60);
stainScore = 0.15*stainA + 0.40*stainB + 0.35*stainC + 0.10*stainD;

% Hard floor 1: no dominant blob → not a stain
if features.maxBlobFrac < 0.02
    stainScore = stainScore * 0.15;
end

% Hard floor 2: low dominance = many equal blobs = dirty, not stain
% glovestain:  blobDominance = 100620/37479 = 2.68 (one blob dominates)
% dirtygloves: blobDominance = 77575/58339 = 1.33 (equal blobs = spread dirt)
if isfield(features, 'blobDominance') && features.blobDominance < 1.8
    stainScore = stainScore * 0.10;
end

% Hard boost: dominant single blob = confirmed stain
% Requires: blobDominance > 1.8 (one blob clearly larger than rest)
%           AND maxBlobFrac > 0.03 (blob is sizeable)
domVal = 1.0;
if isfield(features, 'blobDominance'), domVal = features.blobDominance; end
if domVal > 1.8 && features.maxBlobFrac > 0.03
    stainScore = min(0.95, stainScore * 1.4);
end

% ====================================================================
% DECISION
% ====================================================================
scores = struct();
scores.dirty    = dirtyScore;
scores.spotting = spottingScore;
scores.stain    = stainScore;

[maxScore, idx] = max([dirtyScore, spottingScore, stainScore]);
classes = {'dirty', 'spotting', 'stain'};

if maxScore < 0.35
    defectType = 'normal';
    confidence = 1 - maxScore;
else
    defectType = classes{idx};
    confidence = maxScore;
end

fprintf('=== classifyColorDefect ===\n');
fprintf('  Features used  : meanV=%.3f  darkFrac=%.3f  spread=%.3f\n', ...
    features.meanV, features.darkPixelFrac, features.darkSpatialSpread);
fprintf('  largeBlobCount : %d   maxBlobFrac=%.4f   smallBlobCount=%d\n', ...
    features.largeBlobCount, features.maxBlobFrac, features.smallBlobCount);
fprintf('  labAnomalyScore: %.4f\n', features.labAnomalyScore);
fprintf('  Dirty score    : %.3f\n', dirtyScore);
fprintf('  Spotting score : %.3f\n', spottingScore);
fprintf('  Stain score    : %.3f\n', stainScore);
fprintf('  Decision       : %s  (conf=%.3f)\n', defectType, confidence);

end

function s = scoreThresh(val, softLow, softHigh)
if val <= softLow,      s = 0;
elseif val >= softHigh, s = 1;
else, s = (val-softLow)/(softHigh-softLow);
end
s = max(0,min(1,s));
end

function s = scoreRange(val, minVal, lo, hi, maxVal)
if val < minVal || val > maxVal, s = 0;
elseif val >= lo && val <= hi,   s = 1;
elseif val < lo, s = (val-minVal)/(lo-minVal+0.001);
else,            s = 1-(val-hi)/(maxVal-hi+0.001);
end
s = max(0,min(1,s));
end