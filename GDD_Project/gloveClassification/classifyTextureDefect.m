function [defectType, confidence, scores] = classifyTextureDefect(features)
% classifyTextureDefect  Classify WS3 texture defect type.
%
% Uses texture features to distinguish:
%   'wrinkle' — soft surface deformations, high gradient variance
%   'fold'    — single hard crease line, strong linear edge
%   'normal'  — no significant texture defect

% ====================================================================
% WRINKLE SCORE
%   High gradient magnitude, distributed across surface
%   High GLCM contrast (surface roughness)
% ====================================================================
% features from extractTextureFeatures:
% features(1) = GLCM contrast
% features(2) = GLCM correlation
% features(3) = GLCM energy
% features(4) = GLCM homogeneity
% features(5) = mean gradient magnitude
% features(6) = std gradient magnitude

if isstruct(features)
    % Handle struct input
    meanGrad = getField(features, 'meanGradient', 0.05);
    stdGrad  = getField(features, 'stdGradient',  0.02);
    contrast = getField(features, 'contrast',     50);
else
    % Handle array input (legacy)
    meanGrad = safeGet(features, 5, 0.05);
    stdGrad  = safeGet(features, 6, 0.02);
    contrast = safeGet(features, 1, 50);
end

% Wrinkle: moderate-high gradient, distributed variance
wrinkleA = scoreThresh(meanGrad, 0.04, 0.15);
wrinkleB = scoreThresh(stdGrad,  0.02, 0.08);
wrinkleScore = 0.60*wrinkleA + 0.40*wrinkleB;

% Fold: very high gradient (hard crease) but low variance
% (one strong line vs many soft wrinkles)
foldA = scoreThresh(meanGrad, 0.08, 0.25);
foldB = scoreRange(stdGrad, 0, 0.01, 0.04, 0.08); % lower variance = fold
foldScore = 0.50*foldA + 0.50*foldB;

% Hard gate: folds need a stronger signal than wrinkles
if meanGrad < 0.06
    foldScore = foldScore * 0.3;
end

scores = struct();
scores.wrinkle = wrinkleScore;
scores.fold    = foldScore;

[maxScore, idx] = max([wrinkleScore, foldScore]);
classes = {'wrinkle', 'fold'};

if maxScore < 0.30
    defectType = 'normal';
    confidence = 1 - maxScore;
else
    defectType = classes{idx};
    confidence = maxScore;
end

fprintf('=== classifyTextureDefect ===\n');
fprintf('  Mean gradient : %.4f\n', meanGrad);
fprintf('  Std gradient  : %.4f\n', stdGrad);
fprintf('  Wrinkle score : %.3f\n', wrinkleScore);
fprintf('  Fold score    : %.3f\n', foldScore);
fprintf('  Decision      : %s (conf=%.3f)\n', defectType, confidence);

end

function v = getField(s, f, def)
if isfield(s, f), v = s.(f); else, v = def; end
end

function v = safeGet(arr, idx, def)
if numel(arr) >= idx, v = arr(idx); else, v = def; end
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