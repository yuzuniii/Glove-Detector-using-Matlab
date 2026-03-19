function features = extractColorFeatures(segmentedImg, mask)
% extractColorFeatures  Extract colour and intensity features for WS1
% surface defect classification (Stain, Dirty, Spotting).
%
% Features extracted:
%   1.  meanV              — mean brightness of glove (HSV Value)
%   2.  stdV               — brightness standard deviation
%   3.  darkPixelFrac      — fraction of pixels significantly darker than mean
%   4.  meanSat            — mean saturation (dirty gloves gain saturation)
%   5.  labAnomalyScore    — mean LAB colour anomaly across glove surface
%   6.  largeBlobCount     — number of large dark blobs (>1% glove area)
%   7.  smallBlobCount     — number of small dark blobs (5-200px)
%   8.  darkSpatialSpread  — how spread out dark pixels are (0=clustered,1=uniform)
%   9.  maxBlobFrac        — largest dark blob as fraction of glove area
%   10. meanVar            — mean local texture variance (dirt raises this)
%
% Inputs:
%   segmentedImg - RGB, background zeroed
%   mask         - logical, true = glove pixel
%
% Output:
%   features     - struct with all feature fields

features = struct();

if ~any(mask(:))
    features = defaultFeatures();
    return;
end

imgDouble = im2double(segmentedImg);
gloveArea = nnz(mask);

% ====================================================================
% HSV FEATURES
% ====================================================================
hsvImg = rgb2hsv(imgDouble);
V = hsvImg(:,:,3);
S = hsvImg(:,:,2);

gloveV = V(mask);
gloveS = S(mask);

features.meanV    = mean(gloveV);
features.stdV     = std(gloveV);
features.meanSat  = mean(gloveS);

% Dark pixel fraction
% Dark = significantly below mean brightness
darkThresh            = max(features.meanV - 2.0 * features.stdV, 0.30);
features.darkPixelFrac = mean(gloveV < darkThresh);
features.darkThresh    = darkThresh;

% ====================================================================
% LAB ANOMALY SCORE
%    Use large window (20% of glove diameter) so stain is anomalous
%    relative to surrounding clean glove, not its own neighbourhood.
% ====================================================================
labImg = rgb2lab(imgDouble);
L = labImg(:,:,1);
A = labImg(:,:,2);
B = labImg(:,:,3);

L(~mask) = 0; A(~mask) = 0; B(~mask) = 0;

winSize = round(0.20 * sqrt(gloveArea));
winSize = max(61, min(201, winSize));
if mod(winSize,2)==0, winSize = winSize+1; end

Lmean = imfilter(L, fspecial('average',winSize), 'replicate');
Amean = imfilter(A, fspecial('average',winSize), 'replicate');
Bmean = imfilter(B, fspecial('average',winSize), 'replicate');

Ldiff = abs(L - Lmean); Ldiff(~mask) = 0;
Adiff = abs(A - Amean); Adiff(~mask) = 0;
Bdiff = abs(B - Bmean); Bdiff(~mask) = 0;

labScore = (0.5*mat2gray(Ldiff)) + (0.25*mat2gray(Adiff)) + (0.25*mat2gray(Bdiff));
features.labAnomalyScore = mean(labScore(mask));

% ====================================================================
% BLOB ANALYSIS — Direct LAB colour thresholding
%
%   PROBLEM WITH OLD APPROACH (local dark anomaly map):
%   winSize=89px fragments the large stain into hundreds of tiny
%   pieces → maxBlobFrac stays tiny → classifier misses stain.
%
%   NEW APPROACH: Use clean glove baseline (top-60% bright pixels)
%   and find pixels that are BOTH dark (low L) AND warm-coloured
%   (high A or B). This directly segments the stain as one blob.
%   Same logic as detectStain — keeps features consistent.
% ====================================================================
maskEroded = imerode(mask, strel('disk', max(5, round(0.02*sqrt(gloveArea)))));

% Clean baseline from bright pixels (top 60% — definitely not stain)
Lvals = L(maskEroded);
brightnessThresh = prctile(Lvals(Lvals > 0), 40);
cleanMask = maskEroded & (L >= brightnessThresh);
if nnz(cleanMask) < 100, cleanMask = maskEroded; end

cleanL = median(L(cleanMask));
cleanA = median(A(cleanMask));
cleanB = median(B(cleanMask));

% Stain score: dark drop × colour shift (AND logic — both must be true)
darkDrop    = max(0, cleanL - L); darkDrop(~maskEroded) = 0;
colourShift = max(0, (A - cleanA)) + max(0, (B - cleanB));
colourShift(~maskEroded) = 0;

darkNorm    = mat2gray(darkDrop);
colourNorm  = mat2gray(colourShift);
stainMap    = 0.6 * (darkNorm .* colourNorm) + 0.4 * darkNorm;
stainMap(~maskEroded) = 0;

% Threshold
maskedStain = stainMap(maskEroded);
if ~isempty(maskedStain) && max(maskedStain) > 1e-4
    otsu    = graythresh(maskedStain);
    binMask = (stainMap > otsu * 0.80) & maskEroded;
    binMask = bwareaopen(binMask, 20);
else
    binMask = false(size(mask));
end

stats = regionprops(binMask, 'Area', 'Centroid');

% Blob counts
% largeBlobThresh raised to 1% of glove (from 0.5%) to avoid counting
% stain fragments as separate large blobs.
largeBlobThresh         = 0.010 * gloveArea;
smallBlobMax            = max(300, 0.003 * gloveArea);
if ~isempty(stats)
    areas    = sort([stats.Area], 'descend');
    features.largeBlobCount = sum(areas > largeBlobThresh);
    features.smallBlobCount = sum(areas >= 5 & areas <= smallBlobMax);
    features.maxBlobFrac    = areas(1) / gloveArea;

    % Dominance ratio: largest blob vs second largest.
    % Stain: one dominant blob → ratio HIGH (e.g. 100K / 37K = 2.7)
    % Dirty: many equal blobs  → ratio LOW  (e.g. 77K / 58K = 1.3)
    if numel(areas) >= 2 && areas(2) > 0
        features.blobDominance = areas(1) / areas(2);
    else
        features.blobDominance = 10;  % only one blob = fully dominant
    end
else
    features.largeBlobCount = 0;
    features.smallBlobCount = 0;
    features.maxBlobFrac    = 0;
    features.blobDominance  = 0;
end

% ====================================================================
% SPATIAL SPREAD OF DARK PIXELS
%    0 = clustered (stain)   1 = spread everywhere (dirty)
% ====================================================================
darkPixels = find(binMask);
if numel(darkPixels) > 10
    [dpRows, dpCols]   = ind2sub(size(binMask), darkPixels);
    spreadR        = std(double(dpRows)) / size(mask,1);
    spreadC        = std(double(dpCols)) / size(mask,2);
    features.darkSpatialSpread = (spreadR + spreadC) / 2;
else
    features.darkSpatialSpread = 0;
end

% ====================================================================
% TEXTURE VARIANCE
% ====================================================================
gray = im2double(rgb2gray(segmentedImg));
gray(~mask) = 0;
varMap           = stdfilt(gray, ones(9));
features.meanVar = mean(varMap(mask));

% ====================================================================
% Console output
% ====================================================================
fprintf('=== extractColorFeatures ===\n');
fprintf('  meanV            : %.3f\n', features.meanV);
fprintf('  stdV             : %.3f\n', features.stdV);
fprintf('  darkPixelFrac    : %.3f\n', features.darkPixelFrac);
fprintf('  meanSat          : %.3f\n', features.meanSat);
fprintf('  labAnomalyScore  : %.4f\n', features.labAnomalyScore);
fprintf('  largeBlobCount   : %d\n',   features.largeBlobCount);
fprintf('  smallBlobCount   : %d\n',   features.smallBlobCount);
fprintf('  maxBlobFrac      : %.4f\n', features.maxBlobFrac);
fprintf('  blobDominance    : %.2f\n',  features.blobDominance);
fprintf('  darkSpatialSpread: %.3f\n', features.darkSpatialSpread);
fprintf('  meanVar          : %.4f\n', features.meanVar);

end


function f = defaultFeatures()
f.meanV             = 1;
f.stdV              = 0;
f.meanSat           = 0;
f.darkPixelFrac     = 0;
f.darkThresh        = 0;
f.labAnomalyScore   = 0;
f.largeBlobCount    = 0;
f.smallBlobCount    = 0;
f.maxBlobFrac       = 0;
f.blobDominance     = 0;
f.darkSpatialSpread = 0;
f.meanVar           = 0;
end