function [mask, segmentedImg] = segmentWhiteGlove(img)
% segmentWhiteGlove  Segment white latex glove on dark grey background.
%
% VERSION 3 — LAB Mahalanobis + Corner Flood Fill
%
% WHY PREVIOUS APPROACHES FAILED:
%   HSV brightness threshold (V > threshold):
%     The stain (V≈0.20) is darker than the background (V≈0.47).
%     Any threshold that includes the stain also includes the background.
%     imfill then floods from the stain hole outward → 99% mask.
%
%   Background subtraction in HSV:
%     vUpper = bgV + vTol ≈ 0.59. Many background pixels have V > 0.59
%     due to lighting gradients → classified as "not background" → glove.
%     Result: still 70-80% mask.
%
% WHY THIS APPROACH WORKS:
%   In LAB colour space, the three pixel types are well-separated:
%
%     Dark grey background:  L≈48, A≈ 0, B≈ 0   (neutral grey)
%     White glove surface:   L≈88, A≈-2, B≈ 4   (bright, nearly neutral)
%     Brown stain on glove:  L≈35, A≈12, B≈18   (dark + warm colour)
%
%   The background is uniquely characterised by ALL THREE:
%     - Mid luminance (not as bright as glove, not as dark as stain)
%     - Near-zero A (not green or red)
%     - Near-zero B (not blue or yellow)
%
%   Mahalanobis distance from the background distribution catches this:
%   a pixel must match the background in ALL channels simultaneously.
%   The brown stain fails on A and B even when its L overlaps with bg.
%
% PIPELINE:
%   1. Sample corner patches → fit Gaussian to bg in LAB (mean, cov)
%   2. Compute Mahalanobis distance of every pixel from bg distribution
%   3. Background = distance < threshold; Glove = distance >= threshold
%   4. Flood fill from corners to catch any background leakage
%   5. Morphological cleanup (close gaps, fill holes)
%   6. Keep largest component + refineMask
%
% Inputs:
%   img  - uint8 RGB image (raw, unsegmented)
%
% Outputs:
%   mask         - logical mask (true = glove pixel)
%   segmentedImg - RGB with background zeroed

img    = im2uint8(img);
[rows, cols, ~] = size(img);
totalPx = rows * cols;

% ====================================================================
% STEP 1: Convert to LAB
%   LAB separates luminance from colour.
%   Grey background has near-zero A and B — unique signature.
% ====================================================================
labImg = rgb2lab(im2double(img));
L = labImg(:,:,1);   % 0-100
A = labImg(:,:,2);   % -128 to +127  (green-red)
B = labImg(:,:,3);   % -128 to +127  (blue-yellow)

% ====================================================================
% STEP 2: Sample background from corners
%   Use 8% of shorter dimension for patch size (same as before).
%   Stack all corner pixels into an Nx3 matrix for Gaussian fitting.
% ====================================================================
pSz = max(5, round(min(rows, cols) * 0.08));

% Extract corner patches as vectors
Lc = [ reshape(L(1:pSz,           1:pSz          ), [], 1);
       reshape(L(1:pSz,           cols-pSz+1:cols), [], 1);
       reshape(L(rows-pSz+1:rows, 1:pSz          ), [], 1);
       reshape(L(rows-pSz+1:rows, cols-pSz+1:cols), [], 1) ];

Ac = [ reshape(A(1:pSz,           1:pSz          ), [], 1);
       reshape(A(1:pSz,           cols-pSz+1:cols), [], 1);
       reshape(A(rows-pSz+1:rows, 1:pSz          ), [], 1);
       reshape(A(rows-pSz+1:rows, cols-pSz+1:cols), [], 1) ];

Bc = [ reshape(B(1:pSz,           1:pSz          ), [], 1);
       reshape(B(1:pSz,           cols-pSz+1:cols), [], 1);
       reshape(B(rows-pSz+1:rows, 1:pSz          ), [], 1);
       reshape(B(rows-pSz+1:rows, cols-pSz+1:cols), [], 1) ];

% Fit Gaussian to corner pixels: mean vector and covariance matrix
bgSamples = [Lc, Ac, Bc];   % Nx3
bgMean    = mean(bgSamples, 1);          % 1x3
bgCov     = cov(bgSamples);              % 3x3

% Regularise covariance to avoid singular matrix
bgCov = bgCov + eye(3) * 0.5;

fprintf('  segmentWhiteGlove: bg LAB mean = [%.1f, %.2f, %.2f]\n', ...
    bgMean(1), bgMean(2), bgMean(3));

% ====================================================================
% STEP 3: Mahalanobis distance from background for every pixel
%
%   Mahalanobis distance = sqrt((x-mu)' * inv(Sigma) * (x-mu))
%   Low distance  → pixel looks like background → background
%   High distance → pixel looks unlike background → glove or stain
%
%   We reshape the image into Nx3, compute distances, reshape back.
% ====================================================================
% Reshape image to Nx3
Lv = L(:);  Av = A(:);  Bv = B(:);
pixels = [Lv, Av, Bv];   % (rows*cols) x 3

% Compute Mahalanobis distance for all pixels at once
covInv = inv(bgCov);
diff   = pixels - bgMean;                    % Nx3
mDist  = sqrt(sum((diff * covInv) .* diff, 2));  % Nx1

% Reshape back to image
mDistMap = reshape(mDist, rows, cols);

fprintf('  segmentWhiteGlove: mDist bg_mean=%.2f  p95=%.2f  p99=%.2f\n', ...
    mean(mDist), prctile(mDist, 95), prctile(mDist, 99));

% ====================================================================
% STEP 4: Threshold Mahalanobis distance — Otsu on mDist histogram
%
% INSIGHT: The mDist map has two pixel populations:
%   Population 1 — Background pixels: low mDist (cluster near bg mean)
%   Population 2 — Glove pixels: high mDist (very different from bg)
%
% Otsu's method finds the optimal threshold between two populations
% automatically — no manual tuning needed, works on any image.
%
% Safety: threshold must be >= p99 of corner distances (background)
% and <= p10 of all image distances (can't classify everything as bg).
% ====================================================================
cornerDists  = mDist(1 : numel(Lc));
bgP99        = prctile(cornerDists, 99);

% Otsu on full mDist image to find bg/glove boundary
otsuThresh   = graythresh(mat2gray(mDistMap)) * max(mDist);

% Hard constraints:
% - Must be at least bgP99 (don't classify any corner pixel as glove)
% - Must leave at least 20% of image as glove (bgThresh < p80 of mDist)
bgThreshMin  = bgP99;
bgThreshMax  = prctile(mDist, 80);
bgThresh     = max(bgThreshMin, min(otsuThresh, bgThreshMax));

% Additional floor: at least 1.5x the bg cluster mean
bgThresh     = max(bgThresh, mean(cornerDists) * 1.5);

fprintf('  segmentWhiteGlove: Otsu=%.2f  bgP99=%.2f  final thresh=%.2f\n', ...
    otsuThresh, bgP99, bgThresh);

% Background = low distance from bg distribution
isBg    = mDistMap < bgThresh;
isBg    = imclose(isBg, strel('disk', 3));   % clean jagged edges

% ====================================================================
% STEP 5: Flood fill from corners
%
%   Any background pixel connected to the image boundary is DEFINITELY
%   background (the glove does not touch corners in a controlled capture).
%   This removes any isolated background islands inside the glove mask
%   without affecting the glove itself.
%
%   Method: imfill on the INVERTED mask treats connected background
%   regions touching the border as "holes" to NOT fill → we subtract them.
% ====================================================================
% Mark border-connected background via flood fill from boundary
borderBg = false(rows, cols);

% Scan each border row/col: if it's background, flood fill from there
borderMask = false(rows + 2, cols + 2);
borderMask(2:rows+1, 2:cols+1) = isBg;

% Pad with true background frame so floodfill can propagate from edges
borderMask(1,:)   = true;
borderMask(end,:) = true;
borderMask(:,1)   = true;
borderMask(:,end) = true;

filledBg   = imfill(borderMask, [1,1]);   % flood from corner
connectedBg = filledBg(2:rows+1, 2:cols+1) & isBg;

% Glove = everything NOT connected to border as background
gloveMask = ~connectedBg;

fprintf('  segmentWhiteGlove: after flood fill coverage = %.1f%%\n', ...
    100 * nnz(gloveMask) / totalPx);

% ====================================================================
% STEP 6: Morphological cleanup
%
%   imclose(disk 20): bridges any remaining gaps at glove boundary
%     (important where stain edge meets bg — boundary can be ragged)
%   imfill('holes'): fills interior holes — the stain region and any
%     small holes from threshold leakage
%   bwareaopen(0.5%): removes noise specks
% ====================================================================
gloveMask = imclose(gloveMask, strel('disk', 20));
gloveMask = imfill(gloveMask, 'holes');
gloveMask = bwareaopen(gloveMask, round(totalPx * 0.005));

% ====================================================================
% STEP 7: Keep largest component
%   After all the above, the glove should be the single dominant blob.
% ====================================================================
gloveMask = keepLargest(gloveMask);
gloveMask = imfill(gloveMask, 'holes');

frac = nnz(gloveMask) / totalPx;
fprintf('  segmentWhiteGlove: pre-refine coverage = %.1f%%\n', frac * 100);

% ====================================================================
% STEP 8: refineMask + apply
% ====================================================================
[mask, ~]    = refineMask(gloveMask);
segmentedImg = gddApplyMask(img, mask);

fprintf('  segmentWhiteGlove: FINAL mask = %d px (%.1f%% of image)\n', ...
    nnz(mask), 100 * nnz(mask) / totalPx);

end


% ------------------------------------------------------------------
function out = keepLargest(bw)
if ~any(bw(:))
    out = bw;
    return;
end
CC       = bwconncomp(bw);
nPx      = cellfun(@numel, CC.PixelIdxList);
[~, idx] = max(nPx);
out      = false(size(bw));
out(CC.PixelIdxList{idx}) = true;
end