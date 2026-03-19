function [gloveType, confidence, debugInfo] = detectGloveType(img)
% detectGloveType  Automatically detect glove type from raw image.
%
% Detects four glove types:
%   'white'  — White latex glove (WS1)
%   'black'  — Black nitrile glove (WS2)
%   'blue'   — Blue nitrile glove (WS3)
%   'cotton' — Cotton/fabric glove (WS4)
%
% Strategy:
%   Analyse the RAW image (before segmentation) using HSV colour space.
%   Detection is done on the full image so segmentation errors cannot
%   affect type detection. Each glove type has a distinct visual signature:
%
%   Blue nitrile : Strong blue hue (H=0.45-0.65), medium saturation
%   Black nitrile: Very low brightness overall (V < 0.35 dominant)
%   White latex  : Very high brightness (V > 0.65 dominant), low saturation
%   Cotton/fabric: Beige/cream/tan hue (H=0.05-0.15), low-medium saturation
%                  OR off-white with visible texture pattern
%
% Input:
%   img       - uint8 RGB image (raw, unsegmented)
%
% Outputs:
%   gloveType  - string: 'blue' | 'black' | 'white' | 'cotton'
%   confidence - 0-1 confidence score for the detected type
%   debugInfo  - struct with intermediate values

debugInfo = struct();

img = im2uint8(img);
hsvImg = rgb2hsv(img);

H = hsvImg(:,:,1);
S = hsvImg(:,:,2);
V = hsvImg(:,:,3);

% ====================================================================
% SAMPLE CENTRE REGION
% Focus on centre 60% of image — avoids background corners
% ====================================================================
[rows, cols, ~] = size(img);
r1 = round(rows * 0.20); r2 = round(rows * 0.80);
c1 = round(cols * 0.20); c2 = round(cols * 0.80);

Hc = H(r1:r2, c1:c2);
Sc = S(r1:r2, c1:c2);
Vc = V(r1:r2, c1:c2);

meanV   = mean(Vc(:));
meanS   = mean(Sc(:));
stdV    = std(Vc(:));

% Background estimate from corners
cSz = max(round(min(rows,cols) * 0.08), 5);
cornerV = [reshape(V(1:cSz,1:cSz),[],1);
           reshape(V(1:cSz,end-cSz+1:end),[],1);
           reshape(V(end-cSz+1:end,1:cSz),[],1);
           reshape(V(end-cSz+1:end,end-cSz+1:end),[],1)];
bgV = mean(cornerV);

debugInfo.meanV  = meanV;
debugInfo.meanS  = meanS;
debugInfo.stdV   = stdV;
debugInfo.bgV    = bgV;

% ====================================================================
% SCORE EACH GLOVE TYPE
% ====================================================================

% --- BLUE NITRILE ---
% Strong blue hue in centre region
bluePct    = mean((Hc(:) >= 0.44) & (Hc(:) <= 0.70) & (Sc(:) >= 0.15));
blueScore  = scoreThresh(bluePct, 0.10, 0.35);
debugInfo.bluePct = bluePct;

% Hard gate — if glove is overwhelmingly dark (darkPct > 0.60),
% it is black nitrile not blue nitrile.
% Black gloves with stains can have bluePct > 0 from the stain colour.
darkPctCheck = mean(Vc(:) < 0.35);
if darkPctCheck > 0.60
    blueScore = blueScore * 0.1;
end

% --- BLACK NITRILE ---
% Very dark overall, low saturation
darkPct    = mean(Vc(:) < 0.35);
blackScore = scoreThresh(darkPct, 0.15, 0.45) * scoreThresh(1-meanS, 0.50, 0.80);
debugInfo.darkPct = darkPct;

% Hard gate — must be significantly darker than background
if meanV > bgV * 0.75
    blackScore = blackScore * 0.2;
end

% --- WHITE LATEX ---
% Very bright, low saturation, significantly brighter than background
brightPct   = mean(Vc(:) > 0.65);
whiteScore  = scoreThresh(brightPct, 0.20, 0.55) * scoreThresh(1-meanS, 0.60, 0.90);
debugInfo.brightPct = brightPct;

% Hard gate — must be brighter than background
if meanV < bgV * 0.85
    whiteScore = whiteScore * 0.3;
end

% --- COTTON / FABRIC ---
% Beige/cream/tan hue OR off-white with visible knit/weave texture.
% Cotton gloves: H typically 0.04-0.18 (warm yellow-brown tones)
% OR cream-white with HIGH texture variance from yarn weave pattern.
%
% KEY DISCRIMINATOR vs white latex:
%   White latex: smooth surface → textureVar LOW  (~0.02-0.05)
%   Cotton knit: yarn grid      → textureVar HIGH (~0.06-0.15)
%   This separates cream cotton from white latex even when hue is similar.
cottonHuePct = mean((Hc(:) >= 0.04) & (Hc(:) <= 0.18) & (Sc(:) >= 0.05));

% Texture variance on centre crop
gray         = im2double(rgb2gray(img));
grayC        = gray(r1:r2, c1:c2);
varMap       = stdfilt(grayC, ones(9));
textureVar   = mean(varMap(:));

% Texture variance gets highest weight — most reliable discriminator
% cottonHuePct is secondary (cream cotton may have borderline hue)
cottonScore  = 0.30 * scoreThresh(cottonHuePct, 0.04, 0.20) + ...
               0.50 * scoreThresh(textureVar,   0.05, 0.12) + ...
               0.20 * scoreThresh(meanS,        0.05, 0.25);

% Hard gate — cotton shouldn't have strong blue
if bluePct > 0.10
    cottonScore = cottonScore * 0.2;
end

% Hard gate — white latex has LOW texture variance (smooth rubber surface)
% Cotton knit weave always produces high textureVar from yarn grid.
% If textureVar is high → definitively NOT white latex → kill white score.
% White latex: textureVar typically 0.02-0.05
% Cotton knit: textureVar typically 0.07-0.15
if textureVar > 0.050
    % Completely suppress white — cotton knit cannot be white latex
    whiteScore = 0;
end

% Cotton boost: high texture variance alone is sufficient for cotton
% even if hue is borderline (cream/beige burn hole glove case)
if textureVar > 0.050 && cottonScore < 0.50
    cottonScore = max(cottonScore, 0.55);
end

debugInfo.cottonHuePct = cottonHuePct;
debugInfo.textureVar   = textureVar;

% ====================================================================
% DECISION
% ====================================================================
scores = [blueScore, blackScore, whiteScore, cottonScore];
types  = {'blue', 'black', 'white', 'cotton'};

[maxScore, idx] = max(scores);
gloveType       = types{idx};
confidence      = maxScore;

debugInfo.isInsideOutOverride = false;  % default
% ====================================================================
% INSIDE-OUT BLUE GLOVE OVERRIDE
%   An inside-out blue glove shows its white inner surface.
%   detectGloveType classifies it as 'white' because meanS is very low.
%   Detection signal: very low overall saturation (< 0.10) AND blue
%   pixels still visible at glove edges (fingertips/cuff border).
%   If both conditions met, override to 'blue' so WS3 runs and
%   detectInsideOut can correctly identify it.
% ====================================================================
if strcmp(gloveType, 'white') && meanS < 0.10
    % Check for blue pixels in the image (hue 190-250 deg in [0,360] space)
    imgHSV   = rgb2hsv(im2double(img));
    H360     = imgHSV(:,:,1) * 360;
    Schan    = imgHSV(:,:,2);
    % Blue pixel: hue 190-250, saturation > 0.15
    bluePxMask = (H360 >= 190) & (H360 <= 250) & (Schan > 0.15);
    % Use centre crop for measurement (exclude background)
    [rows, cols, ~] = size(img);
    cr = round(rows*0.1):round(rows*0.9);
    cc = round(cols*0.1):round(cols*0.9);
    regionBluePct = sum(sum(bluePxMask(cr,cc))) / numel(bluePxMask(cr,cc));
    fprintf('  Inside-out check: meanS=%.3f bluePct=%.3f\n', meanS, regionBluePct);
    if regionBluePct > 0.005
        % Blue pixels present on an otherwise white-looking glove
        % → inside-out blue nitrile glove
        gloveType  = 'blue';
        confidence = 0.90;
        debugInfo.isInsideOutOverride = true;
        fprintf('  --> OVERRIDE: inside-out blue glove detected\n');
    end
end

debugInfo.scores.blue   = blueScore;
debugInfo.scores.black  = blackScore;
debugInfo.scores.white  = whiteScore;
debugInfo.scores.cotton = cottonScore;

% ====================================================================
% Console report
% ====================================================================
fprintf('=== detectGloveType ===\n');
fprintf('  Mean V (centre)  : %.3f\n', meanV);
fprintf('  Mean S (centre)  : %.3f\n', meanS);
fprintf('  BG brightness    : %.3f\n', bgV);
fprintf('  Texture variance : %.4f\n', textureVar);
fprintf('  Blue pct         : %.3f  score=%.3f\n', bluePct,    blueScore);
fprintf('  Dark pct         : %.3f  score=%.3f\n', darkPct,    blackScore);
fprintf('  Bright pct       : %.3f  score=%.3f\n', brightPct,  whiteScore);
fprintf('  Cotton hue pct   : %.3f  score=%.3f\n', cottonHuePct, cottonScore);
fprintf('  --> Glove type   : %s  (conf=%.3f)\n',  gloveType, confidence);

end


% ------------------------------------------------------------------
function s = scoreThresh(val, softLow, softHigh)
if val <= softLow,      s = 0;
elseif val >= softHigh, s = 1;
else, s = (val - softLow) / (softHigh - softLow);
end
s = max(0, min(1, s));
end