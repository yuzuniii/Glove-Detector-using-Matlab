function [mask, segmentedImg, debugInfo] = segmentGlove(img, gloveType)
% segmentGlove  Route to type-specific segmentation function.
%
% If gloveType not provided, auto-detects using detectGloveType().
%
% Inputs:
%   img       - uint8 RGB image
%   gloveType - optional: 'blue'|'white'|'black'|'cotton'
%               if omitted, auto-detected
%
% Outputs:
%   mask         - logical segmentation mask
%   segmentedImg - RGB image with background zeroed
%   debugInfo    - struct with gloveType and pixel count

debugInfo = struct();

if nargin < 2 || isempty(gloveType)
    [gloveType, typeConf, ~] = detectGloveType(img);
    fprintf('  Auto-detected: %s (conf=%.2f)\n', gloveType, typeConf);
    debugInfo.autoDetected = true;
    debugInfo.typeConf     = typeConf;
else
    debugInfo.autoDetected = false;
    debugInfo.typeConf     = 1.0;
end

debugInfo.gloveType   = gloveType;
debugInfo.dominantHue = 0;

switch gloveType
    case 'blue'
        [mask, segmentedImg] = segmentBlueGlove(img);
    case 'white'
        [mask, segmentedImg] = segmentWhiteGlove(img);
    case 'black'
        [mask, segmentedImg] = segmentBlackGlove(img);
    case 'cotton'
        [mask, segmentedImg] = segmentCottonGlove(img);
    otherwise
        warning('segmentGlove: unknown type "%s", using white fallback', gloveType);
        [mask, segmentedImg] = segmentWhiteGlove(img);
        debugInfo.gloveType  = 'white';
end

debugInfo.glovePixels = nnz(mask);

% HSV debug fields for compatibility with figures
hsvImg        = rgb2hsv(im2uint8(img));
debugInfo.H   = hsvImg(:,:,1);
debugInfo.S   = hsvImg(:,:,2);
debugInfo.V   = hsvImg(:,:,3);
debugInfo.rawMask = mask;

% Refine debug stub for backward compatibility
debugInfo.refine.afterAreaOpen = mask;
debugInfo.refine.afterSmooth   = mask;
debugInfo.finalMask            = mask;

fprintf('  segmentGlove: type=%s  px=%d\n', gloveType, nnz(mask));
end