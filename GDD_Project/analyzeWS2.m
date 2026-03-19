function results = analyzeWS2(segmentedImg, mask, origImg)
% analyzeWS2  WS2 Black Nitrile Shape Defect pipeline.
%
% Defects: Stain (shared via analyzeGlove), Tear, Incomplete Beading
%
% Pipeline:
%   1. Run detectTear — looks for bright holes inside black glove mask
%   2. Run detectIncompleteBeading — analyses cuff boundary roughness
%   3. Return all results with confidence scores
%
% Note: Stain detection is handled by analyzeGlove before this is called
%       (shared detector for all glove types).

fprintf('\n--- WS2 Shape Analysis (Black Nitrile) ---\n');

results = struct();

if ~any(mask(:))
    results = emptyResults();
    return;
end

% ====================================================================
% STEP 1: Detect Tear
%   Tear = bright hole visible INSIDE the black glove mask
% ====================================================================
fprintf('  Running detectTear...\n');
if nargin < 3, origImg = []; end
[isTear, tearBboxes, tearDebug] = detectTear(segmentedImg, mask, origImg);

% Confidence: based on how bright/large the hole is relative to glove
tearConf = 0;
if isTear && ~isempty(tearBboxes)
    bboxArea = sum(tearBboxes(:,3) .* tearBboxes(:,4));
    gloveArea = max(nnz(mask), 1);
    tearConf = min(0.95, 0.55 + 0.40 * min(bboxArea / gloveArea * 8, 1));
end

results.isTear     = isTear;
results.tearBboxes = tearBboxes;
results.tearDebug  = tearDebug;
results.tearConf   = tearConf;

% ====================================================================
% STEP 2: Detect Incomplete Beading
%   Incomplete beading = ragged/torn cuff at bottom of glove
% ====================================================================
fprintf('  Running detectIncompleteBeading...\n');
[isBeading, beadingBboxes, beadingDebug] = detectIncompleteBeading(segmentedImg, mask);

beadingConf = 0;
if isBeading
    beadingConf = min(0.95, beadingDebug.beadingScore + 0.30);
    beadingConf = min(0.95, beadingConf);
end

results.isBeading      = isBeading;
results.beadingBboxes  = beadingBboxes;
results.beadingDebug   = beadingDebug;
results.beadingConf    = beadingConf;

% ====================================================================
% Summary
% ====================================================================
results.defectType = 'normal';
results.confidence = 0;
results.scores     = struct('tear', tearConf, 'beading', beadingConf);

if isTear || isBeading
    if tearConf >= beadingConf
        results.defectType = 'tear';
        results.confidence = tearConf;
    else
        results.defectType = 'beading';
        results.confidence = beadingConf;
    end
end

fprintf('  Tear          : %d (conf=%.3f)\n', isTear,    tearConf);
fprintf('  Incomplete    : %d (conf=%.3f)\n', isBeading, beadingConf);

end


function r = emptyResults()
r.defectType     = 'normal';
r.confidence     = 0;
r.scores         = struct('tear', 0, 'beading', 0);
r.isTear         = false;
r.tearBboxes     = [];
r.tearDebug      = struct();
r.tearConf       = 0;
r.isBeading      = false;
r.beadingBboxes  = [];
r.beadingDebug   = struct();
r.beadingConf    = 0;
end