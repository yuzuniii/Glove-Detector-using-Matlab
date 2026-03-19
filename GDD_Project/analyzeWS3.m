function results = analyzeWS3(segmentedImg, mask, stainMask)
% analyzeWS3  WS3 Blue Nitrile Texture Defect pipeline.
%
% Defects: Scratch, Fold Damage, Inside-Out (+ shared Stain)

fprintf('\n--- WS3 Texture Analysis ---\n');

results = struct();

if ~any(mask(:))
    results = emptyResults();
    return;
end

% Exclude stain region from texture analysis
if nargin < 3 || isempty(stainMask)
    stainMask = false(size(mask));
end
cleanMask = mask & ~stainMask;
if nnz(cleanMask) < 0.20 * nnz(mask)
    cleanMask = mask;
end

features = extractTextureFeatures(segmentedImg, cleanMask);
results.features = features;

[defectType, confidence, scores] = classifyTextureDefect(features);
results.defectType = defectType;
results.confidence = confidence;
results.scores     = scores;

% ====================================================================
% SCRATCH
% ====================================================================
fprintf('  Running detectScratch...\n');
[isScratch, scratchBboxes, scratchDebug] = detectScratch(segmentedImg, cleanMask);
scratchConf = 0;
if isScratch && ~isempty(scratchBboxes)
    scratchConf = 0.80;
end
results.isScratch    = isScratch;
results.scratchBboxes = scratchBboxes;
results.scratchDebug  = scratchDebug;
results.scratchConf   = scratchConf;

% ====================================================================
% INSIDE-OUT
% ====================================================================
fprintf('  Running detectInsideOut (primary)...\n');
[isInsideOut, ioConf, ioDebug] = detectInsideOut(segmentedImg, mask);
results.isInsideOut = isInsideOut;
results.ioConf      = ioConf;
results.ioDebug     = ioDebug;
if isInsideOut
    props = regionprops(mask, 'BoundingBox');
    if ~isempty(props)
        results.ioBboxes = props(1).BoundingBox;
    else
        results.ioBboxes = [];
    end
end

% ====================================================================
% FOLD DAMAGE
% ====================================================================
fprintf('  Running detectDamagedByFold...\n');
[isFoldDamage, foldBboxes, foldDebug] = detectDamagedByFold(segmentedImg, cleanMask);
foldConf = 0;
if isFoldDamage && ~isempty(foldBboxes)
    foldConf = 0.85;
end
results.isFoldDamage = isFoldDamage;
results.foldBboxes   = foldBboxes;
results.foldDebug    = foldDebug;
results.foldConf     = foldConf;

fprintf('  Classified   : %s (conf=%.3f)\n', defectType, confidence);
fprintf('  Scratch      : %d (conf=%.3f)\n', isScratch,    scratchConf);
fprintf('  Inside-Out   : %d (conf=%.3f)\n', isInsideOut,  ioConf);
fprintf('  Fold Damage  : %d (conf=%.3f)\n', isFoldDamage, foldConf);

end


function r = emptyResults()
r.features     = struct();
r.defectType   = 'normal';
r.confidence   = 0;
r.scores       = struct();
r.isScratch    = false; r.scratchConf = 0; r.scratchBboxes = []; r.scratchDebug = struct();
r.isInsideOut  = false; r.ioConf = 0; r.ioBboxes = []; r.ioDebug = struct();
r.isFoldDamage = false; r.foldConf = 0; r.foldBboxes = []; r.foldDebug = struct();
end