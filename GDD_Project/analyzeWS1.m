function results = analyzeWS1(segmentedImg, mask)
% analyzeWS1  WS1 White Latex Surface Defect pipeline.
%
% Defects: Stain (shared), Dirty, Spotting
%
% Pipeline:
%   1. Extract colour features
%   2. Classify which surface defect is most likely
%   3. Run ONLY the classified detector
%   Note: Stain is handled by analyzeGlove before this is called

fprintf('\n--- WS1 Surface Analysis (White Latex) ---\n');

results = struct();

if ~any(mask(:))
    results = emptyResults();
    return;
end

% ====================================================================
% STEP 1: Extract colour features
% ====================================================================
features = extractColorFeatures(segmentedImg, mask);
results.features = features;

% ====================================================================
% STEP 2: Classify defect type
% ====================================================================
[defectType, confidence, scores] = classifyColorDefect(features);
results.defectType = defectType;
results.confidence = confidence;
results.scores     = scores;

% ====================================================================
% STEP 3: Run classified detector only
% ====================================================================
results.isDirty         = false;
results.dirtyBboxes     = [];
results.dirtyDebug      = struct();
results.dirtyConf       = 0;

results.isSpotting      = false;
results.spottingBboxes  = [];
results.spottingDebug   = struct();
results.spottingConf    = 0;

switch defectType
    case 'dirty'
        fprintf('  Running detectDirty...\n');
        [results.isDirty, results.dirtyBboxes, results.dirtyDebug] = ...
            detectDirty(segmentedImg, mask);
        results.dirtyConf = condVal(results.isDirty, confidence, 0);

    case 'spotting'
        fprintf('  Running detectSpotting...\n');
        [results.isSpotting, results.spottingBboxes, results.spottingDebug] = ...
            detectSpotting(segmentedImg, mask);
        results.spottingConf = condVal(results.isSpotting, confidence, 0);

    case 'stain'
        fprintf('  Stain handled by shared detector in analyzeGlove.\n');

    case 'normal'
        fprintf('  No surface defect classified.\n');
end

% ====================================================================
% Summary
% ====================================================================
fprintf('  Classified : %s (conf=%.3f)\n', defectType, confidence);
fprintf('  Dirty      : %d (conf=%.3f)\n', results.isDirty,    results.dirtyConf);
fprintf('  Spotting   : %d (conf=%.3f)\n', results.isSpotting, results.spottingConf);

end

function v = condVal(cond, a, b)
if cond, v = a; else, v = b; end
end

function r = emptyResults()
r.features      = struct();
r.defectType    = 'normal';
r.confidence    = 0;
r.scores        = struct('dirty',0,'spotting',0,'stain',0);
r.isDirty       = false; r.dirtyBboxes = [];
r.dirtyDebug    = struct(); r.dirtyConf = 0;
r.isSpotting    = false; r.spottingBboxes = [];
r.spottingDebug = struct(); r.spottingConf = 0;
end