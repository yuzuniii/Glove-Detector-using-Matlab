function results = analyzeWS4(segmentedImg, mask, origImg)
% analyzeWS4  WS4 Cotton Fabric Defect pipeline.
%
% Defects:
%   - Stain                  (shared via analyzeGlove)
%   - Burn Hole / Burn Mark  (detectBurnHole)
%   - Uneven Weave           (detectUnevenWeave)
%   - Loose Thread           (stub, always false)

fprintf('\n--- WS4 Fabric Analysis (Cotton) ---\n');

results = struct();

if ~any(mask(:))
    results = emptyResults();
    return;
end

if nargin < 3, origImg = []; end

% ====================================================================
% 1. BURN HOLE
%   Bright circular void + dark charred ring.
%   Run first — most unambiguous defect.
% ====================================================================
fprintf('  Running detectBurnHole...\n');
[isBurnHole, burnBboxes, burnDebug] = detectBurnHole(segmentedImg, mask, origImg);
burnConf = 0;
if isBurnHole, burnConf = 0.90; end
results.isBurnHole = isBurnHole;
results.burnBboxes = burnBboxes;
results.burnDebug  = burnDebug;
results.burnConf   = burnConf;
fprintf('  Burn Hole     : %d (conf=%.3f)\n', isBurnHole, burnConf);

% ====================================================================
% 2. UNEVEN WEAVE
%   Local texture variance outlier in palm region.
%   Run regardless of burn hole — they are independent defects.
% ====================================================================
fprintf('  Running detectUnevenWeave...\n');
[isUnevenWeave, weaveBboxes, weaveDebug] = detectUnevenWeave(segmentedImg, mask);
weaveConf = 0;
if isUnevenWeave
    weaveConf = 0.85;  % fixed confidence — detector passed all gates
end
results.isUnevenWeave = isUnevenWeave;
results.weaveBboxes   = weaveBboxes;
results.weaveDebug    = weaveDebug;
results.weaveConf     = weaveConf;
fprintf('  Uneven Weave  : %d (conf=%.3f)\n', isUnevenWeave, weaveConf);

% ====================================================================
% 3. LOOSE THREAD — stub
% ====================================================================
results.isLooseThread = false;
results.threadBboxes  = [];
results.threadDebug   = struct();
results.threadConf    = 0;

end


function r = emptyResults()
r.isBurnHole    = false; r.burnConf  = 0; r.burnBboxes  = []; r.burnDebug  = struct();
r.isUnevenWeave = false; r.weaveConf = 0; r.weaveBboxes = []; r.weaveDebug = struct();
r.isLooseThread = false; r.threadConf = 0; r.threadBboxes = []; r.threadDebug = struct();
end