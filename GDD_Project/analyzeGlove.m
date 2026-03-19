function results = analyzeGlove(img)
% analyzeGlove  Master orchestrator for the GDD pipeline.

fprintf('\n========== GDD Analysis ==========\n');

% ====================================================================
% STEP 1: Detect glove type
% ====================================================================
fprintf('Step 1: Detecting glove type...\n');
[gloveType, typeConf, typeDebug] = detectGloveType(img);
results.gloveType = gloveType;
results.typeConf  = typeConf;
results.typeDebug = typeDebug;

% ====================================================================
% STEP 2: Segment glove
% ====================================================================
fprintf('Step 2: Segmenting glove (%s)...\n', gloveType);
if strcmp(gloveType, 'blue') && isfield(typeDebug, 'isInsideOutOverride') && typeDebug.isInsideOutOverride
    fprintf('Step 2: Segmenting glove (inside-out — using white segmenter)...\n');
    [mask, segmentedImg, segDebug] = segmentGlove(img, 'white');
else
    [mask, segmentedImg, segDebug] = segmentGlove(img, gloveType);
end
results.mask         = mask;
results.segmentedImg = segmentedImg;
results.segDebug     = segDebug;

if ~any(mask(:))
    warning('analyzeGlove: segmentation failed');
    results = failedResults(gloveType);
    return;
end

% ====================================================================
% STEP 3: Route to workstream
% ====================================================================
fprintf('Step 3: Running workstream for %s...\n', gloveType);

preStainMask = [];
if strcmp(gloveType, 'blue')
    fprintf('Step 3a: Pre-running detectStain for blue glove...\n');
    [preIsStain, ~, preStainDebug] = detectStain(segmentedImg, mask);
    if preIsStain && isfield(preStainDebug, 'filteredMask')
        pixelStainMask = preStainDebug.filteredMask;
        stainCoverage  = nnz(pixelStainMask) / max(nnz(mask), 1);
        fprintf('Step 3a: Stain pixel coverage=%.2f\n', stainCoverage);
        if stainCoverage > 0.03
            preStainMask = imdilate(pixelStainMask, strel('disk', 15));
            preStainMask = preStainMask & mask;
        end
    end
end

switch gloveType
    case 'white'
        wsResults = analyzeWS1(segmentedImg, mask);
    case 'black'
        wsResults = analyzeWS2(segmentedImg, mask, img);
    case 'blue'
        wsResults = analyzeWS3(segmentedImg, mask, preStainMask);
    case 'cotton'
        wsResults = analyzeWS4(segmentedImg, mask, img);
    otherwise
        wsResults = analyzeWS1(segmentedImg, mask);
end

results.wsResults = wsResults;

% ====================================================================
% STEP 4: Shared stain detection
% ====================================================================
runStainDetector = true;

if strcmp(gloveType, 'white')
    wsDefectType = '';
    if isfield(wsResults, 'defectType'), wsDefectType = wsResults.defectType; end
    if ~strcmp(wsDefectType, 'stain')
        runStainDetector = false;
        fprintf('Step 4: Skipping detectStain (WS1 says ''%s'', not stain)\n', wsDefectType);
    end
end

if strcmp(gloveType, 'black')
    hasTear    = isfield(wsResults,'isTear')    && wsResults.isTear;
    hasBeading = isfield(wsResults,'isBeading') && wsResults.isBeading;
    if hasTear || hasBeading
        runStainDetector = false;
        fprintf('Step 4: Skipping detectStain (WS2 found shape defect)\n');
    end
end

if strcmp(gloveType, 'blue')
    hasFold      = isfield(wsResults,'isFoldDamage') && wsResults.isFoldDamage && wsResults.foldConf > 0.70;
    hasInsideOut = isfield(wsResults,'isInsideOut')  && wsResults.isInsideOut  && wsResults.ioConf   > 0.70;
    if hasFold || hasInsideOut
        runStainDetector = false;
        fprintf('Step 4: Skipping detectStain (WS3 found high-conf texture defect)\n');
    end
end

% Cotton: skip stain if burn hole or uneven weave found
if strcmp(gloveType, 'cotton')
    hasBurn  = isfield(wsResults,'isBurnHole')    && wsResults.isBurnHole    && wsResults.burnConf  > 0.70;
    hasWeave = isfield(wsResults,'isUnevenWeave') && wsResults.isUnevenWeave && wsResults.weaveConf > 0.50;
    if hasBurn || hasWeave
        runStainDetector = false;
        fprintf('Step 4: Skipping detectStain (WS4 found burn hole or uneven weave)\n');
    end
end

if runStainDetector
    fprintf('Step 4: Running shared stain detection...\n');
    [isStain, stainBboxes, stainDebug] = detectStain(segmentedImg, mask);
else
    isStain     = false;
    stainBboxes = [];
    stainDebug  = struct();
end

results.allDefects.isStain     = isStain;
results.allDefects.stainBboxes = stainBboxes;
results.allDefects.stainDebug  = stainDebug;
results.allDefects.stainConf   = getConf(isStain, stainBboxes, mask);

% ====================================================================
% STEP 5: Collect defect results
% ====================================================================
defectList = {};

if isStain
    defectList{end+1} = struct('name','Stain','conf',results.allDefects.stainConf,...
        'bbox',getBbox(stainBboxes),'color',[1 0 1]);
end

wsDefects = getWSDefects(wsResults, gloveType, mask);
for i = 1:numel(wsDefects)
    defectList{end+1} = wsDefects{i};
end

% ====================================================================
% STEP 6: Select highest confidence defect
% ====================================================================
if isempty(defectList)
    results.isDefect   = false;
    results.defectType = 'none';
    results.defectConf = 0;
    results.bbox       = [];
    results.color      = [0 1 0];
    results.verdict    = 'PASS';
else
    confs = cellfun(@(d) d.conf, defectList);
    [~, topIdx] = max(confs);
    topDefect   = defectList{topIdx};
    results.isDefect   = true;
    results.defectType = topDefect.name;
    results.defectConf = topDefect.conf;
    results.bbox       = topDefect.bbox;
    results.color      = topDefect.color;
    results.verdict    = 'FAIL';
end

results.allDefectList = defectList;

fprintf('\n========== Results ==========\n');
fprintf('  Glove type   : %s (conf=%.2f)\n', gloveType, typeConf);
fprintf('  Glove pixels : %d\n', nnz(mask));
fprintf('  Defects found: %d\n', numel(defectList));
if results.isDefect
    fprintf('  TOP DEFECT   : %s (conf=%.3f)\n', results.defectType, results.defectConf);
end
fprintf('  VERDICT      : %s\n', results.verdict);
fprintf('==============================\n\n');

end


% ------------------------------------------------------------------
function conf = getConf(isDetected, bboxes, mask)
if ~isDetected || isempty(bboxes), conf = 0; return; end
gloveArea = max(nnz(mask), 1);
bboxArea  = sum(bboxes(:,3) .* bboxes(:,4));
conf      = min(0.95, 0.50 + 0.45 * min(bboxArea/gloveArea * 10, 1));
end

function bb = getBbox(bboxes)
if isempty(bboxes), bb = [];
elseif size(bboxes,1) == 1, bb = bboxes;
else
    x1 = min(bboxes(:,1)); y1 = min(bboxes(:,2));
    x2 = max(bboxes(:,1)+bboxes(:,3)); y2 = max(bboxes(:,2)+bboxes(:,4));
    bb = [x1, y1, x2-x1, y2-y1];
end
end

function wsDefects = getWSDefects(wsResults, gloveType, mask)
wsDefects = {};
switch gloveType
    case 'white'
        if isfield(wsResults,'isDirty') && wsResults.isDirty
            wsDefects{end+1} = struct('name','Dirty','conf',wsResults.scores.dirty,...
                'bbox',getBbox(wsResults.dirtyBboxes),'color',[1 0.4 0]);
        end
        if isfield(wsResults,'isSpotting') && wsResults.isSpotting
            wsDefects{end+1} = struct('name','Spotting','conf',wsResults.scores.spotting,...
                'bbox',getBbox(wsResults.spottingBboxes),'color',[0.2 0.8 0.2]);
        end

    case 'blue'
        if isfield(wsResults,'isScratch') && wsResults.isScratch
            wsDefects{end+1} = struct('name','Scratch','conf',wsResults.scratchConf,...
                'bbox',getBbox(wsResults.scratchBboxes),'color',[0 1 1]);
        end
        if isfield(wsResults,'isFoldDamage') && wsResults.isFoldDamage
            wsDefects{end+1} = struct('name','Fold Damage','conf',wsResults.foldConf,...
                'bbox',getBbox(wsResults.foldBboxes),'color',[1 1 0]);
        end
        if isfield(wsResults,'isInsideOut') && wsResults.isInsideOut
            allMaskProps = regionprops(mask | wsResults.ioDebug.lowSatClean, 'BoundingBox');
            if ~isempty(allMaskProps)
                areas = regionprops(mask | wsResults.ioDebug.lowSatClean, 'Area');
                [~,bigIdx] = max([areas.Area]);
                ioBB = allMaskProps(bigIdx).BoundingBox;
            else
                ioBB = getBbox(wsResults.ioBboxes);
            end
            wsDefects{end+1} = struct('name','Inside-Out','conf',wsResults.ioConf,...
                'bbox',ioBB,'color',[1 0.5 0]);
        end

    case 'black'
        if isfield(wsResults,'isTear') && wsResults.isTear
            wsDefects{end+1} = struct('name','Tear','conf',wsResults.tearConf,...
                'bbox',getBbox(wsResults.tearBboxes),'color',[1 0.3 0.3]);
        end
        if isfield(wsResults,'isBeading') && wsResults.isBeading
            wsDefects{end+1} = struct('name','Incomplete Beading','conf',wsResults.beadingConf,...
                'bbox',getBbox(wsResults.beadingBboxes),'color',[1 1 0]);
        end

    case 'cotton'
        if isfield(wsResults,'isBurnHole') && wsResults.isBurnHole
            wsDefects{end+1} = struct('name','Burn Hole','conf',wsResults.burnConf,...
                'bbox',getBbox(wsResults.burnBboxes),'color',[1 0.3 0]);
        end
        if isfield(wsResults,'isUnevenWeave') && wsResults.isUnevenWeave
            wsDefects{end+1} = struct('name','Uneven Weave','conf',wsResults.weaveConf,...
                'bbox',getBbox(wsResults.weaveBboxes),'color',[0.2 0.8 1]);
        end
        if isfield(wsResults,'isLooseThread') && wsResults.isLooseThread
            wsDefects{end+1} = struct('name','Loose Thread','conf',wsResults.threadConf,...
                'bbox',getBbox(wsResults.threadBboxes),'color',[0.3 1 1]);
        end
end
end

function r = failedResults(gloveType)
r.gloveType = gloveType; r.typeConf = 0; r.mask = []; r.segmentedImg = [];
r.isDefect = false; r.defectType = 'segmentation_failed'; r.defectConf = 0;
r.bbox = []; r.color = [1 0 0]; r.verdict = 'ERROR';
r.allDefects = struct(); r.wsResults = struct(); r.allDefectList = {};
end