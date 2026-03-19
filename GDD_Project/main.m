clc; clear; close all;
set(0, 'DefaultFigureWindowStyle', 'normal');

% ------------------------------------------------
% Add all project folders to path
% ------------------------------------------------
addpath(genpath(pwd));

% ------------------------------------------------
% STEP 1: Load image
%         Change testImage to test different images:
%
%   WS1 — White latex:
%     'glovestain.png'      → expect: Stain
%     'dirtygloves.png'     → expect: Dirty
%     'glovespotting.png'   → expect: Spotting
%
%   WS2 — Black nitrile:
%     'glovetear.png'       → expect: Tear
%     'incompleteglove.png' → expect: Incomplete Beading
%     'blackstain.jpg'      → expect: Stain
%
%   WS3 — Blue nitrile:
%     'blueglove.jpeg'       → expect: PASS
%     'insideout.png'       → expect: Inside-Out
%     'damagebyfold.png'    → expect: Fold Damage  (file: damagebyfold.png)
%     'bluescratch.png'     → expect: Scratch
%     'bluestain.png'       → expect: Stain
%
%
%   WS4 - Cotton Gloves
%   
%   unevenweave.png
%   cottonstain.png
%   cottonburnhole.png
%
% ------------------------------------------------
testImage = 'bluescratch.png';   % <-- CHANGE THIS LINE TO SWITCH IMAGES
img = imread(testImage);

% ------------------------------------------------
% STEP 2: Run full automatic pipeline
% ------------------------------------------------
results = analyzeGlove(img);

% ================================================
% FIGURES
% ================================================

% FIGURE 1: Segmentation
figure('Name','Segmentation Result','NumberTitle','off','Position',[50 600 900 250]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile; imshow(img);                title('Original');
nexttile; imshow(results.mask);       title(sprintf('Mask (%s)', results.gloveType));
nexttile; imshow(results.segmentedImg); title(sprintf('Segmented (%d px)', nnz(results.mask)));
drawnow;

% FIGURE 2: Type detection scores
figure('Name','Glove Type Detection','NumberTitle','off','Position',[980 600 380 250]);
typeScores = [results.typeDebug.scores.white,  results.typeDebug.scores.black, ...
              results.typeDebug.scores.blue,   results.typeDebug.scores.cotton];
typeLabels = {'White','Black','Blue','Cotton'};
typeCols   = [[0.9 0.9 0.8]; [0.2 0.2 0.2]; [0.2 0.4 0.9]; [0.8 0.6 0.3]];
bh = bar(typeScores,'FaceColor','flat');
for k=1:4, bh.CData(k,:) = typeCols(k,:); end
set(gca,'XTickLabel',typeLabels,'FontSize',10);
ylim([0 1]);
yline(0.35,'--k','Min threshold','FontSize',9);
title(sprintf('Type: %s (conf=%.2f)', upper(results.gloveType), results.typeConf),'FontSize',11);
ylabel('Score');
drawnow;

% FIGURE 3: Classification scores (handles all workstreams)
figure('Name','Defect Classification','NumberTitle','off','Position',[50 350 420 250]);
if isfield(results.wsResults,'scores') && isstruct(results.wsResults.scores)
    s = results.wsResults.scores;
    fn = fieldnames(s);
    classScores = cellfun(@(f) s.(f), fn)';
    classLabels = fn';
    % Capitalise labels
    classLabels = cellfun(@(x) [upper(x(1)) x(2:end)], classLabels, 'UniformOutput', false);
    nCls = numel(classScores);
    defCols = lines(nCls);
    bh2 = bar(classScores, 'FaceColor', 'flat');
    for k = 1:nCls, bh2.CData(k,:) = defCols(k,:); end
    set(gca, 'XTickLabel', classLabels, 'FontSize', 9);
    ylim([0 1]);
    yline(0.35,'--k','Threshold','FontSize',9);
    defType = '';
    if isfield(results.wsResults,'defectType'), defType = results.wsResults.defectType; end
    defConf = 0;
    if isfield(results.wsResults,'confidence'), defConf = results.wsResults.confidence; end
    title(sprintf('Defect class: %s (conf=%.2f)', upper(defType), defConf),'FontSize',11);
    ylabel('Score');
else
    text(0.5,0.5,'No classification data','HorizontalAlignment','center');
    axis off;
end
drawnow;

% FIGURE 4: Final Summary
figure('Name','GDD Result','NumberTitle','off','Position',[400 150 750 540]);
imshow(img); hold on;

if results.isDefect && ~isempty(results.bbox)
    % Draw single highest-confidence bounding box
    rectangle('Position', results.bbox, ...
        'EdgeColor', results.color, 'LineWidth', 3);

    % Label above box
    x = results.bbox(1) + results.bbox(3)/2;
    y = results.bbox(2) - 12;
    if y < 10, y = results.bbox(2) + results.bbox(4) + 18; end
    text(x, y, sprintf('%s (%.0f%%)', results.defectType, results.defectConf*100), ...
        'Color', results.color, ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor', [0 0 0 0.6]);
end

% Verdict banner
if strcmp(results.verdict, 'PASS')
    titleStr = sprintf('PASS  —  %s glove  —  No defect detected', results.gloveType);
    titleCol = [0.1 0.8 0.1];
else
    titleStr = sprintf('FAIL  —  %s glove  —  %s detected (%.0f%% conf)', ...
        results.gloveType, results.defectType, results.defectConf*100);
    titleCol = [1 0.2 0.2];
end
title(titleStr, 'FontSize', 12, 'Color', titleCol, 'FontWeight', 'bold');

annotation('textbox',[0.01 0.01 0.98 0.04], ...
    'String', sprintf('Glove type auto-detected: %s  |  Pipeline: detectGloveType → segmentGlove → detectStain → analyzeWS1', results.gloveType), ...
    'FontSize', 8, 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

hold off; drawnow;

% ================================================
% Console Summary
% ================================================
fprintf('\n========== TEST SUMMARY ==========\n');
fprintf('  Image         : %s\n', testImage);
fprintf('  Glove type    : %s (conf=%.2f)\n', results.gloveType, results.typeConf);
fprintf('  Glove pixels  : %d\n', nnz(results.mask));
fprintf('  Top defect    : %s\n', results.defectType);
fprintf('  Confidence    : %.3f\n', results.defectConf);
fprintf('  Verdict       : %s\n', results.verdict);
fprintf('===================================\n');