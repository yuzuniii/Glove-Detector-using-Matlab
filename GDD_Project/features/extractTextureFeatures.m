function [features, featureNames, featureMap] = extractTextureFeatures(segmentedImg, mask)

gray = rgb2gray(segmentedImg);
gray(~mask) = 0;

featureNames = {'edgeDensity', 'gradientMean', 'gradientStd', ...
    'textureVariance', 'entropy', 'glcmContrast'};

if ~any(mask(:))
    features = zeros(1, numel(featureNames));
    featureMap = cell2struct(num2cell(features), featureNames, 2);
    return;
end

edges = edge(gray, 'sobel');
edges(~mask) = 0;
edgeDensity = sum(edges(:)) / sum(mask(:));

[Gmag, ~] = imgradient(gray);
Gmag(~mask) = 0;
gradientValues = double(Gmag(mask));
gradientMean = mean(gradientValues);
gradientStd = std(gradientValues);

grayValues = double(gray(mask));
textureVariance = var(grayValues);
textureEntropy = entropy(gray(mask));

interiorMask = imerode(mask, strel('disk', 5));
if ~any(interiorMask(:))
    interiorMask = mask;
end

[rowIdx, colIdx] = find(interiorMask);
glcmContrast = 0;

if ~isempty(rowIdx)
    minRow = min(rowIdx);
    maxRow = max(rowIdx);
    minCol = min(colIdx);
    maxCol = max(colIdx);

    patchHeight = min(64, maxRow - minRow + 1);
    patchWidth = min(64, maxCol - minCol + 1);

    centerRow = round(mean(rowIdx));
    centerCol = round(mean(colIdx));

    startRow = max(minRow, centerRow - floor(patchHeight / 2));
    startCol = max(minCol, centerCol - floor(patchWidth / 2));
    endRow = min(maxRow, startRow + patchHeight - 1);
    endCol = min(maxCol, startCol + patchWidth - 1);

    startRow = max(minRow, endRow - patchHeight + 1);
    startCol = max(minCol, endCol - patchWidth + 1);

    patch = gray(startRow:endRow, startCol:endCol);
    patchMask = interiorMask(startRow:endRow, startCol:endCol);

    if any(patchMask(:))
        validRows = find(all(patchMask, 2));
        validCols = find(all(patchMask, 1));

        if ~isempty(validRows) && ~isempty(validCols)
            patch = patch(validRows, validCols);
        else
            patch = patch(patchMask);
            patchSide = floor(sqrt(numel(patch)));
            patch = reshape(patch(1:patchSide * patchSide), patchSide, patchSide);
        end

        patch = double(patch);
        if ~isempty(patch) && numel(unique(patch(:))) > 1
            patchQuantized = uint8(rescale(patch, 0, 7));
            glcm = graycomatrix(patchQuantized, 'NumLevels', 8, 'Offset', [0 1]);
            glcmStats = graycoprops(glcm, 'Contrast');
            glcmContrast = glcmStats.Contrast;
        else
            fprintf('GLCM patch is constant or too small; returning 0.\n');
        end
    else
        fprintf('GLCM patch has no valid interior pixels; returning 0.\n');
    end
else
    fprintf('No interior patch available for GLCM; returning 0.\n');
end

features = [edgeDensity, gradientMean, gradientStd, ...
    textureVariance, textureEntropy, glcmContrast];
featureMap = cell2struct(num2cell(features), featureNames, 2);

end
