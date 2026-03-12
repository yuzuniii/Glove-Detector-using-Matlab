clc;
clear;
close all;

% ---------------------------------------------------------
% MAIN EXECUTION
% ---------------------------------------------------------
img = imread('blueglove.jpeg');

% Call the strictly manual function
[mask, segmentedImg] = getGloveMaskManual(img);

% Visualization
figure;
subplot(1,2,1); imshow(mask); title('Manual Binary Mask');
subplot(1,2,2); imshow(segmentedImg); title('Manual Segmented Glove');

% ---------------------------------------------------------
% STRICTLY MANUAL FUNCTIONS (NO TOOLBOX)
% ---------------------------------------------------------
function [mask, segmentedImg] = getGloveMaskManual(img)
    img = double(img) / 255;
    [rows, cols, dims] = size(img);
    
    % 1. MANUAL GAUSSIAN BLUR (Replacing imfilter)
    sigma = 1; kernelSize = 5;
    [x, y] = meshgrid(-2:2);
    G = exp(-(x.^2 + y.^2) / (2*sigma^2));
    G = G / sum(G(:));
    
    img_blur = zeros(size(img));
    pad = 2;
    for d = 1:dims
        padded = padarray_manual(img(:,:,d), pad);
        for i = 1:rows
            for j = 1:cols
                region = padded(i:i+4, j:j+4);
                img_blur(i,j,d) = sum(sum(region .* G));
            end
        end
    end

    % 2. MANUAL RGB TO HSV (Isolating Saturation)
    % Saturation is excellent for identifying colored gloves [cite: 78]
    S = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            rgb = img_blur(i,j,:);
            Cmax = max(rgb);
            Cmin = min(rgb);
            delta = Cmax - Cmin;
            if Cmax ~= 0
                S(i,j) = delta / Cmax;
            else
                S(i,j) = 0;
            end
        end
    end

    % 3. MANUAL OTSU THRESHOLDING (Replacing imhist/graythresh)
    nbins = 256;
    counts = zeros(nbins, 1);
    S_scaled = round(S * 255);
    for i = 1:rows
        for j = 1:cols
            val = S_scaled(i,j) + 1;
            counts(val) = counts(val) + 1;
        end
    end
    
    p = counts / sum(counts);
    omega = cumsum(p);
    mu = cumsum(p .* (1:nbins)');
    mu_t = mu(end);
    sigma_b_squared = (mu_t * omega - mu).^2 ./ (omega .* (1 - omega) + eps);
    [~, idx] = max(sigma_b_squared);
    level = (idx - 1) / 255;
    bw = S > level;

    % 4. MANUAL MORPHOLOGICAL OPS (Replacing imopen/imclose)
    % Used to clean background and fill holes [cite: 34, 46]
    se = ones(5);
    % Dilation
    dilated = manual_morph(bw, se, 'dilate');
    % Erosion
    mask = manual_morph(dilated, se, 'erode');

    % 5. GENERATE SEGMENTED IMAGE
    segmentedImg = img;
    for d = 1:3
        channel = segmentedImg(:,:,d);
        channel(~mask) = 0;
        segmentedImg(:,:,d) = channel;
    end
end

% Helper: Manual Padding
function B = padarray_manual(A, p)
    [r, c] = size(A);
    B = zeros(r+2*p, c+2*p);
    B(p+1:p+r, p+1:p+c) = A;
end

% Helper: Manual Erosion/Dilation
function out = manual_morph(bw, se, type)
    [r, c] = size(bw);
    out = false(r, c);
    pad = 2;
    padded = padarray_manual(double(bw), pad);
    num_se = sum(se(:));
    for i = 1:r
        for j = 1:c
            region = padded(i:i+4, j:j+4);
            s = sum(sum(region .* se));
            if strcmp(type, 'dilate')
                out(i,j) = s > 0;
            else % erode
                out(i,j) = s == num_se;
            end
        end
    end
end