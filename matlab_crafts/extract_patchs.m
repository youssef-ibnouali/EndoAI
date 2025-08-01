% === PATCH SELECTION & CLASSIFICATION FOR NBI ENDOSCOPY ===
% ALGORITHM OVERVIEW:
% 1. Load the input endoscopic image (NBI format)
% 2. Convert to grayscale and apply adaptive histogram equalization
% 3. Compute local entropy (entropyfilt) to highlight structural texture
% 4. Generate a binary mask for valid tissue areas:
%    • High entropy regions (indicating texture)
%    • Mid-range pixel intensity (to avoid over/under-exposed areas)
% 5. Dilate the mask to consolidate valid areas
% 6. Slide a fixed-size patch window over the image:
%    • Reject patches that are too bright/dark or have low valid area
%    • Score valid patches based on mean entropy × valid ratio
% 7. Apply Non-Maximum Suppression (NMS) to keep top patches with minimal overlap (IoU < 0.01)
% 8. For each selected patch:
%    • Extract multiple features:
%         - Entropy, Saturation, Edge Density
%         - GLCM features (Contrast, Correlation, Homogeneity, Energy)
%         - LBP features (std)
%         - Wavelet energy (Haar)
%         - Rotated wavelet average energy
%         - Gabor filter response
%         - Fractal dimension (box-counting)
%    • Apply TreeBagger trained model for classification
% 9. Draw color-coded rectangles and class labels over the patches
% 10. Report and display the percentage of valid image area covered,
%     as well as class distribution across selected patches



start_path = 'C:/Users/ibnou/OneDrive/Bureau/matlab project/Endoscopic DataBase/';
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp','Image Files (*.jpg, *.png, *.bmp)'}, ...
                                 'Select an Endoscopic Image', start_path);
if isequal(filename,0)
    return;
end

output_dir = 'C:/Users/ibnou/OneDrive/Bureau/matlab project/cnn_patchs/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

img_path = fullfile(pathname, filename);
img = imread(img_path);


% === STEP 1: READ IMAGE ===
img = imread(img_path);
gray = rgb2gray(img);
[h, w] = size(gray);

% PARAMETERS
patch_size = 200;
gray_eq = adapthisteq(gray); % SteEqualize illumination
low_thresh  = prctile(gray_eq(:), 6.5);
high_thresh = prctile(gray_eq(:), 98.5);

step = 5;
iou_thresh = 0.01;

% GLCM Feature Extraction Function
function glcm_feats = extract_glcm_features(patch_gray)
    patch_gray = im2uint8(patch_gray);
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(patch_gray, ...
        'Offset', offsets, ...
        'Symmetric', true, ...
        'NumLevels', 256, ...
        'GrayLimits', [0 255]); 
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

    glcm_feats.Contrast    = mean(stats.Contrast);
    glcm_feats.Correlation = mean(stats.Correlation);
    glcm_feats.Energy      = mean(stats.Energy);
    glcm_feats.Homogeneity = mean(stats.Homogeneity);
end

% LBP Feature Extraction Function
function stats = extract_lbp_features(patch_gray)
    patch_gray = im2uint8(patch_gray);
    [h, w] = size(patch_gray);
    cell_size = [h w];

    lbp_hist = extractLBPFeatures(patch_gray, ...
        'Upright', true, ...
        'Normalization', 'None', ...
        'CellSize', cell_size);

    stats.mean   = mean(lbp_hist);
    stats.std    = std(lbp_hist);
    stats.energy = sum(lbp_hist.^2);
end


% === Wavelet Energy Features (Haar) ===
function energy = extract_wavelet_energy(patch_gray)
    [cA,cH,cV,cD] = dwt2(patch_gray, 'haar');
    energy = struct();
    energy.cH = sum(cH(:).^2);
    energy.cV = sum(cV(:).^2);
    energy.cD = sum(cD(:).^2);
end

% === Rotated Wavelet Features (using rotated patch) ===
function rot_energy = extract_rotated_wavelet_energy(patch_gray)
    angles = [0, 45, 90, 135];
    energy_vals = zeros(1, numel(angles));
    for i = 1:length(angles)
        rotated = imrotate(patch_gray, angles(i), 'crop');
        e = extract_wavelet_energy(rotated);
        energy_vals(i) = mean([e.cH, e.cV, e.cD]);
    end
    rot_energy = mean(energy_vals);
end

% === Fractal Dimension via Box-Counting ===
function fd = compute_fractal_dim(patch_gray)
    bw = imbinarize(mat2gray(patch_gray));
    N = size(bw,1);  % assuming square patch
    sizes = 2.^(1:floor(log2(N)));

    log_counts = [];
    log_sizes = [];

    for i = 1:length(sizes)
        s = sizes(i);
        if mod(N, s) ~= 0
            continue;  % skip non-divisible sizes
        end
        step = N / s;
        blocks = mat2cell(bw, repmat(s, 1, step), repmat(s, 1, step));
        count = sum(cellfun(@(b) any(b(:)), blocks(:)));
        log_counts(end+1) = log(count);
        log_sizes(end+1) = log(1/s);
    end

    % Fit line to log-log curve
    p = polyfit(log_sizes, log_counts, 1);
    fd = p(1);  % fractal dimension
end


% === Gabor Feature Extraction ===
function gabor_feat = extract_gabor_features(patch_gray)
    gaborArray = gabor([4 8], [0 45 90 135]);
    gaborMag = imgaborfilt(patch_gray, gaborArray);
    gabor_feat = mean(gaborMag, [1 2]);
    gabor_feat = mean(gabor_feat); % average over filters
end


% === STEP 2: ENTROPY FILTERING ===
entropy_img = entropyfilt(gray);
entropy_norm = mat2gray(entropy_img);

% === STEP 3: BINARY MASKS (entropy + intensity) ===
entropy_thresh = graythresh(entropy_norm);
mask_entropy = imbinarize(entropy_norm, entropy_thresh);
mask_intensity = gray > low_thresh & gray < high_thresh;
mask_combined = mask_entropy & mask_intensity;

% === STEP 4: SMOOTH MASK ===
se = strel('disk', 3);
mask_final = imdilate(mask_combined, se);

max_patches = floor(h / patch_size) * floor(w / patch_size);


% === ADAPTIVE SHARPNESS THRESHOLD USING WAVELET.cH ===
% cH_all = [];
% for y = 1:step:h-patch_size+1
%     for x = 1:step:w-patch_size+1
%         patch_gray = gray(y:y+patch_size-1, x:x+patch_size-1);
%         features = extract_wavelet_energy(patch_gray);
%         cH_all(end+1) = features.cH;
%     end
% end
% 
% % Compute adaptive threshold
% ch_mean = mean(cH_all);
% ch_std = std(cH_all);
% wavelet_cH_thresh = ch_mean - 0.5 * ch_std;

% === STEP 5: SLIDING WINDOW PATCH SELECTION WITH SCORE ===
candidates = [];
for y = 1:step:h-patch_size+1
    for x = 1:step:w-patch_size+1
        region = mask_final(y:y+patch_size-1, x:x+patch_size-1);
        patch_gray = gray(y:y+patch_size-1, x:x+patch_size-1);
        patch_rgb = img(y:y+patch_size-1, x:x+patch_size-1, :);
        patch_hsv = rgb2hsv(patch_rgb);
        H = patch_hsv(:,:,1); S = patch_hsv(:,:,2);

        % Rejection: too bright or too dark
        if max(patch_gray(:)) > high_thresh || min(patch_gray(:)) < low_thresh
            continue;
        end

        % Rejection: not enough valid mask
        valid_ratio = mean(region(:));
        if valid_ratio < 0.85
            continue;
        end

        % % Reject blurry patches based on adaptive wavelet energy
        % features.wavelet = extract_wavelet_energy(patch_gray);
        % if features.wavelet.cH < wavelet_cH_thresh
        %     continue;
        % end

        % Score = valid_ratio × entropy 
        entropy_mean = mean(entropy_img(y:y+patch_size-1, x:x+patch_size-1), 'all');
        score = valid_ratio * entropy_mean;

        candidates(end+1,:) = [x, y, score]; 

    end
end

% === STEP 6: NON-MAX SUPPRESSION (NMS) ===
compute_iou = @(a,b) ...
    ( max(0,min(a(1)+a(3),b(1)+b(3))-max(a(1),b(1))) * ...
      max(0,min(a(2)+a(4),b(2)+b(4))-max(a(2),b(2))) ) ...
    / (a(3)*a(4) + b(3)*b(4) - ...
       max(0,min(a(1)+a(3),b(1)+b(3))-max(a(1),b(1))) * ...
       max(0,min(a(2)+a(4),b(2)+b(4))-max(a(2),b(2))));

% Sort by score descending
if isempty(candidates)
    final_coords = [];
else
    [~, idx] = sort(candidates(:,3), 'descend');
    candidates = candidates(idx,:);
    final_coords = [];

    for i = 1:size(candidates,1)
        current = candidates(i,1:2);
        rect1 = [current(1), current(2), patch_size, patch_size];
        overlap = false;

        for j = 1:size(final_coords,1)
            rect2 = [final_coords(j,1), final_coords(j,2), patch_size, patch_size];
            if compute_iou(rect1, rect2) > iou_thresh
                overlap = true;
                break;
            end
        end

        if ~overlap
            final_coords(end+1,:) = current;
        end
    end
    % === STEP 6: SAVE PATCHES ===
    [~, base_name, ~] = fileparts(filename);
    for p = 1:size(final_coords,1)
        x = final_coords(p,1);
        y = final_coords(p,2);
        patch = img(y:y+patch_size-1, x:x+patch_size-1, :);
        outname = sprintf('%s_patch_%d.png', base_name, p);
        imwrite(patch, fullfile(output_dir, outname));
    end
end


% === STEP 7: DISPLAY RESULTS ===
figure('Name', 'Patch Extraction Stages', 'NumberTitle', 'off');
subplot(2,3,1); imshow(img); title('Original Image');
%subplot(2,3,2); imshow(gray); title('Grayscale');
subplot(2,3,2); imshow(entropy_norm); title('Entropy Filtered');
%subplot(2,3,4); imshow(mask_entropy); title('Entropy Mask');
subplot(2,3,3); imshow(mask_intensity); title('Intensity Mask');

% Display candidate patch regions (with potential overlaps)
subplot(2,3,4); imshow(img); title('Candidate Patch Regions'); hold on;
for i = 1:size(candidates,1)
    rectangle('Position', [candidates(i,1), candidates(i,2), patch_size, patch_size], ...
              'EdgeColor', 'w', 'LineWidth', 0.5); 
end
hold off;


subplot(2,3,5); imshow(img); title('Best selected Patch Regions'); hold on;

for i = 1:size(final_coords,1)
    rectangle('Position', [final_coords(i,1), final_coords(i,2), patch_size, patch_size], ...
              'EdgeColor', 'w', 'LineWidth', 0.8);
end
hold off;


% === DISPLAY IM CLASSIFICATION RESULTS (Color-Coded Patches) ===
subplot(2,3,6); imshow(img); title('Classified Patches'); hold on;

total_patches = size(final_coords, 1);

% Class counters
im_count = 0;
ag_count = 0;
dysplasia_count = 0;
cancer_count = 0;
normal_count = 0;

load('rf_model.mat', 'rf_model');

parfor i = 1:total_patches
    x = final_coords(i,1);
    y = final_coords(i,2);
    patch_rgb = img(y:y+patch_size-1, x:x+patch_size-1, :);
    patch_gray = rgb2gray(patch_rgb);
    patch_hsv = rgb2hsv(patch_rgb);
    
    % Extract features
    entropy_val = entropy(patch_gray);
    saturation = mean(patch_hsv(:,:,2), 'all');
    edges = edge(patch_gray, 'sobel');
    edge_density = sum(edges(:)) / numel(edges);
    glcm = extract_glcm_features(patch_gray);
    lbp = extract_lbp_features(patch_gray);
    wavelet = extract_wavelet_energy(patch_gray);
    rot_wavelet = extract_rotated_wavelet_energy(patch_gray);
    fractal = compute_fractal_dim(patch_gray);
    gabor = extract_gabor_features(patch_gray);

    % Build feature vector in the same order as used during training
    feature_vector = [...
        entropy_val, saturation, edge_density, ...
        glcm.Contrast, glcm.Correlation, glcm.Homogeneity, glcm.Energy, ...
        lbp.std, lbp.energy, ...
        wavelet.cH, wavelet.cV, wavelet.cD, ...
        rot_wavelet, gabor, fractal];
    
    % Predict label using trained model
    predicted_label = predict(rf_model, feature_vector);
    
    label = predicted_label{1};

    switch label
        case 'Cancer', color = 'm'; cancer_count = cancer_count + 1;
        case 'Dysplasia', color = 'r'; dysplasia_count = dysplasia_count + 1;
        case 'IM', color = 'y'; im_count = im_count + 1;
        case 'AG', color = 'c'; ag_count = ag_count + 1;
        case 'Normal', color = 'g'; normal_count = normal_count + 1;
        otherwise, color = 'w';
    end



    % Display patch info in console
    fprintf(['Patch #%d\n', ...
             '  Entropy         : %.4f\n', ...
             '  Saturation      : %.4f\n', ...
             '  Edge Density    : %.4f\n', ...
             '  GLCM            : Contrast=%.4f | Corr=%.4f | Homo=%.4f | Energy=%.4f\n', ...
             '  LBP             : Mean=%.4f | Std=%.4f | Energy=%.4f\n', ...
             '  Wavelet Energy  : cH=%.4f | cV=%.4f | cD=%.4f\n', ...
             '  Rotated Wavelet : %.4f\n', ...
             '  Gabor Feature   : %.4f\n', ...
             '  Fractal Dim     : %.4f\n\n'], ...
            i, entropy_val, saturation, edge_density, ...
            glcm.Contrast, glcm.Correlation, glcm.Homogeneity, glcm.Energy, ...
            lbp.mean, lbp.std, lbp.energy, ...
            wavelet.cH, wavelet.cV, wavelet.cD, ...
            rot_wavelet, gabor, fractal);


    % Draw rectangle and label
    rectangle('Position', [x, y, patch_size, patch_size], ...
              'EdgeColor', color, 'LineWidth', 1.3);
    text(x+3, y+15, sprintf('%d : %s', i, label), 'Color', color, 'FontSize', 10, 'FontWeight', 'bold');

    
end
hold off;


% === NEW: CALCULATE VALID PIXELS PERCENTAGE ===
% Initialize mask with correct dimensions (use size of gray image)
total_mask = false(size(gray,1), size(gray,2));

% Mark all selected patches
for i = 1:size(final_coords,1)
    x = final_coords(i,1);
    y = final_coords(i,2);
    % Ensure we don't go out of image bounds
    y_end = min(y+patch_size-1, size(gray,1));
    x_end = min(x+patch_size-1, size(gray,2));
    total_mask(y:y_end, x:x_end) = true;
end

% Calculate percentage
total_pixels = numel(gray);
valid_pixels = sum(total_mask(:));
valid_percentage = (valid_pixels / total_pixels) * 100;
fprintf('✅ %d best patches selected after NMS.\n', size(final_coords,1));
fprintf('Percentage of valid pixels in image: %.2f%%\n', valid_percentage);


% === PERCENTAGE SUMMARY FOR EACH CLASS ===
im_percentage        = (im_count        / total_patches) * 100;
ag_percentage        = (ag_count        / total_patches) * 100;
dysplasia_percentage = (dysplasia_count / total_patches) * 100;
cancer_percentage    = (cancer_count    / total_patches) * 100;
normal_percentage    = (normal_count    / total_patches) * 100;

% === DISPLAY ===
fprintf('=> Normal percentage: %.2f%%\n',     normal_percentage);
fprintf('=> AG percentage: %.2f%%\n',         ag_percentage);
fprintf('=> IM percentage: %.2f%%\n',         im_percentage);
fprintf('=> Dysplasia percentage: %.2f%%\n',  dysplasia_percentage);
fprintf('=> Cancer percentage: %.2f%%\n',     cancer_percentage);