% === PATCH SELECTION & CLASSIFICATION FOR NBI ENDOSCOPY ===
% ALGORITHM OVERVIEW:
% 1. Load the input endoscopic image (NBI JPG format)
% 2. Convert to grayscale and compute local entropy using entropyfilt
% 3. Generate a binary mask for valid tissue regions:
%    • High entropy (indicating texture and structure)
%    • Mid-range pixel intensity (not overexposed or underexposed)
% 4. Dilate the binary mask to connect nearby valid areas
% 5. Slide a window of fixed patch size over the image (step-wise):
%    • For each patch, compute: Mean entropy, Mean saturation, Edge density (via edge detector), Contrast (standard deviation of intensity)
%    • Reject patches that are too bright, too dark, or invalid
%    • Score remaining patches based on entropy × valid ratio
% 6. Sort patches by score and apply Non-Maximum Suppression (NMS)
%    • Keeps only top-scoring, non-overlapping regions (IoU < 0.01)
% 7. Classify each remaining patch into one of five classes:
%    • IM (Intestinal Metaplasia): high entropy + low saturation
%    • AG (Atrophic Gastritis): low entropy + low edge density + low contrast
%    • Dysplasia: high entropy + high saturation
%    • Cancer: high entropy + high edge density
%    • Normal: not matching any pathological criteria
% 8. Draw colored rectangles and labels over detected patches
% 9. Summarize the percentages for each class over all valid patches


function [result_img_path, scores] = classify_nbi_image(image_path, output_dir)
    patch_size = 200;
    low_thresh = 7;    
    high_thresh = 220;  
    step = 7;
    iou_thresh = 0.01;

    % === STEP 1: READ IMAGE ===
    img = imread(image_path);
    gray = rgb2gray(img);
    [h, w] = size(gray);

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
            if max(patch_gray(:)) > 235 || min(patch_gray(:)) < 7
                continue;
            end

            % Rejection: not enough valid mask
            valid_ratio = mean(region(:));
            if valid_ratio < 0.85
                continue;
            end

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
    end
    % === Visualization (draw on figure and save image) ===
    figure('Visible','off');
    imshow(img); hold on;

    total = size(final_coords,1);
    im_count = 0;
    ag_count = 0;
    dysplasia_count = 0;
    cancer_count = 0;
    normal_count = 0;

    for i = 1:total
        x = final_coords(i,1);
        y = final_coords(i,2);
        patch = img(y:y+patch_size-1, x:x+patch_size-1, :);
        patch_hsv = rgb2hsv(patch);
        patch_gray = rgb2gray(patch);


        % Extract features
        entropy_val = entropy(patch_gray);
        saturation = mean(patch_hsv(:,:,2), 'all');
        edges = edge(patch_gray, 'sobel');
        edge_density = sum(edges(:)) / numel(edges);
        contrast = std2(patch_gray);

        % Classification rules
        if entropy_val > 8 && edge_density > 0.02 && contrast > 20
            label = 'Cancer'; color = 'm'; cancer_count = cancer_count + 1;
        elseif entropy_val > 8 && edge_density > 0.01 && contrast > 20
            label = 'Dysplasia'; color = 'r'; dysplasia_count = dysplasia_count + 1;
        elseif entropy_val > 6.1 && saturation < 0.6
            label = 'IM'; color = 'y'; im_count = im_count + 1;
        elseif entropy_val <= 5.6 && edge_density < 0.05 && contrast < 14
            label = 'AG'; color = 'c'; ag_count = ag_count + 1;
        else
            label = 'Normal'; color = 'g'; normal_count = normal_count + 1;
        end

        rectangle('Position', [x, y, patch_size, patch_size], ...
                  'EdgeColor', color, 'LineWidth', 1.2);
        text(x+3, y+15, label, 'Color', color, 'FontSize', 9, 'FontWeight', 'bold');
    end

    hold off;

    % Save the figure as image
    frame = getframe(gca);
    output_img = frame.cdata;
    result_img_path = fullfile(output_dir, 'result.png');
    imwrite(output_img, result_img_path);

    % === Patch Class Percentages ===
    scores = struct();
    scores.IM = (im_count / total) * 100;
    scores.AG = (ag_count / total) * 100;
    scores.Dysplasia = (dysplasia_count / total) * 100;
    scores.Cancer = (cancer_count / total) * 100;
    scores.Normal = (normal_count / total) * 100;
end
