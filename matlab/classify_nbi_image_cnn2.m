
function [result_img_path, scores] = classify_nbi_image_cnn2(image_path, output_dir)
    patch_size = 200;
    step = 5;
    iou_thresh = 0.01;

    % === STEP 1: READ IMAGE ===
    img = imread(image_path);
    gray = rgb2gray(img);
    [h, w] = size(gray);
    gray_eq = adapthisteq(gray);
    low_thresh  = prctile(gray_eq(:), 6.5);
    high_thresh = prctile(gray_eq(:), 98.5);

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
            if max(patch_gray(:)) > high_thresh || min(patch_gray(:)) < low_thresh
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
    
    % === Préparer dossier temporaire ===
    temp_dir = "temp_patches";
    if ~exist(temp_dir, 'dir')
        mkdir(temp_dir);
    end

    % === Sauver tous les patches dans un dossier ===
    coords = final_coords;
    total = size(coords, 1);
    patch_paths = strings(total, 1);

    for i = 1:total
        x = coords(i,1);
        y = coords(i,2);
        patch_rgb = img(y:y+patch_size-1, x:x+patch_size-1, :);
        patch_path = fullfile(temp_dir, sprintf("patch_%d.png", i));
        imwrite(patch_rgb, patch_path);
        patch_paths(i) = patch_path;
    end

    % === Appeler Python pour tout prédire ===
    system("python matlab/predict_batch_folder.py");

    % === Lire les prédictions ===
    pred_file = fullfile(temp_dir, "predictions.txt");
    labels = readlines(pred_file);

    % === Visualisation ===
    figure('Visible','off');
    imshow(img); hold on;

    im_count = 0; ag_count = 0; dysplasia_count = 0; cancer_count = 0; normal_count = 0;

    for i = 1:total
        label = strtrim(labels(i));
        x = coords(i,1);
        y = coords(i,2);

        switch label
            case 'Cancer', color = 'm'; cancer_count = cancer_count + 1;
            case 'Dysplasia', color = 'r'; dysplasia_count = dysplasia_count + 1;
            case 'IM', color = 'y'; im_count = im_count + 1;
            case 'AG', color = 'c'; ag_count = ag_count + 1;
            case 'Normal', color = 'g'; normal_count = normal_count + 1;
            otherwise, color = 'w';
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
