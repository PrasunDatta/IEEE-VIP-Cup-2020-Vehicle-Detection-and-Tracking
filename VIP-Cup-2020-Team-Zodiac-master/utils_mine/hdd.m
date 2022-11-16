clc
clear all
close all

%% Iterate over the night directory
myDir = 'F:\VIP Cup 2020 Resources\Big Data\night_test\images'
myFiles = dir(fullfile(myDir,'*.jpg'));
patch = 4;
for p=1:length(myFiles)
    baseFileName = myFiles(p).name;
    %disp(baseFileName);
    fullFileName = fullfile(myDir, baseFileName);
    
    % reading the image and color thresholding
    img = imread(fullFileName);
    img_size = size(img);
    width = img_size(2);
    if width==1056
        disp(p)
    else
        pause(10)
    end
end
%     %% 
% %     if width==1024
% %         if patch==4
% %             patch_size = 256;
% %         elseif patch==8
% %             patch_size = 128;
% %         end
% %     elseif width==1280
% %         if patch==4
% %             patch_size = 320;
% %         elseif patch==8
% %             patch_size = 160;
% %         end
% %     end
%     patch_size = width/4;
%     gray_img = rgb2gray(img);
%     bin_img = imbinarize(gray_img, 0.95);
%     
%     % Finding out which patches to blur
%     gray_img_mod = gray_img;
%     to_blur = zeros(4,4);
%     for i=1:4
%         for j=1:4
%             cropped_image = bin_img( (i-1)*patch_size+1:i*patch_size, (j-1)*patch_size+1:j*patch_size );
%             stats = regionprops(cropped_image);
%             element = zeros(1,length(stats));
%             for k=1:length(stats)
%                 element(k) = stats(k).Area;
%             end
%             element = sort(element);
%             index = find(element>=4);
%             element = element(index);
%             if isempty(element) %full black
%                 to_blur(i,j) = 1;
%             elseif (length(element)<4 && max(element)<100)
%                 to_blur(i,j) = 1; %blur these regions
%             else
%                 to_blur(i,j) = 0;
%             end
%         end
%     end
%     
%     % Opening a text file with the same name as the image and saving to_blur
%     filename = strcat('F:\VIP Cup 2020 Resources\Big Data\blurred_labels_44_constraint_3\test\', baseFileName);
%     filename = filename(1:length(filename)-3);
%     filename = strcat(filename, 'txt')
%     writematrix(to_blur, filename);
%     
% end
