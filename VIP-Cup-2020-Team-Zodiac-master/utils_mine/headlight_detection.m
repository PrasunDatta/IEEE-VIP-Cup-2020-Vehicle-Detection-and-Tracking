clc
clear all
close all

%% Image processing
img = imread('F:\VIP Cup 2020 Resources\Big Data\night\images\CLIP_20200610-213240_000643.jpg');
r =4;
c = 2;
size(img)
patch_size = 256;
figure
imshow(img)
subplot(1,3,1), imshow(img);
title('Original image');
gray_img = rgb2gray(img);
subplot(1,3,2), imshow(gray_img);
title('Grayscale image');
bin_img = imbinarize(gray_img, 0.55);
subplot(1,3,3), imshow(bin_img);
title('Binary image after thresholding');

%% First Row
figure
for i=1:r
    for j=1:c
        subplot(r,c,(i-1)*c+1), imshow (gray_img(1:patch_size, (i-1)*patch_size+1:i*patch_size));
        subplot(r,c,(i-1)*c+2), imshow (bin_img(1:patch_size, (i-1)*patch_size+1:i*patch_size));
    end
end

% subplot(r,c,1), imshow(gray_img(1:256, 1:256));
% subplot(r,c,2), imshow(bin_img(1:256, 1:256));
% 
% subplot(r,c,3), imshow(gray_img(1:256, 256:256*2));
% subplot(r,c,4), imshow(bin_img(1:256, 256:256*2));
% 
% subplot(r,c,5), imshow(gray_img(1:256, 256*2:256*3));
% subplot(r,c,6), imshow(bin_img(1:256, 256*2:256*3));
% 
% subplot(r,c,7), imshow(gray_img(1:256, 256*3:1024));
% subplot(r,c,8), imshow(bin_img(1:256, 256*3:1024));

suptitle('First Row');

%% Second Row
figure
for i=1:r
    for j=1:c
        subplot(r,c,(i-1)*c+1), imshow (gray_img(patch_size:2*patch_size, (i-1)*patch_size+1:i*patch_size));
        subplot(r,c,(i-1)*c+2), imshow (bin_img(patch_size:2*patch_size, (i-1)*patch_size+1:i*patch_size));
    end
end

% subplot(r,c,1), imshow(gray_img(256:256*2, 1:256));
% subplot(r,c,2), imshow(bin_img(256:256*2, 1:256));
% 
% subplot(r,c,3), imshow(gray_img(256:256*2, 256:256*2));
% subplot(r,c,4), imshow(bin_img(256:256*2, 256:256*2));
% 
% subplot(r,c,5), imshow(gray_img(256:256*2, 256*2:256*3));
% subplot(r,c,6), imshow(bin_img(256:256*2, 256*2:256*3));
% 
% subplot(r,c,7), imshow(gray_img(256:256*2, 256*3:1024));
% subplot(r,c,8), imshow(bin_img(256:256*2, 256*3:1024));

suptitle('Second Row');

%% Third Row
figure
for i=1:r
    for j=1:c
        subplot(r,c,(i-1)*c+1), imshow (gray_img(patch_size*2:3*patch_size, (i-1)*patch_size+1:i*patch_size));
        subplot(r,c,(i-1)*c+2), imshow (bin_img(patch_size*2:3*patch_size, (i-1)*patch_size+1:i*patch_size));
    end
end

% figure
% subplot(r,c,1), imshow(gray_img(256*2:256*3, 1:256));
% subplot(r,c,2), imshow(bin_img(256*2:256*3, 1:256));
% 
% subplot(r,c,3), imshow(gray_img(256*2:256*3, 256:256*2));
% subplot(r,c,4), imshow(bin_img(256*2:256*3, 256:256*2));
% 
% subplot(r,c,5), imshow(gray_img(256*2:256*3, 256*2:256*3));
% subplot(r,c,6), imshow(bin_img(256*2:256*3, 256*2:256*3));
% 
% subplot(r,c,7), imshow(gray_img(256*2:256*3, 256*3:1024));
% subplot(r,c,8), imshow(bin_img(256*2:256*3, 256*3:1024));
% 
suptitle('Third Row');

%% Fourth Row
figure
for i=1:r
    for j=1:c
        subplot(r,c,(i-1)*c+1), imshow (gray_img(patch_size*3:4*patch_size, (i-1)*patch_size+1:i*patch_size));
        subplot(r,c,(i-1)*c+2), imshow (bin_img(patch_size*3:4*patch_size, (i-1)*patch_size+1:i*patch_size));
    end
end
% figure
% subplot(r,c,1), imshow(gray_img(256*3:1024, 1:256));
% subplot(r,c,2), imshow(bin_img(256*3:1024, 1:256));
% 
% subplot(r,c,3), imshow(gray_img(256*3:1024, 256:256*2));
% subplot(r,c,4), imshow(bin_img(256*3:1024, 256:256*2));
% 
% subplot(r,c,5), imshow(gray_img(256*3:1024, 256*2:256*3));
% subplot(r,c,6), imshow(bin_img(256*3:1024, 256*2:256*3));
% 
% subplot(r,c,7), imshow(gray_img(256*3:1024, 256*3:1024));
% subplot(r,c,8), imshow(bin_img(256*3:1024, 256*3:1024));

suptitle('Fourth Row');
%% Average pooling on binary image
cropped_image = bin_img(200:500, 200:500);
figure
imshow(cropped_image);

idx = zeros(355,2);
count = 0;
for i=1:size(cropped_image,1)
    for j=1:size(cropped_image,2)
        if(cropped_image(i,j)==1)
            count = count+1;
            %fprintf("At index (%d, %d)\n", i,j);
            idx(count,1) = i;
            idx(count,2) = j;
        end
    end
end
count

%% 
cropped_gray_image = gray_img(256*2:256*3, 256*2:256*3);
cropped_bin_image = imbinarize(cropped_gray_image, 0.95);
%cropped_image = bin_img(1:200, 800:1024);
subplot(211), imshow(cropped_gray_image);
subplot(212), imshow(cropped_bin_image);
%disp('Applying regionprops over row 2, column 3')
stats = regionprops(cropped_bin_image);
element = zeros(1,length(stats));
for k=1:length(stats)
    element(k) = stats(k).Area;
end
element = sort(element)
% while element(1)<5
%     element = element(2:end);
% end
disp('Max area is equal to');
disp(max(element));

%% Applying a gaussian blur over the whole grayscale image
h = fspecial('gaussian', 25,5);
C = conv2(gray_img, h);
imshow(C/256);


%% Applying a gaussian blur over crops
gray_img_mod = gray_img;
to_blur = zeros(4,4);
flag = 0;
for i=1:4
    for j=1:4
        cropped_image = bin_img( (i-1)*patch_size+1:i*patch_size, (j-1)*patch_size+1:j*patch_size );
        stats = regionprops(cropped_image);
        element = zeros(1,length(stats));
        for k=1:length(stats)
            element(k) = stats(k).Area;
        end
        element = sort(element); 
        index = find(element>=4);
        element = element(index);
        if isempty(element) %full black
            to_blur(i,j) = 1;
        elseif (length(element)<4 && max(element)<100)
            to_blur(i,j) = 1; %blur these regions
        else
            to_blur(i,j) = 0;
        end
    end
end

%% Opening a text file with the same name as the image and saving to_blur
%dir = 'F:\VIP Cup 2020 Resources\Big Data\night\images\CLIP_20200610-213240_000000'
filename = 'CLIP_20200610-213240_000000.txt';
writematrix(to_blur, filename);

%% 
i = 2;
j = 1;
cropped_image = bin_img( (i-1)*256+1:i*256, (j-1)*256+1:j*256 );
stats = regionprops(cropped_image);
element = zeros(1,length(stats));
for k=1:length(stats)
    element(k) = stats(k).Area;
end
element
max(element)
imshow(cropped_image)

%% 

myDir = 'F:\VIP Cup 2020 Resources\Big Data\night\images'
myFiles = dir(fullfile(myDir,'*.jpg'));
for p=1:length(myFiles)
    baseFileName = myFiles(p).name;
    disp(baseFileName);
    fullFileName = fullfile(myDir, baseFileName);
    
    % reading the image and color thresholding
    img = imread(fullFileName);
    img_size = size(img);
    width = img_size(1);
    if width==1024
        disp('1024');
        continue
    elseif width==1280
        disp('1280');
        continue
    else
        disp(fullFileName);
        pause();
    end
        
end


%% 
figure(1)
imshow(img)
title('Original Image')
figure(2)
imshow(bin_img(256:256*2, 256:256*2))
title('Second Row, Second Column')



%% dividing the image into grids
img = imread('F:\VIP Cup 2020 Resources\Big Data\night\images\CLIP_20200610-213240_000016.jpg');
height = size(img,1)/4
width = size(img,2)/4
linewidth = 1.5
fontsize = 8
subplot(1,3,1)
imshow(img)
hold on
line([1 width*4], [height height], 'LineWidth', linewidth)
line([1 width*4], [height*2 height*2], 'LineWidth', linewidth)
line([1 width*4], [height*3 height*3], 'LineWidth', linewidth)
line([width width], [1 height*4], 'LineWidth', linewidth)
line([width*2 width*2], [1 height*4], 'LineWidth', linewidth)
line([width*3 width*3], [1 height*4], 'LineWidth', linewidth)
hold off
title('Original Image Divided Into Grids', 'FontSize', fontsize)

cropped_orig_image = img(256*2:256*3, 256:256*2, :);
cropped_gray_image = gray_img(256*2:256*3, 256:256*2);
cropped_bin_image = imbinarize(cropped_gray_image, 0.95);
subplot(1,3,2)
imshow(cropped_orig_image)
title('Third Row, Second Column','FontSize', fontsize);
subplot(1,3,3)
imshow(cropped_bin_image)
title('Third Row, Second Column, Binarized Grid','FontSize', fontsize);


%% second try
height = size(img,1)/4
width = size(img,2)/4
linewidth = 1
fontsize = 8
subplot(1,3,1)
imshow(img)
hold on
line([1 width*4], [height height], 'LineWidth', linewidth)
line([1 width*4], [height*2 height*2], 'LineWidth', linewidth)
line([1 width*4], [height*3 height*3], 'LineWidth', linewidth)
line([width width], [1 height*4], 'LineWidth', linewidth)
line([width*2 width*2], [1 height*4], 'LineWidth', linewidth)
line([width*3 width*3], [1 height*4], 'LineWidth', linewidth)
hold off
title('Original Image Divided Into Grids', 'FontSize', fontsize)

cropped_orig_image_1 = img(256*2:256*3, 256*2:256*3, :);
cropped_gray_image_1 = gray_img(256*2:256*3, 256*2:256*3);
cropped_bin_image_1 = imbinarize(cropped_gray_image_1, 0.95);

cropped_orig_image_2 = img(256:256*2, 1:256, :);
cropped_gray_image_2 = gray_img(256:256*2, 1:256);
cropped_bin_image_2 = imbinarize(cropped_gray_image_2, 0.95);

subplot(1,3,2)
imshow(cropped_bin_image_2)
title('Second Row, First Column','FontSize', fontsize);
subplot(1,3,3)
imshow(cropped_bin_image_1)
title('Third Row, Third Column','FontSize', fontsize);


%% third

figure
subplot(221), imshow(img)
title('Original')
img_c1 = imread('constraint_1.jpg');
subplot(222), imshow(img_c1)
title('After Constraint 1')
img_c2 = imread('constraint_2.jpg');
subplot(223), imshow(img_c2)
title('After Constraint 2')
img_c3 = imread('constraint_3.jpg');
subplot(224), imshow(img_c3)
title('After Constraint 3')

%% 
img = imread('F:\VIP Cup 2020 Resources\Big Data\night\images\CLIP_20200610-213240_000016.jpg');
height = size(img,1)/4
width = size(img,2)/4
linewidth = 1.5
fontsize = 8
imshow(img)
hold on
line([width width*3], [height-50 height-50], 'LineWidth', linewidth , 'Color','r')
line([width width*3], [height*3 height*3], 'LineWidth', linewidth ,'Color','r')
line([width width], [height-50 height*3], 'LineWidth', linewidth,'Color','r')
line([width*3 width*3], [height-50 height*3], 'LineWidth', linewidth,'Color','r')
% line([1 width*4], [height*2 height*2], 'LineWidth', linewidth)
% line([1 width*4], [height*3 height*3], 'LineWidth', linewidth)
% line([width width], [1 height*4], 'LineWidth', linewidth)
% line([width*2 width*2], [1 height*4], 'LineWidth', linewidth)
% line([width*3 width*3], [1 height*4], 'LineWidth', linewidth)
hold off


%% 
img = imread('F:\VIP Cup 2020 Resources\Big Data\night\images\CLIP_20200610-213240_000016.jpg');
imshow(img)









