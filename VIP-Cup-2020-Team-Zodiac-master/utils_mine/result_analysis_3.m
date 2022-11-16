clc
clear all
close all

%% Effect of changing constraint for all levels, and mechanisms
x = [1 2];
%level = [50 100];
rb_50 = [47.1 45.2]; %1st element-C1, 2nd element-C2
rg5_50 = [50.2 44.9];
rg15_50 = [48.2 47.1];

rb_200 = [42.8 38.7];
rg35_200 = [49.1 38.3];
rg45_200 = [44.6 42.4];

plot(x, rb_50, 'ro-');
hold on
plot(x, rg5_50, 'bo-');
plot(x, rg15_50, 'go-');
plot(x, rb_200, 'mo-');
plot(x, rg35_200, 'co-');
plot(x, rg45_200, 'ko-');

legend('BoxBlur, KernelWidth=50', 'GaussianBlur(Std=5), KernelWidth=50', 'GaussianBlur(Std=15), KernelWidth=50', 'BoxBlur, KernelWidth=200', 'GaussianBlur(Std=35), KernelWidth=200', 'GaussianBlur(Std=45), KernelWidth=200');
xticks([1 2]);
xticklabels({'C1','C2'});
xlim([0 3]);
xlabel('Constraints');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of changing constraints(Yolov5x)');

%% Effect of changing constraint for all levels, and mechanisms
x = [1 2];
%level = [50 100];
rb_50 = [43.4 43.3];
rg5_50 = [42.2 39.5];
rg15_50 = [42.2 38.7];

rb_200 = [40.6 41.8];
rg35_200 = [44.8 44.3];
rg45_200 = [42.8 43.6];

plot(x, rb_50, 'ro-');
hold on
plot(x, rg5_50, 'bo-');
plot(x, rg15_50, 'go-');
plot(x, rb_200, 'mo-');
plot(x, rg35_200, 'co-');
plot(x, rg45_200, 'ko-');

legend('BoxBlur, KernelWidth=50', 'GaussianBlur(Std=5), KernelWidth=50', 'GaussianBlur(Std=15), KernelWidth=50', 'BoxBlur, KernelWidth=200', 'GaussianBlur(Std=35), KernelWidth=200', 'GaussianBlur(Std=45), KernelWidth=200');
xticks([1 2]);
xticklabels({'C1','C3'});
xlim([0 3]);
xlabel('Constraints');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of changing constraints(Yolov5s)');



%% Effect of changing mechanism for all levels and all constraints
x = [1 2 3];
%level = [50 100];
r50C1 = [50.2 48.2 47.1];
r200C1 = [49.1 44.6 42.8];
r50C3 = [44.9 47.1 45.2];
r200C3 = [38.3 42.4 38.7];
plot(x, r50C1, 'ro-');
hold on
plot(x, r200C1, 'bo-');
plot(x,r50C3, 'go-');
plot(x,r200C3, 'ko-');
legend('KernelSize=50, Constraint=1', 'KernelSize=200, Constraint=1', 'KernelSize=50, Constraint=3', 'KernelSize=200, Constraint=3');
xticks([1 2 3]);
xticklabels({'Gaussian(Std=Low)','Gaussian(Std=High)', 'Box Blur'});
xlim([0 4]);
xlabel('Blur Mechanism');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of changing blur mechanism(Yolov5x)');

%% Effect of changing mechanism for all levels under C1
x = [1 2 3];
%level = [50 100];
r50C1 = [42.2 42.2 43.4];
r200C1 = [44.8 42.8 40.6];
r50C3 = [39.5 38.7 43.3];
r200C3 = [44.3 43.6 41.8];
plot(x, r50C1, 'ro-');
hold on
plot(x, r200C1, 'bo-');
plot(x,r50C3, 'go-');
plot(x,r200C3, 'ko-');
legend('KernelSize=50, Constraint=1', 'KernelSize=200, Constraint=1', 'KernelSize=50, Constraint=3', 'KernelSize=200, Constraint=3');
xticks([1 2 3]);
xticklabels({'Gaussian(Std=Low)','Gaussian(Std=High)', 'Box Blur'});
xlim([0 4]);
ylim([37 45]);
xlabel('Blur Mechanism');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of changing mechanism(Yolov5s)');

%% Effect of changing kernel size yolov5x
x = [1 2];
%level = [50 100];
rb_C1 = [47.1 42.8];
rg5_C1 = [50.2 49.1];
rg15_C1 = [48.2 44.6];

rb_C3 = [45.2 38.7];
rg35_C3 = [44.9 38.3];
rg45_C3 = [47.1 42.4];

plot(x, rb_C1, 'ro-');
hold on
plot(x, rg5_C1, 'bo-');
plot(x, rg15_C1, 'go-');
plot(x, rb_C3, 'mo-');
plot(x, rg35_C3, 'co-');
plot(x, rg45_C3, 'ko-');


legend('BoxBlur, Constraint=C1','GaussianBlur(Std=5), Constraint=C1','GaussianBlur(Std=15), Constraint=C1', 'BoxBlur, Constraint=C3','GaussianBlur(Std=35), Constraint=C3', 'GaussianBlur(Std=45), Constraint=C3');
xticks([1 2]);
xticklabels({'KernelSize = 50','KernelSize = 200'});
xlim([0 3]);
xlabel('Width of kernel');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of increasing kernel width(yolov5x)');

%% Effect of changing kernel size yolov5s
x = [1 2];
%level = [50 100];
rb_C1 = [43.4 40.6];
rg5_C1 = [42.2 44.8];
rg15_C1 = [42.2 42.8];

rb_C3 = [43.3 41.8];
rg35_C3 = [39.5 44.3];
rg45_C3 = [38.7 43.6];

plot(x, rb_C1, 'ro-');
hold on
plot(x, rg5_C1, 'bo-');
plot(x, rg15_C1, 'go-');
plot(x, rb_C3, 'mo-');
plot(x, rg35_C3, 'co-');
plot(x, rg45_C3, 'ko-');


legend('BoxBlur, Constraint=C1','GaussianBlur(Std=5), Constraint=C1','GaussianBlur(Std=15), Constraint=C1', 'BoxBlur, Constraint=C3','GaussianBlur(Std=35), Constraint=C3', 'GaussianBlur(Std=45), Constraint=C3');
xticks([1 2]);
xticklabels({'KernelSize = 50','KernelSize = 200'});
xlim([0 3]);
xlabel('Width of kernel');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of increasing kernel width(yolov5s)');








