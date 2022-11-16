%% Effect of changing kernel size keeping mechanism fixed
x = [1 2];
%level = [50 100];
rC1 = [47.1 47.8];
rC3 = [45.2 46.2];
plot(x, rC1, 'ro-');
hold on
plot(x, rC3, 'bo-');
legend('C1', 'C3');
xticks([1 2]);
xticklabels({'KernelSize = 50','KernelSize = 200'});
xlim([0 3]);
xlabel('Width of kernel');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of increasing kernel width for box blur');


%% Effect of changing blur mechanism keeping kernel size fixed at 50
x = [1 2 3];
%level = [50 100];
rC1 = [51 51.3 47];
rC3 = [51.6 47.1 45.2];
plot(x, rC1, 'ro-');
hold on
plot(x, rC3, 'bo-');
legend('C1', 'C3');
xticks([1 2 3]);
xticklabels({'Gaussian(Std=5)','Gaussian(Std=15)', 'Box Blur'});
xlim([0 4]);
xlabel('Blur Mechanism');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of changing mechanism for kernel size 50');


%% Effect of changing constraints for same level and mechanism
x = [1 2];
%level = [50 100];
rb = [47 45.2];
rg5 = [51 51.6];
rg15 = [51.3 47.1];
plot(x, rb, 'ro-');
hold on
plot(x, rg5, 'bo-');
plot(x, rg15, 'go-');
legend('BoxBlur', 'GaussianBlur(a)', 'GaussianBlur(b)');
xticks([1 2]);
xticklabels({'C1','C3'});
xlim([0 3]);
xlabel('Constraints');
ylabel('mAP@IoU=0.5:0.95');
title('Effect of changing constraint for kernel size 50');







