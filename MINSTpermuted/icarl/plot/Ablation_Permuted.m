%% plot results 
clear all; close all; clc;

%% baselines
dataset = 'permuted_MNIST_incremental_domain';
baseline = {'oursFewShotT2','oursFewShot','oursFewShotT05','oursFullT2','oursFull','oursFullT05','oursPartial20','oursPeusdo20'}; %'NormalNN',
lineformat = {'--','','-.'};

colorcode = {[1 0.2 0.2], ... %red
    [0, 0.4470, 0.7410], ... %blue
    [0.8500, 0.3250, 0.0980], ... %orange
    [0.9290, 0.6940, 0.1250], ... %yellow
    [0.4940, 0.1840, 0.5560], ... %purple
    [0.4660, 0.6740, 0.1880], ... %green
    [0.3010, 0.7450, 0.9330], ... %light blue
    [0.6350, 0.0780, 0.1840], ... %date
    [0, 0.5, 0], ... %dark green
    [0, 0.75, 0.75], ... %cyan
    [0.75, 0, 0.75], ... %magenta
    [0.75, 0.75, 0], ... %dark yellow
    [0.25, 0.25, 0.25]}; %dark grey 

linewidth = 2.0;
folderbase = '/home/mengmi/bossmount/zhangmengmi/proj_CL/code/MINSTpermuted/Continual-Learning-Benchmark-master/outputs/';
n_class = 10;
n_task = 20;

%plot performances on 1st task
hb = figure; hold on;
for i = 1:3
    load([folderbase dataset '/' baseline{i} '-precision_record.mat']);
    plot([1:n_task], prec(:,1)-prec(1,1), lineformat{i}, 'Color',colorcode{1},'LineWidth',linewidth);
end

for i = 1:3
    load([folderbase dataset '/' baseline{i+3} '-precision_record.mat']);
    plot([1:n_task], prec(:,1)-prec(1,1), lineformat{i}, 'Color',colorcode{2},'LineWidth',linewidth);
end

for i = 1:1
    load([folderbase dataset '/' baseline{i+6} '-precision_record.mat']);
    plot([1:n_task], prec(:,1)-prec(1,1), lineformat{i}, 'Color',colorcode{3},'LineWidth',linewidth);
end

for i = 1:1
    load([folderbase dataset '/' baseline{i+7} '-precision_record.mat']);
    plot([1:n_task], prec(:,1)-prec(1,1), lineformat{i}, 'Color',colorcode{4},'LineWidth',linewidth);
end

legend(baseline,'Location','southwest');
xlabel('Task Number');
ylabel('Change of First Task Accuracy (%)');
ylim([-10 3]);
title('Ablation in Permuted MNIST incremental task');

%plot averaged task performance
hb = figure; hold on;
for i = 1:3
    load([folderbase dataset '/' baseline{i} '-precision_record.mat']); 
    vect = nanmean(prec,2);
    plot([1:n_task], vect - vect(1), lineformat{i}, 'Color',colorcode{1},'LineWidth',linewidth);
end
for i = 1:3
    load([folderbase dataset '/' baseline{i+3} '-precision_record.mat']); 
    vect = nanmean(prec,2);
    plot([1:n_task], vect - vect(1), lineformat{i}, 'Color',colorcode{2},'LineWidth',linewidth);
end

for i = 1:1
    load([folderbase dataset '/' baseline{i+6} '-precision_record.mat']); 
    vect = nanmean(prec,2);
    plot([1:n_task], vect - vect(1), lineformat{i}, 'Color',colorcode{3},'LineWidth',linewidth);
end

for i = 1:1
    load([folderbase dataset '/' baseline{i+7} '-precision_record.mat']); 
    vect = nanmean(prec,2);
    plot([1:n_task], vect - vect(1), lineformat{i}, 'Color',colorcode{4},'LineWidth',linewidth);
end

legend(baseline,'Location','southwest');
xlabel('Task Number');
ylabel('Change of Task Average Accuracy (%)');
title('Ablation in Permuted MNIST incremental task');
ylim([-5 3]);



