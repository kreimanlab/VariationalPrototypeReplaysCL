%% plot results 
clear all; close all; clc;

%% baselines
dataset = 'permuted_MNIST_incremental_domain';
baseline = {'oursFullT2','EWC_online','EWC','L2','MAS','SI','SGD'}; %'NormalNN',

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
for i = 1:length(baseline)
    load([folderbase dataset '/' baseline{i} '-precision_record.mat']);
    plot([1:n_task], prec(:,1)-prec(1,1), 'Color',colorcode{i}, 'LineWidth',linewidth);
end

legend(baseline,'Location','southwest');
xlabel('Task Number');
ylabel('Change of First Task Accuracy (%)');
ylim([-50 3]);
title('Permuted MINST incremental domain');

%plot averaged task performance
hb = figure; hold on;
for i = 1:length(baseline)
    load([folderbase dataset '/' baseline{i} '-precision_record.mat']); 
    vect = nanmean(prec,2);
    plot([1:n_task], vect - vect(1), 'Color',colorcode{i},'LineWidth',linewidth);
end
legend(baseline,'Location','southwest');
xlabel('Task Number');
ylabel('Change of Task Average Accuracy (%)');
ylim([-40 3]);
title('Permuted MINST incremental domain');

%parameters of model
ParamBaselines = 2036010;
OurStoreAvg = 28*28*100;
NumProtos = ParamBaselines*2/OurStoreAvg;

ParamModel = 1*256*3*3+256*256*3*3+256*256*3*3+256*256*3*3;
