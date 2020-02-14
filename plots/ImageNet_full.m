%% plot results 
clear all; close all; clc;

netsize = 3*64*3*3+64*128*3*3+128*128*3*3+128*256*3*3+1024*500+500*100;
display(['network size = ' num2str(netsize)]);
numproto = netsize/(3*32*32*1*100);
display(['network protos = ' num2str(numproto)]);

printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = ''; %'-fillpage'
printdirprefix = 'Figures/';
printflag = 0;
labelsize = 14;

% data = repmat(sin(1:0.01:2*pi),2,1);                            %
% data = data + randn(size(data));                                  %
% fcn_plot_areaerrorbar(data);

% %% baselines
dataset = 'split_IMAGENET_incre_class_VP';
baseline = {'VPNet','EWC_online_mnist','MAS','L2','SI','NormalNN'}; %'NormalNN',

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

linewidth = 4.0;
folderbase = '/home/mengmi/HMS/Projects/Proj_CL/code/IMAGENETincrement/Continual-Learning-Benchmark-master/outputs/';
n_class = 10;
n_task = 10;
n_repeats = 10;

hb = figure; hold on;
chance = [];
counter = 10;
for i = 1:n_task
    chance = [chance 1/counter];
    counter = counter+10;
end
for i = 1:length(baseline)
    
    DATA = [];
    for r = 1: n_repeats
        load([folderbase dataset '/' baseline{i} '_' num2str(r) '-precision_record.mat']);
        if size(prec,1) > size(prec,2)
            prec = prec';
        end
        vec = prec;
        DATA = [DATA; vec];
    end
    meanData = mean(DATA,1);
    stdData = nanstd(DATA,0,1)/sqrt(size(DATA,1));
    errorbar([1:size(DATA,2)],meanData, stdData,'color',colorcode{i}, 'Linewidth', linewidth);
end
plot([1:size(DATA,2)], chance*100, 'k--', 'Linewidth', linewidth);
legend(baseline,'Location','northeast','FontWeight','Bold');
xlabel('Task Number','FontSize',labelsize,'FontWeight','Bold');
ylabel('Average Task Classification Accuracy (%)','FontSize',labelsize,'FontWeight','Bold');
xlim([0.5 10.5]);
%ylim([-70 3]);
%title('Few Shot in Split CIFAR10 incremental class');
set(gca,'TickDir','out');
set(gca,'Box','Off');
legend({'ours','EWConline','MAS','L2','SI','SGD','Chance'},'FontSize',16,'FontWeight','Bold');

if printflag == 1
    printfilename = 'fewshot_CIFAR_class_avg';
    set(hb,'Position',[1361         669         560         420]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end