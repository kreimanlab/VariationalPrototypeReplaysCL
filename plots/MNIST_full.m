%% plot results 
clear all; close all; clc;

printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = ''; %'-fillpage'
printdirprefix = '/media/mengmi/MimiDrive/Publications/NIPS_CL2019/nfigure/';
printflag = 0;
labelsize = 12;

%% baselines
dataset = 'permuted_MNIST_incre_domain_VP';
baseline = {'VPNet','EWC_online','EWC','L2','MAS','SI','NormalNN'}; %'NormalNN',

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
folderbase = '/home/mengmi/HMS/Projects/Proj_CL/code/MINSTpermuted/Continual-Learning-Benchmark-master/outputs/';
n_class = 10;
n_task = 50;
n_repeats = 10;

%plot performances on 1st task
firstavg = [];

hb = figure; hold on;
for i = 1:length(baseline)
    
    DATA = [];
    for r = 1: n_repeats
        filename = [folderbase dataset '/' baseline{i} '_' num2str(r) '-precision_record.mat'];
        if exist(filename, 'file') == 2  
            load(filename);
            vect = prec(:,1)';
            DATA = [DATA; vect];   
        end
    end
    meanData = nanmean(DATA,1);
    %firstavg = [firstavg nanmean(meanData)];
    stdData = nanstd(DATA,0,1)/sqrt(size(DATA,1));
    errorbar([1:size(DATA,2)],meanData, stdData,'color',colorcode{i}, 'Linewidth', linewidth);
end
plot([1:size(DATA,2)],10*ones(1,size(DATA,2)),'k--', 'Linewidth', linewidth);
xlabel('Task Number','FontSize',labelsize);
ylabel('First Task Classification Accuracy (%)','FontSize',labelsize);
legend(baseline,'Location','southwest');

ylim([0 100]);
%title('Permuted MINST incremental domain');
xlim([0.5 size(DATA,2)]);
set(gca,'TickDir','out');
set(gca,'Box','Off');
legend({'ours','EWConline','EWC','L2','MAS','SI','SGD','Chance'},'Location','southwest');

if printflag == 1
    printfilename = 'main_MNIST_domain_first';
    set(hb,'Position',[680         634        1233         330]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');    
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end


%plot averaged task performance
hb = figure; hold on;
for i = 1:length(baseline)
    DATA = [];
    for r = 1: n_repeats
        
        filename = [folderbase dataset '/' baseline{i} '_' num2str(r) '-precision_record.mat'];
        if exist(filename, 'file') == 2  
            load(filename);
            %load([folderbase dataset '/' baseline{i} '_' num2str(r) '-precision_record.mat']); 
            vect = nanmean(prec,2)';
            DATA = [DATA; vect];  
        end
    end
    meanData = nanmean(DATA,1);
    stdData = nanstd(DATA,0,1)/sqrt(size(DATA,1));
    errorbar([1:size(DATA,2)],meanData, stdData,'color',colorcode{i}, 'Linewidth', linewidth);
end
plot([1:size(DATA,2)],10*ones(1,size(DATA,2)),'k--', 'Linewidth', linewidth);

legend(baseline,'Location','southwest');
xlabel('Task Number','FontSize',labelsize);
ylabel('Average Task Classification Accuracy (%)','FontSize',labelsize);
ylim([0 100]);
%title('Permuted MINST incremental domain');
xlim([0.5 size(DATA,2)]);
set(gca,'TickDir','out');
set(gca,'Box','Off');
legend({'ours','EWConline','EWC','L2','MAS','SI','SGD','Chance'},'Location','southwest');

if printflag == 1
    printfilename = 'main_MNIST_domain_avg';
    set(hb,'Position',[680         634        1233         330]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');    
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end