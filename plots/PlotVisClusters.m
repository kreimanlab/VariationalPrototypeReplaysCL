clear all; close all; clc;

addpath('tsne');

printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = ''; %'-fillpage'
printdirprefix = 'Figures/';
printflag = 1;
labelsize=12;

prefixmodel = 'vis_representation'; 
%ProtoCLFCNetFull50T2, ProtoCLFCNetFewShotT2, ProtoCLFCNetFull50Pseudo, ProtoCLFCNetFull50UniformMem
directory = '../../../CIFARincrement/vizClusters_vProtoCL/results_viz/';
tasklist = [1 3];
no_dims = 2;
initial_dims = 500;
perplexity = 6;
classlist=[2:1:10];
nsamples = 100;
counter = 1;
classes = {'plane', 'car', 'bird', 'cat',...
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};

colorcode = {[1 0.2 0.2], ... %red
    [0, 0.5, 0], ... %dark green
    [0, 0.75, 0.75], ... %cyan
    [0.75, 0, 0.75], ... %magenta
    [0.75, 0.75, 0], ... %dark yellow
    [0.8500, 0.3250, 0.0980], ... %orange
    [0.9290, 0.6940, 0.1250], ... %yellow
    [0.4940, 0.1840, 0.5560], ... %purple
    [0.4660, 0.6740, 0.1880], ... %green
    [0.3010, 0.7450, 0.9330], ... %light blue
    [0.6350, 0.0780, 0.1840], ... %date    
    [0.25, 0.25, 0.25]}; %dark grey
       
colorcode = cell2mat(colorcode);
colorcode = reshape(colorcode, 3, length(colorcode)/3)';

%hb = figure;
for i = tasklist
    load([directory prefixmodel '_' num2str(i) '.mat']);
    train_X = reshape(protosmu,classlist(i)*nsamples,initial_dims);
    load(['/home/mengmi/Projects/Proj_CL/code/CIFARincrement/vizClusters_vProtoCL/results_viz/protos_' num2str(i) '.mat']);
    train_X = [train_X; protosmu];
    
    labels = [1:classlist(i)]';
    labels = repmat(labels,1,nsamples);
    train_labels = reshape(labels, classlist(i)*nsamples,1);
    train_labels = [train_labels; [1:classlist(i)]'];
    
    % Run t−SNE
    %mappedX = tsne(train_X,train_labels,no_dims );
    mappedX = tsne(train_X,[],no_dims );
    % Plot results
    %subplot(1,2,counter);
    
    hb = figure;
    hold on;
    if counter < 2
        gscatter(mappedX(1:end-classlist(i),1), mappedX(1:end-classlist(i),2), train_labels(1:end-classlist(i)),colorcode,'.',18,'off');
        gscatter(mappedX(end-classlist(i)+1:end,1), mappedX(end-classlist(i)+1:end,2), train_labels(end-classlist(i)+1:end),colorcode,'s',18,'off');
    else
        gscatter(mappedX(1:end-classlist(i),1), mappedX(1:end-classlist(i),2), train_labels(1:end-classlist(i)),colorcode,'.',18,'off');
        gscatter(mappedX(end-classlist(i)+1:end,1), mappedX(end-classlist(i)+1:end,2), train_labels(end-classlist(i)+1:end),colorcode,'s',18,'off');
        legend(classes(1:classlist(i)),'Location','southwest','FontSize',14);
    end
    
    if printflag == 1
        printfilename = 'visModelCluster';
        set(hb,'Position',[1361         669         560         420]);
        set(hb,'Units','Inches');
        pos = get(hb,'Position');
        set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
        print(hb,[printdirprefix printfilename '_task_' num2str(i)  printpostfix],printmode,printoption);
    end

    %scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3));
    %
    %title(['Task = ' num2str(i)],'FontSize',labelsize);
    counter = counter + 1;
end

