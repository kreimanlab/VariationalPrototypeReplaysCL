clear all; close all; clc;

printflag = 1;
printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = ''; %'-fillpage'
printdirprefix = 'Figures/';

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

classes = {'plane', 'car', 'bird', 'cat',...
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
       
%% compute PCAs
load(['/home/mengmi/Projects/Proj_CL/code/CIFARincrement/vizClusters_vProtoCL/results_viz/vis_representation_1.mat']);
PCNum = 3;
zdim = 500;
protosmu = reshape(protosmu, 2*100, zdim);
[coeff,score,~,~,explained,mu] = pca(protosmu);
PCAselected = coeff(:,1:PCNum);
NumTask = 9;
NumClass = 10;
COOR = nan(NumClass, NumTask, PCNum);

for T = 1: NumTask
    load(['/home/mengmi/Projects/Proj_CL/code/CIFARincrement/vizClusters_vProtoCL/results_viz/protos_' num2str(T) '.mat']);
    conv = (protosmu - repmat(mu, size(protosmu,1), 1)) * PCAselected;
    
    for c = 1:size(conv)
        COOR(c,T,:) = conv(c,:);
    end
end

hb = figure; hold on;


for c = 1:NumClass
    cc = squeeze(COOR(c,:,:));
    plot3(cc(:,1), cc(:,2), cc(:,3), '-o', 'Color',colorcode{c},'LineWidth' ,2,'MarkerSize',10);
end

%plot one more time for color gradients
for c = 1:NumClass
    cc = squeeze(COOR(c,:,:));
    startcolor = colorcode{c};
    overcolor = [0 0 0];
    %cc
    if c == 1
        NumConds = 9;
    else
        NumConds = 11-c;
    end
    barcolor = [linspace(startcolor(1),overcolor(1),NumConds)', linspace(startcolor(2),overcolor(2),NumConds)', linspace(startcolor(3),overcolor(3),NumConds)'];

    for cond = NumConds
        plot3(cc(end,1), cc(end,2), cc(end,3), 's', 'Color','k','MarkerSize',15);
    end
end

grid on;
xlabel('PC1','FontSize',14); ylabel('PC2','FontSize',14); zlabel('PC3','FontSize',14);
legend(classes, 'FontSize', 14, 'Location','northeast');
view(45,25);

if printflag == 1
    printfilename = 'ProtoTrajectory';
    set(hb, 'Position',[842   427   560   440]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end






