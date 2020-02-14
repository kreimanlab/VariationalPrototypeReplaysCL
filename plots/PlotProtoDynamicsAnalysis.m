clear all; close all; clc;

printpostfix = '.pdf';
printmode = '-dpdf'; %-depsc
printoption = ''; %'-fillpage'
printdirprefix = 'Figures/';
printflag = 1;
labelsize = 14;

classes = {'plane', 'car', 'bird', 'cat',...
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};

%plot feature similarity confusion mat
load(['/home/mengmi/Projects/Proj_CL/code/CIFARincrement/FeatureSimilarity_vProtoCL/results_featureSimilarity/vis_representation_9.mat']);
Feature = double(represents);
nclass= 10;
nsample = 100;
zdim = 1000;

Feature = squeeze(mean(Feature,2));
D = pdist2(Feature, Feature);
% Feature = reshape(Feature, nclass*nsample, zdim);
% D = pdist2(Feature, Feature);
% D = blockproc(D,[nsample nsample],@(x)sum(x.data(:)));
% D = D/(nsample*nsample);

%normalize D using softamx
% Dnorm = [];
% for c = 1:nclass
%     Dnorm = [Dnorm; softmax(D(c,:)')'];
% end
Dnorm = 1-mat2gray(D);
DFeature = Dnorm(1:nclass-1,1:nclass-1);
DFeature = DFeature(:);
%imshow(imresize(mat2gray(D),[400 400]))
hb = figure;
imagesc(Dnorm);
%colormap(default);
caxis([0 max(max(Dnorm))]);

set(gca,'XTick',1:nclass,...
    'XTickLabel',classes,...
    'YTick',1:nclass,...
    'YTickLabel',classes);
set(gca,'fontsize',14);
%xticklabel_rotate([],30,[],'Fontsize',14);
%hc=colorbar('eastoutside');
axis square;
set(gca, 'TickDir', 'out');
set(gca, 'Box','off');

if printflag == 1
    printfilename = 'featureConfMat';
    set(hb,'Position',[523   231   780   507]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end

%% plot prototype dynamics similarity
% compute PCAs
load(['/home/mengmi/Projects/Proj_CL/code/CIFARincrement/vizClusters_vProtoCL/results_viz/vis_representation_1.mat']);
PCNum = 3;
zdim = 500;
protosmu = reshape(protosmu, 2*100, zdim);
[coeff,score,~,~,explained,mu] = pca(protosmu);
PCAselected = coeff(:,1:PCNum);
NumTask = 9;
COOR = nan(nclass, NumTask, PCNum);

for T = 1: NumTask
    load(['/home/mengmi/Projects/Proj_CL/code/CIFARincrement/vizClusters_vProtoCL/results_viz/protos_' num2str(T) '.mat']);
    conv = (protosmu - repmat(mu, size(protosmu,1), 1)) * PCAselected;
    
    for c = 1:size(conv)
        COOR(c,T,:) = conv(c,:);
    end
end

MOV = [];
for c = [1:(nclass-1)]
    cc = squeeze(COOR(c,:,:));
    cc(isnan(cc)) = [];
    cc = reshape(cc,[],3);
    mov = cc(end,:) - cc(1,:);    
    MOV = [MOV; mov];
end
Feature =  MOV;
D = pdist2(Feature, Feature);
% Feature = reshape(Feature, nclass*nsample, zdim);
% D = pdist2(Feature, Feature);
% D = blockproc(D,[nsample nsample],@(x)sum(x.data(:)));
% D = D/(nsample*nsample);

%normalize D using softamx
% Dnorm = [];
% for c = 1:nclass
%     Dnorm = [Dnorm; softmax(D(c,:)')'];
% end
Dnorm = 1-mat2gray(D);
DProto = Dnorm(:);
%imshow(imresize(mat2gray(D),[400 400]))
hb = figure;
imagesc(Dnorm);
%colormap(default);
caxis([0 max(max(Dnorm))]);

set(gca,'XTick',1:nclass,...
    'XTickLabel',classes,...
    'YTick',1:nclass,...
    'YTickLabel',classes);
set(gca,'fontsize',14);
%xticklabel_rotate([],30,[],'Fontsize',14);
hc=colorbar('eastoutside');
axis square;
DFeature(find(DFeature == 0)) = [];
DProto(find(DProto == 0)) = [];
display(['proto and feature coorelation: ' num2str(corr(DFeature, DProto))]);

title(['Correlation = ' num2str(corr(DFeature, DProto))],'FontSize',14);
set(gca, 'TickDir', 'out');
set(gca, 'Box','off');

if printflag == 1
    printfilename = 'protoConfMat';
    set(hb,'Position',[523   231   780   507]);
    set(hb,'Units','Inches');
    pos = get(hb,'Position');
    set(hb,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(hb,[printdirprefix printfilename printpostfix],printmode,printoption);
end