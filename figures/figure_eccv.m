%% Histogram
load('DrosteEffect-BrewerMap-213e65d\hist_syn.mat')
map = brewermap(3,'Set1'); 
figure
histf(dist_pos,0:.05:3,'facecolor',map(1,:),'facealpha',.5,'edgecolor','none')
hold on
histf(dist_neg,0:.05:3,'facecolor',map(2,:),'facealpha',.5,'edgecolor','none')
box off
axis([0,3,0,150])
%legalpha('H1','H2','H3','location','northwest')
legend('GANFaces Positive Pairs','GANFaces Negative Pairs')
xlabel('Euclidean Distance')
ylabel('Frequency')
title('Distribution of pairwise distances')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
print('ours.eps','-dwinc')

load('DrosteEffect-BrewerMap-213e65d\hist_lsfm.mat')
map = brewermap(3,'Set1'); 
figure
histf(dist_pos,0:.05:3,'facecolor',map(1,:),'facealpha',.5,'edgecolor','none')
hold on
histf(dist_neg,0:.05:3,'facecolor',map(2,:),'facealpha',.5,'edgecolor','none')
box off
axis([0,3,0,150])
%legalpha('H1','H2','H3','location','northwest')
legend('3DMM Positive Pairs','3DMM Negative Pairs')
xlabel('Euclidean Distance')
ylabel('Frequency')
title('Distribution of pairwise distances')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
print('syn.eps','-dwinc')

load('DrosteEffect-BrewerMap-213e65d\hist_vgg.mat')
map = brewermap(3,'Set1'); 
figure
histf(dist_pos,0:.05:3,'facecolor',map(1,:),'facealpha',.5,'edgecolor','none')
hold on
histf(dist_neg,0:.05:3,'facecolor',map(2,:),'facealpha',.5,'edgecolor','none')
box off
axis([0,3,0,150])
%legalpha('H1','H2','H3','location','northwest')
legend('VGG Positive Pairs','VGG Negative Pairs')
xlabel('Euclidean Distance')
ylabel('Frequency')
title('Distribution of pairwise distances')
set(findall(gcf,'-property','FontSize'),'FontSize',14)
print('vgg.eps','-dwinc')

%%

func = @(in) strtok(in,'+-');
final2 = str2double(cellfun(func,final(:,[2,7,8]),'UniformOutput',false));
acc1 = cell2mat(final(end-2:-2:1,[1,4]));
acc2 = cell2mat(final(end-3:-2:1,[1,4]));
far1 = final2(end-2:-2:1,[1,2,3]);
far2 = final2(end-3:-2:1,[1,2,3]);

figure
axes('ColorOrder',brewermap(2,'Set1'),'NextPlot','replacechildren')
x = [20,50,100];
plot(1:3,acc2,1:3,acc1,'--','linewidth',1);
xticks(1:3)
xticklabels(x) 
legend('VGG+GANF Acc.','VGG+GANF 1-EER','VGG Acc.','VGG 1-EER','Location','southeast')
xlabel('% of VGG dataset')
axis([1,3,0.855,0.955])
set(findall(gcf,'-property','FontSize'),'FontSize',12)
title('LFW Scores')
grid on
print('acc.eps','-dwinc')


figure
axes('ColorOrder',brewermap(3,'Set1'),'NextPlot','replacechildren')
x = [20,50,100];
plot(1:3,far2,1:3,far1,'--','linewidth',1);
xticks(1:3)
xticklabels(x) 
legend('VGG+GANF IJB-A@FAR=1e-2','VGG+GANF IJB-A@FAR=1e-3','VGG+GANF LFW@FAR=1e-3','VGG IJB-A@FAR=1e-2','VGG IJB-A@FAR=1e-3','VGG LFW@FAR=1e-3','Location','northwest')
xlabel('% of VGG dataset')
ylabel('True Positive Rates')
%axis([1,4,0.79,0.955])
set(findall(gcf,'-property','FontSize'),'FontSize',10)
title('Verification Scores (TPR) on LFW and IJB-A')
grid on
print('tpr.eps','-dwinc')

nIm = [359705 902264 1803991];
nImSyn = [499555 499555 499555];
nId = [525 1311 2622];
nIdSyn = [10000 10000 10000];

figure;
hold on
colormap(brewermap(3,'Pastel1'))
bar(1:3,[nIm;nImSyn]','stacked')
legend('VGG','GANFaces','Location','northwest')
%bar([nId;nIdSyn]','stacked')
xlabel('% of VGG dataset')
ylabel('Number of Images')
xticks(1:3)
xticklabels(x) 
title('Distribution of the datasets when combined')
set(findall(gcf,'-property','FontSize'),'FontSize',12)
grid on
print('dist.eps','-dwinc')

close all






