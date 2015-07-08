addpath('./Yael Library');

W=randn(784,300);

m=8;

subFeatSz=784/m;

k=50;

for jj=1:m
[C(:,:,jj),idx(:,jj)]=yael_kmeans(single(W((jj-1)*subFeatSz+1:jj*subFeatSz,:)),...
                        k, 'niter', 100, 'verbose', 0);
QVData(:,:,jj)=C(:,idx(:,jj),jj);
end