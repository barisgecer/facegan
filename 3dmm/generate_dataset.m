
temp = load('BaselFaceModel_2.mat');
load('BaselFaceModel_1.mat');
BFM.shapePC = temp.BFM.shapePC;
clear temp;

nId = 500;
nImages = 200; % per Id
data_dir = 'c:/data/3dmm_norm';
mkdir(data_dir);
rng('default');
for id =1:nId
    mkdir([data_dir '/' sprintf('%05d',id)]);
    for f=1:nImages
        im = gen_face(BFM, id, id);
        im = padarray(im,[(size(im,2)-size(im,1))/2 0],0);        
        imwrite(im,[data_dir '/' sprintf('%05d',id) '/' sprintf('%05d',f) '.jpg']);
    end
end