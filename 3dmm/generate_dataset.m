
temp = load('BaselFaceModel_2.mat');
load('BaselFaceModel_1.mat');
BFM.shapePC = temp.BFM.shapePC;
clear temp;

nId = 10;
nImages = 10; % per Id
data_dir = 'data';
mkdir(data_dir);
rng('default');
for id =1:nId
    mkdir([data_dir '/' sprintf('%05d',id)]);
    for f=1:nImages
        im = gen_face(BFM, id, id);
        imwrite(im,[data_dir '/' sprintf('%05d',id) '/' sprintf('%05d',f) '.jpg']);
    end
end