n_id = 50;
n_im = 20;
L=[];
names = {};

accepted = [dir('C:\data\300W-3D-aligned'); dir('C:\data\AFLW2000-aligned')];

for d = {'AFW','HELEN','IBUG','LFPW'}
    data_path = ['../' d{:} '/'];
    %mkdir([data_path '3dmm/'])
    for file = dir([data_path  '/*.mat'])'
        if find(cell2mat(strfind({accepted.name},file.name(1:end-4))))
            sample_name = file.name;
            load([data_path sample_name]);
            names{end+1} = sample_name;
            latent = [Shape_Para; Exp_Para; Tex_Para; Illum_Para'; Pose_Para'; Color_Para'];
            L =[L; latent'];
            %dlmwrite([data_path '3dmm/' sample_name(1:end-4) '.txt'], latent);
        end
    end
end

data_path = 'C:\data\AFLW2000/';
%mkdir([data_path '3dmm/'])
for file = dir([data_path  '/*.mat'])'
    if find(cell2mat(strfind({accepted.name},file.name(1:end-4))))
        sample_name = file.name;
        load([data_path sample_name]);
        names{end+1} = sample_name;
        latent = [Shape_Para; Exp_Para; Tex_Para; Illum_Para'; Pose_Para'; Color_Para'];
        L =[L; latent'];
        %dlmwrite([data_path '3dmm/' sample_name(1:end-4) '.txt'], latent);
    end
end

load('Model_Shape.mat');
load('Model_Exp.mat');
%%
load('all_all_all_scaled.mat');
w = [w zeros(size(w,1),199-size(w,2))];
sigma = [sqrt(eigenvalues) zeros(1,199-size(eigenvalues,2))]';
%mu_shape = mean';
mu_shape = mu_shape';
clear mean;
%%
data_dir = 'c:/data/eccv_sup1';
mkdir(data_dir);
rng('default');

ind_shp = 1:199;
ind_exp = 200:228;
ind_tex = 229:427;
ind_ill = 428:437;
ind_pos = 438:444;
% ind_pos1 = 438:440;
% ind_pos2 = 441:443;
% ind_pos3 = 444:444;
ind_col = 445:451;
ind_rest = 428:451;

mu = mean(L);
covar = cov(L);
sig = std(L);
startfrom = 0;
id =1;
tic
shp = randn(1,length(ind_shp));
%Texture
%     n_tex_dim = length(ind_tex);
%     tex = randn(n_tex_dim, 1)*3;
%     tex(randperm(n_tex_dim,n_tex_dim-20)) = 0;
%     tex(tex>5) = 5;
%     tex(tex<-5) = -5;
%     tex(15:end) = 0;
%     tex(1) = rand*12-4;
%     tex = tex';
%shp = mvnrnd(mu(ind_shp),covar(ind_shp,ind_shp));
tex = mvnrnd(mu(ind_tex),covar(ind_tex,ind_tex));
%tex(15:end) = 0;
if id>startfrom
    mkdir([data_dir '/' sprintf('%05d',id)]);
end
mkdir([data_dir '/' sprintf('%05d',id)]);
pos1 = linspace(-0.2,0.2,10);
pos2 = linspace(1,-1,20);
pos3 = linspace(-1.2,1.2,10);
ill1 = linspace(0,1,20);
ill2 = linspace(0,-3.14/2,1);
exps = [0,-3];
for i=1:50
    exp = mvnrnd(mu(ind_exp),covar(ind_exp,ind_exp)/2);
    ill = mvnrnd(mu(ind_ill),covar(ind_ill,ind_ill)/3);
    pos = mvnrnd(mu(ind_pos),covar(ind_pos,ind_pos)/3);
    col = mvnrnd(mu(ind_col),covar(ind_col,ind_col)/3);
end
shp(2,:) = randn(1,length(ind_shp));
tex(2,:) = mvnrnd(mu(ind_tex),covar(ind_tex,ind_tex));

for e = 1:length(pos2)
    for im =1:length(ill1)
        exp = mu(ind_exp);
        %exp(1) = exps(e);
        %exp = randn(1,length(ind_exp)).*sigma(ind_exp) + mu(ind_exp);
        ill =mu(ind_ill);
        %ill(1:3) = ill1(im);
        %ill(7)= ill2(e); % altitude
        %ill(8)= ill1(im); % azimuth
        pos = mu(ind_pos);
        pos(1) = 0;%pos1(im);
        pos(2) = 0;
        pos(3) =  pos2(e);
        shp2 = shp(1,:)*ill1(im) + shp(2,:)*(1-ill1(im));
        tex2 = tex(1,:)*ill1(im) + tex(2,:)*(1-ill1(im));
        %pos(2) = randn(1).*sig(ind_pos(2))*1.5 + mu(ind_pos(2));
        col = mu(ind_col);
        latent = [shp2.*sigma',exp,tex2,ill,pos,col]; % NEED NORMALIZATION
        im_out = render_face(latent',mu_shape + mu_exp,mu_tex,w,w_exp,w_tex,tri);
        latent_norm = latent;
        latent_norm(ind_shp) = latent_norm(ind_shp)./sigma';
        latent_norm(ind_exp) = latent_norm(ind_exp)./sigma_exp';
        latent_norm(ind_tex) = latent_norm(ind_tex)./sigma_tex';
        latent_norm(ind_rest(14:16)) = latent_norm(ind_rest(14:16))*latent_norm(ind_rest(17));
        imwrite(im_out,[data_dir '/' sprintf('%05d',id) '/' sprintf('%02d_%01d',im,e) '.jpg']);
        %dlmwrite([data_dir '/' sprintf('%05d',id) '/' sprintf('%05d',im) '.txt'], latent');
    end
end