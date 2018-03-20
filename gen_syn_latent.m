n_id = 10000;
n_im = 50;
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
data_dir = 'c:/data/syn_lsfm21';
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
for id = 1:n_id
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
    for im =1:n_im
        exp = mvnrnd(mu(ind_exp),covar(ind_exp,ind_exp)/2);
        %exp = randn(1,length(ind_exp)).*sigma(ind_exp) + mu(ind_exp);
        ill = mvnrnd(mu(ind_ill),covar(ind_ill,ind_ill)/3);
        pos = mvnrnd(mu(ind_pos),covar(ind_pos,ind_pos)/3);
        pos(3) =pos(3)*1.5;
        pos(2) =0;
        col = mvnrnd(mu(ind_col),covar(ind_col,ind_col)/3);
        if id>startfrom
            latent = [shp.*sigma',exp,tex,ill,pos,col]; % NEED NORMALIZATION
            im_out = render_face(latent',mu_shape + mu_exp,mu_tex,w,w_exp,w_tex,tri);
            latent_norm = latent;
            latent_norm(ind_shp) = latent_norm(ind_shp)./sigma';
            latent_norm(ind_exp) = latent_norm(ind_exp)./sigma_exp';
            latent_norm(ind_tex) = latent_norm(ind_tex)./sigma_tex';
            latent_norm(ind_rest(14:16)) = latent_norm(ind_rest(14:16))*latent_norm(ind_rest(17));
            imwrite(im_out,[data_dir '/' sprintf('%05d',id) '/' sprintf('%05d',im) '.jpg'],'Quality',100);
            %dlmwrite([data_dir '/' sprintf('%05d',id) '/' sprintf('%05d',im) '.txt'], latent');
        end
    end
end