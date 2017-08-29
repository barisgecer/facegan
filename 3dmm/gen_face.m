function [im, alpha, beta, gamma, rp] = gen_face (BFM, shp_seed, tex_seed, exp_seed, light_seed, view_seed, brig_seed)
%% Generate a Random Face with discrite Identities
n_shape_dim = size(BFM.shapePC,2);
n_exp_dim = size(BFM.expPC,2);
n_tex_dim = size(BFM.texPC,2);

rng(round(rand*2^32));
if ~exist('shp_seed') shp_seed = round(rand*2^32); end
if ~exist('tex_seed') tex_seed = round(rand*2^32); end
if ~exist('exp_seed') exp_seed = round(rand*2^32); end
if ~exist('light_seed') light_seed = round(rand*2^32); end
if ~exist('view_seed') view_seed = round(rand*2^32); end
if ~exist('brig_seed') brig_seed = round(rand*2^32); end

%ID
rng(shp_seed);
alpha = randn(n_shape_dim, 1);
%Texture
rng(tex_seed);
beta = randn(n_tex_dim, 1)*3;
beta(randperm(n_tex_dim,n_tex_dim-20)) = 0;
beta(beta>5) = 5;
beta(beta<-5) = -5;
beta(15:end) = 0;
beta(1) = rand*12-4;

%Expression
rng(exp_seed);
gamma = randn(n_exp_dim, 1)*1.8;
gamma(gamma>2.5) = 2.5;
gamma(gamma<-2.5) = -2.5;
gamma(randperm(n_exp_dim,n_exp_dim-7)) = 0;
if gamma(1)>1 gamma(1)=1;end
%Light
rng(light_seed);
rp.light = rand(1,2).*[120 30]-[60 30];
%View
rng(view_seed);
rp.phi = rand*1.2-0.6;
rp.rho = rand*0.4-0.2;
%Light Color
rng(brig_seed);
rp.light_color = rand(1,1)/5+0.8 + randn(1,3)/50;
rp.light_color(rp.light_color>1) = 1;

% Render Face here
render_face(BFM, alpha, gamma, beta,rp);
im = print('-RGBImage','-opengl','-r0');