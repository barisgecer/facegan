%% Generate faces with controlled parameters
mkdir('summary');
temp = load('BaselFaceModel_2.mat');
load('BaselFaceModel_1.mat');
BFM.shapePC = temp.BFM.shapePC;
clear temp;

n_shape_dim = size(BFM.shapePC,2);
n_exp_dim = size(BFM.expPC,2);
n_tex_dim = size(BFM.texPC,2);
alpha = zeros(n_shape_dim, 1);
gamma = zeros(n_exp_dim,1);
beta  = zeros(n_tex_dim, 1);
rp.phi = 0.3;

%% All random with fixed ids on rows

mkdir('temp/all');

ims = [];
for dim = 1:10
    im_row = [];
    %ID
    alpha = randn(n_shape_dim, 1);
    %Texture
    beta = randn(n_tex_dim, 1)*3;
    beta(randperm(n_tex_dim,n_tex_dim-20)) = 0;
    beta(beta>5) = 5;
    beta(beta<-5) = -5;
    beta(15:end) = 0;
    beta(1) = rand*12-4;
    
    for n = 1:10
        %Expression
        gamma = randn(n_exp_dim, 1)*1.8;
        gamma(gamma>2.5) = 2.5;
        gamma(gamma<-2.5) = -2.5;
        gamma(randperm(n_exp_dim,n_exp_dim-7)) = 0;
        if gamma(1)>1 gamma(1)=1;end
        %Light
        rp.light = rand(1,2).*[120 30]-[60 30];
        %View
        rp.phi = rand*1.2-0.6;
        rp.rho = rand*0.4-0.2;
        %Light Color
        rp.light_color = rand(1,1)/5+0.8 + randn(1,3)/50;
        rp.light_color(rp.light_color>1) = 1;
        

        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/all/' num2str(dim) '_' num2str(n)],'-djpeg','-r0');
        im =(imread(['temp/all/' num2str(dim) '_' num2str(n) '.jpg']));
        im_row = [im_row im];
        
        %Reset
        gamma = zeros(n_exp_dim,1);
        rp.light = [0 0];
        rp.phi = 0;
        rp.rho = 0;
        rp.light_color = [1 1 1];
    end
    alpha = zeros(n_shape_dim, 1);
    beta = zeros(n_tex_dim, 1);
    ims = [ims;im_row];
end
imwrite(ims,['summary/all.jpg']);

%% ID controlled

mkdir('temp/id');

ims = [];
for dim = 1:10
    im_row = [];
    for coef = -6:2:6

        alpha(dim) = coef;
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/id/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/id/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        alpha = zeros(n_shape_dim, 1);
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/id.jpg']);

%% ID random

mkdir('temp/id');

ims = [];
for dim = 1:6
    im_row = [];
    for coef = -2:2:2

        alpha = randn(n_shape_dim, 1);
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/id/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/id/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        alpha = zeros(n_shape_dim, 1);
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/id_rand.jpg']);

%% Expression

mkdir('temp/exp');

ims = [];
for dim = 1:29
    im_row = [];
    for coef = -3:1:3

        gamma(dim) = coef;
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/exp/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/exp/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        gamma = zeros(n_exp_dim, 1);
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/exp.jpg']);

%% Expression random
mkdir('temp/exp');

ims = [];
for dim = 1:6
    im_row = [];
    for coef = -1:1:1

        gamma = randn(n_exp_dim, 1)*1.8;
        gamma(gamma>2.5) = 2.5;
        gamma(gamma<-2.5) = -2.5;
        gamma(randperm(n_exp_dim,n_exp_dim-7)) = 0;
        if gamma(1)>1 gamma(1)=1;end
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/exp/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/exp/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        gamma = zeros(n_exp_dim, 1);
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/exp_rand.jpg']);


%% Texture

mkdir('temp/tex');

ims = [];
for dim = 1:10
    im_row = [];
    for coef = -6:2:6

        beta(dim) = coef;
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/tex/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/tex/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        beta = zeros(n_tex_dim, 1);
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/tex.jpg']);

%% Texture random
mkdir('temp/exp');

ims = [];
for dim = 1:6
    im_row = [];
    for coef = -1:1:1

        beta = randn(n_tex_dim, 1)*3;
        beta(randperm(n_tex_dim,n_tex_dim-20)) = 0;
        beta(beta>5) = 5;
        beta(beta<-5) = -5;
        beta(15:end) = 0;
        beta(1) = rand*12-4;
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/tex/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/tex/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        beta = zeros(n_tex_dim, 1);
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/tex_rand.jpg']);

%% Light controlled
mkdir('temp/light');

ims = [];
for az = -120:20:120
    im_row = [];
    for el = -60:20:60

        rp.light = [az el];
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/light/' num2str(az) '_' num2str(el)],'-djpeg','-r0');
        im =(imread(['temp/light/' num2str(az) '_' num2str(el) '.jpg']));
        im_row = [im_row;im];
        rp.light = [0 0];
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/light.jpg']);


%% Angle controlled
mkdir('temp/angle');

ims = [];
for phi = -1.2:0.2:1.2
    im_row = [];
    for rho = -0.2:0.05:0.2

        if abs(phi)<1e-6 phi= 0; end
        rp.phi = phi;
        rp.rho = rho;
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/angle/' num2str(phi*10) '_' num2str(rho*100)],'-djpeg','-r0');
        im =(imread(['temp/angle/' num2str(phi*10) '_' num2str(rho*100) '.jpg']));
        im_row = [im_row;im];
        rp.phi = 0;
        rp.rho = 0;
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/angle.jpg']);

%% Random Light
mkdir('temp/color');

ims = [];
for dim = 1:6
    im_row = [];
    for coef = -1:1:1

        rp.light_color = rand(1,1)/2+0.5 + randn(1,3)/20;
        rp.light_color(rp.light_color>1) = 1;
        render_face(BFM, alpha, gamma, beta,rp);
        print('-opengl',['temp/color/' num2str(dim) '_' num2str(coef)],'-djpeg','-r0');
        im =(imread(['temp/color/' num2str(dim) '_' num2str(coef) '.jpg']));
        im_row = [im_row;im];
        rp.light_color = [1 1 1];
    end
    ims = [ims im_row];
end
imwrite(ims,['summary/color.jpg']);