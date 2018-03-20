
path_c = 'C:\data\syn_lsfm-108';
path_s = 'C:\data\gen-108-3';
path_d = 'C:\data\gen-abl-2'; 
path_f = 'C:\data\gen-abl-1'; 
path_p = 'C:\data\gen-abl-3'; 

n_im = 10;
folders = dir(path_c);
while(true)
    fol = folders(randi(length(folders),1));
    if fol.isdir & fol.name ~='.'
        files = dir([fol.folder,'/',fol.name,'/'])  ;      
        fil = files(randi(length(files),n_im,1));
        if ~any([fil.isdir])
            title_str = [fol.name,'/'];
            for i =1:n_im
                subplot(5,n_im,i);
                imshow(imread([fil(i).folder,'/',fil(i).name]))
                subplot(5,n_im,i+n_im);
                imshow(imread([strrep(fil(i).folder,path_c,path_s),'/',fil(i).name]))
                subplot(5,n_im,i+2*n_im);
                imshow(imread([strrep(fil(i).folder,path_c,path_d),'/',fil(i).name]))
                subplot(5,n_im,i+3*n_im);
                imshow(imread([strrep(fil(i).folder,path_c,path_f),'/',fil(i).name]))
                subplot(5,n_im,i+4*n_im);
                imshow(imread([strrep(fil(i).folder,path_c,path_p),'/',fil(i).name]))
                title_str = [title_str fil(i).name,','];
            end
            suptitle(title_str)
            axis tight
            pause
        end
    end
end

%%
target = 'C:\data\eccv-comp'; 
%path_r = 'C:\data\syn64_3';
id = [9932,3687,3377,7297,7087,5823,6804,4245,4647,7979];
im1 = [7,25,11,22,48,48,32,47,33,29];
im2 = [4,46,37,40,40,49,7,19,7,16];
im3 = [46,28,42,34,14,29,30,2,21,1];
counter = 1;
mkdir(target);
for i = 1:length(id)
    copyfile([path_p,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im1(i)),'.jpg'],[target,'/','r',num2str(counter),'a.jpg']);
    copyfile([path_p,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im2(i)),'.jpg'],[target,'/','r',num2str(counter),'b.jpg']);
    copyfile([path_p,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im3(i)),'.jpg'],[target,'/','r',num2str(counter),'c.jpg']);    
    
	copyfile([path_c,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im1(i)),'.jpg'],[target,'/','c',num2str(counter),'a.jpg']);
    copyfile([path_c,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im2(i)),'.jpg'],[target,'/','c',num2str(counter),'b.jpg']);
    copyfile([path_c,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im3(i)),'.jpg'],[target,'/','c',num2str(counter),'c.jpg']);
    
    copyfile([path_s,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im1(i)),'.jpg'],[target,'/','s',num2str(counter),'a.jpg']);
    copyfile([path_s,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im2(i)),'.jpg'],[target,'/','s',num2str(counter),'b.jpg']);
    copyfile([path_s,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im3(i)),'.jpg'],[target,'/','s',num2str(counter),'c.jpg']);
    
    copyfile([path_d,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im1(i)),'.jpg'],[target,'/','d',num2str(counter),'a.jpg']);
    copyfile([path_d,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im2(i)),'.jpg'],[target,'/','d',num2str(counter),'b.jpg']);
    copyfile([path_d,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im3(i)),'.jpg'],[target,'/','d',num2str(counter),'c.jpg']);
    
    copyfile([path_f,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im1(i)),'.jpg'],[target,'/','f',num2str(counter),'a.jpg']);
    copyfile([path_f,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im2(i)),'.jpg'],[target,'/','f',num2str(counter),'b.jpg']);
    copyfile([path_f,'/',sprintf('%05d',id(i)),'/',sprintf('%05d',im3(i)),'.jpg'],[target,'/','f',num2str(counter),'c.jpg']);
    counter = counter+1;
end
