function[phi,r_final]= kernel_na(infile,workDir,m_ubm,v_ubm,w_ubm)
%% generates phi vectors

file = fopen(infile);
data = textscan(file, '%s');
fclose(file);


phi =[];
r_final =[];
for i = 1:length(data{1})
     wavFile = cell2mat(strcat(workDir, '/', data{1}(i)));
     [pathstr,name,ext] = fileparts(wavFile);
	 workFile = strcat(workDir,'/',name);
     x =load(workFile);
     
     m = m_ubm;
     v = v_ubm;
     w = w_ubm;
     [h,r]=calc_h(x,m,v,w);
     phi =[phi;h];
     r_final =[r_final;r];
%      if mod(500,i)==0
%          fprintf('%d files are processed out of %d\n',i,length(data{1}));
%      end

end
