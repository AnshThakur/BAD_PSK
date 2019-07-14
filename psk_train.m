function[kernel_gram_matrix_train,S,phi_train,model]= psk_train(ubm,workDir_bird_train,...
        infile_bird_train,workDir_nbird_train,infile_nbird_train,name)
%%PURPOSE: To build Probabilistic Sequence kernel using UBM and the adpated
%%models
%INPUTS
%          ubm:                      ubm path
%          adaptedgmm:               adapted gmm path
%          
%          workDir_bird_train:       feature path of bird files(train)
%          infile_bird_train:        file list of bird files(train)
%          workDir_nbird_train:      feature path of non-bird files(train) 
%          infile_nbird_train:       file list of non-bird files(train)
%          workDir_bird_test:        feature path of bird files(test)
%          infile_bird_test:         file list of bird files(test)
%          workDir_nbird_test:       feature path of non-bird files(test)              
%          infile_nbird_test:        file list of non-bird files(test)           
%OUTPUTS

load(ubm);
%m_ubm = model.centers{1};
%v_ubm = model.variance{1};
%w_ubm = model.weight{1};
m_ubm = model.means;
v_ubm =model.covariances;
w_ubm = model.weights;

disp('DEBUG: Loaded UBM data');
fprintf('DEBUG: Weight of the first component = %f',w_ubm(1,1));

% load(adaptedgmm);
% adaptedMean_bird = model.centers;
% adaptedVar_bird = model.variance;
% adaptedW_bird = model.weight;


%%Training
[bird_phi_train,r_bird_train]=kernel_na(infile_bird_train,workDir_bird_train,m_ubm,v_ubm,w_ubm);



%% other variation : storing only relevant components 
% 
% [V I]=sort_rows(bird_phi_train);
% a = unique(I(:,1));
% out = [a,histc(I(:,1),a)];
% [~,sorted_inds] = sort( out(:,2),'descend' );
% b = out(sorted_inds,:);
% ind=b(1:10,1);
% 
% bird_phi_train=bird_phi_train(:,ind);
% r_bird_train=r_bird_train(:,ind);

%%
[nbird_phi_train,r_nbird_train]=kernel_na(infile_nbird_train,workDir_nbird_train,m_ubm,v_ubm,w_ubm);

% 
% nbird_phi_train=nbird_phi_train(:,ind);
% r_nbird_train=r_nbird_train(:,ind);
%%

R = [r_bird_train;r_nbird_train];  
m1= size(R,1);
S =(R'*R)/m1;
S_save =sprintf('S_%s',name);
save(S_save,'S');
phi_train = [bird_phi_train;nbird_phi_train];
phi_save =sprintf('phi_train_%s',name);
save(phi_save,'phi_train');
kernel_gram_matrix_train = phi_train*pinv(S)*phi_train';
kgm_train_save =sprintf('kgm_train_%s',name);
save(kgm_train_save,'kernel_gram_matrix_train');
fprintf('kernel gram matrix is built...\n');
%%

%% SVM model
addpath('./matlab'); %% adding libsvm matlab implementation
trainclass = [ones(1,201) -1*ones(1,201)];
num_train = [1:size(kernel_gram_matrix_train,1)];
kernel_train_append = [num_train', kernel_gram_matrix_train];
model = svmtrain(trainclass',kernel_train_append,'-t 4');
model_save =sprintf('model_%s',name);
save(model_save,'model');
