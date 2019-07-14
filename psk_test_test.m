function[kernel_gram_matrix_test,decVals,p_bird,p_nbird] = psk_test_test(ubm,workDir,infile,...
                          phi_train,S,model_svm,name)
%%
%%PURPOSE: To build Probabilistic Sequence kernel using UBM and the adpated
%%models
%INPUTS
%          ubm:            ubm path
%             
%          infile:         file list 
%          workDir:        feature path 
%                      
%         
%          phi_train,S,model:        Probabilistic train vector,correlation
%          matrix and SVM model
%OUTPUTS

load(ubm);
m_ubm = model.means;
v_ubm = model.covariances;
w_ubm = model.weights;
% load(adaptedgmm);
% adaptedMean_bird = model.centers;
% adaptedVar_bird = model.variance;
% adaptedW_bird = model.weight;    
%%Testing
[phi,r]=kernel_na(infile,workDir,m_ubm,v_ubm,w_ubm);
phi_test = phi;
kernel_gram_matrix_test = phi_test*pinv(S)*phi_train';
kgm_test_save =sprintf('kgm_test_%s',name);
save(kgm_test_save,'kernel_gram_matrix_test');
fprintf('kernel gram matrix is built...\n');
%%
%%SVM model
addpath('./matlab'); %% libsvm path
testclass = rand(1,size(kernel_gram_matrix_test,1));
num_test = [1:size(kernel_gram_matrix_test,1)];
kernel_test_append = [num_test', kernel_gram_matrix_test];
[~,~, decVals] = svmpredict(testclass', kernel_test_append, model_svm);
for i=1:length(decVals)
p_bird(i) = 1/(1+exp(-decVals(i)));
end
p_nbird = 1-p_bird;

save('p_bird.mat','p_bird');
save('p_nbird.mat','p_nbird');
