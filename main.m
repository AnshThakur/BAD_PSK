
%% load GMM for PSK
ubm='./ubm_wabler_100_gauss_128.mat'; %% GMM built using bird activity class
%% list of bird files and path to training bird examples
wbir ='../gauss_features/feature_warbler_bird_mfcc_gauss';
b_list='../features_melspec/adaptation_bird_warblrb_200.list';
%% %% list of non-bird files and path to training non-bird examples

wnbdir='../gauss_features/feature_warbler_nonbird_mfcc_gauss';
nb_list='../features_melspec/adaptation_non_bird_warblrb_200.list';
%%
name ='psk';
disp('training begins');
[kernel_gram_matrix_train,S,phi_train,model]= psk_train(ubm,wbir,b_list,wnbdir,nb_list,name);
disp('training ends');

%% path to test examples
wbdir_test = '../gauss_features/test';
test_list='../ubm/test_list';
%%
disp('testing');
[kernel_gram_matrix_test,decVals,p_bird,p_nbird] = psk_test_test(ubm,wbdir_test,test_list,phi_train,S,model,name);







