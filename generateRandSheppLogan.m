clear
clc
clf
rng(0)
n = 32;              % image dimension n x n
n_train = 10000;     % number of training images
n_test = 2000;       % number of testing images
noiseLevel = 0.05;   % noise level
%% load random shepp logan images 
X = randomSheppLogan(n,{'pad', 0; 'M', n_train});
X_t = randomSheppLogan(n,{'pad', 0; 'M', n_test});
% reshape images
%% load tomo A
angles = 1:5:179;
p = round(sqrt(2)*n);
options.angles = angles;
options.p = p;
A = PRtomo(n,options);

%% Training data
for j = 1:n_train
    btrue      = A*X(:,j);
    bnoisy     = btrue + WhiteNoise(btrue,noiseLevel);
    X_train{j} = reshape(X(:,j),n,n);
    B_train{j} = reshape(bnoisy,p,length(angles));
end
%% Testing data
for j = 1:n_test
    btrue      = A*X_t(:,j);
    bnoisy     = btrue + WhiteNoise(btrue,noiseLevel);
    X_test{j} = reshape(X_t(:,j),n,n);
    B_test{j} = reshape(bnoisy,p,length(angles));
end

%%
target_train = zeros(n_train,n,n);
target_test = zeros(n_test,n,n);
input_train = zeros(n_train,p,length(angles));
input_test = zeros(n_test,p,length(angles));

for k = 1:n_train
    target_train(k,:,:) = X_train{1,k};
    input_train(k,:,:) = B_train{1,k};
end
for k = 1:n_test
    target_test(k,:,:) = X_test{1,k};
    input_test(k,:,:) = B_test{1,k};
end
    
%% Normalizing
input_test = input_test - min(input_test(:));
input_test = input_test / max(input_test(:));
input_train = input_train - min(input_train(:));
input_train = input_train / max(input_train(:));

target_test = target_test - min(target_test(:));
target_test = target_test / max(target_test(:));
target_train = target_train - min(target_train(:));
target_train = target_train / max(target_train(:));

%% Horrible Idea
input_test_48 = zeros(n_test,48,36);
input_train_48 = zeros(n_train,48,36);

for i = 1:n_train
    input_train_48(i,1:45,:) = input_train(i,:,:);
end

for i = 1:n_test
    input_test_48(i,1:45,:) = input_test(i,:,:);
end

%% Saving

save('SheppLoganData10000.mat','input_test','target_test','input_train','target_train', 'input_test_48', 'input_train_48')

%%
A = input_train(1,:,:);
max(A(:))
% A = A - min(A(:));
% A = A / max(A(:));
% max(A(:))