
function [reducedData,transformationMat,meanData] = pca(x)
%% Zero-mean the data (by row)

meanData = mean(x,2);
x = bsxfun(@minus,x,meanData);


%% Implement PCA to obtain xRot
%  Implement PCA to obtain xRot, the matrix in which the data is expressed
%  with respect to the eigenbasis of sigma, which is the matrix U.

% xRot = zeros(size(x)); 

cov_x = cov(x');
[eig_vec,eig_val] = eig(cov_x);

[eig_val_sorted,val_max_idx] = sort(diag(eig_val),'descend');


u =  eig_vec(:,val_max_idx);

xRot = u'*x;

%%  Check your implementation of PCA
%  The covariance matrix for the data expressed with respect to the basis U
%  should be a diagonal matrix with non-zero entries only along the main
%  diagonal. We will verify this here.
%  Write code to compute the covariance matrix, covar. 
%  When visualised as an image, you should see a straight line across the
%  diagonal (non-zero entries) against a blue background (zero entries).

% -------------------- YOUR CODE HERE -------------------- 
% covar = zeros(size(x, 1)); % You need to compute this
covar = cov(xRot');
% Visualise the covariance matrix. You should see a line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix');
% imagesc(covar);
imshow(covar);


%% Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.

% -------------------- YOUR CODE HERE -------------------- 
k = 0; % Set k accordingly


denominator = sum(eig_val_sorted);
variance_retained = sum(eig_val_sorted(1:k))/denominator ;
while (k <= numel(eig_val_sorted)-1)&&(variance_retained <=.99)
    k=k+1
    variance_retained = sum(eig_val_sorted(1:k))/denominator
end

transformationMat = u(:,1:k) ;

%% Reducing the dimention to k

xHat = u(:,1:k)' * x ;


reducedData = xHat;

end