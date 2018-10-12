%Reading data from file
inputFile = fopen('data.txt','r');
A= fscanf(inputFile,'%f'); %Double values chosen for data
A = A.';    % Make input vector 1 row
X = 0:18/884:18;
totalSum = sum(A);
[row,column] = size(A);
mu = totalSum / column;

total_diff = 0;
for iter = A
    total_diff = total_diff +(iter-mu)^2;
end 
variance = total_diff / column ;
std = sqrt(variance);

B = zeros(0,length(A));
for iter =1:length(X)
    B(iter) = (1/(std*sqrt(2*pi)) )*exp( (-1/2) * ((X(iter)- mu)/std)^2 );
end

hold on
title(sprintf('MLE results: mu = %f , std = %f',mu,std));

h = histogram(A,'Normalization','probability','FaceColor','g');
p= plot(X,B,'k','LineWidth',3);
legend('data','MLE fixed distribution')
