
 % Ex - 5



    % Computation and cognition undergrad - ex5
    % See PDF document for instructions

    clear; close all; clc;
    
 
 %% Get the data

data_dir = 'PCA_data';  % Data directory

load('PCA_data.mat', 'X');
load('PCA_data.mat', 'M');

%% set dimenssions 
size_X = size(X);

input_d = size_X(1); %How many input neurons there are
output_d = 2; %How many output neurons there are

%% Training parameteres
s_eta         = 1e-3;	% start learning rate, in the loop the learning rate changes 
n_epochs    = 1500;	% number of training epochs

%% Sangler Rule -Training Online learning
 W = 0.1*randn(output_d,input_d);           %preduce random weights.

for epoch= 1:n_epochs
    eta         = s_eta/(epoch);            %Reduce learning rate each epoch
    idx = randperm(length(X));              %produce a random index vector
        for  n_ex = 1:length(X)             %for each example
            x = (X(:,idx(n_ex)));           %Get a random sample
            Y = W * x;                      %code the example according to the W's.
        
            %Update rule
                for i = 1:output_d
                    for  j = 1:input_d
                        sigma = 0;
                        for k = 1:i
                            sigma = sigma + (W (k,j) * Y(k));
                        end
                dW = eta * Y(i) * (x(j) - sigma);
                W(i,j)= W(i,j) + dW;        %updating the current W
                    end
                end   
        end 
end
%% Calculations
%convert the W matrix to 2 vectors
VW1 = W(1,:);                   
VW2 = W(2,:);

%calculating magintude and angle
mag_W1 = sqrt(sum(VW1.*VW1));
mag_W2 = sqrt(sum(VW2.*VW2));

angle = acosd((dot(VW1,VW2))/((norm(VW1)*norm(VW2))));

%% Graphics
    
%Data graphs (Input)
    hold on
    scatter3(X(1,:),X(2,:),X(3,:))                                 %graph of dots for X
    patch('faces',[1 3 4 2],'Vertices',M','facecolor','green')     %the original rectangle
   
%Vectors graphs (Output)
    scaler = 10; % varible to make the vactors longer - Can be modify
    W1 = quiver3(0,0,0,scaler*W(1,1),scaler*W(1,2),scaler*W(1,3),'linewidth',3);
    W2 = quiver3(0,0,0,scaler*W(2,1),scaler*W(2,2),scaler*W(2,3),'linewidth',3);
    
    
%Text    
    str= {'W1' , 'W2'};
    text(scaler*W(:,1) ,scaler*W(:,2),scaler*W(:,3) , str) %Text function will place the vector name at the right place in the 3D space 
    title(['||W1|| = ' num2str(mag_W1)  ,  '    ||W2|| = ' num2str(mag_W2)  , '  <(w1,w2) = ' num2str(angle) char(176)])
    xlabel('x1')
    ylabel('x2')
    zlabel('x3')
