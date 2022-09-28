%this is for training neural network of HARP
% The path in the following code is just for an example, you shuold replace them with your downloaded dataset path 

%% MFCC should use origin system
% Normalise MFCC Data
M=readNPY('ellen/train_noduplication/mfcc.npy');
S=M;
for k=1:size(M,1)
    for i=1:size(M,2)
        vq=S(k,i,:);
        S(k,i,:)=(vq-mean(vq))/(max(vq-mean(vq))+eps);
    end
end
writeNPY(S,'ellen/train_noduplication/mfcc_normalised.npy')

%% Preprocess using interpolation and loess smooth
A=readNPY('ellen/train_noduplication/body.npy');
M=readNPY('ellen/train_noduplication/mfcc_normalised.npy');
s=size(M);
clear M;
[num,T,featnum]=size(A);
B=single(zeros(num,featnum,s(3)));
for k=1:num
    for i=1:featnum
        vq = interp1(1:T,A(k,:,i),1:(T-1)/(s(3)-1):T,'spline');
        B(k,i,:)=single(vq');
    end
end
writeNPY(B,'ellen/train_noduplication/body_matched.npy');
                        
% classify training data for diferent styles according to move amplitude for example, you can classify them in different way to distinguish various styles
% in this case, we have three classes: gentle, moderate, fierce
X=readNPY('ellen\train_noduplication\body_matched.npy');
X=[X;readNPY('ellen\valid/body_matched.npy')];
M=readNPY('ellen\train_noduplication\mfcc_normalised.npy');
M=[M;readNPY('ellen\valid\mfcc_normalised.npy')];
A=[];MA=[];
B=[];MB=[];
C=[];MC=[];
for k=1:size(X,1)
    tmp=shiftdim(X(k,:,:));
    vs=max(std(tmp'));
    if vs<=0.08
        A=cat(1,A,X(k,:,:));
        MA=cat(1,MA,M(k,:,:));
    elseif vs >0.08 && vs<=0.115
        B=cat(1,B,X(k,:,:));
        MB=cat(1,MB,M(k,:,:));
    else
        C=cat(1,C,X(k,:,:));
        MC=cat(1,MC,M(k,:,:));
    end
end
writeNPY(A,'ellen\train_noduplication\body_matched_gentle.npy');
writeNPY(B,'ellen\train_noduplication\body_matched_moderate.npy');
writeNPY(C,'ellen\train_noduplication\body_matched_fierce.npy');
writeNPY(MA,'ellen\train_noduplication\mfcc_normalised_gentle.npy');
writeNPY(MB,'ellen\train_noduplication\mfcc_normalised_moderate.npy');
writeNPY(MC,'ellen\train_noduplication\mfcc_normalised_fierce.npy');

% Calculate the distances among the adjcent key points and yield key point tree
A=readNPY('ellen\train_noduplication\body_matched_gentle.npy');
tree=[11 10;
    11 9;
    11 12;
    12 2;
    2 3;
    2 13;
    2 6;
    13 1;
    3 4;
    4 5;
%     5 14;
    14 21;
    21 22;
    22 23;
    23 32;
    14 24;
    24 25;
    25 26;
    26 33;
    14 18;
    18 19;
    19 20;
    20 31;
    14 15;
    15 16;
    16 17;
    17 30;
    14 27;
    27 28;
    28 29;
    29 34;
    6 7;
    7 8;
%     8 35;
    35 48;
    48 49;
    49 50;
    50 55;
    35 36;
    36 37;
    37 38;
    38 51;
    35 39;
    39 40;
    40 41;
    41 52;
    35 45;
    45 46;
    46 47;
    47 54;
    35 42;
    42 43;
    43 44;
    44 53];
index=randi(size(A,1));
D=shiftdim(A(index,:,:));
D2=zeros(length(tree),size(D,2));
len=55;
for k=1:size(D,2)
    for i=1:length(tree)
        D2(i,k)=sqrt(sum((D(tree(i,1):len:len*3,k)-D(tree(i,2):len:len*3,k)).^2));
    end
end
D2=mean(D2');
writeNPY(tree,'ellen\train_noduplication\body_tree.npy');
writeNPY(D2,'ellen\train_noduplication\body_part_length.npy');

% Transfer key point coordinates into angles
A=readNPY('ellen\train_noduplication\body_matched_gentle.npy');
A2=zeros([size(A,1),length(tree)*2,size(A,3)],'single');
for k=1:size(A,1)
    for i=1:size(A,3)
        tmp=zeros([length(tree),2],'single');
        a=reshape(shiftdim(A(k,:,i)),[55,3]);
        for j=1:length(tree)
            p=a(tree(j,2),:)-a(tree(j,1),:);
            tmp2=sqrt(sum(p.^2));
            gamma=asin(p(2)/tmp2);
            if gamma==pi/2||gamma==-pi/2
                alpha=0;
            else
                tmp2=sqrt(p(1)^2+p(3)^2);
                if p(3)>=0 && p(1)>=0
                    alpha=asin(p(1)/tmp2);
                elseif p(3)>=0 && p(1)<0
                    alpha=asin(p(1)/tmp2);
                elseif p(3)<0 && p(1)>=0
                    alpha=acos(p(3)/tmp2);
                else
                    alpha=-acos(p(3)/tmp2);
                end
            end
            tmp(j,:)=[alpha/pi,gamma*2/pi];
        end
        A2(k,:,i)=tmp(:);
    end
end
writeNPY(A2,'ellen\train_noduplication\body_matched_gentle_angle.npy');
