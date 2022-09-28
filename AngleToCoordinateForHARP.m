function D2=AngleToCoordinateForHARP(D,len,tree,constraint,docorrect)

D(1:size(D,1)/2,:)=D(1:size(D,1)/2,:)+1;
D(size(D,1)/2+1:end,:)=-D(size(D,1)/2+1:end,:);

% preprocess the first frame
D=D(:,2:end);

constraint=[constraint(:,1:2)/180+1;constraint(:,3:4)/90];
if docorrect
    for i=1:length(constraint)
        D(i,:)=RegulateInRange2(D(i,:),constraint(i,1),constraint(i,2));
%         D(i,:)=RegulateInRange(D(i,:),AngleConstraint2(i,1),AngleConstraint2(i,2),1.6,1);
        D(i,:)=smooth(D(i,:),'loess');
    end
end

D2=zeros([55*3,size(D,2)],'single');
a=zeros([55,3],'single');

for k=1:size(D,2)
    tmp=reshape(D(:,k),[52,2]);
    a(11,:)=single([0,0,0]);
    
    for i=1:length(tree)
        a(tree(i,2),:)=a(tree(i,1),:)+len(i)*[cos(tmp(i,2)*pi/2)*sin(tmp(i,1)*pi),...
            sin(tmp(i,2)*pi/2),cos(tmp(i,2)*pi/2)*cos(tmp(i,1)*pi)];
        if tree(i,2)==5
            a(14,:)=a(5,:);
        end
        if tree(i,2)==8
            a(35,:)=a(8,:);
        end
    end
    D2(:,k)=a(:);
end
end

function s=RegulateInRange2(s,lower,upper)
if (lower==0 && upper==0) || (lower==1 && upper==1)
    return;
end
a=max(s);
b=min(s);
s=(s-b)*(upper-lower)/(a-b)+lower;
end

function s=RegulateInRange(s,lower,upper,b,c)
if lower==0 && upper==0
    return;
end
while max(s)>upper || min(s)<lower
    s=upper-FakeAbs2(upper-(lower+FakeAbs2(s-lower,b,c)),b,c);
end
a=(upper-lower)/(max(s)-min(s));
b=upper-a*max(s);
s=a*s+b;
end

function y=FakeAbs(x)
if x==0
    x=eps;
end
y=4*(x-sin(x))/(2-sin(2))./x;
end

function y=FakeAbs2(x,b,c)
y=2*(cos(b*x)-c)/(cos(2*b)-c);
end