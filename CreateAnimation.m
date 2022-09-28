function fmat=CreateAnimation(A,warp)
A=shiftdim(A);
a=A(:,1);
if warp
    a=reshape(a,[3,55])';
else
    a=reshape(a,[55,3]);
end
DrawSeke(-a);
theAxes=axis;
clf;
fmat=moviein(size(A,2));
for k=1:size(A,2)
    axis(theAxes);
    a=A(:,k);
    if warp
        a=reshape(a,[3,55])';
    else
        a=reshape(a,[55,3]);
    end
    DrawSeke(-a);
    title(strcat('frame index: ',int2str(k)))
    fmat(:,k)=getframe;
    clf;
end