% Render a Matlab Animation
A2=readNPY('D:\kw3\KstFakeMan\Gesture\generated.npy');
len=readNPY('ellen\train_noduplication\body_part_length.npy');
tree=readNPY('ellen\train_noduplication\body_tree.npy');
D=shiftdim(A2(13,:,:));
A=AngleToCoordinateForHARP(D,len,tree,0,0);
DrawSeke(reshape(A(:,1),[55,3]));
theAxes=axis;
clf;
fmat=moviein(size(A,2));
for k=1:size(A,2)
    axis(theAxes);
    view(-30,50);
    DrawSeke(reshape(A(:,k),[55,3]));
    title(strcat('frame index: ',int2str(k)))
    fmat(:,k)=getframe;
    clf;
end

% write video
v = VideoWriter('UserStudy\harp3.mp4','MPEG-4');
v.FrameRate=60;
open(v);
writeVideo(v,fmat);
close(v);
