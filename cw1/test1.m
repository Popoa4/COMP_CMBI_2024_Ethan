load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);
%display the Middle slice of the 1st image volume, which has b=0
imshow(flipud(squeeze(dwis(1,:,:,72))'), []);
% Middle slice of the 2nd image volume, which has b=1000
imshow(flipud(squeeze(dwis(2,:,:,72))'), []);

qhat = load('bvecs');
bvals = 1000*sum(qhat.*qhat);

