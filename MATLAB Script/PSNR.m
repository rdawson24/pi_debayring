function peaksnr = PSNR(ref)
refPath = append("C:\Users\dawso.DESKTOP-SK6U4T8\Desktop\Thesis\pi_debayring_repo\", ref);
refImg = imread(refPath);
postImg = imread('C:\Users\dawso.DESKTOP-SK6U4T8\Desktop\Thesis\pi_debayring_repo\Output.png');
[peaksnr, snr] = psnr(postImg, refImg);
%fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
%fprintf('\n The SNR value is %0.4f \n', snr);