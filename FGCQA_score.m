function Score = FGCQA_score(img1,img2)
%%Tranform the RGB channels to YCbCr channels
%img1---Reference, img2---Distorted
img1 = double(img1);
img2 = double(img2);
YCbCr1 = rgb2ycbcr(img1);
YCbCr2 = rgb2ycbcr(img2);
%% Feture Group1: Gradient-based Features
Y1 = YCbCr1(:,:,1);
Y2 = YCbCr2(:,:,1);
T = 170; 
Down_step = 1;
% gradient template
dx = [1/4 0 -1/4; 1/2 0 -1/2; 1/4 0 -1/4];
%dx = [1 0 -1; 1 0 -1; 1 0 -1]/3;
dy = dx';
% get gradient maps
aveKernel = fspecial('average',2);
aveY1 = conv2(Y1, aveKernel,'same');
aveY2 = conv2(Y2, aveKernel,'same');
Y1 = aveY1(1:Down_step:end,1:Down_step:end);
Y2 = aveY2(1:Down_step:end,1:Down_step:end);

IxY1 = conv2(Y1, dx, 'same');     
IyY1 = conv2(Y1, dy, 'same');    
gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);

IxY2 = conv2(Y2, dx, 'same');     
IyY2 = conv2(Y2, dy, 'same');

gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);

% calculate simialrity
gradientSIM = ((2*gradientMap1.*gradientMap2+170)./(gradientMap1.^2+gradientMap2.^2+170));
% define selected regions
highFrequencyArea = (gradientMap1 > mean2(gradientMap1)) | (gradientMap2 > mean2(gradientMap2));
lowFrequencyArea = (abs(gradientMap2-gradientMap1) > abs(mean2(gradientMap2-gradientMap1))) & gradientMap1 < mean2(gradientMap1);
area = highFrequencyArea | lowFrequencyArea;
if sum(area(:))==0
    area = ones(size(gradientSIM));
end
%% Feature Group 2: Texture-based Features
[M, N, L] = size(YCbCr1);
TextureMap = zeros( M, N, 3);
weight       = [0.5 0.75 1 5 6];
weight       = weight./sum(weight(:));
% get texture similarity maps of YCbCr
for i =1:3
    gabRef  = gaborconvolve( double( YCbCr1(:,:,i) ) );
    gabDst  = gaborconvolve( double( YCbCr2(:,:,i) ) );
    for gb_i =1:5 % five scales
        for gb_j = 1:4 % four orientations
            An_ref = abs(gabRef{gb_i,gb_j});
            An_dst = abs(gabDst{gb_i,gb_j});
            TextureMap(:,:,i) = TextureMap(:,:,i) + weight(gb_i).*((2.*An_ref.*An_dst+1)./(An_ref.^2+An_dst.^2+1));
        end
    end
end
% Get final texture map
FinalTextureMap = sqrt(TextureMap(:,:,1).^2 + 0.25 * TextureMap(:,:,2).^2 + 0.25 * TextureMap(:,:,3).^2);
features = [mean2(gradientSIM(area)),std2(gradientSIM(area)),mean2(FinalTextureMap(:)),std2(FinalTextureMap(:))];
Score = features(1)^(0.5)*features(2)^(-0.5)*features(3)^(0.5)*features(4)^(-0.5); %for fgiqa
%Score = features(1)^(1)*features(2)^(-1)*features(3)^(1)*features(4)^(-1); %for fgiqa
end


function EO = gaborconvolve( im )

% GABORCONVOLVE - function for convolving image with log-Gabor filters
%
%   Usage: EO = gaborconvolve(im,  nscale, norient )
%
% For details of log-Gabor filters see:
% D. J. Field, "Relations Between the Statistics of Natural Images and the
% Response Properties of Cortical Cells", Journal of The Optical Society of
% America A, Vol 4, No. 12, December 1987. pp 2379-2394
% Notes on filter settings to obtain even coverage of the spectrum
% dthetaOnSigma 1.5
% sigmaOnf  .85   mult 1.3
% sigmaOnf  .75   mult 1.6     (bandwidth ~1 octave)
% sigmaOnf  .65   mult 2.1
% sigmaOnf  .55   mult 3       (bandwidth ~2 octaves)
%
% Author: Peter Kovesi
% Department of Computer Science & Software Engineering
% The University of Western Australia
% pk@cs.uwa.edu.au  www.cs.uwa.edu.au/~pk
%
% May 2001
% Altered, 2008, Eric Larson
% Altered precomputations, 2011, Eric Larson

nscale          = 5;      %Number of wavelet scales.
norient         = 4;      %Number of filter orientations.
minWaveLength   = 6;      %Wavelength of smallest scale filter.
mult            = 2;      %Scaling factor between successive filters.
sigmaOnf        = 0.55;   %Ratio of the standard deviation of the
%Gaussian describing the log Gabor filter's transfer function
%in the frequency domain to the filter center frequency.
%Orig: 3 6 12 27 64
wavelength      = [minWaveLength ...
  minWaveLength*mult^1 ...
  minWaveLength*mult^2 ...
  minWaveLength*mult^3 ...
  minWaveLength*mult^4 ...
   ];

dThetaOnSigma   = 1.2;    %Ratio of angular interval between filter orientations
% 			       and the standard deviation of the angular Gaussian
% 			       function used to construct filters in the
%                              freq. plane.


[rows cols] = size( im );
imagefft    = fft2( im );            % Fourier transform of image

EO = cell( nscale, norient );        % Pre-allocate cell array

% Pre-compute to speed up filter construction
x = ones(rows,1) * (-cols/2 : (cols/2 - 1))/(cols/2);
y = (-rows/2 : (rows/2 - 1))' * ones(1,cols)/(rows/2);
radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
radius(round(rows/2+1),round(cols/2+1)) = 1; % Get rid of the 0 radius value in the middle
radius = log(radius);
% so that taking the log of the radius will
% not cause trouble.

% Precompute sine and cosine of the polar angle of all pixels about the
% centre point

theta = atan2(-y,x);              % Matrix values contain polar angle.
% (note -ve y is used to give +ve
% anti-clockwise angles)
sintheta = sin(theta);
costheta = cos(theta);
clear x; clear y; clear theta;      % save a little memory

%lp = lowpassfilter([rows,cols],.45,15); 

thetaSigma = pi/norient/dThetaOnSigma;  % Calculate the standard deviation of the
% angular Gaussian function used to
% construct filters in the freq. plane.
rows = round(rows/2+1);
cols = round(cols/2+1);
% precompute the scaling filters
logGabors = cell(1,nscale);
for s = 1:nscale                  % For each scale.
    
    % Construct the filter - first calculate the radial filter component.
    fo = 1.0/wavelength(s);                  % Centre frequency of filter.
    rfo = fo/0.5;                         % Normalised radius from centre of frequency plane
    % corresponding to fo.
    tmp = -(2 * log(sigmaOnf)^2);
    tmp2= log(rfo);
    logGabors{s} = exp( (radius-tmp2).^2 /tmp  );
    
    %logGabors{s} = logGabors{s}.*lp;
    logGabors{s}( rows, cols ) = 0; % Set the value at the center of the filter
    % back to zero (undo the radius fudge).
end


% The main loop...
for o = 1:norient,                   % For each orientation.
  fprintf('.');
  angl = (o-1)*pi/norient;           % Calculate filter angle.
  
  % Pre-compute filter data specific to this orientation
  % For each point in the filter matrix calculate the angular distance from the
  % specified filter orientation.  To overcome the angular wrap-around problem
  % sine difference and cosine difference values are first computed and then
  % the atan2 function is used to determine angular distance.
  
  ds = sintheta * cos(angl) - costheta * sin(angl);     % Difference in sine.
  dc = costheta * cos(angl) + sintheta * sin(angl);     % Difference in cosine.
  dtheta = abs(atan2(ds,dc));                           % Absolute angular distance.
  spread = exp((-dtheta.^2) / (2 * thetaSigma^2));      % Calculate the angular filter component.
  
  for s = 1:nscale,                  % For each scale.

    filter = fftshift( logGabors{s} .* spread ); % Multiply by the angular spread to get the filter
    % and swap quadrants to move zero frequency
    % to the corners.
    
    % Do the convolution, back transform, and save the result in EO
    EO{s,o} = ifft2( imagefft .* filter );
    
    
  end  % ... and process the next scale
  
end  % For each orientation

end

function YCbCr_data = rgb2ycbcr(RGB_data)
R_data =    RGB_data(:,:,1);
G_data =    RGB_data(:,:,2);
B_data =    RGB_data(:,:,3); 
[ROW,COL, DIM] = size(RGB_data); %提取图片的行列数 
Y_data = zeros(ROW,COL);
Cb_data = zeros(ROW,COL);
Cr_data = zeros(ROW,COL);
Gray_data = RGB_data;
%YCbCr_data = RGB_data;
for r = 1:ROW 
     for c = 1:COL
         Y_data(r, c) = 0.299*R_data(r, c) + 0.587*G_data(r, c) + 0.114*B_data(r, c);
        Cb_data(r, c) = -0.172*R_data(r, c) - 0.339*G_data(r, c) + 0.511*B_data(r, c) + 128;
         Cr_data(r, c) = 0.511*R_data(r, c) - 0.428*G_data(r, c) - 0.083*B_data(r, c) + 128;
     end
end
 
YCbCr_data(:,:,1)=Y_data;
YCbCr_data(:,:,2)=Cb_data;
YCbCr_data(:,:,3)=Cr_data;
end

function f = lowpassfilter(sze, cutoff, n)
    
    if cutoff < 0 || cutoff > 0.5
	error('cutoff frequency must be between 0 and 0.5');
    end
    
    if rem(n,1) ~= 0 || n < 1
	error('n must be an integer >= 1');
    end

    if length(sze) == 1
	rows = sze; cols = sze;
    else
	rows = sze(1); cols = sze(2);
    end

    % Set up X and Y matrices with ranges normalised to +/- 0.5
    % The following code adjusts things appropriately for odd and even values
    % of rows and columns.
    if mod(cols,2)
	xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
    else
	xrange = [-cols/2:(cols/2-1)]/cols;	
    end

    if mod(rows,2)
	yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
    else
	yrange = [-rows/2:(rows/2-1)]/rows;	
    end
    
    [x,y] = meshgrid(xrange, yrange);
    radius = sqrt(x.^2 + y.^2);        % A matrix with every pixel = radius relative to centre.
    f = ifftshift( 1 ./ (1.0 + (radius ./ cutoff).^(2*n)) );   % The filter
    return;
    end