%% Demo for FGCQA
ref = imread('1.bmp');
dis = imread('1_b1.jpg');
Score = FGCQA_score(ref,dis);