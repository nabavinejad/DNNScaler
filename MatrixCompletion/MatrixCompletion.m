clc;
rng(234923);    % for reproducible results
N   = 18;       % the matrix Number of Row
M   = 10;       % the matrix Number of Column
r   = 2;        % the rank of the matrix
df  = 2*M*r - r^2;  % degrees of freedom of a N x N rank r matrix
nSamples    = 172; % number of observed entries

% For this demo, we will use a matrix with integer entries
% because it will make displaying the matrix easier.
            %1   2   3   4   5   6   7   8   9
X       = [10	9	8	7	5	5	5	6	7	7
10	9	9	7	5	6	7	8	8	10
10	10	9	7	8	10	11	13	15	16
10	14	18	24	29	35	41	47	53	59
10	11	14	17	21	25	29	34	38	42
7	9	10	13	16	20	23	26	29	32
7	7	10	12	13	15	17	20	22	24
5	5	4	5	4	4	5	6	6	7
5	5	4	4	5	6	7	7	8	9
6	5	5	6	8	10	11	13	14	16
10	11	13	17	21	25	29	34	38	42
11.03	12.45	16.29	23.30	23.14	25.38	33.74	47.11	50.38	52.82
9.77	11.65	15.71	18.67	21.70	30.55	37.14	37.81	53.62	61.01
11	16	22	29	36	43	51	58	66	73
22	27	25	31	38	46	52	59	66	73
28	26	29	37	41	47	55	63	71	79
10	9	10	13	16	19	23	26	29	32
10	9	10	12	13	15	17	20	22	28


]; % Our target matrix
%rPerm   = randperm(N*M); % use "randsample" if you have the stats toolbox
%the base and complete rPerm
%rPerm = [ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44  45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88];

%the missed rPerm
rPerm = [1	19	37	55	73	91	109	127	145	163 2	20	38	56	74	92	110	128	146	164 3	21	39	57	75	93	111	129	147	165 4	22	40	58	76	94	112	130	148	166 5	23	41	59	77	95	113	131	149	167 6	24	42	60	78	96	114	132	150	168 7	25	43	61	79	97	115	133	151	169 8	26	44	62	80	98	116	134	152	170 9	27	45	63	81	99	117	135	153	171 10	28	46	64	82	100	118	136	154	172 11	29	47	65	83	101	119	137	155	173 12	30	48	66	84	102	120	138	156	174 13	31	49	67	85	103	121	139	157	175 14	32	50	68	86	104	122	140	158	176 15	33	51	69	87	105	123	141	159	177 16	34	52	70	88	106	124	142	160	178 17	35	53	71	89	107	125	143	161	179 18 144];
%disp(rPerm);
omega   = sort( rPerm(1:nSamples) );
Y = nan(N,M);
Y(omega) = X(omega);
disp('The "NaN" entries represent unobserved values');
disp(Y)
fprintf([repmat(sprintf('%% %dd',max(floor(log10(abs(Y(:)))))+2+any(Y(:)<0)),1,size(Y,2)) '\n'],Y');
% Add TFOCS to your path (modify this line appropriately):
addpath C:\Users\Morteza\Dropbox\39_ScaleUp_ScaleOut\Experiments_DNNScaler\DNNScaler\Multi_tenancy_MatrixCompletion\TFOCS\

observations = X(omega);    % the observed entries
mu           = .00001;        % smoothing parameter

% The solver runs in seconds
tic;
Xk = solver_sNuclearBP( {N,M,omega}, observations, mu );
toc;
format shortG
disp('Recovered matrix (rounding to nearest .0001):')
%disp( round(Xk*1000000)/1000000 )
format shortG
%Xk = ceil(Xk);
fprintf([repmat(sprintf('%% %dd',max(floor(log10(abs(Xk(:)))))+2+any(Xk(:)<0)),1,size(Xk,2)) '\n'],Xk');
% and for reference, here is the original matrix:
disp('Original matrix:')
%disp( X )

% The relative error (without the rounding) is quite low:
fprintf('Relative error, no rounding: %.8f%%\n', norm(X-Xk,'fro')/norm(X,'fro')*100 );
fprintf('%d, %d, error (percent) = ,%f \n', X(36),Xk(36), abs((X(36)-Xk(36))/X(36))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(54),Xk(54), abs((X(54)-Xk(54))/X(54))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(72),Xk(72), abs((X(72)-Xk(72))/X(72))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(90),Xk(90), abs((X(90)-Xk(90))/X(90))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(108),Xk(108), abs((X(108)-Xk(108))/X(108))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(126),Xk(126), abs((X(126)-Xk(126))/X(126))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(162),Xk(162), abs((X(162)-Xk(162))/X(162))*100)
fprintf('%d, %d, error (percent) = ,%f \n', X(180),Xk(180), abs((X(180)-Xk(180))/X(180))*100)