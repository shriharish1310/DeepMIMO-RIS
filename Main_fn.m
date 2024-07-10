function [Rate_DL,Rate_OPT]=Main_fn(L,My,Mz,M_bar,K_DL,Pt,kbeams,Training_Size)

params.scenario='O1_28'; % DeepMIMO Dataset scenario: http://deepmimo.net/
params.active_BS=3; % active basestation(/s) in the chosen scenario
D_Lambda = 0.5; % Antenna spacing relative to the wavelength
BW = 100e6; % Bandwidth

Ut_row = 850; % user Ut row` number
Ut_element = 90; % user Ut position from the row chosen above
Ur_rows = [1000 1200]; % user Ur rows

Validation_Size = 6200; % Validation dataset Size
K = 512; % number of subcarriers
miniBatchSize  = 32; % Size of the minibatch for the Deep Learning
% Note: The axes of the antennas match the axes of the ray-tracing scenario
Mx = 1;  % number of LIS reflecting elements across the x axis
M = Mx.*My.*Mz; % Total number of LIS reflecting elements 

% Preallocation of output variables
Rate_DL = zeros(1,length(Training_Size)); 
Rate_OPT = Rate_DL;
LastValidationRMSE = Rate_DL;

%--- Accounting SNR in ach rate calculations
%--- Definning Noisy channel measurements
Gt=3;             % dBi
Gr=3;             % dBi
NF=5;             % Noise figure at the User equipment
Process_Gain=10;  % Channel estimation processing gain
noise_power_dB=-204+10*log10(BW/K)+NF-Process_Gain; % Noise power in dB
SNR=10^(.1*(-noise_power_dB))*(10^(.1*(Gt+Gr+Pt)))^2; % Signal-to-noise ratio
% channel estimation noise
noise_power_bar=10^(.1*(noise_power_dB))/(10^(.1*(Gt+Gr+Pt))); 

No_user_pairs = (Ur_rows(2)-Ur_rows(1))*181; % Number of (Ut,Ur) user pairs            
RandP_all = randperm(No_user_pairs).'; % Random permutation of the available dataset

%% Starting the code
disp('======================================================================================================================');
disp([' Calculating for M = ' num2str(M)]);
Rand_M_bar_all = randperm(M);
    
%% Beamforming Codebook
% BF codebook parameters
over_sampling_x=1;            % The beamsteering oversampling factor in the x direction
over_sampling_y=1;            % The beamsteering oversampling factor in the y direction
over_sampling_z=1;            % The beamsteering oversampling factor in the z direction

% Generating the BF codebook 
[BF_codebook]=sqrt(Mx*My*Mz)*UPA_codebook_generator(Mx,My,Mz,over_sampling_x,over_sampling_y,over_sampling_z,D_Lambda);
codebook_size=size(BF_codebook,2);
    
%% DeepMIMO Dataset Generation
disp('-------------------------------------------------------------');
disp([' Calculating for K_DL = ' num2str(K_DL)]);          
% ------  Inputs to the DeepMIMO dataset generation code ------------ % 
% Note: The axes of the antennas match the axes of the ray-tracing scenario
params.num_ant_x= Mx;             % Number of the UPA antenna array on the x-axis 
params.num_ant_y= My;             % Number of the UPA antenna array on the y-axis 
params.num_ant_z= Mz;             % Number of the UPA antenna array on the z-axis
params.ant_spacing=D_Lambda;          % ratio of the wavelnegth; for half wavelength enter .5        
params.bandwidth= BW*1e-9;            % The bandiwdth in GHz 
params.num_OFDM= K;                   % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;        % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=K_DL*1;         % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels
params.num_paths=L;               % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path
params.saveDataset=0;
disp([' Calculating for L = ' num2str(params.num_paths)]);

% ------------------ DeepMIMO "Ut" Dataset Generation -----------------%
params.active_user_first=Ut_row; 
params.active_user_last=Ut_row;
DeepMIMO_dataset=DeepMIMO_generator(params);
Ht = single(DeepMIMO_dataset{1}.user{Ut_element}.channel);
clear DeepMIMO_dataset

% ------------------ DeepMIMO "Ur" Dataset Generation -----------------%            
%Validation part for the actual achievable rate perf eval
Validation_Ind = RandP_all(end-Validation_Size+1:end);
[~,VI_sortind] = sort(Validation_Ind);
[~,VI_rev_sortind] = sort(VI_sortind);
%initialization
Ur_rows_step = 100; % access the dataset 100 rows at a time
Ur_rows_grid=Ur_rows(1):Ur_rows_step:Ur_rows(2);
Delta_H_max = single(0);
for pp = 1:1:numel(Ur_rows_grid)-1 % loop for Normalizing H
    clear DeepMIMO_dataset
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1; 
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
    for u=1:params.num_user
        Hr = single(conj(DeepMIMO_dataset{1}.user{u}.channel));                                
        Delta_H = max(max(abs(Ht.*Hr)));
        if Delta_H >= Delta_H_max
            Delta_H_max = single(Delta_H);
        end    
    end
end
clear Delta_H
disp('=============================================================');
disp([' Calculating for M_bar = ' num2str(M_bar)]);          
Rand_M_bar =unique(Rand_M_bar_all(1:M_bar));
Ht_bar = reshape(Ht(Rand_M_bar,:),M_bar*K_DL,1);
DL_input = single(zeros(M_bar*K_DL*2,No_user_pairs));
DL_output = single(zeros(No_user_pairs,codebook_size));
DL_output_un=  single(zeros(numel(Validation_Ind),codebook_size));
Delta_H_bar_max = single(0);
count=0;
for pp = 1:1:numel(Ur_rows_grid)-1
    clear DeepMIMO_dataset 
    disp(['Starting received user access ' num2str(pp)]);
    params.active_user_first=Ur_rows_grid(pp);
    params.active_user_last=Ur_rows_grid(pp+1)-1;
    [DeepMIMO_dataset,params]=DeepMIMO_generator(params);
    %% Construct Deep Learning inputs
    u_step=100;
    Htx=repmat(Ht(:,1),1,u_step);
    Hrx=zeros(M,u_step);
    for u=1:u_step:params.num_user                        
        for uu=1:1:u_step
            Hr = single(conj(DeepMIMO_dataset{1}.user{u+uu-1}.channel));               
            Hr_bar = reshape(Hr(Rand_M_bar,:),M_bar*K_DL,1);
            %--- Constructing the sampled channel
            n1=sqrt(noise_power_bar/2)*(randn(M_bar*K_DL,1)+1j*randn(M_bar*K_DL,1));
            n2=sqrt(noise_power_bar/2)*(randn(M_bar*K_DL,1)+1j*randn(M_bar*K_DL,1));
            H_bar = ((Ht_bar+n1).*(Hr_bar+n2));
            DL_input(:,u+uu-1+((pp-1)*params.num_user))= reshape([real(H_bar) imag(H_bar)].',[],1);
            Delta_H_bar = max(max(abs(H_bar)));
            if Delta_H_bar >= Delta_H_bar_max
                Delta_H_bar_max = single(Delta_H_bar);
            end
            Hrx(:,uu)=Hr(:,1);
        end
        %--- Actual achievable rate for performance evaluation
        H = Htx.*Hrx;
        H_BF=H.'*BF_codebook;
        SNR_sqrt_var = abs(H_BF);
        for uu=1:1:u_step
            if sum((Validation_Ind == u+uu-1+((pp-1)*params.num_user)))
                count=count+1;
                DL_output_un(count,:) = single(sum(log2(1+(SNR*((SNR_sqrt_var(uu,:)).^2))),1));
            end
        end
        %--- Label for the sampled channel
        R = single(log2(1+(SNR_sqrt_var/Delta_H_max).^2));
        % --- DL output normalization
        Delta_Out_max = max(R,[],2);
        if ~sum(Delta_Out_max == 0)
           Rn=diag(1./Delta_Out_max)*R; 
        end
        DL_output(u+((pp-1)*params.num_user):u+((pp-1)*params.num_user)+u_step-1,:) = 1*Rn; %%%%% Normalized %%%%%
    end
end
clear u Delta_H_bar R Rn
%-- Sorting back the DL_output_un
DL_output_un = DL_output_un(VI_rev_sortind,:);
%--- DL input normalization 
DL_input= 1*(DL_input/Delta_H_bar_max); %%%%% Normalized from -1->1 %%%%%

%% DL Beamforming

% ------------------ Training and Testing Datasets -----------------%
DL_output_reshaped = reshape(DL_output.', 1, 1, size(DL_output, 2), size(DL_output, 1));
DL_output_reshaped_un = reshape(DL_output_un.', 1, 1, size(DL_output_un, 2), size(DL_output_un, 1));
DL_input_reshaped = reshape(DL_input, size(DL_input, 1), 1, 1, size(DL_input, 2));

for dd=1:1:numel(Training_Size)
    disp([' Calculating for Dataset Size = ' num2str(Training_Size(dd))]);
    Training_Ind   = RandP_all(1:Training_Size(dd));

    XTrain = single(DL_input_reshaped(:,1,1,Training_Ind)); 
    YTrain = single(DL_output_reshaped(1,1,:,Training_Ind));
    XValidation = single(DL_input_reshaped(:,1,1,Validation_Ind));
    YValidation = single(DL_output_reshaped(1,1,:,Validation_Ind));
    YValidation_un = single(DL_output_reshaped_un);
    
     % ------------------ DL Model definition -----------------%
    % Input layer
        inputLayer = imageInputLayer([size(XTrain,1),1,1], ...
            'DataAugmentation', 'none', ...
            'Normalization', 'none', ...
            'Name', 'inputLayer');
        
        % AlexNet layers (from 2:16)
        net = alexnet;
        layersTransfer = net.Layers(2:10);
        %Modified pool1
        pool1_mod = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1_mod', 'Padding', 1);
        clear net;
        
        % Modified conv1 layer
        filterSize = [3, 3];
        numFilters = 96;
        conv1 = convolution2dLayer(filterSize, numFilters, ...
            'Stride', 1, 'Padding', 1, 'WeightL2Factor', 0.8, ...
            'Name', 'conv1_mod', 'WeightLearnRateFactor', 50 / 20, ...
            'BiasLearnRateFactor', 50 / 20);
        %Modified conv2 layer
        filterSize = [3, 3];
        numFilters = 128;
        numGroups = 2;
        conv2 = groupedConvolution2dLayer(filterSize, numFilters, numGroups, ...
            'Stride', 1, 'Padding', 1, 'WeightL2Factor', 0.8, ...
            'Name', 'conv2_mod', 'WeightLearnRateFactor', 50 / 20, ...
            'BiasLearnRateFactor', 50 / 20);
        
        %Modified pool2
        pool2_mod = maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2_mod', 'Padding', 1);

        % Modified conv3 layer
        filterSize = [3, 3];
        numFilters = 384;
        conv3 = convolution2dLayer(filterSize, numFilters, ...
            'Stride', 1, 'Padding', 1, 'WeightL2Factor', 0.8, ...
            'Name', 'conv3_mod', 'WeightLearnRateFactor', 50 / 20, ...
            'BiasLearnRateFactor', 50 / 20);
        


        % Replace the original conv layer with the modified one
        layersTransfer(1) = conv1;
        layersTransfer(4) = pool1_mod;
        layersTransfer(5) = conv2;
        layersTransfer(8) = pool2_mod;
        layersTransfer(9) = conv3;

        outSize = size(YTrain,3);
        % Fully connected layers (same as before)
        fc6 = fullyConnectedLayer(outSize*16, 'Name', 'fc6');
        relu6 = reluLayer('Name', 'relu6');
        drop6 = dropoutLayer(0.5, 'Name', 'drop6');
        
        fc7 = fullyConnectedLayer(outSize*8, 'Name', 'fc7');
        relu7 = reluLayer('Name', 'relu7');
        drop7 = dropoutLayer(0.5, 'Name', 'drop7');
        
        fc8 = fullyConnectedLayer(outSize*4, 'Name', 'fc8');
        relu8 = reluLayer('Name', 'relu8');
        drop8 = dropoutLayer(0.5, 'Name', 'drop8');
        
        
        fc9 = fullyConnectedLayer(outSize, 'Name', 'fc9');
        
        % Regression layer
        regLayer = regressionLayer('Name', 'reg');
        
        % Arrange layers
        layers = [
    inputLayer;
    layersTransfer;
    fc6; relu6; drop6; fc7; relu7; drop7; fc8; relu8; drop8; fc9; regLayer
    ];

    if Training_Size(dd) < miniBatchSize
        validationFrequency = Training_Size(dd);
    else
        validationFrequency = floor(Training_Size(dd)/miniBatchSize);
    end
    VerboseFrequency = validationFrequency;
    options = trainingOptions('rmsprop', ...   
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',10, ...
        'InitialLearnRate',1e-4, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateDropPeriod',3, ...
        'L2Regularization',1e-5,...
        'Shuffle','every-epoch', ...
        'ValidationData',{XValidation,YValidation}, ...
        'ValidationFrequency',validationFrequency, ...
        'Plots','none', ... % 'training-progress'
        'Verbose',0, ...    % 1  
        'ExecutionEnvironment', 'gpu', ...
        'VerboseFrequency',VerboseFrequency);   
    % ------------- DL Model Training and Prediction -----------------%
    [~,Indmax_OPT]= max(YValidation,[],3);
    Indmax_OPT = squeeze(Indmax_OPT); %Upper bound on achievable rates
    MaxR_OPT = single(zeros(numel(Indmax_OPT),1));                      
        [trainedNet,traininfo]  = trainNetwork(XTrain,YTrain,layers,options);               
    YPredicted = predict(trainedNet,XValidation);

    % --------------------- Achievable Rate --------------------------%                    
    [~,Indmax_DL] = maxk(YPredicted,kbeams,2);
    MaxR_DL = single(zeros(size(Indmax_DL,1),1)); %True achievable rates    
    for b=1:size(Indmax_DL,1)
        MaxR_DL(b) = max(squeeze(YValidation_un(1,1,Indmax_DL(b,:),b)));
        MaxR_OPT(b) = squeeze(YValidation_un(1,1,Indmax_OPT(b),b));
    end
    Rate_OPT(dd) = mean(MaxR_OPT);          
    Rate_DL(dd) = mean(MaxR_DL);
    LastValidationRMSE(dd) = traininfo.ValidationRMSE(end);                                          
    clear trainedNet traininfo YPredicted
    clear layers options Rate_DL_Temp MaxR_DL_Temp Highest_Rate
end

end