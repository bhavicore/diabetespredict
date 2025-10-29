
clearvars;
close all;
clc;



categoryMap = containers.Map('KeyType','char','ValueType','any');
maxVarsEstimate = 2000;
categoricalCols = cell(1, maxVarsEstimate);
catIdx = 0;

try
    %% STEP 1: Read dictionary 
    
    dictFile = 'DataDictionaryWiDS2021.csv';
    sampleSubFile = 'SampleSubmissionWiDS2021.csv';
    if ~isfile(dictFile), error('Missing %s', dictFile); end
    if ~isfile(sampleSubFile), error('Missing %s', sampleSubFile); end

    dictOpts = detectImportOptions(dictFile, 'TextType','string');
    dictionary = readtable(dictFile, dictOpts);
    sampleSub = readtable(sampleSubFile);

    idxTarget = find(contains(string(dictionary.Category),"Target","IgnoreCase",true),1);
    if isempty(idxTarget)
        targetVarName = sampleSub.Properties.VariableNames{2};
        warning('Using fallback target: %s', targetVarName);
    else
        targetVarName = char(dictionary.VariableName(idxTarget));
    end
    idVarName = sampleSub.Properties.VariableNames{1};
    disp(['Target: ', targetVarName, ' | ID: ', idVarName]);

    %% STEP 2: training data
    
    fullSet = readtable('TrainingWiDS2021.csv');
    if ~ismember(targetVarName, fullSet.Properties.VariableNames)
        alt = {'diabetes_mellitus','diabetes'};
        for k=1:numel(alt)
            if ismember(alt{k}, fullSet.Properties.VariableNames)
                targetVarName = alt{k};
                disp(['Using alternative target: ', targetVarName]);
                break;
            end
        end
    end

    Y = fullSet.(targetVarName);
    if ~isnumeric(Y)
        try
            Y = double(categorical(Y));
        catch ex
            warning(ex.identifier,'%s',ex.message);
            Y = str2double(string(Y));
        end
    end

    colsToRemove_IDs = intersect({targetVarName,idVarName,'patient_id'}, fullSet.Properties.VariableNames,'stable');
    X = removevars(fullSet, colsToRemove_IDs);

    %% STEP 3: Feature engg
    
    if ismember('bmi', X.Properties.VariableNames)
        X.bmi = str2double(string(X.bmi));
    end
    if all(ismember({'gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache'}, X.Properties.VariableNames))
        X.gcs_total = X.gcs_eyes_apache + X.gcs_motor_apache + X.gcs_verbal_apache;
    end

    %% STEP 4: Cleaning data
    
    percentMissing = mean(ismissing(X),1)*100;
    threshold = 40;
    toDrop = X.Properties.VariableNames(percentMissing>threshold);
    X = removevars(X,toDrop);
    combinedTable = addvars(X,Y,'NewVariableNames',{'Target'});
    cleanedTable = rmmissing(combinedTable);
    Y = cleanedTable.Target;
    X = removevars(cleanedTable,'Target');

   
    figure('Name','Missing Data Overview');
    bar(percentMissing);
    title('Percentage of Missing Values per Feature');
    ylabel('% Missing');
    xlabel('Features');
    grid on;

   %Step5: Encode
    varNames = X.Properties.VariableNames;
    for i=1:numel(varNames)
        cName = varNames{i};
        cData = X.(cName);
        if iscell(cData) || isstring(cData) || ischar(cData) || iscategorical(cData)
            catIdx = catIdx + 1;
            categoricalCols{catIdx} = cName;
            temp_cat = categorical(cData);
            categoryMap(cName) = cellstr(categories(temp_cat));
            X.(cName) = double(temp_cat);
        end
    end
    categoricalCols = categoricalCols(1:catIdx);

    varNames2 = X.Properties.VariableNames;
    for i=1:numel(varNames2)
        v = varNames2{i};
        if ~isnumeric(X.(v))
            conv = str2double(string(X.(v)));
            if any(~isnan(conv))
                X.(v) = conv;
            else
                warning('Dropping non-numeric col: %s', v);
                X = removevars(X,v);
            end
        end
    end

    finalTrainVarNames = X.Properties.VariableNames;
    X_encoded = table2array(X);

    %% STEP 6: Split train/test
   
    cv = cvpartition(Y,'HoldOut',0.2);
    XTrain = X_encoded(training(cv),:);
    YTrain = Y(training(cv));
    XValid = X_encoded(test(cv),:);
    YValid = Y(test(cv));

    
    figure('Name','Target Distribution');
    histogram(Y,'FaceColor',[0.2 0.6 0.8]);
    title('Distribution of Target Variable');
    xlabel('Target (0 = No Diabetes, 1 = Diabetes)');
    ylabel('Count');
    grid on;

    %% STEP 7: Training 
  
    t = templateTree('Reproducible',true);
    model = fitcensemble(XTrain,YTrain,'Method','Bag','Learners',t,'NumLearningCycles',100);
    disp('Model training completed.');

    %% STEP 8: Evaluate
    disp('Step 8: Validation...');
    [predLabelsVal, scoresVal] = predict(model,XValid);
    if size(scoresVal,2)>=2
        posScores = scoresVal(:,2);
        [Xroc,Yroc,~,AUC] = perfcurve(YValid,posScores,1);
        disp(['Validation AUC: ',num2str(AUC)]);

       
        figure('Name','ROC Curve');
        plot(Xroc,Yroc,'LineWidth',2);
        hold on;
        plot([0 1],[0 1],'--r');
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(['ROC Curve (AUC = ', num2str(AUC), ')']);
        grid on;

        
        figure('Name','Predicted Probability Distribution');
        histogram(posScores,20,'FaceColor',[0.4 0.7 0.3]);
        title('Predicted Probability Distribution on Validation Set');
        xlabel('Predicted Probability of Diabetes');
        ylabel('Number of Patients');
        grid on;
    end

    %% STEP 9: submission file
    
    XSubmit_table = readtable('UnlabeledWiDS2021.csv');
    submit_IDs = XSubmit_table.(idVarName);
    XSubmit_processed = removevars(XSubmit_table,intersect({'patient_id',idVarName},XSubmit_table.Properties.VariableNames,'stable'));

    % Apply same preprocessing
    if ismember('bmi', XSubmit_processed.Properties.VariableNames)
        XSubmit_processed.bmi = str2double(string(XSubmit_processed.bmi));
    end
    if all(ismember({'gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache'}, XSubmit_processed.Properties.VariableNames))
        XSubmit_processed.gcs_total = XSubmit_processed.gcs_eyes_apache + XSubmit_processed.gcs_motor_apache + XSubmit_processed.gcs_verbal_apache;
    end
    XSubmit_processed = removevars(XSubmit_processed,intersect(XSubmit_processed.Properties.VariableNames,toDrop,'stable'));

    for i=1:numel(categoricalCols)
        cName = categoricalCols{i};
        if ~ismember(cName,XSubmit_processed.Properties.VariableNames), continue; end
        trainCats = categoryMap(cName);
        testStr = string(XSubmit_processed.(cName));
        [lia,loc] = ismember(testStr,trainCats);
        loc(~lia)=NaN;
        XSubmit_processed.(cName)=double(loc);
    end
    sVars = XSubmit_processed.Properties.VariableNames;
    for i=1:numel(sVars)
        v=sVars{i};
        if ~isnumeric(XSubmit_processed.(v))
            tmp=str2double(string(XSubmit_processed.(v)));
            if any(~isnan(tmp))
                XSubmit_processed.(v)=tmp;
            end
        end
    end

    % Align columns
    trainCols = finalTrainVarNames;
    submitCols = XSubmit_processed.Properties.VariableNames;
    missCols = setdiff(trainCols,submitCols,'stable');
    for i=1:numel(missCols)
        XSubmit_processed.(missCols{i}) = zeros(height(XSubmit_processed),1);
    end
    extraCols = setdiff(submitCols,trainCols,'stable');
    XSubmit_processed = removevars(XSubmit_processed,extraCols);
    XSubmit_processed = XSubmit_processed(:,trainCols);
    XSubmit_processed = fillmissing(XSubmit_processed,'constant',0);

    %% Predict probabilities
    
    [finalLabels, finalScores] = predict(model, table2array(XSubmit_processed));
    diabetesProbability = finalScores(:,2);

    submissionTable = table(submit_IDs, diabetesProbability, ...
        'VariableNames', {idVarName, targetVarName});
    writetable(submissionTable,'submission.csv');

   
    figure('Name','Final Prediction Probability Distribution');
    histogram(diabetesProbability,25,'FaceColor',[0.7 0.4 0.4]);
    title('Predicted Diabetes Probabilities (Test Set)');
    xlabel('Probability of Diabetes');
    ylabel('Number of Encounters');
    grid on;

    

catch ME
    if isprop(ME,'identifier')
        warning(ME.identifier,'Error: %s',ME.message);
    else
        warning('MATLAB:RuntimeError','%s',ME.message);
    end
    if ~isempty(ME.stack)
        fprintf(1,'At %s (line %d)\n',ME.stack(1).name,ME.stack(1).line);
    end
    rethrow(ME);
end
