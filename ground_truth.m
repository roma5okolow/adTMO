files = dir("hdr");

TMOs = {@AshikhminTMO, @BanterleTMO, @BestExposureTMO, ...
        @BruceExpoBlendTMO, @ChiuTMO, @DragoTMO, ...
        @DurandTMO, @ExponentialTMO, @FerwerdaTMO, ...
        @GammaTMO, @KimKautzConsistentTMO, @KrawczykTMO, ...
        @KuangTMO, @LischinskiTMO, @LogarithmicTMO, ...
        @MertensTMO, @NormalizeTMO, @PattanaikTMO, ...
        @RamanTMO, @ReinhardDevlinTMO, @ReinhardTMO, ...
        @SchlickTMO, @TumblinTMO, @VanHaterenTMO, ...
        @WardGlobalTMO, @WardHistAdjTMO, @YPFerwerdaTMO, ...
        @YPTumblinTMO, @YPWardGlobalTMO, @tonemap};

scores = zeros(length(files) - 2, 30);

for i = 3:length(files)

    file = files(i);
    fprintf('\n');
    fprintf('\n');
    disp(file.name);
    name_no_ext = split(file.name, '.hdr');
    new_name = strcat(name_no_ext(1), '.jpg');
    %mkdir(sprintf('ground_ldr/%s', name_no_ext{1}));

    imgHDR = hdrimread(convertStringsToChars(strcat("hdr/", file.name)));
    imgHDR = imresize(imgHDR, [256, 256], 'bilinear');
    disp('read successfull');

    for j = 1:30
        disp(functions(TMOs{j}).function);

        if isequal(TMOs{j}, @AshikhminTMO)  
            imgOut = ClampImg(TMOs{j}(imgHDR), 0, 1);
            imgOut = GammaTMO(imgOut, 2.2, 0);
        %doesn't require gamma adjustment
        elseif isequal(TMOs{j}, @BestExposureTMO) || isequal(TMOs{j}, @BruceExpoBlendTMO) ...
               || isequal(TMOs{j}, @GammaTMO) || isequal(TMOs{j}, @MertensTMO) ...
               || isequal(TMOs{j}, @PattanaikTMO) || isequal(TMOs{j}, @RamanTMO) ...
               || isequal(TMOs{j}, @VanHaterenTMO) || isequal(TMOs{j}, @tonemap)
            imgOut = TMOs{j}(imgHDR);
        %drago
        elseif isequal(TMOs{j}, @DragoTMO)
            imgOut = GammaDrago(TMOs{j}(imgHDR));
        %another gamma value
        elseif isequal(TMOs{j}, @ReinhardDevlinTMO)
            imgOut = GammaTMO(TMOs{j}(imgHDR), 1.6, 0, 1);
        %default
        else
            imgOut = GammaTMO(TMOs{j}(imgHDR), 2.2, 0);
        end

        %imwrite(imgOut, sprintf('ground_ldr/%s/%s.jpg', name_no_ext{1}, functions(TMOs{j}).function));
        
        if ~isequal(TMOs{j}, @tonemap)
            imgOut = imgOut * 255;
        end
        
        scores(i - 2, j) = TMQI(imgHDR, real(imgOut));
    end

    [argvalue, argmax] = max(scores(i-2, :));
    best = TMOs{argmax};
    disp(['best TMO is ', functions(best).function]);

    if isequal(best, @AshikhminTMO)  
        imgGT = ClampImg(best(imgHDR), 0, 1);
        imgGT = GammaTMO(imgGT, 2.2, 0);
    %doesn't require gamma adjustment
    elseif isequal(best, @BestExposureTMO) || isequal(best, @BruceExpoBlendTMO) ...
           || isequal(best, @GammaTMO) || isequal(best, @MertensTMO) ...
           || isequal(best, @PattanaikTMO) || isequal(best, @RamanTMO) ...
           || isequal(best, @VanHaterenTMO) || isequal(best, @tonemap)
        imgGT = best(imgHDR);
    %drago
    elseif isequal(best, @DragoTMO)
        imgGT = GammaDrago(best(imgHDR));
    %another gamma value
    elseif isequal(best, @ReinhardDevlinTMO)
        imgGT = GammaTMO(best(imgHDR), 1.6, 0, 1);
    %default handler
    else
        imgGT = GammaTMO(best(imgHDR), 2.2, 0);
    end

    imwrite(imgGT, sprintf('ground_ldr/%s', new_name{1}));
end

writematrix(scores, 'comparison_test_full.txt');