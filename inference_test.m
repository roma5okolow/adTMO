fid = fopen('test/file_names.txt','r');

scores = zeros(105);

for i = 1:105
    file = fgetl(fid);
    fprintf('\n');
    fprintf('\n');
    disp(file);
    gen_name = strcat(file, '_gen.jpg');
    hdr_name = strcat(file, '.hdr');
    %mkdir(sprintf('ground_ldr/%s', name_no_ext{1}));

    imgHDR = hdrimread(convertStringsToChars(strcat("hdr/", hdr_name)));
    imgHDR = imresize(imgHDR, [256, 256], 'bilinear');
    imgGen = imread(convertStringsToChars(strcat("test/", gen_name)));
    disp('read successfull');

    
        
    scores(i) = TMQI(imgHDR, imgGen);
    disp(scores(i));
end

writematrix(scores, 'inference_test.txt');