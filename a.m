% Download  http://ufldl.stanford.edu/housenumbers/train.tar.gz
% Extract digitStruct to folder
% Run block below to generate number length histogram

%imshow(rgb2gray(X(:,:,:,16)))
mydf=zeros(length(digitStruct),1);
for i = 1:length(digitStruct)
    %im = imread([digitStruct(i).name]);
    for j = 1:length(digitStruct(i).bbox)
       % [height, width] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
       % bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc = max(digitStruct(i).bbox(j).left+1,1);
       % dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        
      %  imshow(im(aa:bb, cc:dd, :));
      mydf(i)=max(size(digitStruct(i).bbox));
        %fprintf('%d\n',max(size(digitStruct(i).bbox)));
       % pause;
    end
end
%plot(mydf)
 histogram(mydf)