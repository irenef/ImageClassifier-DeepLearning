function [imdb] = cnn_cifar_data(imdbPath)
% CNN_CIFAR_DATA   Loads CIFAR-10 data
% use:
% [imdb] = cnn_cifar_data(imdbPath)
% or
% [imdb] = cnn_cifar_data(opts)

if(isstruct(imdbPath))
  opts = imdbPath;
  imdbPath = opts.imdbPath;
end

if exist(imdbPath, 'file')
  imdb = load(imdbPath) ;
else
  imdb = getCifarImdb(opts);
  mkdir(fileparts(imdbPath));
  save(imdbPath, '-struct', 'imdb') ;
end

imdb.getBatchFunction = @getBatchFunction;
imdb.getBatch = @getSimpleNNBatch;
imdb.getOrigImage = @getExample;
imdb.showExample = @showExample;
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
function fn = getBatchFunction(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    if(~opts.disturbLabel)
      fn = @(x,y) getSimpleNNBatch(x,y) ;
    else
      fn = @(x,y) getSimpleNNBatchDisturbLabel(x,y,opts.disturbLabel) ;
    end
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function [images, labels] = getExample(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
images = uint8(bsxfun(@plus, images, imdb.meta.dataMean));

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatchDisturbLabel(imdb, batch, fracDisturb)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
dl = rand(size(labels))>(1-fracDisturb) & imdb.images.set(1,batch)==1;
labels(dl) = ceil(rand(sum(dl),1)*10);
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

function showExample(imdb, idx, rescls)

  [im, label] = getExample(imdb, idx);
  figure(11); clf; 
  subplot(1,2,1);
  image(im); axis image;
  subplot(1,2,2);
  barh(rescls); 
  hold on; 
  plot(rescls(label), label, 'rx');
  set(gca, 'YTicKLabel', imdb.meta.classes)
  
% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.meta.dataMean = dataMean;

