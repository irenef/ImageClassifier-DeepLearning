function [net, info, opts, imdb] = cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
opts.runName = '1'; % user defined name 
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(vl_rootnn, 'data', ...
  sprintf('cifar-%s-%s', opts.modelType, opts.runName)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
fprintf('Model Load/Save directory: %s \n', opts.expDir);

opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.imdbPath = fullfile(vl_rootnn, 'data', 'cifar', 'imdb.mat');
opts.whitenData = false ;
opts.contrastNormalization = false ;
opts.networkType = 'simplenn' ;
opts.disturbLabel = 0;
opts.train = struct();
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'lenet'
    net = cnn_cifar_init('networkType', opts.networkType) ;
  case 'nin'
    net = cnn_cifar_init_nin('networkType', opts.networkType) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

[imdb] = cnn_cifar_data(opts);

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, imdb.getBatchFunction(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

