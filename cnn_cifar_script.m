[net, info, opts, imdb] = cnn_cifar('train', struct('numEpochs', 10), 'runName', 'baseVersion');

%batch = datasample(find(imdb.images.set==3), 100, 'Replace', false);
batch = find(imdb.images.set==3);
batch = batch(1:100);
[ims, labels] = imdb.getBatch(imdb, batch);

net.layers{end}.type=string('softmax');
res = vl_simplenn(net, ims);
net.layers{end}.type=string('softmaxloss');

scores = squeeze(gather(res(end).x));
[bestScore, best] = max(scores);

% Show BEST result 
imdb.showExample(imdb, batch(39), scores(:,39));
% Show WORST result 
imdb.showExample(imdb, batch(87), scores(:,87));
% Show AVERAGE results
imdb.showExample(imdb, batch(32), scores(:,32));
imdb.showExample(imdb, batch(22), scores(:,22));
imdb.showExample(imdb, batch(70), scores(:,70));


[net2, info2, opts2, imdb2] = cnn_cifar('train', struct('numEpochs', 15), 'runName', 'baseVersion');

[ims2, labels2] = imdb2.getBatch(imdb2, batch);

net2.layers{end}.type=string('softmax');
res2 = vl_simplenn(net2, ims2);
net2.layers{end}.type=string('softmaxloss');

scores2 = squeeze(gather(res2(end).x));
[bestScore2, best2] = max(scores2);

% Show BEST result for imdb2
imdb2.showExample(imdb2, batch(39), scores(:,39));
% Show WORST result for imdb2
imdb2.showExample(imdb2, batch(87), scores(:,87));
% Show AVERAGE results for imdb2
imdb2.showExample(imdb2, batch(32), scores(:,32));
imdb2.showExample(imdb2, batch(22), scores(:,22));
imdb2.showExample(imdb2, batch(70), scores(:,70));
