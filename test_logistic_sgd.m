%
% Princeton University, COS 429, Fall 2016
%
% test_logistic_sgd.m
%   Test logistic regression on some simple data in 'training_data.txt'
%   and 'test_data.txt'
%

function test_logistic_sgd

    % Load the training data
    training = load('training_data.txt');

    % Do the training
    num_pts = size(training, 1);
    X = [ones(num_pts,1) training(:,1:2)];
    z = training(:,3);
    num_epochs = 5;
    params = logistic_sgd(X, z, num_epochs);

    % Apply the learned model to the training data and print out statistics
    % (Notice that the training_accuracy = ... line doesn't have a
    % semicolon, hence the value is printed out.)
    predicted = logistic_predict(X, params);
    training_accuracy = sum(predicted == z) / num_pts

    % Plot "ground truth" and predictions
    set(figure(1), 'Name', 'Training ground truth');
    plot_classes(training, z);
    set(figure(2), 'Name', 'Training predicted');
    plot_classes(training, predicted);

    % Apply the learned model to the test data
    testing = load('test_data.txt');
    num_pts = size(testing, 1);
    X = [ones(num_pts,1) testing(:,1:2)];
    z = testing(:,3);
    predicted = logistic_predict(X, params);
    testing_accuracy = sum(predicted == z) / num_pts

    % Plot "ground truth" and predictions
    set(figure(3), 'Name', 'Testing ground truth');
    plot_classes(testing, z);
    set(figure(4), 'Name', 'Testing predicted');
    plot_classes(testing, predicted);

end


%
% Create a scatterplot of the given data.  It is assumed that the input data
% is 2-dimensional.
%
% Inputs:
%   data: datapoints (one per row, only first two columns used)
%   z: labels (0/1)
%

function plot_classes(data, z)

    positive = data(z > 0, 1:2);
    negative = data(z == 0, 1:2);

    scatter(positive(:,1), positive(:,2), 'red');
    hold on;  % Next scatter command will be added to the same figure
    scatter(negative(:,1), negative(:,2), 'blue');
    hold off;

end

