%
% Princeton University, COS 429, Fall 2016
%
% tinynet_predict.m
%   Given a tinynet model and some new data, predicts classification
%
% Inputs:
%   X: datapoints (one per row, should include a column of ones
%                  if the model is to have a constant)
%   params: vector of parameters 
% Output:
%   z: predicted labels (0/1)
%

function z = tinynet_predict(X, params)

    [num_pts, num_vars] = size(X);

    % Indices into params of weights for hidden nodes u and v, as well as output
    ubegin = 1;
    uend = num_vars;
    vbegin = uend + 1;
    vend = uend + num_vars;
    obegin = vend + 1;
    oend = vend + 3;

    % Forward propagation pass to evaluate the network
    u = relu(X * params(ubegin:uend));
    v = relu(X * params(vbegin:vend));
    z_hat = logistic([ones(num_pts, 1) u v] * params(obegin:oend));

    z = (z_hat > 0.5);

end

%
% The logistic "sigmoid" function
%

function val = logistic(x)
    val = 1 ./ (1 + exp(-x));
end

%
% The "RELU" function
%

function val = relu(x)
    val = max(x, 0);
end

