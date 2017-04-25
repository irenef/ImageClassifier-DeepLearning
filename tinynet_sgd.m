%
% Princeton University, COS 429, Fall 2016
%
% tinynet_sgd.m
%   Trains a tiny (2 hidden-node + 1 output) network with SGD
%
% Inputs:
%   X: datapoints (one per row, should include a column of ones
%                  if the model is to have a constant)
%   z: ground-truth labels (0/1)
%   num_epochs: number of epochs to train over
% Output:
%   params: vector of parameters 
%

function params = tinynet_sgd(X, z, num_epochs)

    [num_pts, num_vars] = size(X);

    % Indices into params of weights for hidden nodes u and v, as well as output
    ubegin = 1;
    uend = num_vars;
    vbegin = uend + 1;
    vend = uend + num_vars;
    obegin = vend + 1;
    oend = vend + 3;

    % Initial (random) estimate of params.
    mean = 0;
    usigma = 1 / sqrt(num_vars / 2);
    vsigma = 1 / sqrt(num_vars / 2);
    osigma = 1 / sqrt(3 / 2);
    params = zeros(oend, 1);
    params(ubegin:uend) = normrnd(mean, usigma, num_vars, 1);
    params(vbegin:vend) = normrnd(mean, vsigma, num_vars, 1);
    params(obegin:oend) = normrnd(mean, osigma, num_vars, 1);

    % Loop over epochs
    for ep = 1:num_epochs

        % Permute the data rows
        permutation = randperm(num_pts);
        X = X(permutation, :);
        z = z(permutation);

        % Iterate over the points
        for i = 1:num_pts

            % Forward propagation pass to evaluate the network
            u = relu(X(i,:) * params(ubegin:uend));
            v = relu(X(i,:) * params(vbegin:vend));
            z_hat = logistic([1 u v] * params(obegin:oend));

            % Backward propagation pass to evaluate gradient
            gradient = zeros(oend, 1);

            
            output_w_u = params(obegin+1);
            output_w_v = params(oend);
            %{
            shared_exp = (2*(z_hat-z(i))) * (z_hat*(1-z_hat));
            value_u = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_u) .* (devRelu(u).*X(i));
            value_v = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_v) .* (devRelu(v).*X(i));
            gradient(ubegin:uend) = value_u;
            gradient(vbegin:vend) = value_v;
            %}
            
            dRelu_u = devRelu(u);
            dRelu_v = devRelu(v);
            subexp_gen = 2*(z_hat-z(i)) * (z_hat*(1-z_hat));
            subexp_u = 2*(z_hat-z(i)) * (z_hat*(1-z_hat)*output_w_u); 
            subexp_v = 2*(z_hat-z(i)) * (z_hat*(1-z_hat)*output_w_v); 
            
            gradient(1) = subexp_u * (dRelu_u*X(i,1));
            gradient(2) = subexp_u * (dRelu_u*X(i,2));
            gradient(3) = subexp_u * (dRelu_u*X(i,3));
            gradient(4) = subexp_v * (dRelu_v*X(i,1));
            gradient(5) = subexp_v * (dRelu_v*X(i,2));
            gradient(6) = subexp_v * (dRelu_v*X(i,3));
            gradient(7) = subexp_gen*v;
            gradient(8) = subexp_gen*u;
            gradient(9) = subexp_gen*1;
            
            %{
            gradient(1) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_u) * (devRelu(u)*X(i,1));
            gradient(2) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_u) * (devRelu(u)*X(i,2));
            gradient(3) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_u) * (devRelu(u)*X(i,3));
            gradient(4) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_v) * (devRelu(v)*X(i,1));
            gradient(5) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_v) * (devRelu(v)*X(i,2));
            gradient(6) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*output_w_v) * (devRelu(v)*X(i,3));
            gradient(7) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*v);
            gradient(8) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*u);
            gradient(9) = (2*(z_hat-z(i))) * (z_hat*(1-z_hat)*1);
            %}
            
            % Gradient descent step
            params = params - 1/num_epochs .* gradient(:);

        end

    end

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

%
% The "Z" function (derivative of RELU
%

function val = devRelu(x) 
    if (x > 0) 
        val = 1;
    else 
        val = 0;
    end 
end

