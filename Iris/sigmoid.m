function g = sigmoid(g_k)
    dim = size(g_k, 1);
    g = zeros(dim, 1);
    for i = 1:dim
        g_i = 1/(1 + exp(-g_k(i,:)));
        g(i,:) = g_i;
    end
end