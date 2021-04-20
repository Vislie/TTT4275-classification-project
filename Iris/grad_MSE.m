function grad = grad_MSE(g_k, t_k, x_k)
    grad = ((g_k - t_k).*g_k.*(1 - g_k))*x_k.';
end