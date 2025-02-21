function visualize_results(results, method_name)
    figure('Name',method_name);
    subplot(2,3,1); imagesc(results.S0'); axis image; title('S0'); colorbar;
    subplot(2,3,2); imagesc(results.d'); axis image; title('d'); colorbar;
    subplot(2,3,3); imagesc(results.f'); axis image; title('f'); colorbar;
    subplot(2,3,4); imagesc(results.RESNORM'); axis image; title('RESNORM'); colorbar;
    
    [X,Y] = meshgrid(1:size(results.S0,2), 1:size(results.S0,1));
    u = results.f .* sin(results.theta) .* cos(results.phi);
    v = results.f .* sin(results.theta) .* sin(results.phi);

    subplot(2,3,5); 
    quiver(X,Y,flipud(u),flipud(v),2); 
    title('Fiber Directions'); 
    axis image;
end
