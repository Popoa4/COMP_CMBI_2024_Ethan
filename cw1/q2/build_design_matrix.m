function Y = build_design_matrix(bvals, qhat)
    Y = zeros(length(bvals),7);
    for i = 1:length(bvals)
        b = bvals(i);
        q = qhat(i,:);
        Y(i,:) = [1, -b*q(1)^2, -2*b*q(1)*q(2), -2*b*q(1)*q(3),...
                 -b*q(2)^2, -2*b*q(2)*q(3), -b*q(3)^2];
    end
end