function [Y, objHistory] = update_Y_fast(A, B, Y)
%
%  min tr( Y' A Y ) + tr(Y' B)
%  s.t.   Y is ind
%

nSmp = size(Y, 1);

fAf = sum(Y .* (A * Y))'; % c * 1
ff = sum(Y .* B)'; % c * 1
ff2 = sum(Y)'; % c * 1

m_all = vec2ind(Y')'; % n * 1


objHistory = sum(fAf) + sum(ff);
iter = 5;

for z=1:iter
    for i = 1:nSmp
        m = m_all(i);
        if ff2(m) == 1
            % avoid generating empty cluster
            continue;
        end
        
        %*********************************************************************
        % The following matlab code is O(nc)
        % With the loop in n here, it is O(n) actually.
        %*********************************************************************
        Y_A = Y' * A(:, i);
        
        fAf_s = fAf + 2 * Y_A + A(i, i); % assign i to all clusters and update
        fAf_s(m) = fAf(m); % cluster m keep the same
        ff_k = ff + B(i, :)'; % all cluster + b
         ff_k(m) = ff(m); % cluster m keep the same
        ff2_k = ff2 + 1; % all cluster + 1
        ff2_k(m) = ff2(m); % cluster m keep the same
        
        fAf_0 = fAf;
        fAf_0(m) = fAf(m) - 2 * Y_A(m) + A(i, i); % remove i from m
        ff_0 = ff;
        ff_0(m) = ff(m) - B(i, m); % remove i from m
        ff2_0 = ff2;
        ff2_0(m) = ff2(m) - 1; % remove i from m
        e1 = fAf_s + ff_k;
        e0 = fAf_0 + ff_0;
        delta = e1 - e0;
        
        [~, p] = min(delta);
        if p ~= m % sample i is moved from cluster m to cluster p
            fAf([m, p]) = [fAf_0(m), fAf_s(p)];
            ff([m, p]) = [ff_0(m), ff_k(p)];
            ff2([m, p]) = [ff2_0(m), ff2_k(p)];
            Y(i, [p, m]) = [1, 0];
            m_all(i) = p;
        end
        obj = sum(fAf) + sum(ff);
        objHistory = [objHistory; obj];%#ok
    end
end

end