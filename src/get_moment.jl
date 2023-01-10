function get_mom(N,a;domain="ball",radi=1)
    if domain=="ball"
        if any(isodd,a)
            return 0
        else
            return radi^(N + sum(a))/(N+sum(a))*2*prod(gamma.((a.+1)./2))/gamma((N+sum(a))/2)
        end
#     elseif domain=="unit_ball"
#         return 2*prod(gamma((a[i]+1)/2) for i in 1:N)/gamma((sum(a)+N)/2)/(sum(a)+N)
#     elseif N==1 && R==1
#         return 1^(a[1]+1)/(a[1]+1)-(-1)^(a[1]+1)/(a[1]+1)
    end
end

function get_mom_simplex(N,a)
    
    t=sum(a)
    if t>0
        xi=ones(Float64,N)/(prod(N+j for j=1:t)^(1/t))

        return prod(xi[i]^a[i] for i=1:N)/factorial(N)
    else
        return 1/factorial(N)
    end

end