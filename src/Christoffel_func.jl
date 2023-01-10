function christoffel_func(N,Y,t,d;eps=0.0)
    
    println("****Method based on Christoffel function****")
    
    println("number of attributes: ",N)
    println("sample size for traint set: ",t)
    println("degree of polynomial estimation: ",d)
    println("pertubation parameter for moment matrix: ",eps)
    
    
    
    v=get_basis(N,2*d)
    s2d=size(v,2)
    
    sort_v=sortslices(v,dims=2)
    re_ind=Vector{UInt64}(undef,s2d)
    @fastmath @inbounds @simd for j in 1:s2d
        re_ind[bfind(sort_v,s2d,v[:,j],N)]=j
    end
    
    Order(alpha::SparseVector{UInt64})=re_ind[bfind(sort_v,s2d,alpha,N)]
    
    mom_vec=Vector{Float64}(undef,s2d)
    sd=binomial(d+N,N)
    
    for i in 1:s2d
        mom_vec[i]=0
        for j in 1:t
            mom_vec[i]+=eval_power(Y[j,:],v[:,i],N)
        end
        mom_vec[i]/=t
    end
    M=Matrix{Float64}(undef,sd,sd)
    for a in 1:sd
        for b in a:sd
            M[a,b]=mom_vec[Order(v[:,a]+v[:,b])]
            if a==b
                M[a,a]+=eps
            else
                M[b,a]=M[a,b]
            end
        end
    end
    
    println("size of moment matrix: ",sd)
    
    invM=inv(M)
    
    function Lambda(y)
        
        eval_momo=Vector{Float64}(undef,sd)
        
        for i in 1:sd
            eval_momo[i]=eval_power(y,v[:,i],N)
        end
        val=eval_momo'*invM*eval_momo
        
        return 1/val
    end
    return Lambda
end


function christoffel_func_Lebesgue(N,R,d)
    v=get_basis(N,2*d)
    s2d=size(v,2)
    
    sort_v=sortslices(v,dims=2)
    re_ind=Vector{UInt64}(undef,s2d)
    @fastmath @inbounds @simd for j in 1:s2d
        re_ind[bfind(sort_v,s2d,v[:,j],N)]=j
    end
    
    Order(alpha::SparseVector{UInt64})=re_ind[bfind(sort_v,s2d,alpha,N)]
    
    mom_vec=Vector{Float64}(undef,s2d)
    sd=binomial(d+N,N)
    
    for i in 1:s2d
        mom_vec[i]=get_mom(N,v[:,i],domain="ball",radi=R)/(R^N*pi^(N/2)/gamma(1+N/2))
    end
    
    M=Matrix{Float64}(undef,sd,sd)
    for a in 1:sd
        for b in a:sd
            M=mom_vec[Order(v[:,a]+v[:,b])]
        end
    end
    
    invM=inv(M)
    
    function Lambda_Lebesgue(y)
        
        eval_momo=Vector{Float64}(undef,sd)
        
        for i in 1:sd
            eval_momo[i]=eval_power(y,v[:,i],N)
        end
        val=eval_momo'*invM*eval_momo
        
        return 1/val
    end
    return Lambda_Lebesgue
end
    
    