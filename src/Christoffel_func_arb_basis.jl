function christoffel_func_arb_basis(N,Y,t,d,r;eps=0.0)
    
    println("****Method based on Christoffel function with additional monomials****")
    
    println("number of attributes: ",N)
    println("sample size for traint set: ",t)
    println("degree of polynomial estimation: ",d+1)
    println("number of additional monomials: ",length(r))
    println("pertubation parameter for moment matrix: ",eps)
    
    
    
    
    
    v=get_basis(N,d+1)
    lv=size(v,2)
    sd=binomial(d+N,N)
    v=v[:,union(1:sd,r.+sd)]
    lv=size(v,2)
    
    println("size of moment matrix: ",lv)
    
    w=1
    v_plus_v=Matrix{UInt64}(undef,N,lv^2)
    for i in 1:lv 
        for j in 1:lv
            v_plus_v[:,w]=v[:,i]+v[:,j] 
            w+=1
        end
    end
    v_plus_v=unique(v_plus_v,dims=2)
    lv_plus_v=size(v_plus_v,2)
    v_plus_v=sortslices(v_plus_v,dims=2)
    
    
    Order(alpha::SparseVector{UInt64})=bfind(v_plus_v,lv_plus_v,alpha,N)
    
    mom_vec=Vector{Float64}(undef,lv_plus_v)
    for i in 1:lv_plus_v
        mom_vec[i]=0
        for j in 1:t
            mom_vec[i]+=eval_power(Y[j,:],v_plus_v[:,i],N)
        end
        mom_vec[i]/=t
    end
    M=Matrix{Float64}(undef,lv,lv)
    for a in 1:lv
        for b in a:lv
            M[a,b]=mom_vec[Order(v[:,a]+v[:,b])]
            if a==b
                M[a,a]+=eps
            else
                M[b,a]=M[a,b]
            end
        end
    end
    
    invM=inv(M)
    
    function Lambda(y)
        
        eval_momo=Vector{Float64}(undef,lv)
        
        for i in 1:lv
            eval_momo[i]=eval_power(y,v[:,i],N)
        end
        val=eval_momo'*invM*eval_momo
        
        return 1/val
    end
    return Lambda
end


