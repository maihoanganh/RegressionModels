function init(N,R,d;ball_cons=true)
    if ball_cons
        eta=1
    else
        eta=0
    end
    
    lmon_g=Vector{UInt64}(undef,eta)
    coe_g=Vector{Vector{Float64}}(undef,eta)
    supp_g=Vector{SparseMatrixCSC{UInt64}}(undef,eta)
    dg=Vector{UInt64}(undef,eta)
    
    if eta>0
        lmon_g[1]=N+1
        coe_g[1]=[R;-ones(Float64,N)]
        supp_g[1]=[spzeros(UInt64,N) 2*SparseMatrixCSC{UInt64}(I, N, N)]
        dg[1]=2
    end
    
    sd=binomial(d+N,N)
    sd_g=Vector{Int64}(undef,eta)
    
    ceil_g=Int64(0)
    for i in 1:eta
        ceil_g=ceil(Int64,dg[i]/2)
        sd_g[i]=binomial(d-ceil_g+N,N)
    end
    return eta,lmon_g,coe_g,supp_g,sd,sd_g
end


function model_volume(N,Y,t,R,d;ball_cons=true,bound=Inf,delt=1,bound_coeff=Inf,Stokes_constraint=false)
    println("****Method based on Volume Computation****")
    println()
    
    println("number of attributes: ",N)
    println("sample size for traint set: ",t)
    println("degree of polynomial estimation: ",d)
    println("radius of the ball centered at the origin containing the samples: ",R)
    
    eta,lmon_g,coe_g,supp_g,sd,sd_g=init(N,R,d,ball_cons=ball_cons)
    
    println("maximal size of matrix variables: ",sd)
        
    v=get_basis(N,2*d)
    s2d=size(v,2)
    
    sort_v=sortslices(v,dims=2)
    re_ind=Vector{UInt64}(undef,s2d)
    @fastmath @inbounds @simd for j in 1:s2d
        re_ind[bfind(sort_v,s2d,v[:,j],N)]=j
    end
    
    Order(alpha::SparseVector{UInt64})=re_ind[bfind(sort_v,s2d,alpha,N)]
                
    model=Model(optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false))
    coeff=[AffExpr(0) for i=1:s2d]


    if sd==1
        G0=@variable(model, lower_bound=0)
        add_to_expression!(coeff[Order(v[:,1]+v[:,1])],G0)
        if bound!=Inf
            @constraint(model, G0 <= bound)
        end
    else
        G0=@variable(model, [1:sd, 1:sd],PSD)
        if bound!=Inf
            @constraint(model, bound*Matrix(I,sd,sd)-G0 in PSDCone())
        end
        for a=1:sd
            for b=a:sd
                if a==b
                    add_to_expression!(coeff[Order(v[:,a]+v[:,b])],G0[a,b])
                else
                    add_to_expression!(coeff[Order(v[:,a]+v[:,b])],2*G0[a,b])
                end
            end
        end
    end

    G=Vector{Union{VariableRef,Symmetric{VariableRef}}}(undef, eta)

    for i=1:eta
        if sd_g[i]==1
            G[i]=@variable(model, lower_bound=0)
            if bound!=Inf
                @constraint(model, G[i] <= bound)
            end
            for j=1:lmon_g[i]
                add_to_expression!(coeff[Order(v[:,1]+v[:,1]+supp_g[i][:,j])],G[i]*coe_g[i][j])
            end
        else
            G[i]=@variable(model, [1:sd_g[i],1:sd_g[i]],PSD)
            if bound!=Inf
                @constraint(model, bound*Matrix(I,sd_g[i],sd_g[i])-G[i] in PSDCone())
            end
            for a=1:sd_g[i]
                for b=a:sd_g[i]
                    for j=1:lmon_g[i]
                        if a==b
                            add_to_expression!(coeff[Order(v[:,a]+v[:,b]+supp_g[i][:,j])],G[i][a,b]*coe_g[i][j])
                        else
                            add_to_expression!(coeff[Order(v[:,a]+v[:,b]+supp_g[i][:,j])],2*G[i][a,b]*coe_g[i][j])
                        end
                    end
                end
            end
        end
    end
    
    if bound_coeff!=Inf
        @constraint(model, coeff.>=-bound_coeff)
        @constraint(model, coeff.<= bound_coeff)
    end
    
    
    eval_samp=[AffExpr(0) for i=1:t]
    
    for i in 1:t        
        for j in 1:s2d
            add_to_expression!(eval_samp[i],eval_power(Y[i,:],v[:,j],N)*coeff[j])
        end
    end
    
    @constraint(model, eval_samp.>=delt)
    
    alpha=zeros(UInt64,N)
    
    if Stokes_constraint
        eval_derivative_samp=[[AffExpr(0) for i=1:t] for a=1:N]
        for a=1:N
            for i in 1:t        
                for j in 1:s2d
                    alpha=v[:,j]
                    if alpha[a]>=1
                        alpha[a]-=1
                    end
                    add_to_expression!(eval_derivative_samp[a][i],eval_power(Y[i,:],alpha,N)*coeff[j]*v[a,j])
                end
            end

            @constraint(model, sum(eval_derivative_samp[a])==0)
        end
    end
    
    lebesgue_mom=zeros(Float64,s2d)
    
    for j=1:s2d
        lebesgue_mom[j]=get_mom(N,v[:,j],domain="ball",radi=R)
    end

    @objective(model, Min, coeff'*lebesgue_mom)
    optimize!(model)
    
    val_coeff=value.(coeff)

    int = val_coeff'*lebesgue_mom
    println("termination status :", termination_status(model))
    println("integration :",int)
    
    function eval_PDF(y)
        val=0
        for j=1:s2d
            val+=eval_power(y,v[:,j],N)*val_coeff[j]
        end
        
        return val/int
    end
    
    return eval_PDF
end