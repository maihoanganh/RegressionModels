
function model_volume_Handelman(N,Y,t,R,d,s;ball_cons=true,bound=Inf,delt=1,bound_coeff=Inf,Stokes_constraint=false)
    
    println("number of attributes: ",N)
    println("sample size for traint set: ",t)
    println("degree of polynomial estimation: ",d)
    println("upper bound on factor width: ",s)
    println("radius of the ball centered at the origin containing the samples: ",R)
    
    m=1
    
    k=d
    L=R
    n=N
    
    lmon_g=[1]
    supp_g=[zeros(UInt64,n,1)]
    coe_g=[ones(Float64,1)]
    
    dg=[0]

    lmon_bcons_power=Vector{Int64}(undef,k+1)
    supp_bcons_power=Vector{Matrix{UInt64}}(undef,k+1)
    coe_bcons_power=Vector{Vector{Float64}}(undef,k+1)
    
    lmon_bcons_power[1]=Int64(1)
    supp_bcons_power[1]=zeros(UInt64,n,1)
    coe_bcons_power[1]=ones(Float64,1)
    
    lmon_bcons=n+1
    supp_bcons=[spzeros(UInt64,n,1) 2*sparse(I,n,n)]
    coe_bcons=[L;-ones(Float64,n)]
    
    
    for i in 1:k
        lmon_bcons_power[i+1],supp_bcons_power[i+1],coe_bcons_power[i+1]=mulpoly(n,lmon_bcons_power[i],supp_bcons_power[i],coe_bcons_power[i],lmon_bcons,supp_bcons,coe_bcons)
    end
    
    
    
    v=get_basis(n,k)
        
    supp_U=2*v
    
    
    supp_U=sortslices(supp_U,dims=2)
    lsupp_U=size(supp_U,2)   
   
     
    sk=binomial(k+n,n)
    sk_g=Vector{Vector{Int64}}(undef,m)

    for i in 1:m
        sk_g[i]=Vector{Int64}(undef,k-dg[i]+1)
        for r in 0:k-dg[i]
            sk_g[i][r+1]=binomial(k-dg[i]-r+n,n)
        end
        supp_g[i]*=2
    end
    
    
    
   vmod=mod.(v,2)
    
    r=1
    q=1
    maxsize=0
    
    block_G=Vector{Vector{Vector{Vector{Int64}}}}(undef,m)
    len_block_G=Vector{Vector{Vector{Int64}}}(undef,m)
    for i in 1:m
        block_G[i]=Vector{Vector{Vector{Int64}}}(undef,k-dg[i]+1)
        len_block_G[i]=Vector{Vector{Int64}}(undef,k-dg[i]+1)
        for a in 0:k-dg[i]
            block_G[i][a+1]=Vector{Vector{Int64}}(undef,sk_g[i][a+1])
            len_block_G[i][a+1]=Vector{Int64}(undef,sk_g[i][a+1])
            for j in 1:sk_g[i][a+1]
                block_G[i][a+1][j]=[]
                len_block_G[i][a+1][j]=0
                r=j

                while len_block_G[i][a+1][j] <= s-1 && r <= sk_g[i][a+1]
                    #if all(el->iseven(el)==true, v[:,j]+v[:,r])#
                    if norm(vmod[:,j]-vmod[:,r],1)==0
                        append!(block_G[i][a+1][j],r)
                        len_block_G[i][a+1][j]+=1
                    end
                    r+=1
                end

                q=1
                while !issubset(block_G[i][a+1][j],block_G[i][a+1][q]) && q<=j-1
                    q+=1
                end

                if q<j
                    block_G[i][a+1][j]=[]
                    len_block_G[i][a+1][j]=0
                end
                #println(block_G[i][j])
                if maxsize<len_block_G[i][a+1][j]
                    maxsize=len_block_G[i][a+1][j]
                end
            end
        end
    end
        
        
   
    
    println("Maximal matrix size:", maxsize)
    
    
    #error()
        
    #ENV["MATLAB_ROOT"] = "/usr/local/MATLAB/R2018a/toolbox/local"
    
    model=Model(optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false))
    
    
    cons=[AffExpr(0) for i=1:lsupp_U]

    G=Vector{Vector{Vector{Union{VariableRef,Symmetric{VariableRef,Array{VariableRef,2}}}}}}(undef, m)



    for i=1:m
        G[i]=Vector{Vector{Union{VariableRef,Symmetric{VariableRef,Array{VariableRef,2}}}}}(undef, k-dg[i]+1)
        for r in 0:k-dg[i]
            G[i][r+1]=Vector{Union{VariableRef,Symmetric{VariableRef,Array{VariableRef,2}}}}(undef, sk_g[i][r+1])
            for j in 1:sk_g[i][r+1]
                if len_block_G[i][r+1][j]>=1
                    if len_block_G[i][r+1][j]==1
                        G[i][r+1][j]=@variable(model, lower_bound=0)
                        for z=1:lmon_g[i]
                            for a=1:lmon_bcons_power[r+1]
                                @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,supp_g[i][:,z]+supp_bcons_power[r+1][:,a]+2*v[:,block_G[i][r+1][j]],n)],coe_g[i][z]*coe_bcons_power[r+1][a]*G[i][r+1][j])
                            end
                        end
                    else 
                        G[i][r+1][j]=@variable(model,[1:len_block_G[i][r+1][j],1:len_block_G[i][r+1][j]],PSD)
                        for p in 1:len_block_G[i][r+1][j]
                            for q in p:len_block_G[i][r+1][j]
                                for z in 1:lmon_g[i]
                                    for a=1:lmon_bcons_power[r+1]
                                        if p==q
                                            @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,v[:,block_G[i][r+1][j][p]]+v[:,block_G[i][r+1][j][q]]+supp_g[i][:,z]+supp_bcons_power[r+1][:,a],n)],coe_g[i][z]*G[i][r+1][j][p,q]*coe_bcons_power[r+1][a])
                                        else
                                            @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,v[:,block_G[i][r+1][j][p]]+v[:,block_G[i][r+1][j][q]]+supp_g[i][:,z]+supp_bcons_power[r+1][:,a],n)],2*coe_g[i][z]*G[i][r+1][j][p,q]*coe_bcons_power[r+1][a])
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    if bound_coeff!=Inf
        @constraint(model, cons.>=-bound_coeff)
        @constraint(model, cons.<= bound_coeff)
    end
    
    
    eval_samp=[AffExpr(0) for i=1:t]
    
    for i in 1:t        
        for j in 1:lsupp_U
            add_to_expression!(eval_samp[i],eval_power(Y[i,:],supp_U[:,j]/2,N)*cons[j])
        end
    end
    
    @constraint(model, eval_samp.>=delt)
    
    alpha=zeros(UInt64,N)
    
    if Stokes_constraint
        eval_derivative_samp=[[AffExpr(0) for i=1:t] for a=1:N]
        for a=1:N
            for i in 1:t        
                for j in 1:lsupp_U
                    alpha=supp_U[:,j]
                    if alpha[a]>=1
                        alpha[a]-=1
                    end
                    add_to_expression!(eval_derivative_samp[a][i],eval_power(Y[i,:],alpha,N)*cons[j]*supp_U[a,j])
                end
            end

            @constraint(model, sum(eval_derivative_samp[a])==0)
        end
    end
    
    lebesgue_mom=zeros(Float64,lsupp_U)
    
    for j=1:lsupp_U
        #lebesgue_mom[j]=get_mom_simplex(N,supp_U[:,j]/2)
        lebesgue_mom[j]=get_mom(N,Vector{UInt64}(supp_U[:,j]/2),domain="ball",radi=R)
    end

    @objective(model, Min, cons'*lebesgue_mom)
    optimize!(model)
    
    val_coeff=value.(cons)

    int = val_coeff'*lebesgue_mom
    println("termination status :", termination_status(model))
    println("integration :",int)
    
    function eval_PDF(y)
        val=0
        for j=1:lsupp_U
            val+=eval_power(y,supp_U[:,j]/2,N)*val_coeff[j]
        end
        
        return val/int
    end
    
    return eval_PDF
end