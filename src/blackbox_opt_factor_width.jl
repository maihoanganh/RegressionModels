
function RelaxDense(n::Int64,m::Int64,l::Int64,lmon_g::Vector{UInt64},supp_g::Vector{Matrix{UInt64}},coe_g::Vector{Vector{Float64}},lmon_h::Vector{UInt64},supp_h::Vector{Matrix{UInt64}},coe_h::Vector{Vector{Float64}},lmon_f::Int64,supp_f::Matrix{UInt64},coe_f::Vector{Float64},dg::Vector{Int64},dh::Vector{Int64},k::Int64,s::Int64;solver="Mosek",comp_opt_sol=false)
    
    println("**Interrupted relaxation based on Putinar-Vasilescu's Positivstellensatz**")
    println("Relaxation order: k=",k)
    println("Sparsity order: s=",s)
    
    m+=1
    
    lmon_g=[lmon_g;1]
    supp_g=[supp_g;[zeros(UInt64,n,1)]]
    coe_g=[coe_g;[ones(Float64,1)]]
    
    dg=[dg;0]
    
    df=Int64(maximum([sum(supp_f[:,i]) for i in 1:lmon_f]))#+1
    
    supp_f*=2
    
  
    
    lmon_thetak=Int64(1)
    supp_thetak=zeros(UInt64,n,1)
    coe_thetak=ones(Float64,1)
    
    supp_theta=2*[spzeros(UInt64,n,1) sparse(I,n,n)]
    coe_theta=ones(Float64,n+1)
    
    
    for i in 1:k
        lmon_thetak,supp_thetak,coe_thetak=mulpoly(n,lmon_thetak,supp_thetak,coe_thetak,n+1,supp_theta,coe_theta)
    end
    
    
    
    lmon_thetakf,supp_thetakf,coe_thetakf=mulpoly(n,lmon_thetak,supp_thetak,coe_thetak,lmon_f,supp_f,coe_f)
    
    
    sk=binomial(k+df+n,n)
    sk_g=Vector{UInt64}(undef,m)
    sk_h=Vector{UInt64}(undef,l)
    
    
    
    v=get_basis(n,k+df)
        
    supp_U=2*v
    
    
    supp_U=sortslices(supp_U,dims=2)
    lsupp_U=size(supp_U,2)   
   
     
    


    @fastmath @inbounds @simd for i in 1:m
        sk_g[i]=binomial(k+df-dg[i]+n,n)
        supp_g[i]*=2
    end
    
    @fastmath @inbounds @simd for i in 1:l
        sk_h[i]=binomial(k+df-dh[i]+n,n)
        supp_h[i]*=2
    end
    
    
   vmod=mod.(v,2)
    
    r=1
    q=1
    maxsize=0
    
    block_G=Vector{Vector{Vector{Int64}}}(undef,m)
    len_block_G=Vector{Vector{Int64}}(undef,m)
    for i in 1:m
        block_G[i]=Vector{Vector{Int64}}(undef,sk_g[i])
        len_block_G[i]=Vector{Int64}(undef,sk_g[i])
        for j in 1:sk_g[i]
            block_G[i][j]=[]
            len_block_G[i][j]=0
            r=j
            
            while len_block_G[i][j] <= s-1 && r <= sk_g[i]
                #if all(el->iseven(el)==true, v[:,j]+v[:,r])#
                if norm(vmod[:,j]-vmod[:,r],1)==0
                    append!(block_G[i][j],r)
                    len_block_G[i][j]+=1
                end
                r+=1
            end
           
            q=1
            while !issubset(block_G[i][j],block_G[i][q]) && q<=j-1
                q+=1
            end
                
            if q<j
                block_G[i][j]=[]
                len_block_G[i][j]=0
            end
            #println(block_G[i][j])
            if maxsize<len_block_G[i][j]
                maxsize=len_block_G[i][j]
            end
        end
    end
        
        
   
    
    println("Maximal matrix size:", maxsize)
    
    
    #error()
        
    #ENV["MATLAB_ROOT"] = "/usr/local/MATLAB/R2018a/toolbox/local"
    
    if solver=="Mosek"
        model=Model(optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false))
    elseif solver=="SDPT3"
        model=Model(SDPT3.Optimizer)
    elseif solver=="SDPNAL"
        model=Model(SDPNAL.Optimizer)
    elseif solver=="COSMO"
        model=Model(COSMO.Optimizer)
    else
        error("No SDP solver!!!")
    end
    
    
    cons=[AffExpr(0) for i=1:lsupp_U]

    G=Vector{Vector{Union{VariableRef,Symmetric{VariableRef,Array{VariableRef,2}}}}}(undef, m)
    H=Vector{Vector{VariableRef}}(undef, l)



    for i=1:m
        G[i]=Vector{Union{VariableRef,Symmetric{VariableRef,Array{VariableRef,2}}}}(undef, sk_g[i])
        for j in 1:sk_g[i]
            
            
            if len_block_G[i][j]>=1
                if len_block_G[i][j]==1
                    G[i][j]=@variable(model, lower_bound=0)
                    for z=1:lmon_g[i]
                        @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,supp_g[i][:,z]+2*v[:,block_G[i][j]],n)],coe_g[i][z]*G[i][j])
                    end
                else 
                    G[i][j]=@variable(model,[1:len_block_G[i][j],1:len_block_G[i][j]],PSD)
                    for p in 1:len_block_G[i][j]
                        for q in p:len_block_G[i][j]
                            for z in 1:lmon_g[i]
                                if p==q
                                    @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,v[:,block_G[i][j][p]]+v[:,block_G[i][j][q]]+supp_g[i][:,z],n)],coe_g[i][z]*G[i][j][p,q])
                                else
                                    @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,v[:,block_G[i][j][p]]+v[:,block_G[i][j][q]]+supp_g[i][:,z],n)],2*coe_g[i][z]*G[i][j][p,q])
                                end
                            end
                        end
                    end
                end
            end
        end
    end
   
    

    for i in 1:l
        H[i]=@variable(model, [1:sk_h[i]])
        for p in 1:sk_h[i]
            for z in 1:lmon_h[i]
                  @inbounds add_to_expression!(cons[bfind(supp_U,lsupp_U,2*v[:,p]+supp_h[i][:,z],n)],coe_h[i][z]*H[i][p])
            end
        end
    end


    for i in 1:lmon_thetakf
        cons[bfind(supp_U,lsupp_U,supp_thetakf[:,i],n)]-=coe_thetakf[i]
    end
    
    @variable(model, lambda)

    for i in 1:lmon_thetak
        cons[bfind(supp_U,lsupp_U,supp_thetak[:,i],n)]+=coe_thetak[i]*lambda
    end
    
    @constraint(model, cons.==0)
    @objective(model, Max, lambda)
    optimize!(model)

    opt_val = value(lambda)
    println("Termination status = ", termination_status(model))
    println("Primal status = ", primal_status(model))
    println("Optimal value = ",opt_val)
    
  
    
    if comp_opt_sol
    
        Gr=zeros(Float64,sk_g[m],sk_g[m])

        for j in 1:sk_g[m]
            if len_block_G[m][j]>1
                Gr[block_G[m][j],block_G[m][j]]+=value.(G[m][j])
            elseif len_block_G[m][j]==1
                Gr[block_G[m][j],block_G[m][j]]+=[value.(G[m][j])]
            end

        end

  
        opt_sol=extract_optimizer(Gr,Int64(sk_g[m]),v[:,1:sk_g[m]],opt_val,n,m,l,lmon_g,supp_g,coe_g,lmon_h,supp_h,coe_h,lmon_f,supp_f,coe_f)
    
    else
        opt_sol=Vector{Float64}([])
    end
    
    return opt_val,opt_sol

end        
            
         







function init_factor_width(N,R,d;ball_cons=true)
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
        supp_g[1]=[spzeros(UInt64,N) SparseMatrixCSC{UInt64}(I, N, N)]
        dg[1]=1
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
    
function model_factor_width(N,Y,t,R,d;ball_cons=true)
    
    eta,lmon_g,coe_g,supp_g,sd,sd_g=init(N,R,d,ball_cons=ball_cons)
        
    m+=1
    
    lmon_g=[lmon_g;1]
    supp_g=[supp_g;[zeros(UInt64,n,1)]]
    coe_g=[coe_g;[ones(Float64,1)]]
    
    dg=[dg;0]
    
    df=Int64(maximum([sum(supp_f[:,i]) for i in 1:lmon_f]))#+1
    
    supp_f*=2
    
  
    
    lmon_thetak=Int64(1)
    supp_thetak=zeros(UInt64,n,1)
    coe_thetak=ones(Float64,1)
    
    supp_theta=2*[spzeros(UInt64,n,1) sparse(I,n,n)]
    coe_theta=ones(Float64,n+1)
    
    
    for i in 1:k
        lmon_thetak,supp_thetak,coe_thetak=mulpoly(n,lmon_thetak,supp_thetak,coe_thetak,n+1,supp_theta,coe_theta)
    end
    
    
    
    lmon_thetakf,supp_thetakf,coe_thetakf=mulpoly(n,lmon_thetak,supp_thetak,coe_thetak,lmon_f,supp_f,coe_f)
    
    
    sk=binomial(k+df+n,n)
    sk_g=Vector{UInt64}(undef,m)
    sk_h=Vector{UInt64}(undef,l)
    
    
    
    v=get_basis(n,k+df)
        
    supp_U=2*v
    
    
    supp_U=sortslices(supp_U,dims=2)
    lsupp_U=size(supp_U,2)   
   
     
    


    @fastmath @inbounds @simd for i in 1:m
        sk_g[i]=binomial(k+df-dg[i]+n,n)
        supp_g[i]*=2
    end
    
    @fastmath @inbounds @simd for i in 1:l
        sk_h[i]=binomial(k+df-dh[i]+n,n)
        supp_h[i]*=2
    end
    
    
   vmod=mod.(v,2)
    
    r=1
    q=1
    maxsize=0
    
    block_G=Vector{Vector{Vector{Int64}}}(undef,m)
    len_block_G=Vector{Vector{Int64}}(undef,m)
    for i in 1:m
        block_G[i]=Vector{Vector{Int64}}(undef,sk_g[i])
        len_block_G[i]=Vector{Int64}(undef,sk_g[i])
        for j in 1:sk_g[i]
            block_G[i][j]=[]
            len_block_G[i][j]=0
            r=j
            
            while len_block_G[i][j] <= s-1 && r <= sk_g[i]
                #if all(el->iseven(el)==true, v[:,j]+v[:,r])#
                if norm(vmod[:,j]-vmod[:,r],1)==0
                    append!(block_G[i][j],r)
                    len_block_G[i][j]+=1
                end
                r+=1
            end
           
            q=1
            while !issubset(block_G[i][j],block_G[i][q]) && q<=j-1
                q+=1
            end
                
            if q<j
                block_G[i][j]=[]
                len_block_G[i][j]=0
            end
            #println(block_G[i][j])
            if maxsize<len_block_G[i][j]
                maxsize=len_block_G[i][j]
            end
        end
    end
        
        
   
    
    println("Maximal matrix size:", maxsize)
    
    
    
    
    
    
                
    simat0=Int64(0.5*(sd+1)*sd)
    simat=[Int64(0.5*(sd_g[i]+1)*sd_g[i]) for i in 1:eta]
    
    n=s2d+simat0+sum(simat)
    
    n_new=n-s2d
    
    l=s2d+1
    
    Q=zeros(Float64,s2d,n_new)
    ind=1
    
    for a in 1:sd
        for b in a:sd
            Q[Order(v[:,a]+v[:,b]),ind]+=1*sqrt(1+(b>a))
            ind+=1
        end
    end
    
    for i in 1:eta
        for a in 1:sd_g[i]
            for b in a:sd_g[i]
                for j in 1:lmon_g[i]
                    Q[Order(v[:,a]+v[:,b]+supp_g[i][:,j]),ind]+=coe_g[i][j]*sqrt(1+(b>a))
                end
                ind+=1
            end
        end   
    end
    
    C=zeros(Float64,s2d)
    
    for a in 1:s2d
        C[Order(v[:,a])]=get_mom(N,v[:,a],domain="ball",radi=R)
    end
    
    l_new=1
    A_new=zeros(Float64,l_new,n_new)
    A_new[1,:]=C'*Q
    
    
    Y_power=Matrix{Float64}(undef,t,s2d)        
    
    for i in 1:t        
        for j in 1:s2d
            Y_power[i,j]=eval_power(Y[i,:],v[:,j],N)
        end
    end
    
    Y_powerQ=Y_power*Q
    
    b_new=zeros(Float64,l_new)
    b_new[l_new]=1
    

    
    m_new=eta+1
    
    
    
    return n_new,m_new,l_new,A_new,b_new,simat0,simat,Y_powerQ,eta,sd,sd_g
end

function input_opt(N,Y,t,R,d;tol_eig=1e-3,ball_cons=true)
    
    n_new,m_new,l_new,A_new,b_new,simat0,simat,Y_powerQ,eta,sd,sd_g=model(N,Y,t,R,d,ball_cons=ball_cons)
    
    
    function f0_new(x)
        vf0=0.0
        gf0=zeros(Float64,n_new)
        val_q=0.0
        grad_q=zeros(Float64,n_new)
        for i in 1:t
            val_q=Y_powerQ[i,:]'*x   
            vf0-=log(abs(val_q))
            gf0-=Y_powerQ[i,:]/val_q

        end
        vf0/=t
        gf0/=t

        return vf0,gf0
    end
    
    
    function f_new(x)
                    
        eigval=Vector{Float64}(undef,eta+1)
        eigvec=Vector{Vector{Float64}}(undef,eta+1)
        gf=zeros(Float64,n_new,eta+1)
                    
        ind=0
        
        eigval[1],eigvec[1]=SmallEig_block_dense(x[ind+1:ind+simat0],sd,tol_eig=tol_eig)
        gf[ind+1:ind+simat0,1]=-getmatvec_dense(eigvec[1],sd)
        ind+=simat0
        for i in 1:eta
            eigval[i+1],eigvec[i+1]=SmallEig_block_dense(x[ind+1:ind+simat[i]],sd_g[i],tol_eig=tol_eig)
            gf[ind+1:ind+simat[i],i+1]=-getmatvec_dense(eigvec[i+1],sd_g[i])
            ind+=simat[i]
        end
        vf=-eigval
        
        return vf,gf
    
    end
    
    return n_new,m_new,l_new,f0_new,f_new,A_new,b_new
end
            

function solve_opt(N,Y,t,R,d;delta=0.5,s=2,rho=0.5,numiter=1000,eps=1e-2,tol_eig=1e-3,ball_cons=true)
    
    n,m,l,f0,f,A,b=input_opt(N,Y,t,R,d,tol_eig=tol_eig,ball_cons=ball_cons)
    x0=1e-5*ones(Float64,n)#2*rand(Float64,n).-1
    #x0=zeros(Float64,n)
    #x0[1]=1/(R^N*pi^(N/2)/gamma(1+N/2))
    lamb0=ones(Float64,m)
    nu0=ones(Float64,l)
       
    
    return solve_convex_program(n,m,l,f0,f,A,b,x0,lamb0,nu0,delta=delta,s=s,rho=rho,numiter=numiter,eps=eps)
    
end


function func_eval_PDF(x,N,d,R,;ball_cons=true)
    eta,lmon_g,coe_g,supp_g,sd,sd_g=init(N,R,d,ball_cons=ball_cons)
    
    v=get_basis(N,d)
    
    function eval_PDF(y)
    
        val=0.0
        ind=1
        for a in 1:sd
            for b in a:sd
                val+=eval_power(y,v[:,a]+v[:,b],N)*x[ind]*sqrt(1+(b>a))
                ind+=1
            end
        end

        for i in 1:eta
            for a in 1:sd_g[i]
                for b in a:sd_g[i]
                    for j in 1:lmon_g[i]
                        val+=eval_power(y,v[:,a]+v[:,b]+supp_g[i][:,j],N)*coe_g[i][j]*x[ind]*sqrt(1+(b>a))
                    end
                    ind+=1
                end
            end   
        end
        return val
    end
    return eval_PDF
end