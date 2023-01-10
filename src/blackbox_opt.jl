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
    
function model(N,Y,t,R,d;ball_cons=true)
    
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
            

function solve_opt(N,Y,t,R,d;delta=0.5,s=2,rho=0.5,numiter=1000,eps=1e-2,tol_eig=1e-3,ball_cons=true,feas_start=false)
    
    println("****Method based on Maximum likelihood estimation****")
    
    n,m,l,f0,f,A,b=input_opt(N,Y,t,R,d,tol_eig=tol_eig,ball_cons=ball_cons)
    #x0=1e-5*ones(Float64,n)#2*rand(Float64,n).-1
    if feas_start
        #x0=zeros(Float64,n)
        #x0[1]=1/(R^N*pi^(N/2)/gamma(1+N/2))
        #x0[2:end]=1e-5*ones(Float64,n-1)
        x0=starting_point(N,Y,t,d,R,n;eps=1e-2)
        #x0*=1e-2
        #x0-=ones(Float64,n)
    else
        x0=1e-5*ones(Float64,n)
        #x0=2*rand(Float64,n).-1
    end
    lamb0=ones(Float64,m)
    nu0=ones(Float64,l)
      
    #println(A)
    println()
    
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


function starting_point(N,Y,t,d,R,n;eps=0.0)
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
    
    invM=inv(M)
    
    
    mom_ball=[get_mom(N,v[:,a],domain="ball",radi=R) for a=1:s2d]
    
    
    int_ball=0.0
    
    for a in 1:sd
        for b in a:sd
            if a==b
                int_ball+=invM[a,a]*mom_ball[Order(v[:,a]+v[:,b])]
            else
                int_ball+=2*invM[a,a]*mom_ball[Order(v[:,a]+v[:,b])]
            end
        end
    end
    
    x0=zeros(Float64,n)
    
    ind=1
    for a in 1:sd
        for b in a:sd
            x0[ind]=invM[a,b]/int_ball
            ind+=1
        end
    end
    
    return x0
end