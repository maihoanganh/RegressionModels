function model_arb_basis(N,Y,t,R,d,r)
    
    println("number of attributes: ",N)
    println("sample size for traint set: ",t)
    println("degree of polynomial estimation: ",d+1)
    println("radius of the ball centered at the origin containing the samples: ",R)
    println("number of additional monomials: ",length(r))
        
    v=get_basis(N,d+1)
    lv=size(v,2)
    sd=binomial(d+N,N)
    v=v[:,union(1:sd,r.+sd)]
    lv=size(v,2)
    
    println("maximal size of matrix variables: ",lv)
    
    
    
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
                
    simat0=Int64(0.5*(lv+1)*lv)
    
    n=lv_plus_v+simat0
    
    n_new=n-lv_plus_v
    
    l=lv_plus_v+1
    
    Q=zeros(Float64,lv_plus_v,n_new)
    ind=1
    
    for a in 1:lv
        for b in a:lv
            Q[Order(v[:,a]+v[:,b]),ind]+=1*sqrt(1+(b>a))
            ind+=1
        end
    end
    
    
    C=zeros(Float64,lv_plus_v)
    
    for a in 1:lv_plus_v
        C[a]=get_mom(N,v_plus_v[:,a],domain="ball",radi=R)
    end
    
    l_new=1
    A_new=zeros(Float64,l_new,n_new)
    A_new[1,:]=C'*Q
    
    
    Y_power=Matrix{Float64}(undef,t,lv_plus_v)        
    
    for i in 1:t        
        for j in 1:lv_plus_v
            Y_power[i,j]=eval_power(Y[i,:],v_plus_v[:,j],N)
        end
    end
    
    Y_powerQ=Y_power*Q
    
    b_new=zeros(Float64,l_new)
    b_new[l_new]=1
    

    
    m_new=1
    
    
    
    return n_new,m_new,l_new,A_new,b_new,simat0,Y_powerQ,lv
end

function input_opt_arb_basis(N,Y,t,R,d,r;tol_eig=1e-3)
    
    n_new,m_new,l_new,A_new,b_new,simat0,Y_powerQ,lv=model_arb_basis(N,Y,t,R,d,r)
    
    
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
                    
        eigval=Vector{Float64}(undef,1)
        eigvec=Vector{Vector{Float64}}(undef,1)
        gf=zeros(Float64,n_new,1)
                    
        ind=0
        
        eigval[1],eigvec[1]=SmallEig_block_dense(x[ind+1:ind+simat0],lv,tol_eig=tol_eig)
        gf[ind+1:ind+simat0,1]=-getmatvec_dense(eigvec[1],lv)
        ind+=simat0
        
       
        vf=-eigval
        
        return vf,gf
    
    end
    
    return n_new,m_new,l_new,f0_new,f_new,A_new,b_new
end
            

function solve_opt_arb_basis(N,Y,t,R,d,r;delta=0.5,s=2,rho=0.5,numiter=1000,eps=1e-2,tol_eig=1e-3)
    
    println("****Method based on Maximum likelihood estimation with additional monomials****")
    
    n,m,l,f0,f,A,b=input_opt_arb_basis(N,Y,t,R,d,r,tol_eig=tol_eig)
    #x0=1e-5*ones(Float64,n)#2*rand(Float64,n).-1
    x0=1e-5*ones(Float64,n)
    #x0=2*rand(Float64,n).-1
    lamb0=ones(Float64,m)
    nu0=ones(Float64,l)
      
    #println(A)
    
    println()
    
    return solve_convex_program(n,m,l,f0,f,A,b,x0,lamb0,nu0,delta=delta,s=s,rho=rho,numiter=numiter,eps=eps)
    
end


function func_eval_PDF_arb_basis(x,N,d,R,r)
    
    v=get_basis(N,d+1)
    lv=size(v,2)
    sd=binomial(d+N,N)
    v=v[:,union(1:sd,r.+sd)]
    lv=size(v,2)
    
    function eval_PDF(y)
    
        val=0.0
        ind=1
        for a in 1:lv
            for b in a:lv
                val+=eval_power(y,v[:,a]+v[:,b],N)*x[ind]*sqrt(1+(b>a))
                ind+=1
            end
        end

        return val
    end
    return eval_PDF
end