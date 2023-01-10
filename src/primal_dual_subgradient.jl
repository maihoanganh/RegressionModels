function F(x,f,n,m)
    vf,gf=f(x)
    vF=zeros(Float64,m)
    gF=zeros(Float64,n,m)
    for i in 1:m
        if vf[i]>0
            vF[i]=vf[i]
            gF[:,i]=gf[:,i]
        end
    end
    return vF,gF
end

function param(y,norm_y,s,length_y)
    if norm_y != 0
        return s*norm_y^(s-2)*y
    else
        return zeros(Float64,length_y)
    end
end
    

function solve_convex_program(n,m,l,f0,f,A,b,x0,lamb0,nu0;delta=0.5,s=2,rho=0.5,numiter=1000,eps=1e-2)
    x=x0
    lamb=lamb0
    nu=nu0
    
    vf0,gf0=f0(x)
    vF,gF=F(x,f,n,m)
    Axb=A*x-b

    norm_vF=norm(vF)
    norm_Axb=norm(Axb)

    varrho=param(vF,norm_vF,s,m)
    varsigma=param(Axb,norm_Axb,s,l)

    Tx=gf0+sum((lamb[i]+ rho*varrho[i])*gF[:,i] for i=1:m)+A'*(nu+rho*varsigma)
    norm_T=sqrt(sum(Tx.^2)+norm_vF^2+norm_Axb^2)

    gamma=1
    alpha=gamma/norm_T

    axpy!(-alpha,Tx,x)
    axpy!(alpha,vF,lamb)
    axpy!(alpha,Axb,nu)
    
    for k in 0:numiter       
        vf0,gf0=f0(x)
        vF,gF=F(x,f,n,m)
        Axb=A*x-b
        
        norm_vF=norm(vF)
        norm_Axb=norm(Axb)
        
        varrho=param(vF,norm_vF,s,m)
        varsigma=param(Axb,norm_Axb,s,l)
        
        Tx=gf0+sum((lamb[i]+ rho*varrho[i])*gF[:,i] for i=1:m)+A'*(nu+rho*varsigma)
        norm_T=sqrt(sum(Tx.^2)+norm_vF^2+norm_Axb^2)
        
        if k> numiter-50
            @printf("iter=%0.0f  val=%0.4f  norm_vF=%0.4f  norm_Axb=%0.4f  norm_T=%0.4f\n",k,vf0,norm_vF,norm_Axb,norm_T)
        end
        if norm_T<eps
            break
        end
        
        gamma=(k+1)^(-1+delta/2)
        alpha=gamma/norm_T
        
        axpy!(-alpha,Tx,x)
        axpy!(alpha,vF,lamb)
        axpy!(alpha,Axb,nu)
    end
    
    #=
    println("A=",A)
    println()
    println("b=",b)
    println()
    println("vf0=",vf0)
    println()
    println("gf0=",gf0)
    println()
    println("vF=",vF)
    println()
    println("gF=",gF)
    println()
    println(count(!iszero, gF))
    println(length(gF))
    =#
    
    return x
end
        
   