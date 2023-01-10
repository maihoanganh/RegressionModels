

function solve_opt_using_Convex(N,Y,t,R,d;ball_cons=true,feas_start=false,esp=0.0,solver="COSMO")
    
    n,m,l,A,b,simat0,simat,Y_powerQ,eta,sd,sd_g=model(N,Y,t,R,d,ball_cons=ball_cons)
    
    x = Variable(n)
    
    if feas_start
        x0=starting_point(N,Y,t,d,R,n;eps=esp)
    else
        x0=1e-5*ones(Float64,n)
    end
    
    x_new=x+x0
    
    
    data="/home/hoanganh/Desktop/math-topics/algebraic_statistics/codes/DensityEstimation/src/data_Convex"
    
    
    
    output_file = open(data*"/data.jl","w")

    write(output_file, "function SDPcons(x,eta,sd,sd_g) \n \n")
    
    #write(output_file, "sdpcons=Vector{Convex.Constraint}([]) \n \n")

    write(output_file, "G = [ ")
    ind=0
    r=1
    Indmat=zeros(Int64,sd,sd)
    for i=1:sd, j=i:sd
        Indmat[i,j]=r+ind
        Indmat[j,i]=r+ind
        r+=1
    end
    
    ind+=simat0
    
    for i=1:sd
        for j=1:sd
            write(output_file, "x[$(Indmat[i,j])]/sqrt($(1+(j!=i))) ")
        end
        if i==sd
            write(output_file, "] \n \n")
        else
            write(output_file, "; \n")
        end
    end

    

    write(output_file, "sdpcons=G in :SDP \n \n")
    
    
    
    
    for w in 1:eta
        write(output_file, "G = [ ")
        
        r=1
        Indmat=zeros(Int64,sd_g[w],sd_g[w])
        for i=1:sd_g[w], j=i:sd_g[w]
            Indmat[i,j]=r+ind
            Indmat[j,i]=r+ind
            r+=1
        end
        
        ind+=simat[w]

        for i=1:sd_g[w]
            for j=1:sd_g[w]
                write(output_file, "x[$(Indmat[i,j])]/sqrt($(1+(j!=i))) ")
            end
            if i==sd_g[w]
                write(output_file, "] \n \n")
            else
                write(output_file, "; \n")
            end
        end
        
        if sd_g[w]==1
            write(output_file, "sdpcons=[sdpcons; G[1]>=0] \n \n")
        else
            write(output_file, "sdpcons=[sdpcons; G in :SDP] \n \n")
        end
        
    end
    
    write(output_file, "return sdpcons \n \n")
    
    write(output_file, "end")
    
    #SDPcons=read(output_file)
    
    close(output_file)
    
    include(data*"/data.jl")
    
    obj=sum(log(Y_powerQ[i,:]'*x_new) for i=1:t)/t

    problem = maximize(obj, [SDPcons(x_new,eta,sd,sd_g);A*x_new == b])
 
    eps = 1e-8
    maxiter=Int64(1e5)
    
    if solver=="COSMO"
        solve!(problem, COSMO.Optimizer(verbose=true,eps_abs=eps,eps_rel=eps,max_iter=maxiter); warmstart=true)
    elseif solver=="SCS"
        solve!(problem, SCS.Optimizer; silent_solver = false)
    else
        println("no solver!")
    end

    

    problem.status 
    
    return evaluate(x_new)
end
           