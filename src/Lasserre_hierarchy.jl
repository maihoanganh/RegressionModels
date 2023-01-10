function Pu_hierarchy(x,f,g,h,k,r)

    n=length(x) # Number of variables
    m=length(g) # Number of constraints
    l=length(h) # Number of constraints

    u = Vector{UInt8}(undef,m)
    w = Vector{UInt8}(undef,l)

    lf,supp_f,coe_f=info(f,x,n)
   
    A=supp_f

    supp_g=Vector{Array{UInt8,2}}(undef,m)
    coe_g=Vector{Vector{Float64}}(undef,m)
    lg=Vector{UInt8}(undef,m)

    lu0=binomial(k+n,n)
    lu=Vector{UInt16}(undef,m)
    lw=Vector{UInt16}(undef,l)

    basis_sigma0=get_basis(n,k)
    basis_sigma=Vector{Array{UInt8,2}}(undef,m)
    basis_psi=Vector{Array{UInt8,2}}(undef,l)

    
    #A=[A basis_sigma0.*2]
    
    for i=1:m
        lg[i],supp_g[i],coe_g[i]=info(g[i],x,n)
        @inbounds A=[A supp_g[i]]

        @inbounds u[i]=ceil(UInt8,maxdegree(g[i])/2)
        @inbounds lu[i]=binomial(k-u[i]+n,n)
        @inbounds basis_sigma[i]=basis_sigma0[:,1:lu[i]]
    end

    supp_U=[A zeros(n,1)]

    supp_U,lsupp_U,block_sigma0,block_sigma,block_psi,lblock_sigma0,lblock_sigma,lblock_psi,lt_block_sigma0,lt_block_sigma,lt_block_psi=get_blocks(r,n,m,l,supp_U,lu0,lu,lw,lg,lh,supp_g,supp_h,coe_g,coe_h,basis_sigma0,basis_sigma,basis_psi)

    println("block_sigma0 = ",block_sigma0)
    println("-----------------------------")
    println("block_sigma = ",block_sigma)
    println("-----------------------------")
    println("block_psi = ",block_psi)

    model=Model(with_optimizer(Mosek.Optimizer, QUIET=true))
    cons=[AffExpr(0) for i=1:lsupp_U]

    G0=Vector{Union{VariableRef,Symmetric{VariableRef}}}(undef, lblock_sigma0)
    G=Vector{Vector{Union{VariableRef,Symmetric{VariableRef}}}}(undef, m)
    H=Vector{Vector{Union{VariableRef,Symmetric{VariableRef}}}}(undef, l)

    for j=1:lblock_sigma0
        if lt_block_sigma0[j]==1
            @inbounds G0[j]=@variable(model, lower_bound=0)
            @inbounds nota=UInt8(2)*basis_sigma0[:,block_sigma0[j]]
            Locb=bfind(supp_U,lsupp_U,nota,n)
            @inbounds add_to_expression!(cons[Locb],G0[j])
        else
            @inbounds G0[j]=@variable(model, [1:lt_block_sigma0[j], 1:lt_block_sigma0[j]],PSD)
            for p=1:lt_block_sigma0[j]
                for q=p:lt_block_sigma0[j]
                    @inbounds nota=basis_sigma0[:,block_sigma0[j][p]]+basis_sigma0[:,block_sigma0[j][q]]
                    Locb=bfind(supp_U,lsupp_U,nota,n)
                    if p==q
                        @inbounds add_to_expression!(cons[Locb],G0[j][p,q])
                    else
                        @inbounds add_to_expression!(cons[Locb],2*G0[j][p,q])
                    end
                end
            end
        end
    end


    for i=1:m
        G[i]=Vector{Union{VariableRef,Symmetric{VariableRef}}}(undef, lblock_sigma[i])
        for j=1:lblock_sigma[i]
            if lt_block_sigma[i][j]==1
                G[i][j]=@variable(model, lower_bound=0)
                for z=1:lg[i]
                    @inbounds nota=supp_g[i][:,z]+UInt8(2)*basis_sigma[i][:,block_sigma[i][j]]
                    Locb=bfind(supp_U,lsupp_U,nota,n)
                    @inbounds add_to_expression!(cons[Locb],coe_g[i][z]*G[i][j])
                end
            else
                G[i][j]=@variable(model, [1:lt_block_sigma[i][j], 1:lt_block_sigma[i][j]],PSD)
                for p=1:lt_block_sigma[i][j]
                    for q=p:lt_block_sigma[i][j]
                        for z=1:lg[i]
                            @inbounds nota=basis_sigma[i][:,block_sigma[i][j][p]]+basis_sigma[i][:,block_sigma[i][j][q]]+supp_g[i][:,z]
                            Locb=bfind(supp_U,lsupp_U,nota,n)
                            if p==q
                              @inbounds add_to_expression!(cons[Locb],coe_g[i][z]*G[i][j][p,q])
                            else
                              @inbounds add_to_expression!(cons[Locb],2*coe_g[i][z]*G[i][j][p,q])
                            end
                        end
                    end
                end
            end
        end
    end

    for i=1:l
        H[i]=Vector{Union{VariableRef,Symmetric{VariableRef}}}(undef, lblock_psi[i])
        for j=1:lblock_psi[i]
            if lt_block_psi[i][j]==1
                H[i][j]=@variable(model)
                for z=1:lh[i]
                    @inbounds nota=supp_h[i][:,z]+UInt8(2)*basis_psi[i][:,block_psi[i][j]]
                    Locb=bfind(supp_U,lsupp_U,nota,n)
                    @inbounds add_to_expression!(cons[Locb],coe_h[i][z]*H[i][j])
                end
            else
                H[i][j]=@variable(model, [1:lt_block_psi[i][j], 1:lt_block_psi[i][j]],Symmetric)
                for p=1:lt_block_psi[i][j]
                    for q=p:lt_block_psi[i][j]
                        for z=1:lh[i]
                            @inbounds nota=basis_psi[i][:,block_psi[i][j][p]]+basis_psi[i][:,block_psi[i][j][q]]+supp_h[i][:,z]
                            Locb=bfind(supp_U,lsupp_U,nota,n)
                            if p==q
                              @inbounds add_to_expression!(cons[Locb],coe_h[i][z]*H[i][j][p,q])
                            else
                              @inbounds add_to_expression!(cons[Locb],2*coe_h[i][z]*H[i][j][p,q])
                            end
                        end
                    end
                end
            end
        end
    end

    bc=zeros(lsupp_U,1)
    for i=1:lf
        Locb=bfind(supp_U,lsupp_U,supp_f[:,i],n)
        bc[Locb]=coe_f[i]
    end
    @constraint(model, cons[2:end].==bc[2:end])
    @variable(model, lambda)
    @constraint(model, cons[1]+lambda==bc[1])
    @objective(model, Max, lambda)
    optimize!(model)

    opt_val = value(lambda)
    println("Termination status = ", termination_status(model))
    println("Optimal value = ",opt_val)

    Gr=zeros(lu0,lu0)
    for j=1:lblock_sigma0
        if lt_block_sigma0[j]==1
            Gr[block_sigma0[j][1],block_sigma0[j][1]]=value(G0[j])
        else
            Gr[block_sigma0[j],block_sigma0[j]]=value.(G0[j])
        end
    end

    sol=extract_optimizers(Gr,lu0,basis_sigma0,n,n,m,l,opt_val,f,g,h,x,1:n)

    return opt_val, sol

end