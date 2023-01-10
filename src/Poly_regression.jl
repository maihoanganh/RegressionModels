function model_pol_class(n,s,X,t,c,k,eta;additional_monomials=false,lamb=0.0)
    println("****Method based on polynomial regression****")
    println()
    
    println("number of attributes: n=",n)
    println("sample sizes for traint sets: t=",t)
    println("degree of separating polynomial: k=",k)
    println("perturbed parameter: c=",c)
    
    v=get_basis(n,k)
    sk=size(v,2)
    if additional_monomials
        v=[v 2*v[:,2:end] v[:,2:end-1]+v[:,3:end]]
        sk=size(v,2)
    end
                
    model=Model(optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false))
    
    if c==Inf
	q=@variable(model, [1:sk])
    else
        q=@variable(model, [1:sk],lower_bound=-c,upper_bound=c)
    end
    
    eval_q=Vector{Vector{JuMP.AffExpr}}(undef,s)
    
    for r=1:s
        eval_q[r]=[AffExpr(0) for i=1:t[r]]

        for i in 1:t[r]
            for j in 1:sk
                add_to_expression!(eval_q[r][i],eval_power(X[r][i,:],v[:,j],n)*q[j])
            end
        end
    end
            
      
    obj=sum(sum((eval_q[r].-eta[r]).^2)/t[r] for r=1:s)+lamb*sum(q.^2)
    
    @objective(model, Min, obj)
    optimize!(model)
    
    val_coeff=value.(q)
            
    
    function eval_pol_approx(y)
        val=0
        for j=1:sk
            val+=eval_power(y,v[:,j],n)*val_coeff[j]
        end
        return val
    end
    
    return eval_pol_approx
end



function model_pol_regress(n,X,t,c,k,phi;additional_monomials=false,lamb=0.0)
    println("****Method based on polynomial regression****")
    println()
    
    println("number of attributes: n=",n)
    println("sample sizes for traint sets: t=",t)
    println("degree of separating polynomial: k=",k)
    println("perturbed parameter: c=",c)
    
    v=get_basis(n,k)
    sk=size(v,2)
    if additional_monomials
        v=[v 2*v[:,2:end] v[:,2:end-1]+v[:,3:end]]
        sk=size(v,2)
    end
                
    model=Model(optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false))
    
    
    if c==Inf
	q=@variable(model, [1:sk])
    else
        q=@variable(model, [1:sk],lower_bound=-c,upper_bound=c)
    end
    
    eval_q=Vector{JuMP.AffExpr}(undef,t)
    
    for i in 1:t
        eval_q[i]=-phi[i]
        for j in 1:sk
            add_to_expression!(eval_q[i],eval_power(X[i,:],v[:,j],n)*q[j])
        end
    end
            
      
    obj=sum(eval_q.^2)/t+lamb*sum(q.^2)
    
    @objective(model, Min, obj)
    optimize!(model)
    
    val_coeff=value.(q)
            
    
    function eval_pol_approx(y)
        val=0
        for j=1:sk
            val+=eval_power(y,v[:,j],n)*val_coeff[j]
        end
        return val
    end
    
    return eval_pol_approx
end
