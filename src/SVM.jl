function model_SVM(n,X,t,c,k;lamb=0.5,additional_monomials=false)
    println("****Method based on Support Vector Machine****")
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
    
    
    q=@variable(model, [1:sk])
    
    eval_q=Vector{Vector{JuMP.AffExpr}}(undef,2)
    
    zeta=Vector{Vector{JuMP.VariableRef}}(undef,2)
    
    obj=lamb*sum(q.^2)
    
    for r=1:2
        zeta[r]=@variable(model, [1:t[r]],lower_bound=0)
        eval_q[r]=[AffExpr(0) for i=1:t[r]]

        for i in 1:t[r]
            for j in 1:sk
                add_to_expression!(eval_q[r][i],eval_power(X[r][i,:],v[:,j],n)*q[j])
            end
        end
        @constraint(model, (-1)^(r-1)*eval_q[r]+zeta[r].>=1+1/c)
    end
            
      
    obj+=sum(sum(zeta[r]) for r=1:2)/sum(t)
    
    @objective(model, Min, obj)
    optimize!(model)
    
    val_coeff=value.(q)
            
    
    function eval_sep_pol(y)
        val=0
        for j=1:sk
            val+=eval_power(y,v[:,j],n)*val_coeff[j]
        end
        
        return val
    end
    
    return eval_sep_pol
end