function test_univariate_regression(data)
    
    include(data*"/plot/univariate_regression_data.jl");

    func(a)=sin(a)+0.2*cos(5*a)#*(2*rand().-1)
    phi = Vector{Float64}(func.(X[:,1]))

    scatter(X[:,1], phi,label ="input data",color="Blue")#,aspect_ratio = 1)
    plot!(func, -r, r, label ="exact",color="Red")

    n=1
    c=Inf
    k=3

    eval_pol_approx=model_pol_regress(n,X,t,c,k,phi,additional_monomials=false)

    p_approx(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx, -1.7, 1.7, label = "k = $(k)",#=legend=:bottomright,=#title = "t = $(t)",color="Orange")

    n=1
    c=Inf
    k=8

    eval_pol_approx=model_pol_regress(n,X,t,c,k,phi,additional_monomials=false)

    p_approx2(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx2, -1.7, 1.7, label = "k = $(k)",color="Green")

end

function test_univariate_regression2(data)

    include(data*"/plot/univariate_regression2_data.jl");

    func(a)=sin(a)+0.2*cos(5*a)#*(2*rand().-1)
    # weight (kg)
    phi = Vector{Float64}(func.(X[:,1]))

    #p_exact(z)=0.5
    scatter(X[:,1], phi,label ="input data",color="Blue")#,aspect_ratio = 1)
    plot!(func, -r, r, label ="exact",color="Red")

    n=1
    c=Inf
    k=3

    eval_pol_approx=model_pol_regress(n,X,t,c,k,phi,additional_monomials=false);

    p_approx(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx, -1.7, 1.7, label = "k = $(k)",#=legend=:bottomright,=#title = "t = $(t)",color="Orange")

    n=1
    c=Inf
    k=8


    eval_pol_approx=model_pol_regress(n,X,t,c,k,phi,additional_monomials=false);

    p_approx2(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx2, -1.7, 1.7, label = "k = $(k)",color="Green")
    
end

function test_univariate_classification_pol(data)

    include(data*"/plot/univariate_classification_pol_data.jl");


    #p_exact(z)=0.5
    scatter(X[1][:,1],eta[1]*ones(Float64,t[1]),label ="Class 1")#,aspect_ratio = 1)
    for r=2:s
        scatter!(X[r][:,1], eta[r]*ones(Float64,t[r]),label ="Class $(r)")
    end
    #plot!(func, -1.5, 1.5, label ="exact")

    n=1
    c=Inf
    k=5


    eval_pol_approx=model_pol_class(n,s,X,t,c,k,eta,additional_monomials=false)

    p_approx(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx, -1.5, 1.5, label = "k = $(k)",#=legend=:bottomright,=#title = "T = $(t[1])")

    n=1
    c=Inf
    k=10

    eval_pol_approx=model_pol_class(n,s,X,t,c,k,eta,additional_monomials=false)

    p_approx2(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx2, -1.5, 1.5, label = "k = $(k)")
end

function test_univariate_classification_pol2(data)

    include(data*"/plot/univariate_classification_pol2_data.jl");

    #p_exact(z)=0.5
    scatter(X[1][:,1],eta[1]*ones(Float64,t[1]),label ="Class 1")#,aspect_ratio = 1)
    for r=2:s
        scatter!(X[r][:,1], eta[r]*ones(Float64,t[r]),label ="Class $(r)")
    end
    #plot!(func, -1.5, 1.5, label ="exact")


    n=1
    c=Inf
    k=5


    eval_pol_approx=model_pol_class(n,s,X,t,c,k,eta,additional_monomials=false)

    p_approx(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx, -1.5, 1.5, label = "k = $(k)",#=legend=:bottomright,=#title = "T = $(t[1])")

    n=1
    c=Inf
    k=20


    eval_pol_approx=model_pol_class(n,s,X,t,c,k,eta,additional_monomials=false)

    p_approx2(z)=eval_pol_approx([z;zeros(n-1)])
    plot!(p_approx2, -1.5, 1.5, label = "k = $(k)")

end

function test_Breast_cancer_polynomial_regression(data,k)

    df = CSV.read(data*"/data.csv",DataFrame)

    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-2)
    for j=1:nr
        for i=3:nc-1
            D[j,i-2]=df[j,i]
        end
        if df[j,2]=="M"
            D[j,nc-2]=1
        else
            D[j,nc-2]=2
        end
    end

    max_col=[maximum(D[:,j]) for j=1:30]

    ind_zero=Vector{Int64}([])
    for j=1:30
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end

    D=D[:,setdiff(1:31,ind_zero)]

    D[:,1:30].-=0.5
    D[:,1:30]*=2

    max_norm_col=maximum(norm(D[j,1:30]) for j=1:nr)

    R=1
    D[:,1:30]/=max_norm_col/R
    

    Y=Vector{Matrix{Float64}}(undef,2)

    for r in 1:2
        Y[r]=D[findall(u -> u == r, D[:,end]),1:30]
    end
    N=30

    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        t[r]=ceil(Int64,0.8*size(Y[r],1))
        Y_train[r]=Y[r][1:t[r],:]
    end
    
    println("Number of samples: ", nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))

    Y_test=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        Y_test[r]=Y[r][(t[r]+1):end,:]
    end

    c=Inf
    s=2

    eta=[1;2]
    @time eval_pol_approx=model_pol_class(N,s,Y_train,t,c,k,eta,
                                                        additional_monomials=false,lamb=0.001);

    function classifier(y)
        return findmin([abs(eval_pol_approx(y)-eta[r]) for r=1:2])[2]
    end

    predict=Vector{Vector{Int64}}(undef,2)

    for r=1:2
        predict[r]=[classifier(Y_test[r][j,:]) for j in 1:size(Y_test[r],1)]
    end

    numcor=Vector{Int64}(undef,2)

    for r=1:2
        numcor[r]=length(findall(u -> u == r, predict[r]))
    end

    sum(numcor)

    accuracy=(sum(numcor))/(sum(size(Y_test[r],1) for r=1:2))
    
    println("Accuracy: ",accuracy)
    
end

function test_house_regression(data,k)

    df = CSV.read(data*"/Boston.csv",DataFrame)

    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-1)
    for j=1:nr
        for i=2:nc
            D[j,i-1]=df[j,i]
        end
    end

    max_col=[maximum(D[:,j]) for j=1:13]

    ind_zero=Vector{Int64}([])
    for j=1:13
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end

    D=D[:,setdiff(1:14,ind_zero)]

    X=D[:,1:13]
    eta=D[:,14]
    n=13

    t=ceil(Int64,0.8*nr)
    X_train=X[1:t,:]
    eta_train=eta[1:t]
    
    println("Number of samples: ", nr)
    println("Cardinality of train set: ",t)
    println("Cardinality of test set: ",nr-t)

    X_test=X[t+1:end,:]
    eta_test=eta[t+1:end]

    c=Inf

    @time eval_pol_approx=model_pol_regress(n,X_train,t,c,k,eta_train,
                                                        additional_monomials=false,lamb=0.005)

    predict=[eval_pol_approx(X_test[j,:]) for j in 1:length(eta_test)]

    error=norm(predict-eta_test)/maximum([norm(predict);norm(eta_test)])
    
    println("Error: ", error)

end

function test()
    r=1.7
    nr=100
    X = zeros(Float64,nr,1)
    for j in 1:nr
        X[j,1] = 2r*rand()-r
    end

    phi(a)=sin(a)+0.2*cos(5*a)
    eta = Vector{Float64}(phi.(X[:,1]))

    t=Int64(0.8*nr)
    X_train=X[1:t,:]
    eta_train=eta[1:t]

    X_test=X[t+1:end,:]
    eta_test=eta[t+1:end]

    println("Number of samples: ", nr)
    println("Cardinality of train set: ",t)
    println("Cardinality of test set: ",nr-t)


    n=1
    c=Inf
    k=3

    eval_pol_approx=model_pol_regress(n,X_train,t,c,k,eta_train,additional_monomials=false)

    predict=[eval_pol_approx(X_test[j,:]) for j in 1:length(eta_test)]

    error=norm(predict-eta_test)/maximum([norm(predict);norm(eta_test)])

    println("Error: ", error)
end