eval_power(y,alp,N)=prod(y[j]^alp[j] for j=1:N)


function mulpoly(n,lmon_g,supp_g,coe_g,lmon_h,supp_h,coe_h)
    lsupp=lmon_g*lmon_h
    supp=Matrix{UInt64}(undef,n,lsupp)
    t=1
    for i in 1:lmon_g, j in 1:lmon_h
        supp[:,t]=supp_g[:,i]+supp_h[:,j]
        t+=1
    end
    
    
    supp=sortslices(supp,dims=2)
    supp=unique(supp,dims=2)
    lsupp=size(supp,2)
    coe=zeros(Float64,lsupp)
    
    for i in 1:lmon_g, j in 1:lmon_h
        coe[bfind(supp,lsupp,supp_g[:,i]+supp_h[:,j],n)]+=coe_g[i]*coe_h[j]
    end
    return lsupp,supp,coe
    
end

function get_basis(n::Int64,d::Int64)
    
    lb=binomial(n+d,d)
    basis=spzeros(UInt64,n,lb)
    i=UInt64(0)
    t=UInt64(1)
    while i<d+1
        if basis[n,t]==i
           if i<d
              @inbounds t+=1
              @inbounds basis[1,t]=i+1
              @inbounds i+=1
           else 
                @inbounds i+=1
           end
        else 
            j=UInt64(1)
             while basis[j,t]==0
                   @inbounds j+=1
             end
             if j==1
                @inbounds t+=1
                @inbounds basis[:,t]=basis[:,t-1]
                @inbounds basis[1,t]=basis[1,t]-1
                @inbounds basis[2,t]=basis[2,t]+1
                else t+=1
                  @inbounds basis[:,t]=basis[:,t-1]
                  @inbounds basis[1,t]=basis[j,t]-1
                  @inbounds basis[j,t]=0
                  @inbounds basis[j+1,t]=basis[j+1,t]+1
             end
        end
    end
    return basis
end

#function bfind(A::Matrix{UInt64},l::Int64,a::Vector{UInt64},n::Int64)
function bfind(A,l,a,n)
    if l==0
        return 0
    end
    low=UInt64(1)
    high=l
    while low<=high
        @inbounds mid=Int(ceil(1/2*(low+high)))
        @inbounds order=comp(A[:,mid],a,n)
        if order==0
           return mid
        elseif order<0
           @inbounds low=mid+1
        else
           @inbounds high=mid-1
        end
    end
    return 0
end

#function comp(a::Vector{UInt64},b::Vector{UInt64},n::Int64)
function comp(a,b,n)
    i=UInt64(1)
    while i<=n
          if a[i]<b[i]
             return -1
          elseif a[i]>b[i]
             return 1
          else
             @inbounds i+=1
          end
    end
    if i==n+1
       return 0
    end
end


function SmallEig_dense(mat::Matrix{Float64},s::Int64;tol_eig=1e-3)
    try
       @fastmath @inbounds E=eigs(mat,nev = 1,which=:SR,tol=tol_eig) 
       return E[1][1],E[2][:,1]
    catch
       @fastmath @inbounds E=eigen(Symmetric(mat),1:1)
       return E.values[1],E.vectors[:,1]
    end
end

function getmat_dense(vec::Vector{Float64},sk::Int64)
    B=zeros(Float64,sk,sk)
    r=1
    @fastmath @inbounds for i in 1:sk, j in i:sk
        B[i,j]=vec[r]/sqrt(1+(j>i))
        B[j,i]= copy(B[i,j])
        r+=1
    end
    return B
end

getmatvec_dense(vec::Array{Float64,1},s::Int64)=[@fastmath @inbounds vec[i]*vec[j]*sqrt(1+(j>i)) for i in 1:s for j in i:s]
                                        
function SmallEig_block_dense(vec::Vector{Float64},sk::Int64;tol_eig=1e-3)
    if sk==1
        return vec[1], ones(Float64,1)
    else
        return SmallEig_dense(getmat_dense(vec,sk),sk,tol_eig=tol_eig)
    end

end