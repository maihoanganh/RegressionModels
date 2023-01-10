module RegressionModels

using LinearAlgebra, SparseArrays, Arpack, Plots, SCS, Convex, COSMO, Printf, DelimitedFiles, CSV, DataFrames, JuMP, MosekTools

#using SpecialFunctions

#import Contour: contours, levels, level, lines, coordinates

#using DynamicPolynomials, LightGraphs, Ipopt


# src
include("./data_Convex/data.jl")
include("./primal_dual_subgradient.jl")
include("./get_moment.jl")
include("./blackbox_opt.jl")
include("./blackbox_opt_using_Convex.jl")
include("./blackbox_opt_arb_basis.jl")
include("./funcs.jl")
include("./Christoffel_func.jl")
include("./Christoffel_func_arb_basis.jl")
include("./test.jl")
include("./DenseModelVolume_Handelman.jl")

include("./DenseModelVolume.jl")

include("./SVM.jl")
include("./Poly_regression.jl")

end


