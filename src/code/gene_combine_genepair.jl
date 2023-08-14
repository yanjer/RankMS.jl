export fgene_to_combine_genepair_kernel,
        all_gene_combine_pair, all_feature_gene_combine_pair, feature_gene_combine_pair

include("compare_genepair.jl")


"""
    All genes are intergenically combined into gene pairs

```jldoctest
julia> mat = [[1 4 2];[3 5 2];[5 6 7]]

julia> all_gene_combine_pair(mat)
  0.000007 seconds (15 allocations: 960 bytes)
3×3×3 BitArray{3}:
[:, :, 1] =
 0  0  0
 0  0  0
 0  0  0

[:, :, 2] =
 0  0  0
 0  0  0
 0  0  0

[:, :, 3] =
 0  0  0
 0  0  0
 0  0  0
```
"""
function all_gene_combine_pair(mat::AbstractMatrix # Expression profiles matrix
                              )
    r,c = size(mat)
    mat_genepair_01 = falses(r,r,c)
    @time [[mat_genepair_01[i,j,:] = broadcast(is_greater,  mat[i,:],  mat[j,:]) for j in 1:r] for i in 1:r]
    return mat_genepair_01
end

"""
    All genes are intergenically combined into gene pairs

```jldoctest
julia> mat = [[1 4 2];[3 5 2];[5 6 7]]

julia> f_l=[1,3]
2-element Vector{Int64}:
 1
 3

julia> f_n = 2
2

julia> all_feature_gene_combine_pair(mat,f_l,f_n)
  0.000004 seconds (10 allocations: 640 bytes)
2×3×3 BitArray{3}:
[:, :, 1] =
 0  0  0
 0  0  0

[:, :, 2] =
 0  0  0
 0  0  0

[:, :, 3] =
 0  1  0
 0  0  0
```
"""
function all_feature_gene_combine_pair(mat::AbstractMatrix, # Expression profiles matrix.
                               f_l::Vector, # The row in which the feature gene is located in the expression profile.
                               f_n::Int # Number of feature gene. 
                               )
    r,c = size(mat)
    mat_genepair_01 = falses(f_n,r,c)
    @time [[mat_genepair_01[i,j,:] = broadcast(is_greater,  mat[i,:],  mat[j,:]) for j in 1:r] for i in 1:f_n]
    return mat_genepair_01
end

"""
    All genes are intergenically combined into gene pairs

```jldoctest
julia> mat = [[1 4 2];[3 5 2];[5 6 7]]

julia> f_l=[1,3]
2-element Vector{Int64}:
 1
 3

julia> f_n = 2
2

julia> feature_gene_combine_pair(mat,f_l,f_n)
  0.000004 seconds (6 allocations: 384 bytes)
2×2×3 BitArray{3}:
[:, :, 1] =
 0  0
 0  0

[:, :, 2] =
 0  0
 0  0

[:, :, 3] =
 0  0
 0  0
```
"""
function feature_gene_combine_pair(mat::AbstractMatrix, # Expression profiles matrix 
                               f_l::Vector, # The row in which the feature gene is located in the expression profile.
                               f_n::Int # Number of feature gene.
                               )
    r,c = size(mat)
    mat_genepair_01 = falses(f_n,f_n,c)
    @time [[mat_genepair_01[i,j,:] = broadcast(is_greater,  mat[i,:],  mat[j,:]) for j in 1:f_n] for i in 1:f_n]
    return mat_genepair_01
end



function fgene_to_combine_genepair_kernel(
                    mat::AbstractMatrix, # Expression profiles matrix 
                    fea::Vector,
                    fn_feature::AbstractString;
                    mode_combine_genepair::AbstractString = "all_gene_combine_pair", # "all_gene_combine_pair", "all_feature_gene_combine_pair", "feature_gene_combine_pair"
                         mode_use_feature::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t'
                   )
    if mode_combine_genepair == "all_gene_combine_pair"
        mat_genepair_01 = all_gene_combine_pair(mat)
        return mat_genepair_01, fea
    else
        if mode_use_feature == "custom"
            features = read_feature(fn_feature; fg_sit = fn_feature_gene_sit, delim = fn_feature_delim)
        else
            println("DEGs方法还未写入")
            features = fea[1:5]
        end
            f_l = map(x -> ∈(x, features), fea)
            f_n = sum(f_l)
            mat_genepair_01 = ((f_n == 0) ? all_gene_combine_pair(mat) : ((mode_combine_genepair == "all_feature_gene_combine_pair") ?  all_feature_gene_combine_pair(mat, f_l, f_n) : feature_gene_combine_pair(mat, f_l, f_n)))
        (mode_combine_genepair == "all_feature_gene_combine_pair") ? fea = fea[.!f_l] : fea
        return mat_genepair_01, features
    end
end



