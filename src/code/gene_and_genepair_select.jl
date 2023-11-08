export fgene_to_genepair_kernel,
        all_gene_pair, all_feature_gene_pair, feature_gene_pair

using DataFrames, CSV


include("compare_genepair.jl")

include("../RankCompV3/RankCompV3.jl")

"""
    Pairing genes into gene pairs.

```jldoctest
julia> mat = [[1 4 2];[3 5 2];[5 6 7]]
3×3 Matrix{Int64}:
 1  4  2
 3  5  2
 5  6  7

julia> f_l1=[1,3]
2-element Vector{Int64}:
 1
 3

julia> f_l2=[1,2,3]
3-element Vector{Int64}:
 1
 2
 3
julia> feature_gene_pair(mat,f_l1,f_l2)
  0.000007 seconds (27 allocations: 1.719 KiB)
2×3×3 BitArray{3}:
[:, :, 1] =
 0  0  0
 1  1  1

[:, :, 2] =
 1  0  0
 1  1  0

[:, :, 3] =
 1  1  0
 1  1  1
```

```jldoctest
feature_gene_pair(mat::AbstractMatrix, # Expression profiles matrix 
                          f_l1::Vector, # The row in which the feature gene is located in the expression profile.
                          f_l2::Vector # The row in which the feature gene is located in the expression profile.
                          )
```

The parameters are: 
- `mat::AbstractMatrix`: Expression profiles matrix.
- `f_l1::Vector`: The row in which the feature gene is located in the expression profile.
- `f_l2::Vector`: The row in which the feature gene is located in the expression profile.
"""

function genepair_select(mat::AbstractMatrix, # Expression profiles matrix 
                               f_l1::Vector, # The row in which the feature gene is located in the expression profile.
                               f_l2::Vector # The row in which the feature gene is located in the expression profile.
                               )
    r,c = size(mat)
    mat1 = mat[f_l1,:]
    mat2 = mat[f_l2,:]
    f_n1 = size(f_l1)[1]
    f_n2 = size(f_l2)[1]
    mat_genepair_01 = falses(f_n1,f_n2,c)
    @time [[mat_genepair_01[i,j,:] = broadcast(is_greater,  mat1[i,:],  mat2[j,:]) for j in 1:f_n2] for i in 1:f_n1]
    return mat_genepair_01
end


function gene_select(mat::AbstractMatrix, # Expression profiles matrix 
                    fea::Vector,
                    ngrp::BitVector,
                    bar::Vector,
                    fn_feature::AbstractString;
                         mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t')
    if mode_gene_select == "custom"
        features = read_feature(fn_feature; fg_sit = fn_feature_gene_sit, delim = fn_feature_delim)
        f_l = map(x -> ∈(x, features), fea)
        f_n = sum(f_l)
    else
        expr = DataFrame(mat, :auto)
        rename!(expr, Symbol.(bar))
        insertcols!(expr,   1, :Name => fea)
        # RankCompV3
        @info "INFO: Run the RankCompV3 algorithm to get the DEGs list."
        outs_RankCompV3 = reoa(expr,DataFrame(Name = bar,Group = ngrp))
        # f_l = (outs_RankCompV3[:,2] .== "up" .|| outs_RankCompV3[:,2] .== "down")
        # features = outs_RankCompV3[f_l,1]
        f_up = (outs_RankCompV3[:,2] .== "up")
        f_down = (outs_RankCompV3[:,2] .== "down")
        if (sum(f_up) >= 20) && (sum(f_down) >= 20)
            features = vcat(outs_RankCompV3[f_up,1][1:20],outs_RankCompV3[f_down,1][1:20])
            f_l = map(x -> ∈(x, features), fea)
            f_n = 40
        else
            println("DEGs的上下调出现小于20的情况，所以取全部degs")
            f_l = (outs_RankCompV3[:,2] .== "up" .|| outs_RankCompV3[:,2] .== "down")
            features = outs_RankCompV3[f_l,1]
            f_n = sum(f_l)
        end
        f_n > 0 || throw(DimensionMismatch("No DEGs was recognized under the current threshold."))
        @info "INFO: A total of $(f_n) DEGs were obtained by RankCompV3 algorithm."
        writedlm("RankCompV3_features.tsv", features, '\t')
    end
    return f_l
end


""" 
Feature gene pairs are paired through this function: 

```jldoctest
fgene_to_genepair_kernel(
                    mat::AbstractMatrix, # Expression profiles matrix 
                    fea::Vector,
                    ngrp::BitVector,
                    bar::Vector,
                    fn_feature::AbstractString,
                    fn_stem::AbstractString;
                    mode_genepair_select::AbstractString = "all_gene_pair", # "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair"
                         mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t'
                   )
```

The parameters are: 
- `mat::AbstractMatrix`: Expression profiles matrix.
- `fea::Vector`: Gene list.
- `ngrp::BitVector`: Sample grouping information, 1 or 0.
- `bar::Vector`: Sample group name.
- `fn_feature::AbstractString`: Feature file name (optional).
- `fn_stem::AbstractString`: Filename.
- `mode_genepair_select::AbstractString`: "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair". Default: "all_gene_pair".
- `mode_gene_select::AbstractString`: "no", "custom", "DEGs_by_RankCompV3". Default: "custom".
- `fn_feature_gene_sit::Int`: The column in which the feature gene is located. Default: 1.
- `fn_feature_delim::AbstractChar`: The separator of the characteristic gene file. Default: '\t'.
"""
function fgene_to_genepair_kernel(
                    mat::AbstractMatrix, # Expression profiles matrix 
                    fea::Vector,
                    ngrp::BitVector,
                    bar::Vector,
                    fn_feature::AbstractString,
                    fn_stem::AbstractString;
                    mode_genepair_select::AbstractString = "all_gene_pair", # "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair"
                         mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t'
                   )
    r,c = size(mat)
    if mode_genepair_select == "all_gene_pair"
        mat_genepair_01 = genepair_select(mat, [1:r...], [1:r...])
        return mat_genepair_01, fea, fea, r
    else
        f_l = gene_select(mat, fea, ngrp, bar, fn_feature, mode_gene_select = mode_gene_select, fn_feature_gene_sit = fn_feature_gene_sit, fn_feature_delim = fn_feature_delim)
        if (mode_genepair_select == "all_feature_gene_pair")
            # 将特征排列在前面
            # mat_genepair_01 = genepair_select(mat, [1:r...][f_l], vcat([1:r...][f_l],[1:r...][.!f_l]))
            l_fea = vcat([1:r...][f_l],[1:r...][.!f_l])
            mat_genepair_01 = genepair_select(mat, l_fea, l_fea)
            return mat_genepair_01, fea[l_fea], fea[l_fea], sum(f_l)
        else
            (mode_genepair_select == "feature_gene_pair") || throw("The mode_genepair_select parameter supports only all_gene_pair, all_feature_gene_pair, or feature_gene_pair. ")
            mat_genepair_01 = genepair_select(mat, [1:r...][f_l], [1:r...][f_l])
            return mat_genepair_01, fea[f_l], fea[f_l], sum(f_l)
        end
    end
end



