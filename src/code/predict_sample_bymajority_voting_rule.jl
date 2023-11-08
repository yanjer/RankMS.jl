export predict_sample_classification, intersect_groudtruth

using DelimitedFiles, SparseArrays

using JLD, HDF5

include("read_file.jl")
include("compare_genepair.jl")
include("sample_intersection.jl")

"""

    Using majority voting rules to classify samples according to characteristic gene pairs.

    predict_sample_classification(fn_expr::AbstractString = "matrix.mtx",
                                                rn_expr::AbstractString = "features.tsv",
                                                cn_expr::AbstractString = "barcodes.tsv",
                                 fn_marker_fea::AbstractString = "marker_feas.tsv")

```jldoctest
julia> pre_sample_classification
37×2 Matrix{Any}:
 "39-3-31-14_S15"    2
 "MGH39_082514-1"    2
 "208-9-10-14_S11"   2
 "MGH42_101714"      2
 "MGH208_051315"     2
 "62-10-2-13_S6"     2
 "115-041514"        2
 "115-031814"        2
 "208-10-22-14_S12"  2
 "MGH208_031115-2"   2
 "MGH42_112414"      2
 ⋮                   
 "MGH42_101714_1"    2
 "62-7-8-14_S8"      2
 "208-5-13-15_S14"   2
 "MGH39_082514-2"    2
 "39-8-25-14_S16"    2
 "MGH39_103114"      2
 "MGH422-110515"     2
 "62-5-27-14_S7"     2
 "42-11-24-14_S4"    2
 "208-3-11-15_S13"   2

```

With known metadata information, the classification effect of the current feature gene pair is calculated

```jldoctest
julia> pre_sample_classification, pre_sample_classification_ct=predict_sample_classification("fn_matrix.txt", 
                                  "rn_matrix.txt", 
                                  "cn_matrix.txt",
                                  "fn_matrix_marker_feas.tsv",
                                  file_format_expr = "read_expr_matrix",
                                  fn_meta="metadata.txt")
  0.779828 seconds (8.64 M allocations: 272.278 MiB, 41.15% gc time)
[ Info: INFO: The size of expression profile was (75253, 37).
  0.000440 seconds (94 allocations: 45.453 KiB)
(Any["39-3-31-14_S15" 2; "MGH39_082514-1" 2; … ; "42-11-24-14_S4" 2; "208-3-11-15_S13" 2], Any["" "True_group1" "True_group2"; "Pre_group1" 0 0; "Pre_group2" 34 3])

julia> pre_sample_classification
37×2 Matrix{Any}:
 "39-3-31-14_S15"    2
 "MGH39_082514-1"    2
 "208-9-10-14_S11"   2
 "MGH42_101714"      2
 "MGH208_051315"     2
 "62-10-2-13_S6"     2
 "115-041514"        2
 "115-031814"        2
 "208-10-22-14_S12"  2
 "MGH208_031115-2"   2
 "MGH42_112414"      2
 ⋮                   
 "MGH42_101714_1"    2
 "62-7-8-14_S8"      2
 "208-5-13-15_S14"   2
 "MGH39_082514-2"    2
 "39-8-25-14_S16"    2
 "MGH39_103114"      2
 "MGH422-110515"     2
 "62-5-27-14_S7"     2
 "42-11-24-14_S4"    2
 "208-3-11-15_S13"   2

julia> pre_sample_classification_ct
3×3 Matrix{Any}:
 ""              "True_group1"   "True_group2"
 "Pre_group1"   0               0
 "Pre_group2"  34               3


    predict_sample_classification(fn_expr::AbstractString = "matrix.mtx",
                                                rn_expr::AbstractString = "features.tsv",
                                                cn_expr::AbstractString = "barcodes.tsv",
                                 fn_marker_fea::AbstractString = "marker_feas.tsv";
                                 fn_meta::AbstractString = "fn_meta.tsv",
                                 fn_meta_delim::AbstractChar = '\t',
                               fn_meta_group::AbstractString = "group",
                                       file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
                                                      T::Type = Int32,
                                            feature_col::Int = 2,
                                            barcode_col::Int = 1,
                                            fn_marker_fea_delim::AbstractChar = '\t',
                                      feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
                                         cell_threshold::Int = 200) # Include profiles (cells) where at least this many features are detected

"""

# Majority voting rule
function predict_sample_classification(fn_expr::AbstractString = "matrix.mtx",
                                                rn_expr::AbstractString = "features.tsv",
                                                cn_expr::AbstractString = "barcodes.tsv",
                                 fn_marker_fea::AbstractString = "marker_feas.tsv";
                                 fn_meta::AbstractString = "fn_meta.tsv",
                                 fn_meta_delim::AbstractChar = '\t',
                               fn_meta_group::AbstractString = "group",
                                       file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
                                                      T::Type = Int32,
                                            feature_col::Int = 2,
                                            barcode_col::Int = 1,
                                            fn_marker_fea_delim::AbstractChar = '\t',
                                      feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
                                         cell_threshold::Int = 200) # Include profiles (cells) where at least this many features are detected
    fn_stem, = splitext(basename(fn_expr))   #filename stem
    @time mat, fea, bar = (file_format_expr == "read_mtx") ? read_mtx(fn_expr, rn_expr, cn_expr; T, feature_col, barcode_col) : read_expr_matrix(fn_expr, rn_expr, cn_expr)
    @info "INFO: The size of expression profile was $(size(mat))."
    @time marker_fea = read_marker_fea(fn_marker_fea; delim = fn_marker_fea_delim)
    # # 过滤表达谱
    # @time mat, kf, kb = filter_expr_matrix(mat, feature_threshold, cell_threshold)
    # @info "INFO: The filtered of expression profile size was $(size(mat))."
    # fea = fea[kf]
    # bar = bar[kb]
    Xnorm = Matrix(mat')
    r, c = size(Xnorm)
    X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_fea))
    vote = sum(X_reo,dims=2)
    m = size(marker_fea)[1]/2
    pre_classification = .!mapreduce(x -> is_greater(x, m, 0),vcat,vote) .+ 1
    # 分类结果，第一列为样本名，第二列为分类类别
    pre_sample_classification = hcat(bar,pre_classification)
    writedlm(join([fn_stem, "majority_voting_rule_pre.tsv"], "_"), pre_sample_classification, "\t")
    if isfile(fn_meta)
        grp, nam, meta_bar = read_meta(fn_meta, fn_meta_group; delim = fn_meta_delim)
        r == length(meta_bar) ||  throw(DimensionMismatch("The number of samples for metadata is not equal to the number of samples for the expression profile."))
        pre_sample_classification_ct = ["" "True_group1" "True_group2"; "Pre_group1" length(intersect_groudtruth(pre_sample_classification,grp,1,1)) length(intersect_groudtruth(pre_sample_classification,grp,1,2)); "Pre_group2" length(intersect_groudtruth(pre_sample_classification,grp,2,1)) length(intersect_groudtruth(pre_sample_classification,grp,2,2))]
        return pre_sample_classification, pre_sample_classification_ct
    end
    return pre_sample_classification
end

"""

    Using majority voting rules to classify samples according to characteristic gene pairs.

    predict_sample_classification(mat::Matrix,
                                       fea::Vector,
                                       bar::Vector,
                                    marker_fea::Matrix)

```jldoctest
julia> pre_sample_classification
37×2 Matrix{Any}:
 "39-3-31-14_S15"    2
 "MGH39_082514-1"    2
 "208-9-10-14_S11"   2
 "MGH42_101714"      2
 "MGH208_051315"     2
 "62-10-2-13_S6"     2
 "115-041514"        2
 "115-031814"        2
 "208-10-22-14_S12"  2
 "MGH208_031115-2"   2
 "MGH42_112414"      2
 ⋮                   
 "MGH42_101714_1"    2
 "62-7-8-14_S8"      2
 "208-5-13-15_S14"   2
 "MGH39_082514-2"    2
 "39-8-25-14_S16"    2
 "MGH39_103114"      2
 "MGH422-110515"     2
 "62-5-27-14_S7"     2
 "42-11-24-14_S4"    2
 "208-3-11-15_S13"   2

```

With known metadata information, the classification effect of the current feature gene pair is calculated

```jldoctest
julia> pre_sample_classification, pre_sample_classification_ct=predict_sample_classification(mat, 
                                                                                            fea, 
                                                                                            bar,
                                                                                            marker_fea,
                                                                                            fn_stem,
                                                                                            grp)
  0.779828 seconds (8.64 M allocations: 272.278 MiB, 41.15% gc time)
[ Info: INFO: The size of expression profile was (75253, 37).
  0.000440 seconds (94 allocations: 45.453 KiB)
(Any["39-3-31-14_S15" 2; "MGH39_082514-1" 2; … ; "42-11-24-14_S4" 2; "208-3-11-15_S13" 2], Any["" "True_group1" "True_group2"; "Pre_group1" 0 0; "Pre_group2" 34 3])

julia> pre_sample_classification
37×2 Matrix{Any}:
 "39-3-31-14_S15"    2
 "MGH39_082514-1"    2
 "208-9-10-14_S11"   2
 "MGH42_101714"      2
 "MGH208_051315"     2
 "62-10-2-13_S6"     2
 "115-041514"        2
 "115-031814"        2
 "208-10-22-14_S12"  2
 "MGH208_031115-2"   2
 "MGH42_112414"      2
 ⋮                   
 "MGH42_101714_1"    2
 "62-7-8-14_S8"      2
 "208-5-13-15_S14"   2
 "MGH39_082514-2"    2
 "39-8-25-14_S16"    2
 "MGH39_103114"      2
 "MGH422-110515"     2
 "62-5-27-14_S7"     2
 "42-11-24-14_S4"    2
 "208-3-11-15_S13"   2

julia> pre_sample_classification_ct
3×3 Matrix{Any}:
 ""              "True_group1"   "True_group2"
 "Pre_group1"   0               0
 "Pre_group2"  34               3
```

    predict_sample_classification(mat::Matrix,
                                       fea::Vector,
                                       bar::Vector,
                                    marker_fea::Matrix,
                                    fn_stem::AbstractString = "psc",
                                    grp::Vector = [])

"""
function predict_sample_classification(mat::Matrix,
                                       fea::Vector,
                                       bar::Vector,
                                    marker_fea::Matrix,
                                    fn_stem::AbstractString = "psc",
                                    grp::Vector = [])
    Xnorm = Matrix(mat')
    r, c = size(Xnorm)
    X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_fea))
    vote = sum(X_reo,dims=2)
    m = size(marker_fea)[1]/2
    pre_classification = .!mapreduce(x -> is_greater(x, m, 0),vcat,vote) .+ 1
    # 分类结果，第一列为样本名，第二列为分类类别
    pre_sample_classification = hcat(bar,pre_classification)
    writedlm(join([fn_stem, "majority_voting_rule_pre.tsv"], "_"), pre_sample_classification, "\t")
    if grp != []
        pre_sample_classification_ct = ["" "True_group1" "True_group2"; "Pre_group1" length(intersect_groudtruth(pre_sample_classification,grp,1,1)) length(intersect_groudtruth(pre_sample_classification,grp,1,2)); "Pre_group2" length(intersect_groudtruth(pre_sample_classification,grp,2,1)) length(intersect_groudtruth(pre_sample_classification,grp,2,2))]
        return pre_sample_classification, pre_sample_classification_ct
    end
    return pre_sample_classification
end




# """
#     predict_sample_classification(fn_classifier::AbstractString = "RandomForest_classifier.jld", # 随机森林结果
#                                                 fn_expr::AbstractString = "matrix.mtx",
#                                                 rn_expr::AbstractString = "features.tsv",
#                                                 cn_expr::AbstractString = "barcodes.tsv",
#                                  fn_marker_fea::AbstractString = "marker_feas.tsv";
#                                        file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
#                                                       T::Type = Int32,
#                                             feature_col::Int = 2,
#                                             barcode_col::Int = 1,
#                                             fn_marker_fea_delim::AbstractChar = '\t',
#                                       feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
#                                          cell_threshold::Int = 200) # Include profiles (cells) where at least this many features are detected

# ```jldoctest
# julia> classifier = run_REO_RandomForest(mat, fea, Y, marker_fea, fn_stem)
# Feature importance (permutation)        Dict(:means => [0.27999999999999997], :stds => [0.19321835661585918])
# RandomForestClassifier{Float64}(n_trees=10, max_depth=nothing, max_features=4, min_samples_leaf=3, random_state=MersenneTwister(UInt32[0x0000002a],...), bootstrap=true, oob_score=true)

# julia> nam = ["group1","group2"]
# 2-element Vector{String}:
#  "group1"
#  "group2" 

# julia> fn_expr = joinpath(@__DIR__, "..", "test", "matrix.mtx")
# "/REO_molecular_markers.jl/src/../test/matrix.mtx"

# julia> rn_expr = joinpath(@__DIR__, "..", "test", "features.tsv")
# "/REO_molecular_markers.jl/src/../test/features.tsv"

# julia> cn_expr = joinpath(@__DIR__, "..", "test", "barcodes.tsv")
# "/REO_molecular_markers.jl/src/../test/barcodes.tsv"

# julia> predict_sample_classification("RandomForest_classifier.jld", "matrix.mtx", "features.tsv", "barcodes.tsv", "marker_feas.tsv")
#   0.065766 seconds (1.08 M allocations: 34.743 MiB)
# [ Info: INFO: The size of expression profile was (36602, 8).
#   0.045254 seconds (642.74 k allocations: 27.426 MiB)
# [ Info: INFO: The filtered of expression profile size was (7549, 8).
# 8×2 Matrix{String}:
#  "AAACCCAAGAAACCAT-1"  "group2"
#  "AAACCCAAGCAACAAT-1"  "group1"
#  "AAACCCAAGCCAGAGT-1"  "group2"
#  "AAACCCAAGGGTTAAT-1"  "group2"
#  "AAACCCAAGTAGACAT-1"  "group2"
#  "AAACCCACAGCAGATG-1"  "group1"
#  "AAACCCACAGCGTGCT-1"  "group1"
#  "AAACCCAGTACGGGAT-1"  "group1"

# ```
# """

# # 随机森林
# function predict_sample_classification(fn_classifier::AbstractString = "RandomForest_classifier.jld", # 随机森林结果
#                                                 fn_expr::AbstractString = "matrix.mtx",
#                                                 rn_expr::AbstractString = "features.tsv",
#                                                 cn_expr::AbstractString = "barcodes.tsv",
#                                  fn_marker_fea::AbstractString = "marker_feas.tsv";
#                                  fn_meta::AbstractString = "fn_meta.tsv",
#                                  fn_meta_delim::AbstractChar = '\t',
#                                fn_meta_group::AbstractString = "group",
#                                        file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
#                                                       T::Type = Int32,
#                                             feature_col::Int = 2,
#                                             barcode_col::Int = 1,
#                                             fn_marker_fea_delim::AbstractChar = '\t',
#                                       feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
#                                          cell_threshold::Int = 200) # Include profiles (cells) where at least this many features are detected
#     fn_stem, = splitext(basename(fn_expr))   #filename stem
#     @time classifier = load_RandomForest_classifier(fn_classifier)
#     @time mat, fea, bar = (file_format_expr == "read_mtx") ? read_mtx(fn_expr, rn_expr, cn_expr; T, feature_col, barcode_col) : read_expr_matrix(fn_expr, rn_expr, cn_expr)
#     @info "INFO: The size of expression profile was $(size(mat))."
#     @time marker_fea = read_marker_fea(fn_marker_fea; delim = fn_marker_fea_delim)
#     # # 过滤表达谱
#     # @time mat, kf, kb = filter_expr_matrix(mat, feature_threshold, cell_threshold)
#     # @info "INFO: The filtered of expression profile size was $(size(mat))."
#     # fea = fea[kf]
#     # bar = bar[kb]
#     Xnorm = Flux.normalise(Matrix(mat'), dims = 1)
#     r, c = size(Xnorm)
#     X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_fea))
#     X = DataFrame(X_reo, :auto)
#     y_pred = predict(classifier, X)
#     # classifier.t_classes
#     # y_pred
#     # 分类结果，第一列为样本名，第二列为分类类别
#     pre_sample_classification = hcat(bar,y_pred)
#     writedlm(join([fn_stem, "RandomForest_pre.tsv"], "_"), pre_sample_classification, "\t")
#     if isfile(fn_meta)
#         grp, nam, meta_bar = read_meta(fn_meta, fn_meta_group; delim = fn_meta_delim)
#         r == length(meta_bar) ||  throw(DimensionMismatch("The number of samples for metadata is not equal to the number of samples for the expression profile."))
#         pre_sample_classification_ct = ["" "True_group1" "True_group2"; "Pre_group1" length(intersect_groudtruth(pre_sample_classification,grp,1,1)) length(intersect_groudtruth(pre_sample_classification,grp,1,2)); "Pre_group2" length(intersect_groudtruth(pre_sample_classification,grp,2,1)) length(intersect_groudtruth(pre_sample_classification,grp,2,2))]
#         return pre_sample_classification, pre_sample_classification_ct
#     end
#     return pre_sample_classification
#     # return hcat(nam[y_pred],nam[y_pred])
# end
    
# function predict_sample_classification(classifier::AbstractClassifier,
#                                                 mat::Matrix,
#                                                 fea::Vector,
#                                                 bar::Vector,
#                                     marker_fea::Matrix,
#                                     fn_stem::AbstractString,
#                                     grp::Vector = [])
#     Xnorm = Flux.normalise(Matrix(mat'), dims = 1)
#     r, c = size(Xnorm)
#     X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_fea))
#     X = DataFrame(X_reo, :auto)
#     y_pred = predict(classifier, X)
#     # classifier.t_classes
#     # y_pred
#     # 分类结果，第一列为样本名，第二列为分类类别
#     pre_sample_classification = hcat(bar,y_pred)
#     writedlm(join([fn_stem, "RandomForest_pre.tsv"], "_"), pre_sample_classification, "\t")
#     if grp != []
#         pre_sample_classification_ct = ["" "True_group1" "True_group2"; "Pre_group1" length(intersect_groudtruth(pre_sample_classification,grp,1,1)) length(intersect_groudtruth(pre_sample_classification,grp,1,2)); "Pre_group2" length(intersect_groudtruth(pre_sample_classification,grp,2,1)) length(intersect_groudtruth(pre_sample_classification,grp,2,2))]
#         return pre_sample_classification, pre_sample_classification_ct
#     end
#     return pre_sample_classification
#     # return hcat(nam[y_pred],nam[y_pred])
# end