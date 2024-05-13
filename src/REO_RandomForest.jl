# module REO_RandomForest

export run_REO_RandomForest, predict_sample_classification

include("code/read_file.jl")

include("code/process_expr.jl")

include("binRF/TreeEnsemble.jl")


using .TreeEnsemble
using .TreeEnsemble:fit!
using .TreeEnsemble:predict
using Test

using CSV, DataFrames
using Random
using Flux

using JLD
using HDF5

"""
    Train a random forest model.

    run_REO_RandomForest(mat::AbstractMatrix, # Expression profiles matrix 
                         fea::Vector,  # 基因名
                           Y::DataFrame, # 真实样本标签
            marker_genepair::Matrix,
                     fn_stem::String)

```jldoctest
julia> mat = [1 2 3 4 5 6;6 5 4 3 2 1;1 2 3 4 5 6;1 2 3 4 5 6;6 5 4 3 2 1;1 2 3 4 5 6]'
6×6 adjoint(::Matrix{Int64}) with eltype Int64:
 1  6  1  1  6  1
 2  5  2  2  5  2
 3  4  3  3  4  3
 4  3  4  4  3  4
 5  2  5  5  2  5
 6  1  6  6  1  6

julia> fea = ["LINC01409", "NOC2L", "HES4", "SDF4", "B3GALT6", "ACAP3"]
6-element Vector{String}:
 "LINC01409"
 "NOC2L"
 "HES4"
 "SDF4"
 "B3GALT6"
 "ACAP3"

 julia> Y = DataFrame([1 2 1 1 2 1]',:auto)
 6×1 DataFrame
  Row │ x1    
      │ Int64 
 ─────┼───────
    1 │     1
    2 │     2
    3 │     1
    4 │     1
    5 │     2
    6 │     1
 
julia> marker_genepair = ["LINC01409" "NOC2L"]
1×2 Matrix{String}:
 "LINC01409"  "NOC2L"

julia> fn_stem = "file_test"
"file_test"

julia> classifier = run_REO_RandomForest(mat, fea, Y, marker_genepair, fn_stem)
Feature importance (permutation)        Dict(:means => [0.27999999999999997], :stds => [0.19321835661585918])
RandomForestClassifier{Float64}(n_trees=10, max_depth=nothing, max_features=4, min_samples_leaf=3, random_state=MersenneTwister(UInt32[0x0000002a],...), bootstrap=true, oob_score=true)
```

The parameters are:
- `nmat::AbstractMatrix`: Expression profiles matrix.
- `fea::Vector`: Gene list.
- `fea::Vector`: Sample grouping information, 1 or 0.
- `marker_genepair::Matrix`: The list of marker gene pairs consists of two columns, the first is gene a, the second is gene b, and the relationship between them is a &gt; b.
- `fn_stem::String`: File name prefix.
"""


# 训练随机森林模型
function run_REO_RandomForest(mat::AbstractMatrix, # Expression profiles matrix 
                              fea::Vector,  # 基因名
                                Y::DataFrame, # 真实样本标签
                 marker_genepair::Matrix,
                          fn_stem::String)
    ## 标准化X（除target外的其他变量）。（数值 - 均值）/标准差
    Xnorm = Flux.normalise(Matrix(mat'), dims = 1)
    r, c = size(Xnorm)

    ## 将X中有关联的特征间计算reo关系（每个值的大小比较）
    X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_genepair))
    # X = DataFrame(convert.(Int64,X_reo), :auto)
    X = DataFrame(X_reo, :auto)
    # Set up RF parameters
    max_features = 4    ## 特征数量，若特征数大于该值，则随机抽取特征
    min_samples_leaf = 3  ## 最少子节点数
    n_trees = 10     ## 树的数量
    ## 定义一个包含多个指定参数的变量
    classifier = RandomForestClassifier(random_state=42,
                                        n_trees=n_trees,
                                        max_features=max_features,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=true,
                                        oob_score=true)

    ## 抽取训练和测试数据集。test_size表示取样本中多少占比作为测试集。
    X_train, y_train, X_test, y_test = split_data(X, Y, rng=classifier.random_state, test_size=0.2)

    # Train
    fit!(classifier, X_train, y_train)
    ## 每个筛选出的单个特征的重要性程度
    rng = MersenneTwister(2)
    fi =  perm_feature_importance(classifier, X_train, y_train, n_repeats=10, random_state = rng)
    println("Feature importance (permutation)\t", fi)

    ## 绘制森林图
    # plot_forest(classifier, marker_genepair, fn_stem)
    return classifier
end

# """
#     predict_sample_classification(fn_classifier::AbstractString = "RandomForest_classifier.jld", # 随机森林结果
#                                                 fn_expr::AbstractString = "matrix.mtx",
#                                                 rn_expr::AbstractString = "features.tsv",
#                                                 cn_expr::AbstractString = "barcodes.tsv",
#                                  fn_marker_genepair::AbstractString = "marker_genepairs.tsv";
#                                        file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
#                                                       T::Type = Int32,
#                                             feature_col::Int = 2,
#                                             barcode_col::Int = 1,
#                                             fn_marker_genepair_delim::AbstractChar = '\t',
#                                       feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
#                                          cell_threshold::Int = 200) # Include profiles (cells) where at least this many features are detected

# ```jldoctest
# julia> classifier = run_REO_RandomForest(mat, fea, Y, marker_genepair, fn_stem)
# Feature importance (permutation)        Dict(:means => [0.27999999999999997], :stds => [0.19321835661585918])
# RandomForestClassifier{Float64}(n_trees=10, max_depth=nothing, max_features=4, min_samples_leaf=3, random_state=MersenneTwister(UInt32[0x0000002a],...), bootstrap=true, oob_score=true)

# julia> nam = ["group1","group2"]
# 2-element Vector{String}:
#  "group1"
#  "group2" 

# julia> fn_expr = joinpath(@__DIR__, "..", "test", "matrix.mtx")
# "/REO_molecular_signatures.jl/src/../test/matrix.mtx"

# julia> rn_expr = joinpath(@__DIR__, "..", "test", "features.tsv")
# "/REO_molecular_signatures.jl/src/../test/features.tsv"

# julia> cn_expr = joinpath(@__DIR__, "..", "test", "barcodes.tsv")
# "/REO_molecular_signatures.jl/src/../test/barcodes.tsv"

# julia> predict_sample_classification("RandomForest_classifier.jld", "matrix.mtx", "features.tsv", "barcodes.tsv", "marker_genepairs.tsv")
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

# function predict_sample_classification(fn_classifier::AbstractString = "RandomForest_classifier.jld", # 随机森林结果
#                                                 fn_expr::AbstractString = "matrix.mtx",
#                                                 rn_expr::AbstractString = "features.tsv",
#                                                 cn_expr::AbstractString = "barcodes.tsv",
#                                  fn_marker_genepair::AbstractString = "marker_genepairs.tsv";
#                                        file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
#                                                       T::Type = Int32,
#                                             feature_col::Int = 2,
#                                             barcode_col::Int = 1,
#                                             fn_marker_genepair_delim::AbstractChar = '\t',
#                                       feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
#                                          cell_threshold::Int = 200) # Include profiles (cells) where at least this many features are detected
#     fn_stem, = splitext(basename(fn_expr))   #filename stem
#     @time classifier = load_RandomForest_classifier(fn_classifier)
#     @time mat, fea, bar = (file_format_expr == "read_mtx") ? read_mtx(fn_expr, rn_expr, cn_expr; T, feature_col, barcode_col) : read_expr_matrix(fn_expr, rn_expr, cn_expr)
#     @info "INFO: The size of expression profile was $(size(mat))."
#     @time marker_genepair = read_marker_genepair(fn_marker_genepair; delim = fn_marker_genepair_delim)
#     # # 过滤表达谱
#     # @time mat, kf, kb = filter_expr_matrix(mat, feature_threshold, cell_threshold)
#     # @info "INFO: The filtered of expression profile size was $(size(mat))."
#     # fea = fea[kf]
#     # bar = bar[kb]
#     Xnorm = Flux.normalise(Matrix(mat'), dims = 1)
#     r, c = size(Xnorm)
#     X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_genepair))
#     X = DataFrame(X_reo, :auto)
#     y_pred = predict(classifier, X)
#     # classifier.t_classes
#     # y_pred
#     # 分类结果，第一列为样本名，第二列为分类类别
#     pre_sample_classification = hcat(bar,y_pred)
#     writedlm(join([fn_stem, "RandomForest_pre.tsv"], "_"), pre_sample_classification, "\t")
#     return pre_sample_classification
#     # return hcat(nam[y_pred],nam[y_pred])
# end
    
# function predict_sample_classification(classifier::AbstractClassifier,
#                                                 mat::Matrix,
#                                                 fea::Vector,
#                                                 bar::Vector,
#                                     marker_genepair::Matrix,
#                                     fn_stem::AbstractString) # Include profiles (cells) where at least this many features are detected
#     Xnorm = Flux.normalise(Matrix(mat'), dims = 1)
#     r, c = size(Xnorm)
#     X_reo = mapreduce(x->(Xnorm[:,(fea .== x[1])] .> Xnorm[:,(fea .== x[2])]),hcat,eachrow(marker_genepair))
#     X = DataFrame(X_reo, :auto)
#     y_pred = predict(classifier, X)
#     # classifier.t_classes
#     # y_pred
#     # 分类结果，第一列为样本名，第二列为分类类别
#     pre_sample_classification = hcat(bar,y_pred)
#     writedlm(join([fn_stem, "RandomForest_pre.tsv"], "_"), pre_sample_classification, "\t")
#     return pre_sample_classification
#     # return hcat(nam[y_pred],nam[y_pred])
# end
# end # module