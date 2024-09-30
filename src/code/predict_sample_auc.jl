export predict_sample_classification, intersect_groudtruth

using DelimitedFiles, SparseArrays

# using JLD, HDF5

include("read_file.jl")
include("compare_genepair.jl")
include("roc.jl")

function predict_sample_classification(fn_expr::AbstractString = "matrix.mtx",
                                                rn_expr::AbstractString = "features.tsv",
                                                cn_expr::AbstractString = "barcodes.tsv",
                                                fn_group_order::AbstractString = "group_order.tsv",
                                 fn_marker_fea::AbstractString = "marker_feas.tsv",
                                 fn_meta::AbstractString = "fn_meta.tsv";
                                 fn_meta_delim::AbstractChar = '\t',
                               fn_meta_group::AbstractString = "group",
                                       file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
                                                      T::Type = Int32,
                                            feature_col::Int = 2,
                                            barcode_col::Int = 1,
                                            fn_marker_fea_delim::AbstractChar = '\t',
                                      feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
                                         cell_threshold::Int = 200，
                                         train_label1::AbstractString = "",
                                         train_label2::AbstractString = "",
                                         verify_label1::AbstractString = "",
                                         verify_label2::AbstractString = "",
                                         ) # Include profiles (cells) where at least this many features are detected
    fn_stem, = splitext(basename(fn_expr))   #filename stem
    @time mat, fea, bar = (file_format_expr == "read_mtx") ? read_mtx(fn_expr, rn_expr, cn_expr; T, feature_col, barcode_col) : read_expr_matrix(fn_expr, rn_expr, cn_expr)
    @info "INFO: The size of expression profile was $(size(mat))."
    @time marker_fea = read_marker_fea(fn_marker_fea; delim = fn_marker_fea_delim)
    group_order = read_group_order(fn_group_order)
    # group_order[group_order .== "regression"] .= "Response"
    # group_order[group_order .== "unregression"] .= "No_Response"
    if train_label1 != ""
        group_order[group_order .== train_label1] .= verify_label1
        group_order[group_order .== train_label2] .= verify_label2
    end
    grp, nam, meta_bar = read_meta(fn_meta, fn_meta_group; delim = fn_meta_delim)
    if nam != group_order
        nam = reverse(nam)
        nam == group_order || throw("The group names of the validation dataset and the training dataset need to be the same.")
        grp = reverse(grp)
    end
    # # # 过滤表达谱
    # # @time mat, kf, kb = filter_expr_matrix(mat, feature_threshold, cell_threshold)
    # # @info "INFO: The filtered of expression profile size was $(size(mat))."
    # # fea = fea[kf]
    # # bar = bar[kb]
    # Xnorm = Matrix(mat')
    r, c = size(mat)
    X_reo = mapreduce(x->(mat[(fea .== x[1]),:] .>= mat[(fea .== x[2]),:]),vcat,eachrow(marker_fea))
    reduce(vcat, [[mat[(fea .== x[1]),:] .> mat[(fea .== x[2]),:]] for x in eachrow(marker_fea)])
    vote = sum(X_reo',dims=2)
    ngrp = (bar .∈ (grp[2], ))
    pre_AUC = roc_kernel(vec(vote), ngrp)
    return pre_AUC
end