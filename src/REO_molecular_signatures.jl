module REO_molecular_signatures

using JLD
using HDF5
using DataFrames

export REO_molecular_signatures_main, 
       hill_climbing_method_kernel,
       fgene_to_genepair_kernel,
       generate_pseudobulk_kernel,
       filter_expr_matrix,
       read_mtx, read_gmt, read_meta,
       # 构建随机森林模型；使用随机森林进行样本分类
       run_REO_RandomForest, predict_sample_classification


include("code/read_file.jl")

include("code/process_expr.jl")

include("code/gene_and_genepair_select.jl")

include("code/train_and_test_sample.jl")

include("code/hill_climbing_method.jl")

include("run_REO_molecular_signatures.jl")

include("code/roc.jl")

include("REO_RandomForest.jl")

include("code/predict_sample_bymajority_voting_rule.jl")

include("binRF/predict_sample_byRandomForest.jl")

include("code/sample.jl")


"""
  Identification of molecular signatures based on REO.
  REO_molecular_signatures_main(fn_expr, rn_expr, cn_expr, fn_meta)

Test with testdata.
```jldoctest
julia> @time REO_molecular_signatures_main(use_testdata = "yes")
  0.968692 seconds (3.16 M allocations: 144.953 MiB, 4.02% gc time, 92.61% compilation time)
[ Info: INFO: The size of expression profile was (36602, 8).
  1.233192 seconds (4.89 M allocations: 249.153 MiB, 3.64% gc time, 96.91% compilation time)
[ Info: INFO: The filtered of expression profile size was (7549, 8).
  0.119223 seconds (241.21 k allocations: 13.016 MiB, 99.76% compilation time)
  0.028087 seconds (452.95 k allocations: 27.934 MiB)
  0.679817 seconds (2.44 M allocations: 127.700 MiB, 4.66% gc time, 95.32% compilation time)
  0.163096 seconds (467.58 k allocations: 23.957 MiB, 99.55% compilation time)
...............................
...............................
 16.750809 seconds (137.61 M allocations: 22.710 GiB, 21.21% gc time, 14.69% compilation time)
  0.000375 seconds (22 allocations: 2.719 KiB)
 21.680632 seconds (152.58 M allocations: 23.441 GiB, 17.13% gc time, 33.40% compilation time)
 10×2 Matrix{SubString{String}}:
 "NOC2L"      "LINC01409"
 "HES4"       "LINC01409"
 "SDF4"       "LINC01409"
 "TSPAN2"     "LINC01409"
 "LINC9"      "NOC2L"
 "HES3"       "NOC2L"
 "SDF4"       "NOC2L"
 "TSPAN2"     "NOC2L"
 "LINC01409"  "HES4"
 "NOC2L"      "HES4"

```
Psudo-bulk mode
```jldoctest
julia> REO_molecular_signatures_main("matrix.mtx", "features.tsv", "barcodes.tsv", "fn_meta.txt", ncell_pseudo = 50)
```

Example
```jldoctest
julia> REO_molecular_signatures_main("matrix.mtx", "features.tsv", "barcodes.tsv", "fn_meta.txt")
  0.056830 seconds (452.95 k allocations: 27.934 MiB, 28.07% gc time)
  1.357235 seconds (2.88 M allocations: 151.763 MiB, 2.36% gc time, 95.23% compilation time)
  0.229058 seconds (174.22 k allocations: 9.222 MiB, 99.67% compilation time)
...............................
...............................
 20.626382 seconds (137.43 M allocations: 22.621 GiB, 17.40% gc time, 17.27% compilation time)
  0.000552 seconds (22 allocations: 2.719 KiB)
10×2 Matrix{SubString{String}}:
 "NOC2L"      "LINC01409"
 "HES4"       "LINC01409"
 "SDF4"       "LINC01409"
 "TSPAN2"     "LINC01409"
 "LINC9"      "NOC2L"
 "HES3"       "NOC2L"
 "SDF4"       "NOC2L"
 "TSPAN2"     "NOC2L"
 "LINC01409"  "HES4"
 "NOC2L"      "HES4"

```
All parameters.
```jldoctest
REO_molecular_signatures_main(fn_expr::AbstractString = "matrix.mtx",
                          rn_expr::AbstractString = "features.tsv",
                          cn_expr::AbstractString = "barcodes.tsv",
                          fn_meta::AbstractString = "fn_meta.txt",
                       fn_feature::AbstractString = "fn_feature.txt";
                                       n_top::Int = 10, # The top n_top genes with the highest scores.
                                     n_train::Int = 13, # Number of training set samples.
                                      n_test::Int = 3,  # Number of test set samples.
                             t_hill_iter_num::Int = 500,  # Hill climbing number of iterations.
                            t_train_iter_num::Int = 15,  # The number of iterations of the climb training.
                                ncell_pseudo::Int = 0, # ncell_pseudo is the number of pseudobulk combined cells in each group
                      fn_meta_delim::AbstractChar = '\t',
                    fn_meta_group::AbstractString = "group",
                 file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
                                          T::Type = Int32,
                                 feature_col::Int = 2,
                                 barcode_col::Int = 1,
                           feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
                              cell_threshold::Int = 200, # Include profiles (cells) where at least this many features are detected
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t',
                   mode_genepair_select::AbstractString = "all_gene_pair", # "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair"
                 mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                               remove_zeros::Bool = true,
                     use_testdata::AbstractString = "no",
                         work_dir::AbstractString = "./")
```

The parameters are:

- `fn_expr::AbstractString`: Expression matrix file path, with rows representing genes and columns representing samples, does not require column names. (Default: "matrix.mtx".
- `rn_expr::AbstractString`: Gene list file path. (Default: "features.tsv".
- `cn_expr::AbstractString`: Sample name list file path. (Default: "barcodes.tsv".
- `fn_meta::AbstractString`: Metadata file path, the first column sample name, the second column group information. (Default = "fn_meta.txt".
- `fn_feature::AbstractString`: List of characteristic genes (optional). Default: = "fn_feature.txt".
- `building_random_forests::AbstractString`: Whether to establish a random forest model. Default = "yes".
- `n_top::Int`: The top n_top genes with the highest scores. Default: 1.
- `n_train::Int`: Number of training set samples. Default: 13.
- `n_test::Int`: Number of test set samples. Default: 3.
- `t_hill_iter_num::Int`: Hill climbing number of iterations. Default: 500.
- `t_train_iter_num::Int`: The number of iterations of the climb training. Default: 15.
- `ncell_pseudo`: pseudo-bulk mode, which combines `ncell_pseudo` cells into one sample. If 0 is set, this mode is disabled Default: 0.
- `fn_meta_delim::AbstractChar`: Delimiter of the metadata file. Default: = '\t'.
- `fn_meta_group::AbstractString`: Specifies the column name of the group information in metadata. Default: "group".
- `file_format_expr`: There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format. Default: "read_mtx".
- `T::Type`: Datatype in the MTX file. Default: Int32.
- `feature_col::Int`: which column is used as feature names. Default: 1 (first).
- `barcode_col::Int`: which column is used as barcode names. Default: 1 (first).
- `feature_threshold::Int`: the least number of cells that a feature must express in, in order to be kept. Default: 30.
- `cell_threshold::Int`: the least number of genes that a cell must express, in order to be kept. Default: 200.
- `fn_feature_gene_sit::Int`: The column in the feature gene file where the feature gene resides. Default: 1.
- `fn_feature_delim::AbstractChar`: The separator of the characteristic gene file. Default: '\t'.
- `mode_genepair_select::AbstractString`: Gene matching method: "all_gene_pair", which means all gene pairs, "all_feature_gene_pair" indicates the pairing of the feature gene with all genes except itself. "feature_gene_pair" indicates the pairing of feature genes. Default: "all_gene_pair".
- `mode_gene_select::AbstractString`: For the use of feature genes, "no" indicates that no feature genes are used, "custom" indicates that custom feature genes are used, and "DEGs_by_RankCompV3" indicates that DEGs (FDR &lt; 0.05) as a characteristic gene. Default: "custom".
- `remove_zeros::Bool`: # Filter values that are all zeros. Default: true.
- `use_testdata::AbstractString`: Whether to use test data. "yes "or "no". Default: "no".
- `work_dir::AbstractString`: Working directory. Default: "./".
"""
function REO_molecular_signatures_main(fn_expr::AbstractString = "matrix.mtx",
                          rn_expr::AbstractString = "features.tsv",
                          cn_expr::AbstractString = "barcodes.tsv",
                          fn_meta::AbstractString = "fn_meta.txt",
                       fn_feature::AbstractString = "fn_feature.txt";
                       fn_candidate_gene::AbstractString = "no", # Keep the gene you want to analyze.
          building_random_forests::AbstractString = "yes",
                                       n_top::Int = 15, # The top n_top genes with the highest scores.
                                     n_train::Int = 13, # Number of training set samples.
                                      n_test::Int = 3,  # Number of test set samples.
                             t_hill_iter_num::Int = 2000,  # Hill climbing number of iterations.
                            t_train_iter_num::Int = 15,  # The number of iterations of the climb training.
                                ncell_pseudo::Int = 0, # ncell_pseudo is the number of pseudobulk combined cells in each group
                      fn_meta_delim::AbstractChar = '\t',
                    fn_meta_group::AbstractString = "group",
                 file_format_expr::AbstractString = "read_mtx", # There are two input modes "read_mtx" and "read_expr_matrix" for the expression profile file format.
                                          T::Type = Int32,
                                 feature_col::Int = 2,
                                 barcode_col::Int = 1,
                           feature_threshold::Int = 30, # Include features (genes) detected in at least this many cells
                              cell_threshold::Int = 200, # Include profiles (cells) where at least this many features are detected
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t',
                    fn_candidate_gene_fg_sit::Int = 1,
                     fn_candidate_gene_delim::AbstractChar = '\t',
                   mode_genepair_select::AbstractString = "all_gene_pair", # "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair"
                 mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                               remove_zeros::Bool = true,
                     use_testdata::AbstractString = "no",
                         work_dir::AbstractString = "./")
    cd(work_dir)
    if use_testdata == "yes"
        fn_expr = joinpath(@__DIR__, "..", "test", "matrix.mtx")
        rn_expr = joinpath(@__DIR__, "..", "test", "features.tsv")
        cn_expr = joinpath(@__DIR__, "..", "test", "barcodes.tsv")
        fn_meta = joinpath(@__DIR__, "..", "test", "fn_meta.txt")
        fn_feature = joinpath(@__DIR__, "..", "test", "fn_feature.txt")
        mode_genepair_select = "all_gene_pair"
        feature_threshold = 1;cell_threshold = 1;n_train = 5;n_test = 3;t_hill_iter_num = 2
    end
    fn_stem, = splitext(basename(fn_expr))   #filename stem
    @time mat, fea, bar = (file_format_expr == "read_mtx") ? read_mtx(fn_expr, rn_expr, cn_expr; T, feature_col, barcode_col) : read_expr_matrix(fn_expr, rn_expr, cn_expr)
    @info "INFO: The size of expression profile was $(size(mat))."
    # 过滤表达谱
    @time mat, kf, kb = filter_expr_matrix(mat, feature_threshold, cell_threshold)
    r,c=size(mat)
    @info "INFO: The filtered of expression profile size was $((r, c))."
    (r != 0 && c != 0) || throw("Under feature_threshold = $(feature_threshold) and cell_threshold = $(cell_threshold) conditions, the filtered expression profiles are empty.")
    fea = fea[kf]
    bar = bar[kb]
    # pseudobulk模式
    grp, nam, meta_bar = read_meta(fn_meta, fn_meta_group; delim = fn_meta_delim)
    c == length(meta_bar) ||  throw(DimensionMismatch("The number of samples for metadata is not equal to the number of samples for the expression profile."))
    @time nmat, ngrp = (ncell_pseudo > 0) ?  generate_pseudobulk_kernel(mat, bar, grp, ncell_pseudo) : (copy(mat),bar .∈ (grp[1], ))
    # 过滤全为0的值
    if remove_zeros
      keep = reshape(sum(nmat, dims = 2), :) .> 0
      nmat = nmat[keep,:]
      fea = fea[keep]
    end
    if fn_candidate_gene != "no"
      candidate_gene = read_feature(fn_candidate_gene; fg_sit = fn_candidate_gene_fg_sit, delim = fn_candidate_gene_delim)
      l_cg = .!isnothing.(indexin(fea, candidate_gene))
      nmat = nmat[l_cg,:]
      fea = fea[l_cg]
    end
    # # 模式：特征基因选择；合并基因对。其中，mat_fea_01表示每个样本中各基因对的REO关系，1为大于，0为小于。
    # @time nmat_fea_01, features1, features2 = fgene_to_genepair_kernel(nmat, fea, fn_feature; mode_genepair_select, mode_gene_select, fn_feature_gene_sit, fn_feature_delim)
    # # 抽训练集和测试集
    # r,c,s = size(nmat_fea_01)
    # s_tt = n_train + n_test
    # @time train_test_set, true_ngrp = train_and_test_sample_kernel(s, ngrp, s_tt, t_hill_iter_num = t_hill_iter_num)
    # # 爬山法筛选特征基因对
    # @time gene_pair_01 = hill_climbing_method_kernel(nmat_fea_01, r, c, n_train, n_test, train_test_set, true_ngrp; t_hill_iter_num = t_hill_iter_num, t_train_iter_num = t_train_iter_num)
    # # 取出得分最高的前n_top个基因对各划分正确的样本数和各基因所在的行索引
    # @time min_score, n_top_gp = n_top_genepair(gene_pair_01, n_top)
    # return hcat.(features1[n_top_gp[:,1]], " > ", features2[n_top_gp[:,2]])
    omarker_fea, wmarker_fea = run_REO_molecular_signatures(nmat, fea, ngrp, bar,fn_stem, t_hill_iter_num, n_train, n_test, t_train_iter_num, fn_feature,mode_genepair_select, mode_gene_select, fn_feature_gene_sit, fn_feature_delim)
    omarker_feas = mapreduce(x -> [x[1] x[2] ">"],vcat,eachrow(omarker_fea))
    n_top <= (r*c - min(r,c)) ||  @info ("The number of features is less than $n_top for the specified output, so all features are output as a result.")
    writedlm(join([fn_stem, "oall_feas_all.tsv"], "_"), omarker_feas, "\t")
    writedlm(join([fn_stem, "omarker_feas.tsv"], "_"), omarker_feas[1:n_top,:], "\t")
    # marker_feas = vcat(["marker_fea"], mapreduce(x -> join(x," > "),vcat,eachrow(marker_fea)))
    wmarker_feas = mapreduce(x -> [x[1] x[2] ">"],vcat,eachrow(wmarker_fea))
    n_top <= (r*c - min(r,c)) ||  @info ("The number of features is less than $n_top for the specified output, so all features are output as a result.")
    writedlm(join([fn_stem, "wall_feas_all.tsv"], "_"), wmarker_feas, "\t")
    writedlm(join([fn_stem, "wmarker_feas.tsv"], "_"), wmarker_feas[1:n_top,:], "\t")
    # 保存组别信息
    writedlm(join([fn_stem, "group_order.tsv"], "_"), nam, "\t")
    # # 随机森林结果
    # (building_random_forests == "yes") ? classifier = run_REO_RandomForest(nmat, fea, DataFrame(reshape(Int.(.!ngrp),size(ngrp)[1],1) .+ 1,:auto), marker_fea[1:n_top,:], fn_stem) : return marker_feas
    # save("RandomForest_classifier.jld", "RandomForest_classifier", classifier)
    # # save(join([fn_stem, "RandomForest_classifier.jld"], "_"), "RandomForest_classifier", classifier)
    # # 随机森林模型运用到本数据集中
    # pre_randomforest = predict_sample_classification(classifier, nmat, fea, bar, marker_feas[1:n_top,:],fn_stem)
    # pre_randomforest_ct = ["" "True_group1" "True_group2"; "Pre_group1" length(intersect(pre_randomforest[(pre_randomforest[:,2] .== 1),1],grp[1])) length(intersect(pre_randomforest[(pre_randomforest[:,2] .== 1),1],grp[2])); "Pre_group2" length(intersect(pre_randomforest[(pre_randomforest[:,2] .== 2),1],grp[1])) length(intersect(pre_randomforest[(pre_randomforest[:,2] .== 2),1],grp[2]))]
    # @info "INFO: The prediction results of the trained random forest model in the training set are shown by 2 x 2 contingency tables: $(pre_randomforest_ct)"
    # #多数投票
    # pre_majorityvote = predict_sample_classification(nmat, fea, bar, omarker_feas[1:n_top,:],fn_stem)
    # pre_majorityvote_ct = ["" "True_group1" "True_group2"; "Pre_group1" length(intersect(pre_majorityvote[(pre_majorityvote[:,2] .== 1),1],grp[1])) length(intersect(pre_majorityvote[(pre_majorityvote[:,2] .== 1),1],grp[2])); "Pre_group2" length(intersect(pre_majorityvote[(pre_majorityvote[:,2] .== 2),1],grp[1])) length(intersect(pre_majorityvote[(pre_majorityvote[:,2] .== 2),1],grp[2]))]
    # @info "INFO: The prediction results of the majority voting rule in the training set are shown by 2 x 2 contingency tables: $(pre_majorityvote_ct)"
    # return marker_feas[1:n_top,:], classifier
    return omarker_feas[1:n_top,:], wmarker_feas[1:n_top,:]
    
    # marker_fea = run_REO_molecular_signatures(nmat, fea, ngrp, bar,fn_stem, t_hill_iter_num, n_train, n_test, t_train_iter_num, fn_feature,mode_genepair_select, mode_gene_select, fn_feature_gene_sit, fn_feature_delim)
    # marker_feas = mapreduce(x -> [x[1] x[2] ">"],vcat,eachrow(marker_fea))
    # n_top <= (r*c - min(r,c)) ||  @info ("The number of features is less than $n_top for the specified output, so all features are output as a result.")
    # writedlm(join([fn_stem, "all_feas_all.tsv"], "_"), marker_feas, "\t")
    # writedlm(join([fn_stem, "marker_feas.tsv"], "_"), marker_feas[1:n_top,:], "\t")
    # # marker_feas = vcat(["marker_fea"], mapreduce(x -> join(x," > "),vcat,eachrow(marker_fea)))
    # return marker_feas[1:n_top,:]
end

end
    

    