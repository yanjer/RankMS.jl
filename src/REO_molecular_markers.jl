module REO_molecular_markers

export REO_molecular_markers_main, 
       hill_climbing_method_kernel,
       fgene_to_combine_genepair_kernel,
       generate_pseudobulk_kernel,
       filter_expr_matrix,
       read_mtx, read_gmt, read_meta

include("code/read_file.jl")

include("code/process_expr.jl")

include("code/gene_combine_genepair.jl")

include("code/train_and_test_sample.jl")

include("code/hill_climbing_method.jl")



"""
  Identification of molecular markers based on REO.
  REO_molecular_markers_main(fn_expr, rn_expr, cn_expr, fn_meta)

Test with testdata.
```jldoctest
julia> @time REO_molecular_markers_main(use_testdata = "yes")
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
10×3 Matrix{AbstractString}:
 "HHAT"    ">"  "LINC01409"
 "TSPAN2"  ">"  "LINC01409"
 "TNN"     ">"  "LINC01409"
 "ATP5F1"  ">"  "LINC01409"
 "MIR553"  ">"  "LINC01409"
 "HHAT"    ">"  "NOC2L"
 "TSPAN2"  ">"  "NOC2L"
 "TNN"     ">"  "NOC2L"
 "ATP5F1"  ">"  "NOC2L"
 "MIR553"  ">"  "NOC2L"
```
Psudo-bulk mode
```jldoctest
julia> REO_molecular_markers_main("matrix.mtx", "features.tsv", "barcodes.tsv", "fn_meta.txt", ncell_pseudo = 50)
```

Example
```jldoctest
julia> REO_molecular_markers_main("matrix.mtx", "features.tsv", "barcodes.tsv", "fn_meta.txt")
1.077946 seconds (3.17 M allocations: 147.227 MiB, 3.35% gc time, 94.54% compilation time)
[ Info: INFO: The size of expression profile was (36602, 8).
  1.303662 seconds (4.50 M allocations: 230.460 MiB, 3.49% gc time, 97.25% compilation time)
[ Info: INFO: The filtered of expression profile size was (7549, 8).
  0.131077 seconds (248.33 k allocations: 13.187 MiB, 99.57% compilation time)
  0.027984 seconds (452.95 k allocations: 27.934 MiB)
  0.778336 seconds (2.59 M allocations: 137.744 MiB, 3.14% gc time, 95.98% compilation time: 7% of which was recompilation)
................................................................................................................................
 35.945972 seconds (278.36 M allocations: 50.403 GiB, 13.91% gc time, 6.13% compilation time)
 41.118347 seconds (293.45 M allocations: 51.157 GiB, 12.54% gc time, 17.61% compilation time: 6% of which was recompilation)
10×3 Matrix{AbstractString}:
 "TNN"     ">"  "HES4"
 "HHAT"    ">"  "SDF4"
 "HHAT"    ">"  "LINC01409"
 "ATP5F1"  ">"  "HES4"
 "TSPAN2"  ">"  "SDF4"
 "TNN"     ">"  "NOC2L"
 "MIR553"  ">"  "HES4"
 "HHAT"    ">"  "HMGN2"
 "HHAT"    ">"  "ARID1A"
 "HHAT"    ">"  "ZDHHC18"
```
All parameters.
```jldoctest
REO_molecular_markers_main(fn_expr::AbstractString = "matrix.mtx",
                          rn_expr::AbstractString = "features.tsv",
                          cn_expr::AbstractString = "barcodes.tsv",
                          fn_meta::AbstractString = "fn_meta.txt",
                       fn_feature::AbstractString = "fn_feature.txt";
                                       n_top::Int = 10, # The top n_top genes with the highest scores.
                                     n_train::Int = 13, # Number of training set samples.
                                      n_test::Int = 3,  # Number of test set samples.
                             t_hill_iter_num::Int = 500,  # Hill climbing number of iterations.
                            t_train_iter_num::Int = 128,  # The number of iterations of the climb training.
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
                   mode_combine_genepair::AbstractString = "all_gene_combine_pair", # "all_gene_combine_pair", "all_feature_gene_combine_pair", "feature_gene_combine_pair"
                 mode_use_feature::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                               remove_zeros::Bool = true,
                     use_testdata::AbstractString = "no",
                         work_dir::AbstractString = "./")
```
"""
function REO_molecular_markers_main(fn_expr::AbstractString = "matrix.mtx",
                          rn_expr::AbstractString = "features.tsv",
                          cn_expr::AbstractString = "barcodes.tsv",
                          fn_meta::AbstractString = "fn_meta.txt",
                       fn_feature::AbstractString = "fn_feature.txt";
                                       n_top::Int = 10, # The top n_top genes with the highest scores.
                                     n_train::Int = 13, # Number of training set samples.
                                      n_test::Int = 3,  # Number of test set samples.
                             t_hill_iter_num::Int = 500,  # Hill climbing number of iterations.
                            t_train_iter_num::Int = 128,  # The number of iterations of the climb training.
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
                   mode_combine_genepair::AbstractString = "all_gene_combine_pair", # "all_gene_combine_pair", "all_feature_gene_combine_pair", "feature_gene_combine_pair"
                 mode_use_feature::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
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
        mode_combine_genepair = "all_feature_gene_combine_pair"
        feature_threshold = 1;cell_threshold = 1;n_train = 2;n_test = 2;t_hill_iter_num = 2;t_train_iter_num=30
    end
    @time mat, fea, bar = (file_format_expr == "read_mtx") ? read_mtx(fn_expr, rn_expr, cn_expr; T, feature_col, barcode_col) : read_expr_matrix(fn_expr, rn_expr, cn_expr)
    @info "INFO: The size of expression profile was $(size(mat))."
    # 过滤表达谱
    @time mat, kf, kb = filter_expr_matrix(mat, feature_threshold, cell_threshold)
    @info "INFO: The filtered of expression profile size was $(size(mat))."
    fea = fea[kf]
    bar = bar[kb]
    # pseudobulk模式
    grp, nam = read_meta(fn_meta, fn_meta_group; delim = fn_meta_delim)
    @time nmat, ngrp = (ncell_pseudo > 0) ?  generate_pseudobulk_kernel(mat, bar, grp, ncell_pseudo) : (copy(mat),bar .∈ (grp[1], ))
    # 过滤全为0的值
    if remove_zeros
      keep = reshape(sum(nmat, dims = 2), :) .> 0
      nmat = nmat[keep,:]
      fea = fea[keep]
    end
    # 模式：特征基因选择；合并基因对
    @time nmat_genepair_01, features = fgene_to_combine_genepair_kernel(nmat, fea, fn_feature; mode_combine_genepair, mode_use_feature, fn_feature_gene_sit, fn_feature_delim)
    # 抽训练集和测试集
    r,c,s = size(nmat_genepair_01)
    s_tt = n_train + n_test
    @time train_test_set = train_and_test_sample_kernel(s, ngrp, s_tt, t_hill_iter_num = t_hill_iter_num)
    # 爬山法筛选特征基因对
    @time gene_pair_01 = hill_climbing_method_kernel(nmat_genepair_01, vcat(trues(s_tt),falses(s_tt)), r, c, n_train, n_test, train_test_set; t_hill_iter_num = t_hill_iter_num, t_train_iter_num = t_train_iter_num)
    # 取出得分最高的前n_top个索引
    @time min_score, n_top_gp = n_top_genepair(gene_pair_01, n_top)
    return ((mode_combine_genepair == "all_gene_combine_pair") ? [fea[n_top_gp[:,1]] ((m_sign=copy(features[n_top_gp[:,1]])) .=">") fea[n_top_gp[:,2]]] : [features[n_top_gp[:,1]] ((g_sign=features[n_top_gp[:,1]]) .=">") fea[n_top_gp[:,2]]])
end

end
    

    