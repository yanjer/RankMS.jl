# module REO_molecular_markers

export run_REO_molecular_markers, 
       hill_climbing_method_kernel,
       fgene_to_genepair_kernel

include("code/gene_and_genepair_select.jl")

include("code/train_and_test_sample.jl")

include("code/hill_climbing_method.jl")

include("code/roc.jl")


"""
  Identification of molecular markers based on REO.
  run_REO_molecular_markers(fn_expr, rn_expr, cn_expr, fn_meta)

Test with testdata.
```jldoctest
julia> @time run_REO_molecular_markers(use_testdata = "yes")
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
julia> run_REO_molecular_markers("matrix.mtx", "features.tsv", "barcodes.tsv", "fn_meta.txt", ncell_pseudo = 50)
```

Example
```jldoctest
julia> run_REO_molecular_markers(nmat, fea, ngrp, t_hill_iter_num, n_train, n_test, t_train_iter_num, n_top, fn_feature)
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
run_REO_molecular_markers(nmat::AbstractMatrix, # Expression profiles matrix 
                                   fea::Vector,
                                  ngrp::BitVector,
                       t_hill_iter_num::Int = 500,  # Hill climbing number of iterations.
                               n_train::Int = 13, # Number of training set samples.
                                n_test::Int = 3,  # Number of test set samples.
                      t_train_iter_num::Int = 15,  # The number of iterations of the climb training.
                                 n_top::Int = 10, # The top n_top genes with the highest scores.
                            fn_feature = "fn_feature.txt",
                 mode_genepair_select::AbstractString = "all_gene_pair", # "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair"
                      mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                   fn_feature_gene_sit::Int = 1,
                      fn_feature_delim::AbstractChar = '\t')
```

The parameters are:

- `nmat::AbstractMatrix`: Expression profiles matrix.
- `fea::Vector`: Gene list.
- `ngrp::BitVector`: Sample grouping information, 1 or 0.
- `bar::Vector`: Sample group name.
- `fn_stem::AbstractString`: Filename.
- `t_hill_iter_num::Int`: Hill climbing number of iterations. Default: 500.
- `n_train::Int`: Number of training set samples. Default: 13.
- `n_test::Int`: Number of test set samples. Default: 3.
- `t_train_iter_num::Int`: The number of iterations of the climb training. Default: 15.
- `n_top::Int`: The top n_top genes with the highest scores. Default: 10.
- `fn_feature::AbstractString`: Feature file name (optional). Default: "fn_feature.txt".
- `mode_genepair_select::AbstractString`: "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair". Default: "all_gene_pair".
- `mode_gene_select::AbstractString`: "no", "custom", "DEGs_by_RankCompV3". Default: "custom".
- `fn_feature_gene_sit::Int`: The column in which the feature gene is located. Default: 1.
- `fn_feature_delim::AbstractChar`: The separator of the characteristic gene file. Default: '\t'.
"""

function run_REO_molecular_markers(nmat::AbstractMatrix, # Expression profiles matrix 
                                   fea::Vector,
                                  ngrp::BitVector,
                                  bar::Vector,
                                  fn_stem::AbstractString,
                       t_hill_iter_num::Int = 500,  # Hill climbing number of iterations.
                               n_train::Int = 13, # Number of training set samples.
                                n_test::Int = 3,  # Number of test set samples.
                      t_train_iter_num::Int = 15,  # The number of iterations of the climb training.
                            fn_feature::AbstractString = "fn_feature.txt",
                 mode_genepair_select::AbstractString = "all_gene_pair", # "all_gene_pair", "all_feature_gene_pair", "feature_gene_pair"
                      mode_gene_select::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                   fn_feature_gene_sit::Int = 1,
                      fn_feature_delim::AbstractChar = '\t')
    # 模式：特征基因选择；合并基因对。其中，mat_fea_01表示每个样本中各基因对的REO关系，1为大于，0为小于。
    @time nmat_fea_01, features1, features2, s_fl = fgene_to_genepair_kernel(nmat, fea, ngrp, bar, fn_feature, fn_stem; mode_genepair_select, mode_gene_select, fn_feature_gene_sit, fn_feature_delim)
    # 抽训练集和测试集
    r,c,s = size(nmat_fea_01)
    # 一次性抽取训练集和测试集（不分组别且不放回抽样）
    # s_tt = n_train + n_test
    # @time train_test_set, true_ngrp = train_and_test_sample_kernel(s, ngrp, s_tt, t_hill_iter_num = t_hill_iter_num)
    @time train_test_set, true_ngrp, l_train, l_test = train_and_test_sample_kernel(s, ngrp, n_train, n_test, t_hill_iter_num = t_hill_iter_num)
    # 爬山法筛选特征基因对
    if t_train_iter_num > (r*c - min(r,c))
      @info ("The threshold of the number of iterations for each sample of the mountain climbing method is $t_train_iter_num, which is greater than the total number of features $(r*c - min(r,c)), so the number of features is taken as the upper limit.")
      t_train_iter_num = (r*c - min(r,c))
    end
    @time mat_first_fea_01, mat_all_fea_01, list_sample_auc = hill_climbing_method_kernel(nmat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, t_hill_iter_num, t_train_iter_num)
    # 从2000次迭代中抽500次
    # original
    scores, fea = sampling(mat_first_fea_01, list_sample_auc, mat_all_fea_01; sp_choice = 0)
    f1 = fea .% r
    f1[f1 .== 0] .= c
    f2 = fea .÷ r
    f2[f2 .== 0] .= 1
    ffo = hcat(features1[f1], features2[f2])
    # weight
    scores, fea = sampling(mat_first_fea_01, list_sample_auc, mat_all_fea_01; sp_choice = 1)
    f1 = fea .% r
    f1[f1 .== 0] .= c
    f2 = fea .÷ r
    f2[f2 .== 0] .= 1
    ffw = hcat(features1[f1], features2[f2])
    return ffo, ffw
    # 返回特征基因对(REO都为">"关系)。
    # f1 = fea .% r
    # f1[f1 .== 0] .= c
    # f2 = fea .÷ r
    # f2[f2 .== 0] .= 1
    # return hcat(features1[f1], features2[f2])
end

    

    