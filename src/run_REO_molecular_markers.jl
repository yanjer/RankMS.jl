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
    scores, l_fea, ss = sampling(mat_first_fea_01, list_sample_auc, mat_all_fea_01; sp_choice = 0)
    writedlm(join([fn_stem, "osample_local.tsv"], "_"), ss, "\t")
    f1 = l_fea .% r
    f2 = (l_fea .- 1) .÷ r .+ 1
    f1[f1 .== 0] .= r
    ffo = hcat(features1[f1], features2[f2])
    # weight
    scores, l_fea, ss = sampling(mat_first_fea_01, list_sample_auc, mat_all_fea_01; sp_choice = 1)
    writedlm(join([fn_stem, "wsample_local.tsv"], "_"), ss, "\t")
    f1 = l_fea .% r
    f2 = (l_fea .- 1) .÷ r .+ 1
    f1[f1 .== 0] .= r
    ffw = hcat(features1[f1], features2[f2])
    # 保存文件
    mat_first_fea_01 = mat_first_fea_01[setdiff(1:end, [1:r...].^2),:]
    mat_all_fea_01 = mat_all_fea_01[setdiff(1:end, [1:r...].^2),:]
    writedlm(join([fn_stem, "mat_first_fea_01.tsv"], "_"), mat_first_fea_01, "\t")
    writedlm(join([fn_stem, "mat_all_fea_01.tsv"], "_"), mat_all_fea_01, "\t")
    writedlm(join([fn_stem, "list_sample_auc.tsv"], "_"), list_sample_auc, "\t")
    return ffo[.!(ffo[:,1] .== ffo[:,2]),:], ffw[.!(ffw[:,1] .== ffw[:,2]),:]
    # 返回特征基因对(REO都为">"关系)。
    # f1 = l_fea .% r
    # f1[f1 .== 0] .= c
    # f2 = l_fea .÷ r
    # f2[f2 .== 0] .= 1
    # return hcat(features1[f1], features2[f2])


    # # # FOLDS
    # s_sample = readdlm("/public/yanj/jupyter_work/molecular_marker/idea_1_climbing_method/IMPRES-codes-master/MAIN_CODES/Feature_selection/Sample_training_set/FOLDS_sample.txt", '\t', header = false)
    # [s_sample[(s_sample .== i[1])] .= i[2] for i in eachrow(Int64.(hcat(sort(unique(s_sample)),[1:108...])))]
    # train_test_set = [Int64.(vec(i)) for i in eachcol(s_sample)]
    # if t_train_iter_num > (r*c - min(r,c))
    #   @info ("The threshold of the number of iterations for each sample of the mountain climbing method is $t_train_iter_num, which is greater than the total number of features $(r*c - min(r,c)), so the number of features is taken as the upper limit.")
    #   t_train_iter_num = (r*c - min(r,c))
    # end
    # t_hill_iter_num=500
    # @time mat_first_fea_01, mat_all_fea_01, list_sample_auc = hill_climbing_method_kernel(nmat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, t_hill_iter_num, t_train_iter_num)
    # writedlm(join([fn_stem, "s500_mat_first_fea_01.tsv"], "_"), mat_first_fea_01, "\t")
    # writedlm(join([fn_stem, "s500_mat_all_fea_01.tsv"], "_"), mat_all_fea_01, "\t")
    # writedlm(join([fn_stem, "s500_list_sample_auc.tsv"], "_"), list_sample_auc, "\t")
    # scores = sum(mat_all_fea_01[:,(list_sample_auc .>= 0.6)],dims=2) .- sum(mat_all_fea_01[:,(list_sample_auc .<= 0.4)],dims=2)
    # o = sortperm(vec(scores), rev = true)
    # scores = scores[o]
    # l_fea = (1:size(mat_all_fea_01)[1])[o]
    # f1 = l_fea .% r
    # f2 = (l_fea .- 1) .÷ r .+ 1
    # f1[f1 .== 0] .= r
    # ff = hcat(features1[f1], features2[f2])
    # # ff_all = mapreduce(x -> hcat.(features2,x),vcat,features1)
    # # ff_all = ff_all[o]
    # # ff = ff_all[l_fea,:]
    # feas = ff[.!(ff[:,1] .== ff[:,2]),:]
    # return feas
    # writedlm(join([fn_stem, "all_feas_all.tsv"], "_"), feas, "\t")
    # writedlm(join([fn_stem, "marker_feas.tsv"], "_"), feas[1:n_top,:], "\t")


    ## 测试和原本结果是否一致，使用和文章中完全一样的样本
    # s_sample = readdlm("/public/yanj/jupyter_work/molecular_marker/idea_1_climbing_method/outcome_REO_molecular_markers/data/train_NEUB/pool_sample.txt", '\t', header = false)
    # s_500index = readdlm("/public/yanj/jupyter_work/molecular_marker/idea_1_climbing_method/outcome_REO_molecular_markers/data/train_NEUB/500index.txt", '\t', header = false)
    # [s_500index[(s_500index .== i[1])] .= i[2] for i in eachrow(Int64.(hcat(sort(unique(s_sample)),[1:108...])))]
    # s_500index = vec(Int64.(s_500index))
    # [s_sample[(s_sample .== i[1])] .= i[2] for i in eachrow(Int64.(hcat(sort(unique(s_sample)),[1:108...])))]
    # train_test_set = [Int64.(vec(i)) for i in eachcol(s_sample)]
    # if t_train_iter_num > (r*c - min(r,c))
    #   @info ("The threshold of the number of iterations for each sample of the mountain climbing method is $t_train_iter_num, which is greater than the total number of features $(r*c - min(r,c)), so the number of features is taken as the upper limit.")
    #   t_train_iter_num = (r*c - min(r,c))
    # end
    # @time mat_first_fea_01, mat_all_fea_01, list_sample_auc = hill_climbing_method_kernel(nmat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, t_hill_iter_num, t_train_iter_num)
    # writedlm(join([fn_stem, "s2000_mat_first_fea_01.tsv"], "_"), mat_first_fea_01, "\t")
    # writedlm(join([fn_stem, "s2000_mat_all_fea_01.tsv"], "_"), mat_all_fea_01, "\t")
    # writedlm(join([fn_stem, "s2000_list_sample_auc.tsv"], "_"), list_sample_auc, "\t")
    # s500_list_sample_auc = list_sample_auc[s_500index]
    # s500_mat_first_fea_01 = mat_first_fea_01[:,s_500index]
    # s500_mat_all_fea_01 = mat_all_fea_01[:,s_500index]
    # writedlm(join([fn_stem, "s500_mat_first_fea_01.tsv"], "_"), s500_mat_first_fea_01, "\t")
    # writedlm(join([fn_stem, "s500_mat_all_fea_01.tsv"], "_"), s500_mat_all_fea_01, "\t")
    # writedlm(join([fn_stem, "s500_list_sample_auc.tsv"], "_"), s500_list_sample_auc, "\t")
    # scores = sum(s500_mat_all_fea_01[:,(s500_list_sample_auc .>= 0.6)],dims=2) .- sum(s500_mat_all_fea_01[:,(s500_list_sample_auc .<= 0.4)],dims=2)
    # o = sortperm(vec(scores), rev = true)
    # scores = scores[o]
    # l_fea = (1:size(mat_all_fea_01)[1])[o]
    # f1 = l_fea .% r
    # f2 = l_fea .÷ r
    # f2[(f1 .== 0) .&& (f2 .!=0 )] .-= 1
    # f2[f2 .== 0] .= 1
    # f1[f1 .== 0] .= r
    # ff = hcat(features1[f1], features2[f2])
    # return ff[.!(ff[:,1] .== ff[:,2]),:]


end

    

    