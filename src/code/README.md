# run_REO_molecular_signatures.jl

### Molecular marker recognition based on relative expression order of gene pairs (REOs).

It is based on julia language development, including feature selection, feature combination, molecular marker gene pair selection and other modules.

## Function

#### main function

Molecular marker gene pairs are obtained by this function: 

`run_REO_molecular_signatures(nmat::AbstractMatrix, fea::Vector, ngrp::BitVector, bar::Vector; fn_stem::AbstractString, t_hill_iter_num::Int = 500, n_train::Int = 13,  n_test::Int = 3, t_train_iter_num::Int = 15, n_top::Int = 10, fn_feature::AbstractString = "fn_feature.txt", mode_combine_genepair::AbstractString = "all_gene_combine_pair", mode_use_feature::AbstractString = "custom", fn_feature_gene_sit::Int = 1, fn_feature_delim::AbstractChar = '\t')`

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
- `mode_combine_genepair::AbstractString`: "all_gene_combine_pair", "all_feature_gene_combine_pair", "feature_gene_combine_pair". Default: "all_gene_combine_pair".
- `mode_use_feature::AbstractString`: "no", "custom", "DEGs_by_RankCompV3". Default: "custom".
- `fn_feature_gene_sit::Int`: The column in which the feature gene is located. Default: 1.
- `fn_feature_delim::AbstractChar`: The separator of the characteristic gene file. Default: '\t'.

#### fgene_to_combine_genepair_kernel

Feature gene pairs are paired through this function: 

`fgene_to_combine_genepair_kernel(mat::AbstractMatrix, fea::Vector, ngrp::BitVector, bar::Vector, fn_feature::AbstractString, fn_stem::AbstractString; mode_combine_genepair::AbstractString = "all_gene_combine_pair", mode_use_feature::AbstractString = "custom", fn_feature_gene_sit::Int = 1, fn_feature_delim::AbstractChar = '\t')`

The parameters are: 

- `mat::AbstractMatrix`: Expression profiles matrix.
- `fea::Vector`: Gene list.
- `ngrp::BitVector`: Sample grouping information, 1 or 0.
- `bar::Vector`: Sample group name.
- `fn_feature::AbstractString`: Feature file name (optional).
- `fn_stem::AbstractString`: Filename.
- `mode_combine_genepair::AbstractString`: "all_gene_combine_pair", "all_feature_gene_combine_pair", "feature_gene_combine_pair". Default: "all_gene_combine_pair".
- `mode_use_feature::AbstractString`: "no", "custom", "DEGs_by_RankCompV3". Default: "custom".
- `fn_feature_gene_sit::Int`: The column in which the feature gene is located. Default: 1.
- `fn_feature_delim::AbstractChar`: The separator of the characteristic gene file. Default: '\t'.

### train_and_test_sample_kernel

Randomly extract the training set and the validation set by this function:

`train_and_test_sample_kernel(n_sample::Int, ngrp::BitVector, n_train_test::Int; t_hill_iter_num::Int = 500)`

The parameters are: 

- `n_sample`: Total number of samples.
- `ngrp::BitVector`: Sample grouping information, 1 or 0.
- `n_train_test::Int`: Number of training set and test set samples.
- `t_hill_iter_num::Int`: Sample extraction times. Default: 500.

### hill_climbing_method_kernel

Mountain climbing method to train molecular marker gene pairs, through this function:

`hill_climbing_method_kernel(mat_genepair_01::BitArray, r::Int, c::Int, n_train::Int, n_test::Int, train_test_set::Vector, true_ngrp::Vector; t_hill_iter_num::Int = 500, gene_pair_01=zeros(Int,r,c), t_train_iter_num::Int = 15)`

The parameters are:

- `mat_genepair_01::BitArray`: For the REO relationship of gene pairs in each sample, 1 is greater than and 0 is less than.
- `r::Int`: Number of lines of expression spectrum.
- `c::Int`: The number of columns of the expression profile.
- `n_train::Int`: Number of training set samples.
- `n_test::Int`: Number of test set samples.
- `train_test_set::Vector`: Sample collection of randomly selected training set and test set.
- `true_ngrp::Vector`: Real grouping label for a sample set of randomly selected training and test sets.
- `t_hill_iter_num::Int`: Hill climbing number of iterations. Default: 500.
- `t_train_iter_num::Int`: The number of iterations of the climb training. Default: 15.

### n_top_genepair

The top gene pair with the highest score was extracted.

`n_top_genepair(matr::Matrix, n_top::Int, gene_pair::Matrix = [0 0])`

The parameters are:

- `matr::Matrix`: The score for each gene pair, behavioral gene 1, is listed as gene 2.
- `n_top::Int`: Number of top gene pairs extracted.

## Test data sets

`REO_molecular_signatures_main(use_testdata = "yes")`

## Dependencies

`DataFrame`, `CSV`,  `Random`
