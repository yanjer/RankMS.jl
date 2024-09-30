## ReoMS.jl

Molecular marker identification and model construction based on gene pair relative Expression order (REOs).

It is based on julia language development, including molecular marker identification and model construction.

## Function

#### main function

Molecular marker gene pairs  and model construction are obtained by this function: 

`ReoMS_main(fn_expr::AbstractString = "matrix.mtx", rn_expr::AbstractString = "features.tsv", cn_expr::AbstractString = "barcodes.tsv", fn_meta::AbstractString = "fn_meta.txt", fn_feature::AbstractString = "fn_feature.txt"; building_random_forests::AbstractString = "yes", n_top::Int = 10,  n_train::Int = 13, n_test::Int = 3, t_hill_iter_num::Int = 500,  t_train_iter_num::Int = 15,  ncell_pseudo::Int = 0, fn_meta_delim::AbstractChar = '\t', fn_meta_group::AbstractString = "group", file_format_expr::AbstractString = "read_mtx",  T::Type = Int32, feature_col::Int = 2, barcode_col::Int = 1, feature_threshold::Int = 30, cell_threshold::Int = 200, fn_feature_gene_sit::Int = 1, fn_feature_delim::AbstractChar = '\t', mode_combine_genepair::AbstractString = "all_gene_combine_pair", mode_use_feature::AbstractString = "custom", remove_zeros::Bool = true, use_testdata::AbstractString = "no", work_dir::AbstractString = "./")`

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
-  `cell_threshold::Int`: the least number of genes that a cell must express, in order to be kept. Default: 200.
- `fn_feature_gene_sit::Int`: The column in the feature gene file where the feature gene resides. Default: 1.
- `fn_feature_delim::AbstractChar`: The separator of the characteristic gene file. Default: '\t'.
- `mode_combine_genepair::AbstractString`: Gene matching method: "all_gene_combine_pair", which means all gene pairs, "all_feature_gene_combine_pair" indicates the pairing of the feature gene with all genes except itself. "feature_gene_combine_pair" indicates the pairing of feature genes. Default: "all_gene_combine_pair".
- `mode_use_feature::AbstractString`: For the use of feature genes, "no" indicates that no feature genes are used, "custom" indicates that custom feature genes are used, and "DEGs_by_RankCompV3" indicates that DEGs (FDR &lt; 0.05) as a characteristic gene. Default: "custom".
- `remove_zeros::Bool`: # Filter values that are all zeros. Default: true.
- `use_testdata::AbstractString`: Whether to use test data. "yes "or "no". Default: "no".
- `work_dir::AbstractString`: Working directory. Default: "./".

### read_mtx

Read in the common 10X single-cell RNA expression file in the MTX format (unzipped).

`read_mtx(fn::AbstractString, rn::AbstractString, cn::AbstractString; T::Type = Int32, feature_col::Int = 2, barcode_col::Int = 1)`

The parameters are:

- `fn::AbstractString`: MTX file path.
- `rn::AbstractString`: features file path.
- `cn::AbstractString`: barcodes file path.
- `T::Type`: Datatype in the MTX file. Default: Int32.
- `feature_col::Int`: which column is used as feature names. Default: 1 (first).
- `barcode_col::Int`: which column is used as barcode names. Default: 1 (first).

### read_expr_matrix

Read in an expression matrix stored in `fn` where its row names are stored in `rn` and column names are stored in `cn`.

It returns (matrix, vector of row names, vector of column names) 

`read_expr_matrix(fn::AbstractString,rn::AbstractString, cn::AbstractString)`

The parameters are:

- `fn::AbstractString`: Expression matrix file path, with rows representing genes and columns representing samples, does not require column names.
- `rn::AbstractString`: Gene list file path. 
- `cn::AbstractString`: Sample name list file path. 

### read_meta

Read in a meta data file with the first row assumed to be the header and the row names assumed to be the profile names (cell barcodes).

Grouping information is specified by the column with the header name of `group`. If `group` is not found, the second column will be used.

It returns the grouped profile names (vector of vectors) and group names.

`read_meta(fn::AbstractString, group::AbstractString = "group"; delim::AbstractChar = '\t')`

The parameters are:

- `fn::AbstractString`: Metadata file path, the first column sample name, the second column group information. (Default = "fn_meta.txt".
- `group::AbstractString`: Specifies the column name of the group information in metadata. (Default: "group".
- `delim::AbstractChar`: Delimiter of the metadata file. (Default: = '\t')

### filter_expr_matrix

Filter an expression matrix `mat`, only keep those genes expressed in greater than `feature_threshold` cells and cells expressing greater than `cell_threshold` features.

Return the filtered matrix and the bit vectors for keeping features and cells.

`filter_expr_matrix(mat::AbstractMatrix, feature_threshold::Int=30, cell_threshold::Int=200)`

The parameters are:

- `mat::AbstractMatrix`: expression matrix (either dense or sparse).
- `feature_threshold::Int`: the least number of cells that a feature must express in, in order to be kept. Default: 30.
-  `cell_threshold::Int`: the least number of genes that a cell must express, in order to be kept. Default: 200.

### generate_pseudobulk_kernel

Generate a matrix of pseudobulk profiles from `mat` which stores single-cell RNA profiles. Each column represents a cell's profile. Each pseudobulk profile is generated from `np` (default: 10) single-cell profiles.

`generate_pseudobulk_kernel(mat::AbstractMatrix, np::Int = 10)`

The parameters are:

- `mat::AbstractMatrix`: Each column is a profile.
- `np::Int`: Number of profiles in each pseudobulk profile. Default: 10.

### run_REO_molecular_signatures

#### 

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

### run_REO_RandomForest

Train a random forest model.

`run_REO_RandomForest(mat::AbstractMatrix, fea::Vector, Y::DataFrame, marker_genepair::Matrix, fn_stem::String)`

The parameters are:

- `nmat::AbstractMatrix`: Expression profiles matrix.
- `fea::Vector`: Gene list.
- `fea::Vector`: Sample grouping information, 1 or 0.
- `marker_genepair::Matrix`: The list of marker gene pairs consists of two columns, the first is gene a, the second is gene b, and the relationship between them is a &gt; b.
- `fn_stem::String`: File name prefix.
