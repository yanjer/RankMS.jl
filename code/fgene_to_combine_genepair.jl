export fgene_to_combine_genepair_kernel

include(joinpath(@__DIR__, "..", "code", "gene_combine_genepair.jl"))

function fgene_to_combine_genepair_kernel(
                    mat::AbstractMatrix, # Expression profiles matrix 
                    fea::Vector,
                    fn_feature::AbstractString;
                    mode_combine_genepair::AbstractString = "all_gene_combine_pair", # "all_gene_combine_pair", "all_feature_gene_combine_pair", "feature_gene_combine_pair"
                         mode_use_feature::AbstractString = "custom", # "no", "custom", "DEGs_by_RankCompV3".
                         fn_feature_gene_sit::Int = 1,
                   fn_feature_delim::AbstractChar = '\t'
                   )
    if mode_combine_genepair == "all_gene_combine_pair"
        mat_genepair_01 = all_gene_combine_pair(mat)
        return mat_genepair_01, fea
    else
        if mode_use_feature == "custom"
            features = read_feature_genes(fn_feature; fg_sit = fn_feature_gene_sit, delim = fn_feature_delim)
        else
            println("DEGs方法还未写入")
            features = fea[1:5]
        end
            f_l = map(x -> ∈(x, features), fea)
            f_n = sum(f_l)
            mat_genepair_01 = ((f_n == 0) ? all_gene_combine_pair(mat) : ((mode_combine_genepair == "all_feature_gene_combine_pair") ?  all_feature_gene_combine_pair(mat, f_l, f_n) : feature_gene_combine_pair(mat, f_l, f_n)))
        (mode_combine_genepair == "all_feature_gene_combine_pair") ? fea = fea[.!f_l] : fea
        return mat_genepair_01, features
    end
end