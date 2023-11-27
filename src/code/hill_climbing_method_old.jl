export hill_climbing_method_kernel, n_top_fea

# using DataFrames

include("compare_genepair.jl")

include("roc.jl")

"""
The top gene pair with the highest score was extracted.

```jldoctest
         n_top_fea(matr::Matrix,
                        n_top::Int,
                        gene_pair::Matrix = [0 0])
```
The parameters are:
- `matr::Matrix`: The score for each gene pair, behavioral gene 1, is listed as gene 2.
- `n_top::Int`: Number of top gene pairs extracted.
"""
# AUC最大的基因对为多个时，选取最大的第一个基因对。
function n_top_fea(matr::Matrix,
                        n_top::Int,
                        s_fl::Int = 0,
                        gene_pair::Matrix = [0 0])
    r, c = size(matr)
    score, ind = findmax(matr)
    if (score == 0)
        gene_pair != [0 0] ||  throw(DimensionMismatch("All gene pairs have scores less than or equal to 0, so there are no marker gene pairs."))
        @info "INFO: The gene pairs with scores greater than 0 are less than $(n_top), so the gene pairs with scores greater than 0 are output as marker gene pairs."
        return score,gene_pair[2:end,:]
    end
    g1,g2 = convert(Tuple, ind)
    matr[g1,g2] = 0
    matr[g2,g1] = 0
    if (s_fl != 0 && (g1 > s_fl) && (g2 > s_fl))
        (return n_top_fea(matr, n_top, s_fl, gene_pair))
    end
    gene_pair = vcat(gene_pair, [g1 g2])
    n_top -= 1
    (n_top == 0) ? (return score,gene_pair[2:end,:]) : (return n_top_fea(matr, n_top, s_fl, gene_pair))
end

# AUC最大的基因对为多个时，随机抽取基因对。
function n_top_fea(matr::Matrix,
                        n_top::Int,
                        s_fl::Int = 0,
                        gene_pair::Matrix = [0 0])
    r, c = size(matr)
    score, ind = findmax(matr)
    if (score == 0)
        gene_pair != [0 0] ||  throw(DimensionMismatch("All gene pairs have scores less than or equal to 0, so there are no marker gene pairs."))
        @info "INFO: The gene pairs with scores greater than 0 are less than $(n_top), so the gene pairs with scores greater than 0 are output as marker gene pairs."
        return score,gene_pair[2:end,:]
    end
    max_num = findall(matr .== score)
    if size(max_num)[1] >= 2
        ind = rand(max_num)
    end
    g1,g2 = convert(Tuple, ind)
    matr[g1,g2] = 0
    matr[g2,g1] = 0
    if (s_fl != 0 && (g1 > s_fl) && (g2 > s_fl))
        (return n_top_fea(matr, n_top, s_fl, gene_pair))
    end
    gene_pair = vcat(gene_pair,  [g1 g2])
    # matr[g1,g2] = 0
    # if ((g2 <= r) && (g1 <= c))
    #     matr[g2,g1] = 0
    # end
    n_top -= 1
    (n_top == 0) ? (return score,gene_pair[2:end,:]) : (return n_top_fea(matr, n_top, s_fl, gene_pair))
end

"""
    Mountain climbing method using majority voting rule model is used to identify feature gene pairs.

    hill_climbing_method_kernel(
                                mat_fea_01::BitArray, # The 01 matrix of gene pairs
                                r::Int,
                                c::Int,
                                n_train::Int,
                                n_test::Int,
                                train_test_set::Vector,
                                true_ngrp::Vector;
                                t_hill_iter_num::Int = 500,
                                gene_pair_01=zeros(Int,r,c),
                                t_train_iter_num::Int = 15)

Examples
```jldoctest
julia> (mat_fea_01 = falses(3,4,3))[3,1,:] .= [1,1,1]; mat_fea_01[1,2,:] .= [1,1,1];mat_fea_01
3×4×3 BitArray{3}:
[:, :, 1] =
 0  1  0  0
 0  0  0  0
 1  0  0  0

[:, :, 2] =
 0  1  0  0
 0  0  0  0
 1  0  0  0

[:, :, 3] =
 0  1  0  0
 0  0  0  0
 1  0  0  0

 julia> true_group_01 = BitVector([1,0,1])
 3-element BitVector:
  1
  0
  1

julia> r
3

julia> c
4

julia> hill_climbing_method_kernel(mat_fea_01,true_group_01,r,c)
.................................................................................................................................
130×2 Matrix{Int64}:
 3  1
 1  1
 2  1
 1  2
 1  3
 1  4
 1  1
 ⋮
 1  1
 1  1
 1  1
 1  1
 1  1
 1  1

julia> hill_climbing_method_kernel(mat_fea_01,BitVector([1,0,1]),r,c;t_train_iter_num = 5)
......
7×2 Matrix{Int64}:
 3  1
 1  1
 2  1
 1  2
 1  3
 1  4
 1  1

```
"""

# function trainset_hill_climbing(
#         mat_fea_01::BitArray, # The 01 matrix of gene pairs
#         true_group_01::BitVector,
#         r::Int,
#         c::Int,
#         s_fl::Int;
#         t_train_iter_num::Int = 15,
#         gene_pair::Matrix = [1 1],
#         f_fea_01::Matrix = [0 0],
#         iter_num::Number = 0)
#     if iter_num == 0
#         # f_auc_max,f_max_auc_fea = findmax(sum(mat_fea_01[:,:,:],dims = 3))
#         # g1,g2 = convert(Tuple, f_max_auc_fea)
#         # # g1,g2 = (sum(mat_fea_01[g1,g2,:]) > s/2) ? (g1,g2) : (g2,g1)
#         # gene_pair = [g1 g2]
#         # f_fea_01 = reshape(mapreduce(x->reshape([mat_fea_01[n,x,:] .+ mat_fea_01[g1,g2,:] for n in 1:r],:,1), hcat, [1:c...]),1,:)
#         f_fea_01 = reshape(mapreduce(x->reshape([mat_fea_01[n,x,:] for n in 1:r],:,1), hcat, [1:c...]),1,:)
#     else
#         g1,g2=gene_pair[iter_num + 1,:]
#         f_fea_01 = mapreduce(x-> [x .+ mat_fea_01[g1,g2,:]], hcat, f_fea_01)
#     end
#     # # 按照多数投票规则，当比例大于0.5则赋值1；小于0.5赋值0；等于0.5随机赋值1或0
#     # sample_actual_label = mapreduce(x -> [is_greater.(x ./ (iter_num + 2),0.5,0)], hcat, f_fea_01)
#     # # 也可以直接使用这个比例作为AUC计算
#     # # sample_actual_label = f_fea_01 ./ (iter_num + 2)
#     # auc_fea = reshape(roc_kernel(sample_actual_label, true_group_01, decreasing  = true,auc_only = true, verbose = false),r,c)
#     auc_fea = reshape(roc_kernel(f_fea_01, true_group_01, decreasing  = true,auc_only = true, verbose = false),r,c)
#     # 将已选到的基因对的AUC赋值为0。避免重复选择。
#     auc_fea[gene_pair[:,1] .+ r*(gene_pair[:,2] .- 1)] .= 0.0
#     auc_fea[gene_pair[:,2] .+ r*(gene_pair[:,1] .- 1)] .= 0.0
#     # 自身和自身配对的基因对的AUC赋值为0。避免错选。
#     [auc_fea[[(i-1)*r + i]] .= 0.0 for i in 1:r]
#     # t_gene_pair = gene_pair[(gene_pair[:,2] .<= r),:]
#     # auc_fea[t_gene_pair[:,2] .+ r*(t_gene_pair[:,1] .- 1)] .= 0.0
#     # 找到AUC取最大值的fea, 贪婪算法
#     if (findmax(auc_fea)[1] != 0)
#         max_auc, max_auc_gene_pair = n_top_fea(auc_fea, 1, s_fl)
#     else
#         @info "INFO: The AUC of all gene pairs in this sampling training set is 0, so the list of marker gene pairs cannot be obtained."
#         # println("本次训练中，训练样本筛选特征基因列表的AUC最大值为0")
#         return [1 1]
#     end

#     # # 对于基因对列表中存在，当“对于scores为“1”（>(max-min)/2）的情况中positives为1的比例小于0.5且positives为1的数目小于总的positives为1的数目时，scores取负值，positives取反。”
#     # gp_s = f_fea_01[max_auc_gene_pair[1] .+ r*(max_auc_gene_pair[2] .- 1)]
#     # t_sp = true_group_01[(gp_s .>= (findmax(gp_s)[1] + findmin(gp_s)[1])/2)]
# 	# if (sum(t_sp) < length(t_sp)/2) && (sum(t_sp) < sum(true_group_01))
#     #     println("reverse")
#     #     max_auc_gene_pair = reverse(max_auc_gene_pair, dims = 2)
#     # end

#     # max_auc_gene_pair = (sum(mat_fea_01[max_auc_gene_pair[1],max_auc_gene_pair[2],:]) > s/2) ? max_auc_gene_pair : reverse(max_auc_gene_pair)
#     gene_pair = vcat(gene_pair,  max_auc_gene_pair)
#     iter_num = iter_num + 1
#     print(".")
#     (max_auc == 1 .|| iter_num == t_train_iter_num .|| iter_num == max(r,c)) ? (return gene_pair[2:end,:]) : (return trainset_hill_climbing(mat_fea_01, true_group_01, r, c, s_fl; t_train_iter_num = t_train_iter_num, gene_pair = gene_pair, f_fea_01 = f_fea_01, iter_num = iter_num))
#     # gene_pair = rule_score(mat_fea_01, true_group_01, gene_pair, r, c, t_train_iter_num = 15)
#     # println(".")
#     # return gene_pair
# end

# 将每轮AUC最大的基因对进行保留。
function trainset_hill_climbing(
        mat_fea_01::BitArray, # The 01 matrix of gene pairs
        true_group_01::BitVector,
        r::Int,
        c::Int,
        s_fl::Int;
        t_train_iter_num::Int = 15,
        gene_pair::Matrix = [1 1],
        ms_mat_fea_01::Vector = [0, 0],
        f_fea_01::Matrix = [0 0],
        iter_num::Number = 0,
        first_fea::Matrix = [0 0])
    if iter_num == 0
        # f_auc_max,f_max_auc_fea = findmax(sum(mat_fea_01[:,:,:],dims = 3))
        # g1,g2 = convert(Tuple, f_max_auc_fea)
        # # g1,g2 = (sum(mat_fea_01[g1,g2,:]) > s/2) ? (g1,g2) : (g2,g1)
        # gene_pair = [g1 g2]
        # f_fea_01 = reshape(mapreduce(x->reshape([mat_fea_01[n,x,:] .+ mat_fea_01[g1,g2,:] for n in 1:r],:,1), hcat, [1:c...]),1,:)
        f_fea_01 = reshape(mapreduce(x->reshape([mat_fea_01[n,x,:] for n in 1:r],:,1), hcat, [1:c...]),1,:)
    else
        g1,g2=gene_pair[iter_num + 1,:]
        f_fea_01 = mapreduce(x-> [x .+ ms_mat_fea_01], hcat, f_fea_01)
    end
    # # 按照多数投票规则，当比例大于0.5则赋值1；小于0.5赋值0；等于0.5随机赋值1或0
    # sample_actual_label = mapreduce(x -> [is_greater.(x ./ (iter_num + 2),0.5,0)], hcat, f_fea_01)
    # # 也可以直接使用这个比例作为AUC计算
    # # sample_actual_label = f_fea_01 ./ (iter_num + 2)
    # auc_fea = reshape(roc_kernel(sample_actual_label, true_group_01, decreasing  = true,auc_only = true, verbose = false),r,c)
    auc_fea = reshape(roc_kernel1(f_fea_01, true_group_01, decreasing  = true,auc_only = true, verbose = false),r,c)
    # 将已选到的基因对的AUC赋值为0。避免重复选择。
    auc_fea[gene_pair[:,1] .+ r*(gene_pair[:,2] .- 1)] .= 0.0
    auc_fea[gene_pair[:,2] .+ r*(gene_pair[:,1] .- 1)] .= 0.0
    # 自身和自身配对的基因对的AUC赋值为0。避免错选。
    [auc_fea[[(i-1)*r + i]] .= 0.0 for i in 1:r]
    # t_gene_pair = gene_pair[(gene_pair[:,2] .<= r),:]
    # auc_fea[t_gene_pair[:,2] .+ r*(t_gene_pair[:,1] .- 1)] .= 0.0
    # 找到AUC取最大值的fea, 贪婪算法
    max_auc = findmax(auc_fea)[1]
    if (max_auc != 0)
        # max_auc, max_auc_gene_pair = n_top_fea(auc_fea, 1, s_fl)
        # 本轮auc是max的基因对的index
        ma_gp = convert.(Tuple, findall(auc_fea .== max_auc))
    else
        @info "INFO: The AUC of all gene pairs in this sampling training set is 0, so the list of marker gene pairs cannot be obtained."
        # println("本次训练中，训练样本筛选特征基因列表的AUC最大值为0")
        return [1 1],[0 0]
    end

    # # 对于基因对列表中存在，当“对于scores为“1”（>(max-min)/2）的情况中positives为1的比例小于0.5且positives为1的数目小于总的positives为1的数目时，scores取负值，positives取反。”
    # gp_s = f_fea_01[max_auc_gene_pair[1] .+ r*(max_auc_gene_pair[2] .- 1)]
    # t_sp = true_group_01[(gp_s .>= (findmax(gp_s)[1] + findmin(gp_s)[1])/2)]
	# if (sum(t_sp) < length(t_sp)/2) && (sum(t_sp) < sum(true_group_01))
    #     println("reverse")
    #     max_auc_gene_pair = reverse(max_auc_gene_pair, dims = 2)
    # end

    # max_auc_gene_pair = (sum(mat_fea_01[max_auc_gene_pair[1],max_auc_gene_pair[2],:]) > s/2) ? max_auc_gene_pair : reverse(max_auc_gene_pair)
    gene_pair = vcat(gene_pair, hcat(first.(ma_gp), last.(ma_gp)))
    # 本次最大AUC的基因对的样本标签的和
    ms_mat_fea_01 = vec(sum(mapreduce(x -> mat_fea_01[x[1],x[2],:], hcat, ma_gp),dims=2))
    # 保存first的fea
    if (iter_num == 0)
        first_fea = gene_pair[2:end,:]
    end
    iter_num = iter_num + 1
    print(".")
    (max_auc == 1 .|| iter_num == t_train_iter_num .|| iter_num == max(r,c)) ? (return gene_pair[2:end,:], first_fea) : (return trainset_hill_climbing(mat_fea_01, true_group_01, r, c, s_fl; t_train_iter_num = t_train_iter_num, gene_pair = gene_pair,ms_mat_fea_01 = ms_mat_fea_01, f_fea_01 = f_fea_01, iter_num = iter_num, first_fea = first_fea))
    # gene_pair = rule_score(mat_fea_01, true_group_01, gene_pair, r, c, t_train_iter_num = 15)
    # println(".")
    # return gene_pair
end

function testset_hill_climbing(
        mat_fea_01::BitArray, # The 01 matrix of gene pairs
        true_group_01::BitVector,
        r::Int,
        c::Int,
        l_train::Vector,
        l_test::Vector,
        s_fl::Int;
        t_train_iter_num::Int = 15)
    # 训练集训练特征
    gene_pair, first_fea = trainset_hill_climbing(mat_fea_01[:,:,l_train], true_group_01[l_train], r, c, s_fl; t_train_iter_num = t_train_iter_num)
    if gene_pair == [1 1]
        return [1 1], 0, [0, 0], [0, 0] 
    end
    # gene_pair = (sum(true_group_01[l_train]) >= size(l_train)[1]/2) ? gene_pair : reverse(gene_pair, dims = 2)
    println(".")
    # 在测试集中提取训练集得到的特征基因对在每个样本中的REO进行求和
    s_test_fea= sum(mapreduce(x -> mat_fea_01[x[1],x[2],l_test] , hcat, eachrow(gene_pair)), dims = 2)
    # # 按照多数投票规则，当比例大于0.5则赋值1；小于0.5赋值0；等于0.5随机负值1或0
    # g_r, g_c = size(gene_pair)
    # sample_actual_label = is_greater.(s_test_fea./g_r,0.5,0)
    # println(roc_kernel(vec(is_greater.(sum(mapreduce(x -> mat_fea_01[x[1],x[2],l_train] , hcat, eachrow(gene_pair)), dims = 2)./g_r,0.5,0)), true_group_01[l_train]))
    # # 计算AUC
    # auc_fea = roc_kernel(vec(sample_actual_label), true_group_01[l_test], decreasing  = true,auc_only = true, verbose = false)
    auc_fea = roc_kernel(vec(s_test_fea), true_group_01[l_test], decreasing  = true,auc_only = true, verbose = false)
    # 保存gene_pair的结果,auc列表，first基因对的位置（01矩阵），全部基因对的位置（01矩阵）
    all_fea_01 = zeros(Int64,r*c,1)
    first_fea_01 = zeros(Int64,r*c,1)
    first_fea_01[r*(first_fea[:,2] .- 1) .+ first_fea[:,1]] .= 1
    all_fea_01[r*(gene_pair[:,2] .- 1) .+ gene_pair[:,1]] .= 1
    return gene_pair, auc_fea, first_fea_01, all_fea_01
end

"""
Mountain climbing method to train molecular marker gene pairs, through this function:

```jldoctest
        hill_climbing_method_kernel(
                                    mat_fea_01::BitArray, # The 01 matrix of gene pairs
                                    r::Int,
                                    c::Int,
                                    n_train::Int,
                                    n_test::Int,
                                    train_test_set::Vector,
                                    true_ngrp::Vector;
                                    t_hill_iter_num::Int = 500,
                                    gene_pair_01=zeros(Int,r,c),
                                    t_train_iter_num::Int = 15)
```

The parameters are:
- `mat_fea_01::BitArray`: For the REO relationship of gene pairs in each sample, 1 is greater than and 0 is less than.
- `r::Int`: Number of lines of expression spectrum.
- `c::Int`: The number of columns of the expression profile.
- `n_train::Int`: Number of training set samples.
- `n_test::Int`: Number of test set samples.
- `train_test_set::Vector`: Sample collection of randomly selected training set and test set.
- `true_ngrp::Vector`: Real grouping label for a sample set of randomly selected training and test sets.
- `t_hill_iter_num::Int`: Hill climbing number of iterations. Default: 500.
- `t_train_iter_num::Int`: The number of iterations of the climb training. Default: 15.
"""
function hill_climbing_method_kernel(
                                    mat_fea_01::BitArray, # The 01 matrix of gene pairs
                                    r::Int,
                                    c::Int,
                                    l_train::Vector,
                                    l_test::Vector,
                                    train_test_set::Vector,
                                    true_ngrp::Vector,
                                    s_fl::Int;
                                    t_hill_iter_num::Int = 500,
                                    gene_pair_01::Matrix = zeros(Int,r,c),
                                    t_train_iter_num::Int = 15,
                                    mat_first_fea_01::Matrix = [0 0],
                                    mat_all_fea_01::Matrix = [0 0],
                                    list_sample_auc::Vector = [0,0],
                                    sample_gene_pair::Matrix = [0 0])
    gene_pair, auc_fea, first_fea_01, all_fea_01 = testset_hill_climbing(mat_fea_01[:,:,train_test_set[t_hill_iter_num]], true_ngrp[t_hill_iter_num], r, c, l_train, l_test, s_fl; t_train_iter_num = t_train_iter_num)
    if gene_pair == [1 1]
        t_hill_iter_num -= 1
        (mat_first_fea_01 == [0 0]) ? (return hill_climbing_method_kernel(mat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, s_fl; t_hill_iter_num = t_hill_iter_num, gene_pair_01 = gene_pair_01,t_train_iter_num = t_train_iter_num)) : (return hill_climbing_method_kernel(mat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, s_fl; t_hill_iter_num = t_hill_iter_num, gene_pair_01 = gene_pair_01,t_train_iter_num = t_train_iter_num, mat_first_fea_01 = mat_first_fea_01, mat_all_fea_01 = mat_all_fea_01, list_sample_auc = list_sample_auc, sample_gene_pair = sample_gene_pair))
    end
    # println(auc_fea)
    # mapreduce(x -> ((auc_fea >= 0.6) ? gene_pair_01[x[1],x[2]] += 1 : ((auc_fea <= 0.4) ? gene_pair_01[x[1],x[2]] -= 1 : gene_pair_01)), hcat, eachrow(gene_pair))

    if (mat_first_fea_01 == [0 0])
        mat_first_fea_01 = copy(first_fea_01)
        mat_all_fea_01 = copy(all_fea_01)
        list_sample_auc = [auc_fea]
        sample_gene_pair = reshape([gene_pair],1,:)
    else
        mat_first_fea_01 = hcat(mat_first_fea_01,first_fea_01)
        mat_all_fea_01 = hcat(mat_all_fea_01,all_fea_01)
        list_sample_auc = vcat(list_sample_auc,auc_fea)
        sample_gene_pair = [sample_gene_pair [gene_pair]]
    end
    t_hill_iter_num -= 1
    (t_hill_iter_num == 0) ? (return gene_pair_01, mat_first_fea_01, list_sample_auc, sample_gene_pair, mat_all_fea_01) : (return hill_climbing_method_kernel(mat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, s_fl; t_hill_iter_num = t_hill_iter_num, gene_pair_01 = gene_pair_01,t_train_iter_num = t_train_iter_num, mat_first_fea_01 = mat_first_fea_01, mat_all_fea_01 = mat_all_fea_01, list_sample_auc = list_sample_auc, sample_gene_pair = sample_gene_pair))
end
