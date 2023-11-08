
export hill_climbing_method_kernel, n_top_fea

# using DataFrames

include("compare_genepair.jl")

include("roc.jl")

function fea_maxAUC(f_fea_01::Matrix,
                    true_grp::BitVector,
                    r::Int,
                    c::Int,
                    feas::Matrix = [1 1])   
    auc_fea = reshape(roc_kernel(f_fea_01, true_grp, decreasing  = true,auc_only = true, verbose = false),r,c)
    # 将已选到的基因对的AUC赋值为0。避免重复选择。
    auc_fea[feas[:,1] .+ r*(feas[:,2] .- 1)] .= 0.0
    auc_fea[feas[:,2] .+ r*(feas[:,1] .- 1)] .= 0.0
    # 自身和自身配对的基因对的AUC赋值为0。避免错选。
    [auc_fea[[(i-1)*r + i]] .= 0.0 for i in 1:r]
    # 找到AUC取最大值的fea, 贪婪算法
    max_auc = findmax(auc_fea)[1]
    if (max_auc != 0)
        # 本轮auc等于maxAUC的基因对的index
        ma_feas = convert.(Tuple, findall(auc_fea .== max_auc))
    else
        @info "INFO: The AUC of all gene pairs in this sampling training set is 0, so the list of marker gene pairs cannot be obtained."
        # println("本次训练中，训练样本筛选特征基因列表的AUC最大值为0")
        return [1 1], 0
    end
    return ma_feas, max_auc
end


function trainset_hill_climbing(
        mat_fea_01::BitArray, # The 01 matrix of gene pairs
        f_fea_01::Matrix,
        true_grp::BitVector,
        feas::Matrix,
        r::Int,
        c::Int,
        last_max_auc::Number,
        ms_mat_fea_01::Vector,
        t_train_iter_num::Int = 15)
    f_fea_01 = mapreduce(x-> [x .+ ms_mat_fea_01], hcat, f_fea_01)
    ## 选取使AUC增大最多的基因对以及最大AUC
    ma_feas, max_auc = fea_maxAUC(f_fea_01, true_grp, r, c, feas)
    # 本轮AUC比上一轮小,则停止迭代，输出结果
    if last_max_auc >= max_auc
        return feas
    end
    feas = vcat(feas, hcat(first.(ma_feas), last.(ma_feas)))
    # 本次最大AUC的基因对的样本标签的和
    ms_mat_fea_01 = vec(sum(mapreduce(x -> mat_fea_01[x[1],x[2],:], hcat, ma_feas),dims=2))
    t_train_iter_num = t_train_iter_num - 1
    print(".")
    # 终止条件，满足其一：1）最大AUC=1；2）迭代t_train_iter_num-1次；3）迭代次数大于特征数
    if (max_auc == 1 .|| t_train_iter_num == 1)
        return feas
    else
        last_max_auc = max_auc
        return trainset_hill_climbing(mat_fea_01, f_fea_01, true_grp, feas, r, c, last_max_auc, ms_mat_fea_01, t_train_iter_num)
    end
end

function testset_hill_climbing(
        mat_fea_01::BitArray, # The 01 matrix of gene pairs
        true_grp::BitVector,
        feas::Matrix,
        r::Int,
        c::Int)
    println(".")
    # 在测试集中提取训练集得到的特征基因对在每个样本中的REO进行求和
    s_test_fea= sum(mapreduce(x -> mat_fea_01[x[1],x[2],:] , hcat, eachrow(feas)), dims = 2)
    auc_fea = roc_kernel(vec(s_test_fea), true_grp, decreasing  = true,auc_only = true, verbose = false)
    return auc_fea
end


"""
Mountain climbing method to train molecular marker gene pairs, through this function:

```jldoctest
        hill_climbing_method_kernel(
                                    mat_fea_01::BitArray, # The 01 matrix of gene pairs
                                    r::Int,
                                    c::Int,
                                    l_train::Vector,
                                    l_test::Vector,
                                    train_test_set::Vector,
                                    true_ngrp::BitVector,
                                    t_hill_iter_num::Int = 500,
                                    gene_pair_01::Matrix = zeros(Int,r,c),
                                    t_train_iter_num::Int = 15,
                                    mat_first_fea_01::Matrix = [0 0],
                                    mat_all_fea_01::Matrix = [0 0],
                                    list_sample_auc::Vector = [0,0],
                                    sample_gene_pair::Matrix = [0 0])
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

julia> true_ngrp, r, c, l_train, l_test, train_test_set, t_hill_iter_num, t_train_iter_num = BitVector([1,0,1]), 3, 4, [1,2,3], [2,3,1], [[3,1,2]], 1, 2
(Bool[1, 0, 1], 3, 4, [1, 2, 3], [2, 3, 1], [[3, 1, 2]], 1, 2)

julia> hill_climbing_method_kernel(mat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, t_hill_iter_num, t_train_iter_num)
[ Info: INFO: The AUC of all gene pairs in this sampling training set is 0, so the list of marker gene pairs cannot be obtained.
.
([0; 1; … ; 1; 1;;], [0; 0; … ; 1; 1;;], [0.5], [[3 1; 1 2; … ; 2 4; 3 4];;])

```
"""
function hill_climbing_method_kernel(
                                    mat_fea_01::BitArray, # The 01 matrix of gene pairs
                                    r::Int,
                                    c::Int,
                                    l_train::Vector,
                                    l_test::Vector,
                                    train_test_set::Vector,
                                    true_ngrp::BitVector,
                                    t_hill_iter_num::Int = 500,
                                    t_train_iter_num::Int = 15,
                                    mat_first_fea_01::Matrix = [0 0],
                                    mat_all_fea_01::Matrix = [0 0],
                                    list_sample_auc::Vector = [0,0],
                                    sample_gene_pair::Matrix = [0 0])
    # 本轮样本
    n_mat_fea_01 = mat_fea_01[:,:,train_test_set[t_hill_iter_num]]
    # 训练集和测试集
    ntr_mat_fea_01 = n_mat_fea_01[:,:,l_train]
    nte_mat_fea_01 = n_mat_fea_01[:,:,l_test]
    # 初始值
    ## 特征的01矩阵
    f_fea_01 = reshape(mapreduce(x->reshape([ntr_mat_fea_01[n,x,:] for n in 1:r],:,1), hcat, [1:c...]),1,:)
    ## 选取使AUC增大最多的基因对以及最大AUC
    first_feas, max_auc = fea_maxAUC(f_fea_01, true_ngrp[l_train], r, c)
    if max_auc == 0
        t_hill_iter_num -= 1
        return hill_climbing_method_kernel(ntr_mat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, t_hill_iter_num, t_train_iter_num, mat_first_fea_01, mat_all_fea_01, list_sample_auc, sample_gene_pair)
    end
    ## 本次最大AUC的基因对的样本标签的和
    ms_mat_fea_01 = vec(sum(mapreduce(x -> ntr_mat_fea_01[x[1],x[2],:], hcat, first_feas),dims=2))
    first_feas = hcat(first.(first_feas), last.(first_feas))
    # 保存feas
    feas = copy(first_feas)
    # 训练集训练
    feas = trainset_hill_climbing(ntr_mat_fea_01, f_fea_01, true_ngrp[l_train], feas, r, c, max_auc, ms_mat_fea_01, t_train_iter_num)
    # 测试集测试
    auc_fea = testset_hill_climbing(nte_mat_fea_01, true_ngrp[l_test], feas, r, c)
    # 保存gene_pair的结果,auc列表，first基因对的位置（01矩阵），全部基因对的位置（01矩阵）
    all_fea_01 = zeros(Int64,r*c,1)
    first_fea_01 = zeros(Int64,r*c,1)
    first_fea_01[r*(first_feas[:,2] .- 1) .+ first_feas[:,1]] .= 1
    all_fea_01[r*(feas[:,2] .- 1) .+ feas[:,1]] .= 1
    # 保存每次迭代结果
    if (mat_first_fea_01 == [0 0])
        mat_first_fea_01 = copy(first_fea_01)
        mat_all_fea_01 = copy(all_fea_01)
        list_sample_auc = [auc_fea]
        sample_gene_pair = reshape([feas],1,:)
    else
        mat_first_fea_01 = hcat(mat_first_fea_01,first_fea_01)
        mat_all_fea_01 = hcat(mat_all_fea_01,all_fea_01)
        list_sample_auc = vcat(list_sample_auc,auc_fea)
        sample_gene_pair = [sample_gene_pair [feas]]
    end
    t_hill_iter_num -= 1
    if (t_hill_iter_num == 0)
        return mat_first_fea_01, mat_all_fea_01, list_sample_auc, sample_gene_pair
    else
        return hill_climbing_method_kernel(mat_fea_01, r, c, l_train, l_test, train_test_set, true_ngrp, t_hill_iter_num, t_train_iter_num, mat_first_fea_01, mat_all_fea_01, list_sample_auc, sample_gene_pair)
    end
end