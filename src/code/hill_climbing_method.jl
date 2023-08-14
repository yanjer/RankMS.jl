export hill_climbing_method_kernel, n_top_genepair

# using DataFrames

include("compare_genepair.jl")

include("roc.jl")


function n_top_genepair(matr::Matrix,
                        n_top::Int,
                        gene_pair::Matrix = [0 0])
    score, ind = findmax(matr)
    g1,g2 = convert(Tuple, ind)
    gene_pair = vcat(gene_pair,  [g1 g2])
    matr[g1,g2] = 0
    n_top -= 1
    (n_top == 0) ? (return score,gene_pair[2:end,:]) : (return n_top_genepair(matr, n_top, gene_pair))
end

"""
    Mountain climbing method using majority voting rule model is used to identify feature gene pairs.

    rule_score(mat_genepair_01::BitArray,
                        true_group_01::BitVector,
                        gene_pair::Matrix,
                        r::Int,
                        c::Int;
                        t_train_iter_num::Int = 128,
                        f_genepair_01::Matrix = [0 0],
                        iter_num::Number = 0)

Examples
```
julia> (mat_genepair_01 = falses(3,4,3))[3,1,:] .= [1,1,1]; mat_genepair_01[1,2,:] .= [1,1,1];mat_genepair_01
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

julia> hill_climbing_method_kernel(mat_genepair_01,true_group_01,r,c)
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

julia> hill_climbing_method_kernel(mat_genepair_01,BitVector([1,0,1]),r,c;t_train_iter_num = 5)
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

function trainset_hill_climbing(
        mat_genepair_01::BitArray, # The 01 matrix of gene pairs
        true_group_01::BitVector,
        r::Int,
        c::Int;
        t_train_iter_num::Int = 128,
        gene_pair::Matrix = [0 0],
        f_genepair_01::Matrix = [0 0],
        iter_num::Number = 0)
    if iter_num == 0
        f_auc_max,f_max_auc_genepair = findmax(sum(mat_genepair_01[:,:,:],dims = 3))
        g1,g2 = convert(Tuple, f_max_auc_genepair)
        gene_pair = [g1 g2]
        f_genepair_01 = reshape(mapreduce(x->reshape([mat_genepair_01[n,x,:] .+ mat_genepair_01[g1,g2,:] for n in 1:r],:,1), hcat, [1:c...]),1,:)
    else
        g1,g2=gene_pair[iter_num + 1,:]
        f_genepair_01 = mapreduce(x-> [x .+ mat_genepair_01[g1,g2,:]], hcat, f_genepair_01)
    end
    # # 按照多数投票规则，当比例大于0.5则赋值1；小于0.5赋值0；等于0.5随机负值1或0
    sample_actual_label = mapreduce(x -> [is_greater.(x ./ (iter_num + 2),0.5,0)], hcat, f_genepair_01)
    # 也可以直接使用这个比例作为AUC计算
    # sample_actual_label = f_genepair_01 ./ (iter_num + 2)
    auc_genepair = reshape(roc_kernel(sample_actual_label, true_group_01, decreasing  = true,auc_only = true, verbose = false),r,c)
    auc_genepair[gene_pair[:,1] .+ r*(gene_pair[:,2] .- 1)] .= 0.0
    # t_gene_pair = gene_pair[(gene_pair[:,2] .<= r),:]
    # auc_genepair[t_gene_pair[:,2] .+ r*(t_gene_pair[:,1] .- 1)] .= 0.0
    # 找到AUC取最大值的genepair, 贪婪算法
    max_auc, max_auc_gene_pair = n_top_genepair(auc_genepair,1)
    gene_pair = vcat(gene_pair,  max_auc_gene_pair)
    iter_num = iter_num + 1
    print(".")
    (max_auc == 1 .|| iter_num == t_train_iter_num) ? (return gene_pair) : (return trainset_hill_climbing(mat_genepair_01, true_group_01, r, c, t_train_iter_num = t_train_iter_num, gene_pair = gene_pair, f_genepair_01 = f_genepair_01, iter_num = iter_num))
    # gene_pair = rule_score(mat_genepair_01, true_group_01, gene_pair, r, c, t_train_iter_num = 128)
    # println(".")
    # return gene_pair
end

function testset_hill_climbing(
        mat_genepair_01::BitArray, # The 01 matrix of gene pairs
        true_group_01::BitVector,
        r::Int,
        c::Int,
        n_train::Int,
        n_test::Int;
        t_train_iter_num::Int = 128)
    # 训练集训练特征
    gene_pair = trainset_hill_climbing(mat_genepair_01[:,:,1:n_train], true_group_01[1:n_train], r, c, t_train_iter_num = t_train_iter_num)
    println(".")
    # 在测试集中提取训练集得到的特征基因对在每个样本中的REO进行求和
    s_test_genepair= sum(mapreduce(x -> mat_genepair_01[x[1],x[2],(n_train+1):end] , hcat, eachrow(gene_pair)), dims = 2)
    # 按照多数投票规则，当比例大于0.5则赋值1；小于0.5赋值0；等于0.5随机负值1或0
    g_r, g_c = size(gene_pair)
    sample_actual_label = is_greater.(s_test_genepair./g_r,0.5,0)
    # 计算AUC
    auc_genepair = roc_kernel(vec(sample_actual_label), true_group_01[(n_train+1):end], decreasing  = true,auc_only = true, verbose = false)
    # 保存gene_pair的结果
    return gene_pair, auc_genepair
end

function hill_climbing_method_kernel(
        mat_genepair_01::BitArray, # The 01 matrix of gene pairs
        true_group_01::BitVector,
        r::Int,
        c::Int,
        n_train::Int,
        n_test::Int,
        train_test_set::Vector;
        t_hill_iter_num::Int = 500,
        gene_pair_01=zeros(Int,r,c),
        t_train_iter_num::Int = 128)
    gene_pair, auc_genepair = testset_hill_climbing(mat_genepair_01[:,:,train_test_set[t_hill_iter_num]], true_group_01, r, c, n_train, n_test; t_train_iter_num = t_train_iter_num)
    mapreduce(x -> ((auc_genepair >= 0.6) ? gene_pair_01[x[1],x[2]] += 1 : ((auc_genepair <= 0.4) ? gene_pair_01[x[1],x[2]] -= 1 : gene_pair_01)), hcat, eachrow(gene_pair))
    t_hill_iter_num -= 1
    (t_hill_iter_num == 0) ? (return gene_pair_01) : (return hill_climbing_method_kernel(mat_genepair_01, true_group_01, r, c, n_train, n_test, train_test_set; t_hill_iter_num = t_hill_iter_num, gene_pair_01=gene_pair_01,t_train_iter_num = t_train_iter_num))
end
