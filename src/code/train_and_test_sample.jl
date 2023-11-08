export train_and_test_sample_kernel

using Random

# function g_train_and_test_sample(n_sample::Int,
#                                     group::Vector,
#                                     n_g::Int,
#                                     ngrp::BitVector,
#                                     n_train_test::Int;
#                                     t_hill_iter_num::Int = 500,
#                                     n_iter::Int = 0,
#                                     train_test_sample::Vector = [0,0],
#                                     true_ngrp::Vector = [0,0])
#     n_g >= n_train_test || throw("The total sample size should be greater than the total number of extracted training sets and test sets.")
#     ind = randperm(n_g)
#     tt_sample = group[ind[1:n_train_test]]
#     true_ngrp = ((n_iter == 0) ? [ngrp[tt_sample]] : vcat(true_ngrp,[ngrp[tt_sample]]))
#     train_test_sample = ((n_iter == 0) ? [tt_sample] : vcat(train_test_sample,[tt_sample]))
#     n_iter = n_iter + 1
#     (n_iter == t_hill_iter_num) ? (return train_test_sample, true_ngrp) : (return g_train_and_test_sample(n_sample, group, n_g, ngrp, n_train_test,t_hill_iter_num = t_hill_iter_num,n_iter = n_iter,train_test_sample = train_test_sample,true_ngrp=true_ngrp))
# end

"""
Randomly extract the training set and the validation set by this function:

```jldoctest
train_and_test_sample_kernel(n_sample::Int,
                                    ngrp::BitVector,
                                    n_train_test::Int;
                                    t_hill_iter_num::Int = 500)
```

The parameters are: 
- `n_sample`: Total number of samples.
- `ngrp::BitVector`: Sample grouping information, 1 or 0.
- `n_train_test::Int`: Number of training set and test set samples.
- `t_hill_iter_num::Int`: Sample extraction times. Default: 500.
"""
function train_and_test_sample_kernel(n_sample::Int,
                                    ngrp::BitVector,
                                    n_train_test::Int;
                                    t_hill_iter_num::Int = 500)
    n_sample >= n_train_test || throw("The total sample size should be greater than the total number of extracted training sets and test sets.")
    # g1_train_test_sample = g_train_and_test_sample(n_sample, [1:n_sample...][ngrp], sum(ngrp), n_train_test, t_hill_iter_num = t_hill_iter_num)
    # g2_train_test_sample = g_train_and_test_sample(n_sample, [1:n_sample...][.!ngrp], sum(.!ngrp), n_train_test, t_hill_iter_num = t_hill_iter_num)
    # return vcat.(g1_train_test_sample,g2_train_test_sample)
    g_train_test_sample, true_ngrp = g_train_and_test_sample(n_sample, [1:n_sample...], n_sample, ngrp, n_train_test, t_hill_iter_num = t_hill_iter_num)
    return g_train_test_sample, true_ngrp 
end

function train_and_test_sample_kernel(n_sample::Int,
                                    ngrp::BitVector,
                                    n_train::Int,
                                    n_test::Int;
                                    t_hill_iter_num::Int = 500)
    n_group1 = sum(ngrp) 
    n_group2 = n_sample - n_group1
    n_train_test = n_train + n_test
    (n_group1 >= n_train_test && n_group2 >= n_train_test) || throw("The number of samples in each group should be greater than the number of samples in the training and test set.")
    true_ngrp = vcat(trues(n_train),trues(n_test),falses(n_train),falses(n_test))
    g_train_test_sample = g_train_and_test_sample(n_group1, n_group2, [1:n_sample...][ngrp], [1:n_sample...][.!ngrp], n_train_test, t_hill_iter_num = t_hill_iter_num)
    # g_test_sample = g_train_and_test_sample(n_sample, n_group1, [1:n_sample...][ngrp], [1:n_sample...][.!ngrp], n_test, t_hill_iter_num = t_hill_iter_num)

    ## 训练集和测试集的位置
    l_train = [1:n_train...,(n_train+n_test+1):(2*n_train+n_test)...]
    l_test = setdiff([1:2*n_train_test...],l_train)
    return g_train_test_sample, true_ngrp, l_train, l_test
end

function g_train_and_test_sample(n_group1::Int,
                                    n_group2::Int,
                                    l_group1::Vector,
                                    l_group2::Vector,
                                    n::Int;
                                    t_hill_iter_num::Int = 500,
                                    n_iter::Int = 0,
                                    t_sample::Vector = [0,0])
    (n_group1 >= n && n_group2 >= n) || throw("The number of samples in each group should be greater than the number of samples in the training and test set.")
    group1_ind = randperm(n_group1)[1:n]
    group2_ind = randperm(n_group2)[1:n]
    group1_sample = l_group1[group1_ind]
    group2_sample = l_group2[group2_ind]
    tt_sample = vcat(group1_sample, group2_sample)
    t_sample = ((n_iter == 0) ? [tt_sample] : vcat(t_sample,[tt_sample]))
    n_iter = n_iter + 1
    (n_iter == t_hill_iter_num) ? (return t_sample) : (return g_train_and_test_sample(n_group1, n_group2, l_group1, l_group2, n,t_hill_iter_num = t_hill_iter_num,n_iter = n_iter,t_sample = t_sample))
end