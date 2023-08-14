export train_and_test_sample_kernel

using Random

function g_train_and_test_sample(n_sample::Int,
                                    group::Vector,
                                    n_g::Int,
                                    n_train_test::Int;
                                    t_hill_iter_num::Int = 500,
                                    n_iter::Int = 0,
                                    train_test_sample::Vector = [0,0])
    n_g >= n_train_test || throw("The total sample size should be greater than the total number of extracted training sets and test sets.")
    ind = randperm(n_g)
    train_test_sample = ((n_iter == 0) ? [group[ind[1:n_train_test]]] : vcat(train_test_sample,[group[ind[1:n_train_test]]]))
    n_iter = n_iter + 1
    (n_iter == t_hill_iter_num) ? (return train_test_sample) : (return g_train_and_test_sample(n_sample, group, n_g, n_train_test,t_hill_iter_num = t_hill_iter_num,n_iter = n_iter,train_test_sample = train_test_sample))
end

function train_and_test_sample_kernel(n_sample::Int,
                                    ngrp::BitVector,
                                    n_train_test::Int;
                                    t_hill_iter_num::Int = 500)
    n_sample >= n_train_test || throw("The total sample size should be greater than the total number of extracted training sets and test sets.")
    g1_train_test_sample = g_train_and_test_sample(n_sample, [1:n_sample...][ngrp], sum(ngrp), n_train_test, t_hill_iter_num = t_hill_iter_num)
    g2_train_test_sample = g_train_and_test_sample(n_sample, [1:n_sample...][.!ngrp], sum(.!ngrp), n_train_test, t_hill_iter_num = t_hill_iter_num)
    return vcat.(g1_train_test_sample,g2_train_test_sample)
end