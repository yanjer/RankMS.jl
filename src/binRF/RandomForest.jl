#=
10  December 2020

Random Forests from scratch. Redo of Python code

Sources
- https://course18.fast.ai/lessonsml1/lesson5.html
- https://github.com/bensadeghi/DecisionTree.jl

=#

using Random, Statistics
using CSV, DataFrames, Printf


## --------------------- Random Forest Classifier  --------------------- ##
"""
    RandomForestClassifier{T}([n_trees=100], [max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG], [bootstrap=true], [oob_score=false])
    RandomForestClassifier([n_trees=100], [max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG], [bootstrap=true], [oob_score=false]) -> T = Float64

A random forest classifier.

Available methods are:
`fit!`, `predict`, `predict_prob`, `score`,
`feature_importance_impurity`, `perm_feature_importance`
"""
mutable struct RandomForestClassifier{T}  <: AbstractClassifier
    #internal variables
    ## 当变量类型为多种，可使用Union定义
    n_features::Union{Int, Nothing}
    n_classes::Union{Int, Nothing}
    features::Vector{String}
    trees::Vector{DecisionTreeClassifier}
    feature_importances::Union{Vector{Float64}, Nothing}

    # external parameters
    n_trees::Int
    max_depth::Union{Int, Nothing}
    max_features::Union{Int, Nothing} # sets n_features_split
    min_samples_leaf::Int
    random_state::Union{AbstractRNG, Int}
    bootstrap::Bool
    oob_score::Bool
    oob_score_::Union{Float64, Nothing}

    RandomForestClassifier{T}(;
            n_trees=100,
            max_depth=nothing,
            max_features=nothing,
            min_samples_leaf=1,
            random_state=Random.GLOBAL_RNG,
            bootstrap=true,
            oob_score=false
        ) where T = new(
            nothing, nothing, [], [], nothing, n_trees,
            max_depth, max_features, min_samples_leaf, check_random_state(random_state), bootstrap, oob_score, nothing
            )
end
RandomForestClassifier(;
        n_trees=100,
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf=1,
        random_state=Random.GLOBAL_RNG,
        bootstrap=true,
        oob_score=false
    ) = RandomForestClassifier{Float64}(
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        bootstrap=bootstrap,
        oob_score=oob_score
    )


function Base.show(io::IO, forest::RandomForestClassifier)
    rng = forest.random_state
    if hasproperty(rng, :seed)
        str_rng = string(typeof(rng), "($(forest.random_state.seed),...)")
    else
        str_rng = string(typeof(rng))
    end
    str_out = string(
        typeof(forest), "(",
        "n_trees=$(forest.n_trees)",
        ", max_depth=$(forest.max_depth)",
        ", max_features=$(forest.max_features)",
        ", min_samples_leaf=$(forest.min_samples_leaf)",
        ", random_state="*str_rng,
        ", bootstrap=$(forest.bootstrap)",
        ", oob_score=$(forest.oob_score)",
        ")"
    )
    println(io, str_out)
end

## --------------------- fitting --------------------- ##
function fit!(forest::RandomForestClassifier, X::DataFrame, Y::DataFrame)
    @assert size(Y, 2) == 1 "Output Y must be an m x 1 DataFrame"

    # set internal variables
    forest.n_features = size(X, 2)
    forest.n_classes = size(unique(Y), 1)
    forest.features = names(X)
    forest.trees = []

    # create decision trees
    ## 创建随机决策树
    rng_states = typeof(forest.random_state)[]  # save the random states to regenerate the random indices for the oob_score
    ## 建立forest.n_trees数目的树
    for i in 1:forest.n_trees
        push!(rng_states, copy(forest.random_state))
        push!(forest.trees, create_tree(forest, X, Y))
    end

    # set attributes
    ## 计算每棵树中特征的平均feature_importances。其中每棵树中每个特征的得分是，根据当前节点的impurity和左右节点的打分（根据左右叶子节点的gini_impurity的值和该左右叶子节点时的样本数计算，score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples）计算获得，feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    forest.feature_importances = feature_importance_impurity(forest)
    if forest.oob_score
        if !forest.bootstrap
            printstyled("Warning: out-of-bag score will not be calculated because bootstrap=false\n", color=:yellow)
        else
            forest.oob_score_ = calculate_oob_score(forest, X, Y, rng_states)
        end
    end

    return
end

## 创建一个决策树分类器，主要分配给DecisionTreeClassifier进行创建。
function create_tree(forest::RandomForestClassifier, X::DataFrame, Y::DataFrame)
    n_samples = nrow(X)

    if forest.bootstrap # sample with replacement
        ## 从样本中有放回的随机抽相同数目的样本
        idxs = [rand(forest.random_state, 1:n_samples) for i in 1:n_samples]
        # println(idxs)
        X_ = X[idxs, :]
        Y_ = Y[idxs, :]
    else
        X_ = copy(X)
        Y_ = copy(Y)
    end

    T = typeof(forest).parameters[1] # use the same type T used for the forest
    new_tree = DecisionTreeClassifier{T}(
            max_depth = forest.max_depth,
            max_features = forest.max_features,
            min_samples_leaf = forest.min_samples_leaf,
            random_state = forest.random_state
    )
    new_tree.n_classes = forest.n_classes
    fit!(new_tree, X_, Y_)

    return new_tree
end

## OOB_SCORE就是袋外估计准确率得分，即对袋外样本正确预测的比例。对应的，OOB_ERROR就是对袋外样本预测错误的比例。
## 如果这意味着每棵树只在数据的一个子集上进行训练。 然后，可以将袋外分数计算为基于未用于训练的树对每个样本的预测。 这是衡量训练准确性的有用指标。 对于替换抽样，样本大小是数据集的大小，我们可以预期平均 63.2% 的样本是唯一的。bootstrap=true5这意味着每棵树有 36.8% 的样本是袋外的，可用于计算 OOB 分数。
function calculate_oob_score(
    forest::RandomForestClassifier, X::DataFrame, Y::DataFrame,
    rng_states::Vector{T}) where T <: AbstractRNG
    n_samples = nrow(X)
    oob_prob  = zeros(n_samples, forest.n_classes)
    oob_count = zeros( n_samples)
    for (i, rng) in enumerate(rng_states)
        idxs = Set([rand(forest.random_state, 1:n_samples) for i in 1:n_samples])
        # note: expected proportion of out-of-bag is 1-exp(-1) = 0.632...
        # so length(row_oob)/n_samples ≈ 0.63
        row_oob =  filter(idx -> !(idx in idxs), 1:n_samples)
        oob_prob[row_oob, :] .+= predict_prob(forest.trees[i], X[row_oob, :])
        oob_count[row_oob] .+= 1.0
    end
    # remove missing values
    valid = oob_count .> 0.0
    oob_prob = oob_prob[valid, :]
    oob_count = oob_count[valid]
    y_test = Y[valid, 1]
    # predict out-of-bag score
    y_pred = mapslices(argmax, oob_prob./oob_count, dims=2)[:, 1]
    return mean(y_pred .==  y_test)
end



## --------------------- prediction --------------------- ##
function predict_prob(forest::RandomForestClassifier, X::DataFrame)
    if length(forest.trees) == 0
        throw(NotFittedError(:forest))
    end
    ## 定义 样本数*Y中类别数的zeros
    probs = zeros(nrow(X), forest.n_classes)
    ##/\ 最终分类是将每棵树进行加权总和后最高的类（森林的预测是通过多数投票完成的。特别是，进行了“软”投票，其中每棵树的投票由其每个类的概率预测进行加权。 因此，最终预测等效于具有最大概率和的类。）
    ## 对每棵树
    for tree in forest.trees
        ## 对每棵树最后节点的各类别样本数进行相加
        probs .+= predict_prob(tree, X)
    end
    return probs ./ forest.n_trees
end

function predict(forest::RandomForestClassifier, X::DataFrame)
    probs = predict_prob(forest, X)
    ## 获得数值最大的类别
    return mapslices(argmax, probs, dims=2)[:, 1]
end

## --------------------- description --------------------- ##
"""
    nleaves(forest::RandomForestClassifier) => Vector{Int}

The number of leaves in each DecisionTreeClassifier in the forest.
"""
nleaves(forest::RandomForestClassifier) = [nleaves(tree.binarytree) for tree in forest.trees]

"""
    feature_importance_impurity(forest::RandomForestClassifier) => Vector{Float64}

Calculate feature importance based on impurity.
This is the mean of all the impurity feature importances for each DecisionTreeClassifier in the forest for each feature.
"""
function feature_importance_impurity(forest::RandomForestClassifier)
    if length(forest.trees) == 0
        throw(NotFittedError(:forest))
    end
    feature_importances = zeros(forest.n_trees, forest.n_features)
    for (i, tree) in enumerate(forest.trees)
        ## tree.feature_importances表示每颗树中每个特征的feature_importances
        feature_importances[i, :] = tree.feature_importances
    end
    ## 对每棵树的特征计算平均feature_importances
    return mean(feature_importances, dims=1)[1, :]
end
