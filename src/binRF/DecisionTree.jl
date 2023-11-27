#=
8 December 2020

Random Forests from scratch. Redo of Python code

Decision tree
- Tree is represent with 2 parallel arrays. This is more compact and requires much recusion than a linked list.
    - left_child_id = tree_.children_left[parent_id]
    - right_child_id = tree_.children_left[parent_id]
    - if id = -1, this node does not exist
- Works for multi-class problems

see https://github.com/bensadeghi/DecisionTree.jl

=#

export DecisionTreeClassifier, predict_row, predict_batch, predict_prob, predict,
feature_importance_impurity, print_tree, node_to_string

using Random
using CSV, DataFrames
using Printf
import Base: size

gini_score(counts) = 1.0 - sum(counts .* counts)/(sum(counts) ^2)

## --------------------- Binary Tree --------------------- ##
"""
    BinaryTree()

A binary tree implemented as 2 parallel arrays.

Available methods are: `add_node!`, `set_left_child`, `set_right_child`, `get_children`, `is_leaf`, `size`, `nleaves`, `find_depths`, `get_max_depth`
"""
mutable struct BinaryTree
    children_left::Vector{Int}
    children_right::Vector{Int}
    BinaryTree() = new([], [])
end

function add_node!(tree::BinaryTree)
    push!(tree.children_left, -1)
    push!(tree.children_right, -1)
    return
end

function set_left_child!(tree::BinaryTree, node_id::Int, child_id::Int)
    tree.children_left[node_id] = child_id
    return
end

function set_right_child!(tree::BinaryTree, node_id::Int, child_id::Int)
    tree.children_right[node_id] = child_id
    return
end

function get_children(tree::BinaryTree, node_id::Int)
    return tree.children_left[node_id], tree.children_right[node_id]
end

function is_leaf(tree::BinaryTree, node_id::Int)
    return tree.children_left[node_id] == tree.children_right[node_id] == -1
end

nleaves(tree::BinaryTree) = count(tree.children_left .== -1)
size(tree::BinaryTree) = length(tree.children_left)

function find_depths(tree::BinaryTree)
    depths = zeros(Int, size(tree))
    depths[1] = -1
    stack = [(1, 1)] # (parent, node_id)
    while !isempty(stack)
        parent, node_id = pop!(stack)
        if node_id == -1
            continue
        end
        depths[node_id] = depths[parent] + 1
        left, right = get_children(tree, node_id)
        push!(stack, (node_id, left))
        push!(stack, (node_id, right))
    end
    return depths
end

"""
    get_max_depth(tree::BinaryTree; node_id=0) => Int

Calculate the maximum depth of the tree
"""
function get_max_depth(tree::BinaryTree; node_id=1)
    if is_leaf(tree, node_id)
        return 0
    end
    left, right = get_children(tree, node_id)
    return max(get_max_depth(tree, node_id=left), get_max_depth(tree, node_id=right)) + 1
end



## --------------------- Decision Tree Classifier --------------------- ##
"""
    DecisionTreeClassifier{T}([max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG])
    DecisionTreeClassifier([max_depth=nothing], [max_features=nothing], [min_samples_leaf=1], [random_state=Random.GLOBAL_RNG]) -> T=Float64

A random forest classifier.

Available methods are:
`fit!`, `predict`, `predict_prob`, `predict_row`, `predict_batch`, `score`,
`feature_importance_impurity`, `perm_feature_importance`
`print_tree`, `node_to_string`
"""
mutable struct DecisionTreeClassifier{T} <: AbstractClassifier
    #internal variables
    num_nodes::Int
    binarytree::BinaryTree
    n_samples::Vector{Int} # total samples per each node
    values::Vector{Vector{Float64}} # samples per class per each node. Float64 to speed up calculations
    impurities::Vector{Float64}
    split_features::Vector{Union{Int, Nothing}}
    split_values::Vector{Union{T, Nothing}} #Note: T is the same for all values
    n_features::Union{Int, Nothing}
    n_classes::Union{Int, Nothing}
    features::Vector{String}
    feature_importances::Union{Vector{Float64}, Nothing}

    # external parameters
    max_depth::Union{Int, Nothing}
    max_features::Union{Int, Nothing} # sets n_features_split
    min_samples_leaf::Int
    random_state::Union{AbstractRNG, Int}

    DecisionTreeClassifier{T}(;
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf=1,
        random_state=Random.GLOBAL_RNG
        ) where T = new(
            0, BinaryTree(), [], [], [], [], [], nothing, nothing, [], nothing,
            max_depth, max_features, min_samples_leaf, check_random_state(random_state)
            )
end
DecisionTreeClassifier(;
    max_depth=nothing,
    max_features=nothing,
    min_samples_leaf=1,
    random_state=Random.GLOBAL_RNG
) = DecisionTreeClassifier{Float64}(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=check_random_state(random_state)
    )


function Base.show(io::IO, tree::DecisionTreeClassifier)
    rng = tree.random_state
    if hasproperty(rng, :seed)
        str_rng = string(typeof(rng), "($(tree.random_state.seed),...)")
    else
        str_rng = string(typeof(rng))
    end
    str_out = string(
        typeof(tree), "(",
        "num_nodes=$(tree.num_nodes)",
        ", max_depth=$(tree.max_depth)",
        ", max_features=$(tree.max_features)",
        ", min_samples_leaf=$(tree.min_samples_leaf)",
        ", random_state="*str_rng,
        ")"
    )
    println(io, str_out)
end

function node_to_string(tree::DecisionTreeClassifier, node_id::Int)
    n_samples = tree.n_samples[node_id]
    value = tree.values[node_id]
    impurity = tree.impurities[node_id]
    s = @sprintf("n_samples: %d; value: %s; impurity: %.4f",
                  n_samples, value, impurity)
    ## 判断当前节点(node_id)是否是叶子节点
    if !is_leaf(tree.binarytree, node_id)
        split_name = tree.features[tree.split_features[node_id]]
        #split_val = tree.split_values[node_id]
        #s *= @sprintf("; split: %s<=%.3f", split_name, split_val)
        s *= @sprintf("; split: %s is false", split_name)
    end
    return s
end

function print_tree(tree::DecisionTreeClassifier)
    depths = find_depths(tree.binarytree)
    for (i, node) in enumerate(1:size(tree.binarytree))
        d = depths[node]
        s = @sprintf("%03d ", i)
        println(s, "-"^d, node_to_string(tree, node))
    end
    return
end

size(tree::DecisionTreeClassifier) = size(tree.binarytree)
get_max_depth(tree::DecisionTreeClassifier) = get_max_depth(tree.binarytree)

## --------------------- fitting --------------------- ##
## 该函数启动一个递归调用，该调用使树增长，直到达到停止条件。
function fit!(tree::DecisionTreeClassifier, X::DataFrame, Y::DataFrame)
    @assert size(Y, 2) == 1 "Output Y must be an m x 1 DataFrame"

    # set internal variables
    tree.n_features = size(X, 2)
    # tree.n_classes = size(unique(Y), 1)
    tree.features = names(X)

    # fit
    split_node!(tree, X, Y, 0)

    # set attributes
    ## 对每个节点计算打分，该打分根据该节点的gini_impurity和该节点的左右叶子节点的impurity进行打分
    tree.feature_importances = feature_importance_impurity(tree)

    return
end

# Y values must be converted into 1,2,3 for categorical levels
function count_classes(Y, n::Int)
    counts = zeros(n)
    for entry in eachrow(Y)
        counts[entry[1]] += 1.0
    end
    return counts
end

function set_defaults!(tree::DecisionTreeClassifier, Y::DataFrame)
    ## 计算每个类的样本数
    values = count_classes(Y, tree.n_classes)
    ## 存储每个类的样本数
    push!(tree.values, values)
    ## gini_score = 1 - (values值的平方)的加和/(values值的加和)的平方
    push!(tree.impurities, gini_score(values))
    push!(tree.split_features, nothing)
    push!(tree.split_values, nothing)
    push!(tree.n_samples, size(Y, 1))
    ## 增加一个node，值取默认值[-1][-1]
    add_node!(tree.binarytree)

end

# x: one feature of the samples, bool (true/false) values
# y: classes (integer values, 1..n_class)
## 对每个特征计算与Y的关系，按照(gini_score(lhs_count) * sum(lhs_count) + gini_score(rhs_count) * sum(rhs_count))/n_samples，其中lhs_count和rhs_count表示x的false和true中y的各类别的数量
function gini_impurity(x::AbstractVector, y::AbstractVector, n_classes::Int)
    n_samples = length(x)
    ## sortperm升序排列x
	order = sortperm(Bool.(x))
    ## 按照x的升序顺序排列x和y
	x_sort, y_sort = (Bool.(x))[order], y[order]
    ## 找到第一个1(ture)所在的行号
	n_falses = findfirst(x_sort) 
	# all features are falses
	if isnothing(n_falses) || n_falses == 1
		return gini_score(count_classes(y_sort, n_classes)) == 0 ? 0 : Inf
	end
	n_falses -= 1
    ## x为false时，y中每个类别的数量
	lhs_count = count_classes(y_sort[1:n_falses],     n_classes)
    ## x为true时，y中每个类别的数量
	rhs_count = count_classes(y_sort[n_falses+1:end], n_classes)
    ## 1 - gini指数越小表示其中Y的类别越均匀
	return (gini_score(lhs_count) * sum(lhs_count) + gini_score(rhs_count) * sum(rhs_count))/n_samples
end

function split_node!(tree::DecisionTreeClassifier, X::DataFrame, Y::DataFrame, depth::Int)
    tree.num_nodes += 1
    node_id = tree.num_nodes
    ## 配置初始节点参数
    set_defaults!(tree, Y)
    ## tree.impurities由gini_score计算得到
    if tree.impurities[node_id] == 0.0
        return # only one class in this node
    end

    # random shuffling ensures a random variable is used if 2 splits are equal or if all features are used
    ## 如果特征数大于tree.max_features，则随机抽取tree.max_features个特征
    n_features_split = isnothing(tree.max_features) ? tree.n_features : min(tree.n_features, tree.max_features)
    ## 随机生成1 ~ tree.n_features中的tree.n_features个随机数
    features = randperm(tree.random_state, tree.n_features)[1:n_features_split]

	# find the best split
    ## 对每个特征计算与Y的关系，按照(gini_score(lhs_count) * sum(lhs_count) + gini_score(rhs_count) * sum(rhs_count))/n_samples，其中lhs_count和rhs_count表示x的false和true中y的各类别的数量
	## 使用features随机分配的数，这样可以随机特征开始
    scores = [gini_impurity(X[:, i], Y[:,1], tree.n_classes) for i in features]
	## 找到最小的scores的值和位置
    best_score, i_best = findmin(scores)
    ## 提取对应的特征
	feature_idx = features[i_best]
	tree.split_features[node_id] = feature_idx
	tree.split_values[node_id] = false

    if best_score == Inf
        return # no split was made
    end

    # make children
    if isnothing(tree.max_depth) || (depth < tree.max_depth)
        ## 提取score分数最小的特征
        x_split = X[:, tree.split_features[node_id]]
        # lhs = x_split .<= tree.split_values[node_id]
        # rhs = x_split .>  tree.split_values[node_id]
		lhs = .!x_split  #lhs =  x_split .== false
		rhs =   x_split  #  x_split .== true
        ## tree.binarytree.children_left[node_id] = tree.num_nodes + 1
        set_left_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        ## 提取X和Y中 X的第tree.split_features[node_id]列为false 的行，输入split_node函数
        ## 其中计算的tree.values表示该得分最高的特征中为false时，Y中各分类的样本数
        split_node!(tree, X[lhs, :], Y[lhs, :], depth + 1)
        ## tree.binarytree.children_right[node_id] = tree.num_nodes + 1
        ## 其中计算的tree.values表示该得分最高的特征中为true时，Y中各分类的样本数
        set_right_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        ## 提取X和Y中 X的第tree.split_features[node_id]列为true 的行，输入split_node函数
        split_node!(tree, X[rhs, :], Y[rhs, :], depth + 1)
    end

    return
end

# NOTE: for REOs, each entry in X is false/true (0/1) only.
# After sorting, x and y look like
# x: 0 0 ... 0 1 1 1 ... 1
# y: 2 1 ... 3 1 2 2 ... 1
## 调用“find_better_split（）”以确定拆分节点的最佳功能。这是一种贪婪的方法，因为它扩展了基于树的方法 关于目前最好的功能。
function find_better_split(feature_idx, X::DataFrame, Y::DataFrame, node_id::Int,
                           best_score::AbstractFloat, tree::DecisionTreeClassifier)
    x = X[:, feature_idx]

    n_samples = length(x)

	#NOTE: by default, 'sort' and 'sortperm' are in the increasing order.
	#For REO outcomes (true and false), this will put all 'falses' (0s) before 'trues (1s)'.
    order = sortperm(x)
    x_sort, y_sort = x[order], Y[order, 1]

	#NOTE: needs new implementation for 'count_classes' for REOs? seems not...
	#rhs_count records number of cases in each class; it has nothing to do with ordering.
    rhs_count = count_classes(y_sort, tree.n_classes)
    lhs_count = zeros(tree.n_classes)

	#NOTE: 'similar' (to obtain DataType)
	#Find the best 
    xi, yi = zero(x_sort[1]), zero(y_sort[1]) # declare variables used in the loop (for optimisation purposes)
    for i in 1:(n_samples-1)
        global xi = x_sort[i]
        global yi = y_sort[i]
        lhs_count[yi] += 1.0; rhs_count[yi] -= 1.0
        if (xi == x_sort[i+1]) || (sum(lhs_count) < tree.min_samples_leaf)
            continue
        end
        if sum(rhs_count) < tree.min_samples_leaf
            break
        end
        # Gini impurity
        curr_score = (gini_score(lhs_count) * sum(lhs_count) + gini_score(rhs_count) * sum(rhs_count))/n_samples
		# NOTE: decide which side the sample will be assigned to, left or child?
		# Here, just record the best split 'feature_idx' and the 'threshold value' (middle point the current and the next one)
		# For REOs, we need to decide true or false for which outcome
        if curr_score < best_score
            best_score = curr_score
            tree.split_features[node_id] = feature_idx
            tree.split_values[node_id]= (xi + x_sort[i+1])/2
        end
    end
    return best_score
end

## --------------------- prediction --------------------- ##
"""
    predict_row(tree::DecisionTreeClassifier, xi<: DataFrameRow) => Vector{Int}

Returns the counts at the leaf node for a sample xi given a fitted DecisionTreeClassifier.
"""
## 返回样本xi在叶节点上的计数，xi表示单个样本
function predict_row(tree::DecisionTreeClassifier, xi::T ) where T <: DataFrameRow
    next_node = 1
    ## 判断当前节点(next_node)是否是叶子节点。对该样本在树中进行判断。
    while !is_leaf(tree.binarytree, next_node)
        ## 提取当前节点的左右叶子节点号
        left, right = get_children(tree.binarytree, next_node)
        #next_node = xi[tree.split_features[next_node]] <= tree.split_values[next_node] ? left : right
        ## tree.split_features[next_node]表示当前节点的特征号
        ## 在该节点对样本中该特征的值进行判断，是true（right）还是flase（left），获得下一个节点号
        next_node = xi[tree.split_features[next_node]] ? right : left # left: false, right true
    end
    ## 获得最后一个节点号时的各类别样本数
    return tree.values[next_node]
end

"""
    predict_batch(tree::DecisionTreeClassifier, X::DataFrame; node_id=1) => Matrix

Predict normalized weighting for each class for a dataset X given a fitted DecisionTreeClassifier.
Predictions are done in batches -> all samples which follow the same path along the tree will be entered at the same time.
node_id is for internal use and should not be given as a parameter.
"""
function predict_batch(tree::DecisionTreeClassifier, X::DataFrame; node_id=1)
    # more complex and no speed increase
    if  (size(X, 1) == 0)
        return
    elseif is_leaf(tree.binarytree, node_id)
        counts = tree.values[node_id]
        return transpose(counts/sum(counts))
    end
    x_split = X[:, tree.split_features[node_id]]
    #lhs = x_split .<= tree.split_values[node_id]
    #rhs = x_split .>  tree.split_values[node_id]
	lhs = .!x_split
	rhs =   x_split

    ## 第node_id节点的左右叶子节点的node_id
    left, right = get_children(tree.binarytree, node_id)

    ## tree.n_classes表示Y中的类别数
    probs = zeros(nrow(X), tree.n_classes)
    ## 判断Bool集合中是否包含true
    if any(lhs)
        probs[lhs, :] .= predict_batch(tree, X[lhs, :], node_id=left)
    end
    ## 判断Bool集合中是否包含true
    if any(rhs)
        probs[rhs, :] .= predict_batch(tree, X[rhs, :], node_id=right)
    end
    return probs
end

"""
    predict_prob(tree::DecisionTreeClassifier, X::DataFrame) => Matrix

Predict normalized weighting for a dataset X given a fitted DecisionTreeClassifier.
Predictions are done individually for each sample in X.
"""
function predict_prob(tree::DecisionTreeClassifier, X::DataFrame)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    probs  = zeros(nrow(X), tree.n_classes)
    ## 计算单棵树中，对每个样本的预测能力。获得最后一个节点号时的各类别样本数
    for (i, xi) in enumerate(eachrow(X))
        ## 计算单棵树中，获得最后一个节点号时的各类别样本数
        counts = predict_row(tree, xi)
        probs[i, :] .= counts/sum(counts)
    end
    return probs
end

"""
    predict(tree::DecisionTreeClassifier, X::DataFrame) => Matrix

Predict classes for a dataset X given a fitted DecisionTreeClassifier.
Predictions are done individually for each sample in X.
"""
function predict(tree::DecisionTreeClassifier, X::DataFrame)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    probs = predict_prob(tree, X)
    ## 计算哪个特征预测率最高
    return mapslices(argmax, probs, dims=2)[:, 1]
end


## --------------------- feature importances --------------------- ##
"""
    nleaves(tree::DecisionTreeClassifier) => Int

Return the number of leaves in the DecisionTreeClassifier.
nleaves = size(binarytree) - nodes(binarytree)
"""
nleaves(tree::DecisionTreeClassifier) = nleaves(tree.binarytree)

"""
    feature_importance_impurity(tree::DecisionTreeClassifier) => Vector{Float64}

Calculate feature importance based on impurity.
For each feature, this is the weighted sum of the decrease in impurity that each node where that feature is used achieves.
The weights are the proportion of training samples present in each node out of the total samples.
"""
## 对每个节点计算打分，该打分根据该节点的gini_impurity和该节点的左右叶子节点的impurity进行打分
function feature_importance_impurity(tree::DecisionTreeClassifier)
    if tree.num_nodes == 0
        throw(NotFittedError(:tree))
    end
    feature_importances = zeros(tree.n_features)
    total_samples = tree.n_samples[1]
    ## 对每个节点计算打分，该打分根据该节点的gini_impurity和该节点的左右叶子节点的impurity进行打分
    for node in 1:length(tree.impurities)
        if is_leaf(tree.binarytree, node)
            continue
        end
        ## 提取出当前节点的信息
        ### tree.split_features[node]表示当前节点选取的特征的列号
        spit_feature = tree.split_features[node]
        impurity = tree.impurities[node]
        n_samples = tree.n_samples[node]
        # calculate score
        ## 第node_id节点的左右叶子节点的node_id
        left, right = get_children(tree.binarytree, node)
        ## 左右叶子节点的gini_impurity的值
        lhs_gini = tree.impurities[left]
        rhs_gini = tree.impurities[right]
        ## 计算该左右叶子节点时的样本数
        lhs_count = tree.n_samples[left]
        rhs_count = tree.n_samples[right]
        ## 计算当前节点的叶子节点的打分
        score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
        # feature_importances      = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
        ## 计算每棵树中特征的平均feature_importances。其中每棵树中每个特征的得分是，根据当前节点的impurity和左右节点的打分（根据左右叶子节点的gini_impurity的值和该左右叶子节点时的样本数计算，score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples）计算获得，feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    end
    # normalise
    ## 标化打分
    feature_importances = feature_importances/sum(feature_importances)
    return feature_importances
end
