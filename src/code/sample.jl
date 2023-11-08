"""
    collect_pool(F1::AbstractMatrix, auc::AbstractVector, aucp::Number=0.6, aucn::Number =0.4)

Summary the occurence of the first features and the AUC for all the samples in the initial pool. 

### Arguments
`F1`::AbstractMatrix` is the matrix to store if a feature (row) appears as the first one in asample (column).
`auc::AbstractVector` is a vector of the best AUCs for each sample.
`aucp::Number` is the lower AUC threshold for a postive model (default 0.6).
`aucn::Number` is the upper AUC threshold for a negative model (default 0.4).
"""

using StatsBase, Random

function collect_pool(F1::AbstractMatrix, auc::AbstractVector, aucp::Number=0.6, aucn::Number =0.4)
	m,n = size(F1) # m, number of features, n number of samples
	n  == length(auc) || throw("`auc` does not match with `F1`.")
	pos = (1:n)[auc .>=  aucp] # Indices to samples
	neg = (1:n)[auc .<=  aucn]
	ran = setdiff(1:n, pos, neg)
	oa  = reshape(sum(F1, dims = 2), :)
	#op  = reshape(sum(F1[:, pos], dims = 2), :)
	#on  = reshape(sum(F1[:, neg], dims = 2), :)
	ma  = (1:m)[oa .> 0]
	#indices to the samples for each feature as the first ones
	mnp = [pos[F1[i, pos] .> 0] for i in ma] # Vector of vectors
	mnn = [neg[F1[i, neg] .> 0] for i in ma]
	return (pos, neg, ran, length.(mnp), length.(mnn), oa[ma], ma, mnp, mnn)
end


"""
    adjust_counts(op, on, oa, np, nn, th)

Adjust the counts for the first occurences for each feature to a value for unbiased sampling.

### Arguments
`op, on, oa` are the vectors the priority values (the times as the first selected features in the model) in the postive (AUC > 0.6),  negative (AUC < 0.4) and all samples, respectively, for the selected features (as the first feature for at least once). They should have equal length.
`np` and `nn` are the numbers of positive and negative draws, respectively.
`th` is a threshold for high-frequency features

"""

function adjust_counts( op::AbstractVector,
						on::AbstractVector,
						oa::AbstractVector,
						np::Integer,
						nn::Integer,
						th::Integer = 50
					   )
	n1 = length(op)
	rp = zeros(Int64, n1) # adjusted priority for postive features
	rn = zeros(Int64, n1) # ... negative ...
	# select high-freq features
	th1 = sample(1:floor(Int64,min(quantile(oa, 0.75), th, min(findmax(op)[1],findmax(on)[1]))))
	sel = op .> th1 .&& on .> th1
	sel = (1:n1)[sel]
	n2 = sample(1:length(sel))
	sel = sample(sel, n2, replace = false) # random subset
	# assign randomly a non-zero priority to this subset
	rp[sel] = sample(1:th1, n2, replace = true)
	rn[sel] = sample(1:th1, n2, replace = true)
	while sum(rp) < np
		z   = ceil(Int64, min(0.5*(np - sum(rp)), 0.5*sum(rp .< op)))
		if z < 1
			np -= 1
		end
		sel = (1:n1)[rp .< op]
		sel = sample(sel, z, replace = false)
		rp[sel] .+= 1
	end
	rn[ rp .== 0] .= 1
	while sum(rn) < nn
		z   = ceil(Int64, min(0.5*(nn - sum(rn)), 0.5*sum(rn .< on)))
		if z < 1
			nn -= 1
		end
		sel = (1:n1)[rn .< on]
		sel = sample(sel, z, replace = false)
		rn[sel] .+= 1
	end
	return rp, rn 
end

"""
    transfer_counts_to_samples(n_samples, rp, rn, mnp, mnn)

Transfer the adjust counts for each sample to the sampling weights for each sample in the pool.

`rp`, `rn`, `mnp`, `mnn` should have the same length.
We transfer the counts stored in `rp` and `rn` to the samples 
for each feature in `mnp` and `mnn`, respectively.
"""

function transfer_counts_to_samples(n_samples::Integer, 
									rp::AbstractVector, 
									rn::AbstractVector,
									mnp::AbstractVector,
									mnn::AbstractVector
									)
	n = length(rp)
	n == length(rn) == length(mnp) == length(mnn) || throw("`rp`, `rn`, `mnp`, `mnn` should have the same length.")
	ns = zeros(Int64, n_samples) # weight vector for each sample
	for i in 1:n
		ns[mnp[i]] .= rp[i]
		ns[mnn[i]] .= rn[i]
	end
	return ns
end

"""
     sample_pool(pool::AbstractVector, n::Integer, w::AbstractVector)

Sample the positive or negative samples using the given weights.
"""

function sample_pool(pool::AbstractVector, n::Integer, w::AbstractVector)
	length(pool) == length(w) || throw("`pool` must have the same length as `w`.")
	n>0 || throw("`n` must be a positive integer.")
	sample(pool, Weights(w), n, replace = true)
end

#Original sampling method
function sample_pool(w::AbstractVector, mn::AbstractVector, s_mn::Vector = [])
	n = length(w)
	length(mn) == n || throw("`mn` must have the same length as `w`.")
	for i in 1:n
		if mn[i] == []
			continue
		end
		s1_mn = sample(mn[i], w[i], replace = true)
		s_mn = vcat(s_mn,s1_mn)
	end
	return s_mn
	# reduce(vcat, [sample(mn[i], w[i], replace = true) for i in 1:n])
end

"""
     get_draw_sizes(n::Integer, npos_min::Integer, npos_max::Integer)

Return the numbers of positive and negative draws.

"""

function get_draw_sizes(n::Integer, npos_min::Integer, npos_max::Integer)
	 n > npos_max > npos_min > 0 || throw("Input numbers is not reasonable.")
	 np = npos_min -1 + sample(1:npos_max-npos_min + 1)
	 nn = sample(1:n-np-1)
	 return np, nn
end

"""
     score_features(ss,fe, pos, neg)

Return the final scores for each feature.

`ss` is the sample index vector. Please note that duplicated indices are expected.
`fe` is the feature matrix for all the samples in the pool (should be 0/1 only).
`pos` is the vector for all the positive samples in the pool.
`neg` is the vector for all the negative samples in the pool.
"""

function score_features(ss::AbstractVector, 
						fe::AbstractMatrix, 
						pos::AbstractVector,
						neg::AbstractVector
						)
	m,n = size(fe)
	scores = sum(fe[:,ss[map(x-> x ∈ pos, ss)]], dims = 2) .- sum(fe[:,ss[map(x-> x ∈ neg, ss)]], dims = 2)
	scores = reshape(scores, :)
	o = sortperm(scores, rev = true)
	return scores[o], (1:m)[o]
end


function sampling(poolFirsts::AbstractMatrix,
				  aucs::AbstractVector,
				  feats::AbstractMatrix;
				  aucp::Number = 0.6, 
				  aucn::Number = 0.4,
				  n::Int64 = 500,
				  sp_choice::Int = 0)
	pos, neg, ran,  op, on, oa, ma, mnp, mnn = collect_pool(poolFirsts, reshape(aucs,:))

	n = 500
	np, nn = get_draw_sizes(n, 101, 300)

	nz = n - np - nn 

	rp, rn = adjust_counts(op, on, oa, np, nn, 50)


	ns = transfer_counts_to_samples(2000, rp, rn, mnp, mnn)
	if sp_choice == 1
		#----------------------
		# choice 1
		# re-sampled samples
		spos = sample_pool(pos, np, ns[pos])
		sneg = sample_pool(neg, nn, ns[neg])
		sran = sample(ran, 500 -length(spos) - length(sneg), replace = true)
	else
		#----------------------
		# choice 2
		#  originial implmentation

		spos = sample_pool(rp, mnp)
		sneg = sample_pool(rn, mnn)

		sran = sample(ran, 500 -length(spos) - length(sneg), replace = true)
	end
	ss = vcat(spos, sneg, sran)

	scores, fea = score_features(ss, feats, pos, neg)

	return scores, fea
end

# 	scores[1:15]
# 	fea[1:15]





# #########################################
# ## Tests
# using DelimitedFiles

# poolFirsts = readdlm("/Users/hwang/Downloads/GET_TRAINING_SET/POOL_7.txt", '\t', Int64, '\n')
# feats = readdlm("/Users/hwang/Downloads/GET_TRAINING_SET/POOL_6.txt", '\t', Int64, '\n')

# aucs = readdlm("/Users/hwang/Downloads/GET_TRAINING_SET/POOL_5.txt", '\t', Float64, '\n')

# pos, neg, ran,  op, on, oa, ma, mnp, mnn = collect_pool(poolFirsts, reshape(aucs,:))

# n = 500
# np, nn = get_draw_sizes(n, 101, 300)

# nz = n - np - nn 

# rp, rn = adjust_counts(op, on, oa, np, nn, 50)


# #----------------------
# # choice 1
# ns = transfer_counts_to_samples(2000, rp, rn, mnp, mnn)

# # re-sampled samples
# spos = sample_pool(pos, np, ns[pos])
# sneg = sample_pool(neg, nn, ns[neg])
# sran = sample(ran, 500 -length(spos) - length(sneg), replace = true)


# #----------------------
# # choice 2
# #  originial implmentation

# spos = sample_pool(rp, mnp)
# sneg = sample_pool(rn, mnn)

# sran = sample(ran, 500 -length(spos) - length(sneg), replace = true)

# ss = vcat(spos, sneg, sran)

# scores, fea = score_features(ss, feats, pos, neg)

# scores[1:15]
# fea[1:15]
