export intersect_groudtruth
# 比较和ground truth的交集
# 比较和ground truth的交集
function intersect_groudtruth(pre_sample::Matrix,
                              groudtruth::Vector,
                              group_pre_sample::Int64,
                              group_groudtruth::Int64)
    return intersect(pre_sample[(pre_sample[:,2] .== group_pre_sample),1],groudtruth[group_groudtruth])
end