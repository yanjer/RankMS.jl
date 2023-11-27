using Plots
import Base: size

## 4,result

function plot_forest(classifier::RandomForestClassifier, features::Matrix,fn_stem::String) #,local_feature_tag::Missing=missing,distance_between_trees::Int=0
    p2 = plot()
    max_tree_j = 0
    for i in 1:classifier.n_trees
        if isnothing(classifier.trees[i].split_features[1])
            continue
        end
        left_i = classifier.trees[i].binarytree.children_left
        right_i = classifier.trees[i].binarytree.children_right
        ## left_i = [2,3,4,-1,-1,-1,8,-1,-1]
        ## right_i = [7,6,5,-1,-1,-1,9,-1,-1]
        l_n = size(left_i)[1]
        x=0
        y=0
        node_coordinate = hcat(1, zeros(Int,1,2), 1, Int64(sum(classifier.trees[i].values[1])), [classifier.trees[i].values[1]], classifier.trees[i].impurities[1],classifier.trees[i].split_features[1])
        for j in 1:l_n
            if (left_i[j] == -1)
                continue
            end
            if j <= 2
                x,y = node_coordinate[j,2:3]
            else
                n_node = (node_coordinate[:,1] .== j) # 当前节点所在的行
                x,y = node_coordinate[n_node,2:3]
                if right_i[right_i[j]] != -1 && left_i[left_i[j]] != -1
                    srow_distance = (x - 1) - findmax(node_coordinate[(node_coordinate[:,3] .== y - 1) .&& (node_coordinate[:,1] .< j),2])[1]
                    if (sum((node_coordinate[:,2] .== x - 1) .&& (node_coordinate[:,3] .== y - 1)) > 0) && srow_distance <= 0
                        node_coordinate[n_node,2] = node_coordinate[n_node,2] .+ 4
                    end
                    x,y = node_coordinate[n_node,2:3]
                    node_coordinate = vcat(node_coordinate,[left_i[j] x-1 y-1 j Int64(sum(classifier.trees[i].values[j])) [classifier.trees[i].values[j]] classifier.trees[i].impurities[j] classifier.trees[i].split_features[j]])
                    node_coordinate = vcat(node_coordinate,[right_i[j] x+3 y-1 j Int64(sum(classifier.trees[i].values[j])) [classifier.trees[i].values[j]] classifier.trees[i].impurities[j] classifier.trees[i].split_features[j]])
                    continue
                end
            end
            node_coordinate = vcat(node_coordinate,[left_i[j] x-1 y-1 j Int64(sum(classifier.trees[i].values[j])) [classifier.trees[i].values[j]] classifier.trees[i].impurities[j] classifier.trees[i].split_features[j]])
            node_coordinate = vcat(node_coordinate,[right_i[j] x+1 y-1 j Int64(sum(classifier.trees[i].values[j])) [classifier.trees[i].values[j]] classifier.trees[i].impurities[j] classifier.trees[i].split_features[j]])
        end
        # node_coordinate包含 当前节点 节点坐标x值，节点坐标y值，samples，value，impurity,split_features
        node_coordinate[:,end-1] = round.(node_coordinate[:,end-1],sigdigits = 3)
        tree_distance = (findmin(node_coordinate[:,2])[1] - max_tree_j)
        if (i != 1) && tree_distance <= 0
            node_coordinate[:,2] = node_coordinate[:,2] .- tree_distance .+ 1
        end
        max_tree_j = findmax(node_coordinate[:,2])[1]
        plot_tree(node_coordinate,features,l_n)
    end
    # yticks!([findmin(node_coordinate[:,3])[1]:1:0;])
    savefig(p2, join([fn_stem,"Forest.pdf"], "_"))
end

function plot_tree(node_coordinate::Matrix,features::Matrix,l_n::Int64)
    dot1_2=[0,0]
    for c in 2:l_n
        dot1 = node_coordinate[c,2:3]
        dot2 = vec(node_coordinate[(node_coordinate[:,1] .== node_coordinate[c,4]),2:3])
        plot!([dot1[1],dot2[1]],[dot1[2],dot2[2]],lc=[:black],label=false,aspect_ratio =:equal)
    end
    for i in unique(node_coordinate[:,4])
        l_features = node_coordinate[(node_coordinate[:,4] .== i),:][end-1:end,:] # 因为第一个节点的父节点设置的是自己，所以要去掉
        f1 = l_features[1,end]
        f2 = l_features[2,end]
        # 该子节点所在的行
        l_1 = node_coordinate[(node_coordinate[:,1] .== i),2:3]
        dot1_2 = (reshape(l_features[1,2:3],1,2) .+ l_1)./2
        dot1_3 = (reshape(l_features[2,2:3],1,2) .+ l_1)./2
        annotate!([(dot1_2[1]-0.1,dot1_2[2],text(string(features[f1,1]," ≤ ",features[f2,2]),2,:right))])
        annotate!([(dot1_3[1]+0.1,dot1_3[2],text(string(features[f1,1]," > ",features[f2,2]),2,:left))])
    end
    for i in 1:l_n
        annotate!([(node_coordinate[i,2],node_coordinate[i,3],text(string("samples: $(node_coordinate[i,5])","\n","values: $(node_coordinate[i,6])","\n","impurity: $(node_coordinate[i,7])"),2))])
    end
    plot!(node_coordinate[:,2],node_coordinate[:,3],seriestype=:scatter,title="Tree",ms = 19,label=false,mc = "#FFFFFF", aspect_ratio =:equal, markershape=:rect) #"#B7E9FA"
end
