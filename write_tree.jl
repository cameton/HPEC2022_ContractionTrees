
using Graphs, TickTock, MatrixMarket, Random, SparseArrays, GraphIO
using LinearOrdering, Coarsening
ENV["TICKTOCK_MESSAGES"] = false

include("./contractiontree.jl")

onesum = PSum(1)


# CALL THIS
# cost = tops, tsize | merge = log2sumexp2, max
function gettree(A, maxiter, maxtime, costfn, mergefn)
    table, position_to_idx = gettable(A, maxiter, maxtime, costfn, mergefn)
    return table[1, size(A, 2)][1], recurtable(table, 1, size(A, 2)), position_to_idx
end

function config_gen(i) 
    return (
        compat_sweeps=10,
        stride_percent=0.5,
        gauss_sweeps=10,
        coarsening=VolumeCoarsening(0.4, 2.0, 5),
        coarsest=10,
        pad_percent=0.05,
        node_window_sweeps=10,
        node_window_size=1,
        seed = i
    )
end
function recurtable(table, i, j)
    if i == j
        return i
    end
    k = table[i, j][2]
    return [recurtable(table, i, k), recurtable(table, k+1, j)]
end


function gettable(A, maxiter, maxtime, costfn, mergefn)
    bestcost, bestseed, besttime = Inf, 1, 0
    G = SimpleGraph(A)
    n = nv(G)
    table = fill((0.0, 0), (n, n))
    between = zeros(n)

    tick()
    for i in 1:maxiter
        config = config_gen(i)
        position_to_idx, idx_to_position = ordergraph(onesum, G; config...);
        pmap = PositionMap(position_to_idx, idx_to_position)

        cost = _iter_width!(table, between, A, pmap; cost=costfn, merge=mergefn)[1, end][1]
        t = peektimer()
        if t >= maxtime
            break
        end
        if cost <= bestcost
            bestcost, bestseed, besttime = cost, i, t
        end
    end
    tok()

    config = config_gen(bestseed)
    position_to_idx, idx_to_position = ordergraph(onesum, G; config...);
    pmap = PositionMap(position_to_idx, idx_to_position)
    cost = _iter_width!(table, between, A, pmap; cost=costfn, merge=mergefn)[1, end][1]
    return table, position_to_idx
end
