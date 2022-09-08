using Graphs, TickTock, MatrixMarket, Random, SparseArrays, GraphIO
using LinearOrdering, Coarsening
ENV["TICKTOCK_MESSAGES"] = false

include("./contractiontree.jl")

onesum = PSum(1)

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

function get_cost_over_time(gnames, graphs, seconds, costfn, mergefn)
    println("Name Cost Seed Time")
    for (name, G) in Iterators.zip(gnames, graphs)
        bestcost, bestseed, besttime = Inf, 1, 0
        n = nv(G)
        A = adjacency_matrix(G)
        table = fill((0.0, 0), (n, n))
        between = zeros(n)

        tick()
        for i in 1:50
            config = config_gen(i)
            position_to_idx, idx_to_position = ordergraph(onesum, G; config...);
            pmap = PositionMap(position_to_idx, idx_to_position)
            cost = _iter_width!(table, between, A, pmap; cost=costfn, merge=mergefn)[1, end][1]
            t = peektimer()
            if t >= seconds
                break
            end
            if cost <= bestcost
                bestcost, bestseed, besttime = cost, i, t
            end
        end
        tok()
        println("$name $bestcost $bestseed $besttime")
    end
end

rrfiles = [
    "./graphs/rr/regular$(d)_32_$(p)_$(seed).mtx" for d in 3:5 for p in 2:5 for seed in 0:9
];
rradj = makeadj.(mmread.(rrfiles));
rr = SimpleGraph.(rradj);

# println("Max Ops (NOTE: Equal to TW + 1)")
# get_cost_over_time(rrfiles, rr, 5, tops, max)
# println("Max Size")
# get_cost_over_time(rrfiles, rr, 5, tsize, max)
# println("Total Ops")
# get_cost_over_time(rrfiles, rr, 5, tops, log2sumexp2)

qbfiles = [
           "./graphs/qubits/regular_qubit_3_$(n)_2_$(seed).mtx" for n in [96] for seed in 0:9
];
qbadj = makeadj.(mmread.(qbfiles));
qb = SimpleGraph.(qbadj);

println("Total Ops - Scaling Qubits")
get_cost_over_time(qbfiles, qb, 10, tops, log2sumexp2)

