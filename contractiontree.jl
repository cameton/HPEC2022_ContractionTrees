using SparseArrays
using Random
using LinearAlgebra
using DataStructures

struct PositionMap
    p_to_v::Vector{Int} # position to vertex
    v_to_p::Vector{Int} # vertex to position
end

struct ConTree{T}
    nodes::Vector{Union{Tuple{UInt, UInt}, UInt}}
    bags::Vector{Int}
end

function _init_table!(table, A, pmap)
    fill!(table, (0.0, 1))
    vals = nonzeros(A)
    for p in size(table, 1)
        acc = 0.0
        for idx in nzrange(A, pmap.p_to_v[p])
            acc += vals[idx]
        end
        table[p] = (acc, p)
    end
    return table
end

# Costs
tsize(outgoing, _) = outgoing
tops(outgoing, shared) = outgoing + shared

# Merge
log2sumexp2(x, y) = max(x, y) + log1p(exp2(min(x, y) - max(x, y))) / log(2)

function weighted_degree(A, pmap, i, k, j)
    rows = rowvals(A)
    vals = nonzeros(A)
    left = right = out = zero(eltype(A))
    v = pmap.p_to_v[k]
    for idx in nzrange(A, v)
        r = pmap.v_to_p[rows[idx]]
        if !(i <= r <= j)
            out += vals[idx]
        elseif i <= r < k
            left += vals[idx]
        elseif k < r <= j
            right += vals[idx]
        end
    end
    return out, left, right
end

function calc_vals!(between, A, pmap, i, j) 
    outgoing = zero(eltype(A))
    outgoing, _, between[i] = weighted_degree(A, pmap, i, i, j)
    for k in (i + 1):j
        out, left, right = weighted_degree(A, pmap, i, k, j)
        outgoing += out
        between[k] = between[k - 1] + right - left
    end
    return outgoing, between
end

function iter_width(A, pmap; cost=tsize, merge=max)
    n = size(A, 1)
    table = fill((0.0, 0), (n, n))
    between = zeros(n)
    return _iter_width!(table, between, A, pmap; cost=cost, merge=merge)
end

function _iter_width!(table, between, A, pmap; cost=tsize, merge=max)
    _init_table!(table, A, pmap)
    n = length(pmap.p_to_v)
    fill!(between, 0)

    for win_size in 2:n
        for i in 1:(n-win_size+1)
            j = i + win_size - 1
            outgoing, between = calc_vals!(between, A, pmap, i, j)
            best = Inf
            bestk = -1
            for k in i:(j-1)
                v = pmap.p_to_v[k]
                b = between[k]

                c = cost(outgoing, b)
                c = merge(merge(c, table[i, k][1]), table[k+1, j][1])
                if c < best
                    best = c
                    bestk = k
                end
            end
            table[i, j] = (best, bestk)
        end
    end
    return table
end

function makeadj(B)
    n = size(B, 2)
    A = zeros(n, n)
    for i in axes(B, 1) 
        nz = findall(x -> x != 0, B[i, :])
        A[nz[1], nz[2]] = 1.0
        A[nz[2], nz[1]] = 1.0 
    end
    return A
end

