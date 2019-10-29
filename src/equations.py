# Equations found at: https://www.frontiersin.org/articles/10.3389/fbioe.2017.00027/full

# For each edge i, j,
#     Q(i, j) = [pi*(r^4)*deltaPI(i, j)] / 8*n*l

# For each node, i with σ(i) neighbors,
#     ∑k∈σ(i)  Q(i,k) = 0

# For each lacunae cycle ϕ(k),
#     ∑(i,j)∈ϕ(k)  ΔP(i,j) = 0


# The rate of nutrient supply along the vessel (i, j) is defined as:
#     partial_N(i,j)/partial_t = p_sub_n * (2*PI*r) * [Q(i,j) / (k_out + Q(i,j))] * [k_l / (k_l + N)]

# The availability of nutrient will activate the producer cells that will begin to consume N and produce X. The controlling equation is given as:
#     partial_X / partial_t = u_p * (N / (N + k_p)) * (k_i / (X + k_i)) * M_p + D_p*(v^2)*X

# The rate of product uptake along the vessel (i, j) is defined as:
#     partial_X(i,j)/partial_t = -P_p * (2*PI*r) * [Q(i,j) / (k_in + Q(i,j))] * [X / (k_p + X)]

# Nutrient will be consumed by the producer cells in direct correspondence to the production of X, but at a different reaction rate μn:
#     partial_N / partial_t = -u_n * (N / (N + k_p)) * (k_i / (X + k_i)) * M_p + D_p*(v^2)*N
