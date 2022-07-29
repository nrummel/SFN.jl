using Plots
using JLD2: load

data = load("results/efficiency_data.jld2")

labels = data["labels"]
features = data["features"]
t_arr = data["time"]
m_arr = data["memory"]

#Plot time
plot(features, t_arr[:,1:3],
    xlabel="Problem Dimension",
    ylabel="Execution Time (ms)",
    w=3,
    label=reshape(labels[1:3], 1, 3),
    legend=:topleft,
    yaxis=:log)
savefig("figures/newton-variants-time-ad.pdf")

plot(features, t_arr[:,[1,4,5]],
    xlabel="Problem Dimension",
    ylabel="Execution Time (ms)",
    w=3,
    label=reshape(labels[[1,4,5]], 1, 3),
    legend=:topleft,
    yaxis=:log)
savefig("figures/newton-variants-time-matrix.pdf")

#Plot memory
plot(features, m_arr[:,1:3],
    xlabel="Problem Dimension",
    ylabel="Memory Usage (MiB)",
    w=3,
    label=reshape(labels[1:3], 1, 3),
    legend=:topleft,
    yaxis=:log)
savefig("figures/newton-variants-memory-ad.pdf")

plot(features, m_arr[:,[1,4,5]],
    xlabel="Problem Dimension",
    ylabel="Memory Usage (MiB)",
    w=3,
    label=reshape(labels[[1,4,5]], 1, 3),
    legend=:topleft,
    yaxis=:log)
savefig("figures/newton-variants-memory-matrix.pdf")
