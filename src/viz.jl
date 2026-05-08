# Interactive "Grok-1 MoE Atmosphere" visualization (GLMakie)
# Four regions:
#   A. Global MoE heatmap (64 blocks × 8 experts) + risk strip
#   B. Selected block routing graph (input → router → experts → output)
#   C. Right inspector panel (logits, probs, entropy, SAAQ, provenance)
#   D. Bottom timeline (token slider + play/pause + seed)
#
# Operates on a real XAIReportBundle parsed from xai-dissect JSON.
# Router dynamics are simulated (synthetic W,h) on top of real metadata so the
# atmosphere stays animated; weights are never loaded from disk.
# CPU always, CUDA optional.

using GLMakie, GraphMakie, Graphs, Makie, Observables

function launch_atmosphere(bundle::XAIReportBundle; backend::ComputeBackend = CPUBackend())
    n_blocks = get(bundle.metadata, "n_blocks", 64)
    n_experts = get(bundle.metadata, "n_experts", 8)

    # --- State ---
    selected_block = Observable(1)
    token_idx = Observable(0)
    activity = Observable(zeros(Float32, n_blocks, n_experts))
    is_playing = Observable(false)
    seed = Observable(42)
    current_frame = Observable(simulate_router_frame(bundle, 1, 0; backend=backend))

    # Seed initial activity from first frame
    on(current_frame) do frame
        activity[][frame.block, :] .= frame.expert_activity
        activity[] = activity[]  # notify
    end

    # --- Figure & Layout ---
    fig = Figure(size = (1280, 820), title = "Grok-1 MoE Atmosphere — XAIDissectViz")
    grid = fig[1,1] = GridLayout(tellwidth = false, tellheight = false)

    # A. Global MoE Map (top-left)
    ax_map = Axis(grid[1, 1],
        title = "Global MoE Map (blocks 1:$n_blocks × experts 1:$n_experts)",
        ylabel = "Transformer Block",
        xlabel = "Expert",
        yreversed = true,
        yticks = 1:8:n_blocks
    )
    hm = heatmap!(ax_map, activity, colormap = :viridis, colorrange = (0, 1))
    Colorbar(grid[1, 2], hm; label = "Expert activity")

    # Risk / SAAQ side strip
    risk_ax = Axis(grid[1, 3], title = "Router Risk", yticksvisible = false, xticksvisible = false)
    risk_vals = [s.risk_score for s in bundle.saaq]
    barplot!(risk_ax, 1:n_blocks, risk_vals; direction = :x, color = risk_vals, colormap = :RdYlGn_5)
    linkyaxes!(ax_map, risk_ax)

    # Click to select block
    on(events(ax_map).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            pos = mouseposition(ax_map)
            block = clamp(round(Int, pos[2]), 1, n_blocks)
            selected_block[] = block
        end
    end

    # B. Selected Block Graph (mid-left)
    ax_graph = Axis(grid[2, 1],
        title = lift(b -> "Block $b — Router Graph", selected_block),
        aspect = DataAspect())
    hidedecorations!(ax_graph); hidespines!(ax_graph)

    # Simple node positions (input, router, 8 experts, output)
    node_pos = [ (0.0, 0.0), (1.0, 0.0),                # input, router
                 [(2.0, y) for y in range(-1.2, 1.2, length = n_experts)]...,  # experts
                 (3.5, 0.0) ]                            # merge/output
    node_colors = Observable(fill(:gray, 2 + n_experts + 1))
    node_sizes  = Observable(fill(18, 2 + n_experts + 1))

    # Edges (input->router, router->experts, experts->output)
    edge_from = Int[]; edge_to = Int[]
    push!(edge_from, 1); push!(edge_to, 2)                    # input -> router
    for e in 1:n_experts
        push!(edge_from, 2); push!(edge_to, 2 + e)            # router -> expert
        push!(edge_from, 2 + e); push!(edge_to, 2 + n_experts + 1)  # expert -> out
    end
    g = SimpleDiGraph(length(node_pos))
    for (f, t) in zip(edge_from, edge_to) add_edge!(g, f, t) end

    # Draw with GraphMakie (or fallback scatter+lines if issues)
    graphplot!(ax_graph, g;
        layout = _ -> node_pos,
        node_color = node_colors,
        node_size = node_sizes,
        edge_color = :slategray,
        edge_width = 1.5,
        arrow_show = true
    )

    # Update graph highlights when selection or frame changes
    function refresh_graph!(frame)
        top2 = frame.topk
        for i in 1:length(node_colors[])
            if i == 2
                node_colors[][i] = :royalblue   # router
                node_sizes[][i] = 26
            elseif i > 2 && i <= 2 + n_experts
                eid = i - 2
                if eid in top2
                    node_colors[][i] = :limegreen
                    node_sizes[][i] = 32 + 6 * sin(2π * (frame.token_idx % 30) / 30)  # pulse
                else
                    node_colors[][i] = :dimgray
                    node_sizes[][i] = 14
                end
            else
                node_colors[][i] = i == 1 ? :skyblue : :orange
                node_sizes[][i] = 20
            end
        end
        node_colors[] = node_colors[]  # notify
        node_sizes[] = node_sizes[]
    end

    onany(selected_block, current_frame) do blk, frame
        if frame.block != blk
            new_frame = simulate_router_frame(bundle, blk, token_idx[]; backend=backend)
            current_frame[] = new_frame
        else
            refresh_graph!(frame)
        end
    end

    # C. Inspector Panel (right)
    inspector = grid[1:2, 4] = GridLayout()
    Label(inspector[1, 1], "Inspector", fontsize = 18, font = :bold)
    Label(inspector[2, 1], lift(b -> "Block: $b", selected_block))
    Label(inspector[3, 1], lift(f -> "Top-2 Experts: $(f.topk)", current_frame))
    Label(inspector[4, 1], lift(f -> "Logits (first 4): $(round.(f.logits[1:4]; digits=3)) …", current_frame))
    Label(inspector[5, 1], lift(f -> "Probs: $(round.(f.probs; digits=3))", current_frame))
    Label(inspector[6, 1], lift(f -> "Entropy: $(round(f.entropy; digits=4))", current_frame))
    Label(inspector[7, 1], lift(selected_block) do b
        idx = clamp(b, 1, length(bundle.saaq))
        r = bundle.saaq[idx]
        "Risk: $(round(r.risk_score; digits=3)) | Readiness: $(round(r.readiness; digits=3)) | $(r.status)"
    end)
    Label(inspector[8, 1], "Provenance: $(bundle.provenance) — simulated router dynamics on real metadata")
    Label(inspector[9, 1], "Click heatmap row or use slider to change block", fontsize = 10, color = :gray)

    # D. Timeline Controls (bottom)
    timeline = grid[3, 1:4] = GridLayout()
    token_slider = Slider(timeline[1, 1], range = 0:300, startvalue = 0, width = 600)
    on(token_slider.value) do v
        token_idx[] = v
        new_frame = simulate_router_frame(bundle, selected_block[], v; backend=backend)
        current_frame[] = new_frame
    end

    play_btn = Button(timeline[1, 2], label = lift(p -> p ? "⏸ Pause" : "▶ Play", is_playing))
    on(play_btn.clicks) do _
        is_playing[] = !is_playing[]
        if is_playing[]
            @async begin
                while is_playing[]
                    token_idx[] += 1
                    if token_idx[] > 300; token_idx[] = 0; end
                    token_slider.value[] = token_idx[]   # triggers the on above
                    sleep(0.08)  # ~12 fps
                end
            end
        end
    end

    seed_box = Textbox(timeline[1, 3], placeholder = "Seed", width = 80)
    on(seed_box.stored_string) do s
        try
            seed[] = parse(Int, s)
            Random.seed!(seed[])
        catch
        end
    end

    # Initial frame
    current_frame[] = simulate_router_frame(bundle, selected_block[], token_idx[]; backend=backend)
    refresh_graph!(current_frame[])

    # Final layout tweaks
    colsize!(grid, 1, Auto())
    colsize!(grid, 4, Relative(0.28))
    rowsize!(grid, 1, Relative(0.42))
    rowsize!(grid, 2, Relative(0.38))

    display(fig)
    @info "Grok-1 MoE Atmosphere launched — use mouse on heatmap, play button, slider. Close window to exit."
    return fig
end