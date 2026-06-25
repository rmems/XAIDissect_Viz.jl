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
#
# IMPORTANT: GLMakie / GraphMakie / Graphs / Observables are intentionally
# NOT imported at module load time. They are loaded on demand inside
# `launch_atmosphere`, so `using XAIDissectViz` works on headless servers
# (no DISPLAY, no OpenGL) and the non-visual API
# (load_report_bundle, simulate_router_frame, router_logits, router_probs,
# topk_experts) stays callable without a UI stack.

# Public, lazy-loading entry point. Loads GLMakie & friends on first call and
# raises a clear error if they (or the underlying display/OpenGL context) are
# unavailable. The actual implementation lives in `_launch_atmosphere`.
function launch_atmosphere(bundle::XAIReportBundle; backend::ComputeBackend = CPUBackend())
    try
        @eval XAIDissectViz using GLMakie, GraphMakie, Graphs, Observables
    catch err
        error("launch_atmosphere requires GLMakie, GraphMakie, Graphs, and Observables, " *
              "plus a working display / OpenGL context. On headless servers wrap the " *
              "Julia process in xvfb-run. Underlying error: $err")
    end
    return Base.invokelatest(_launch_atmosphere, bundle; backend=backend)
end

const _ATMOSPHERE_CUDA_OK = Ref{Union{Nothing,Bool}}(nothing)

# Resolve CUDA functionality for the atmosphere, reusing the package-wide
# soft probe (env-var override + Base.find_package + cached result) so a
# CUDABackend request on a CPU-only host never eagerly imports CUDA.jl.
function _atmosphere_cuda_functional!()::Bool
    _ATMOSPHERE_CUDA_OK[] !== nothing && return _ATMOSPHERE_CUDA_OK[]::Bool
    ok = cuda_available()
    _ATMOSPHERE_CUDA_OK[] = ok
    return ok
end

function _launch_atmosphere(bundle::XAIReportBundle; backend::ComputeBackend = CPUBackend())
    n_blocks = get(bundle.metadata, "n_blocks", 64)::Int
    n_experts = get(bundle.metadata, "n_experts", 8)::Int
    n_tokens_default = 300
    cache_seed = 42

    # --- Frame cache (built once at launch) ---
    # On failure we fall back to the older per-tick `simulate_router_frame`
    # path so the viewer still works on bundles with unusual metadata.
    local cache::Union{Nothing,RouterFrameCache}
    try
        cache = build_frame_cache(bundle; backend=CPUBackend(),
                                  n_tokens=n_tokens_default, seed=cache_seed)
    catch err
        @warn "build_frame_cache failed; falling back to direct simulate_router_frame path" error=err
        cache = nothing
    end

    cuda_avail = if backend isa CUDABackend
        try
            _atmosphere_cuda_functional!()
        catch
            false
        end
    else
        false
    end

    # --- State ---
    selected_block = Observable(1)
    token_idx = Observable(0)
    activity = Observable(zeros(Float32, n_blocks, n_experts))
    is_playing = Observable(false)
    seed = Observable(cache_seed)
    play_task = Ref{Union{Nothing,Task}}(nothing)
    current_frame = Observable(simulate_router_frame(bundle, 1, 0; backend=backend, seed=seed[]))

    # FPS tracking — exponential moving average over recent play ticks.
    fps_obs = Observable(0.0)
    last_tick_time = Ref(time())

    # Reusable CPU top-k buffer (the heatmap update consumes one per tick).
    topk_buf = cache === nothing ? Matrix{Int32}(undef, 0, 0) :
                                    Matrix{Int32}(undef, n_blocks, cache.top_k)

    # Optional CUDA buffers for the per-tick activity update. Allocated only
    # when the user picked CUDABackend AND CUDA is actually functional, so the
    # default CPU path on headless / no-GPU hosts pays nothing extra.
    cuda_act = Ref{Any}(nothing)
    cuda_tk  = Ref{Any}(nothing)
    use_cuda_activity = Ref(false)
    if backend isa CUDABackend && cache !== nothing && cuda_avail
        try
            CuArray_T = Base.invokelatest(getfield, XAIDissectViz, :CuArray)
            cuda_act[] = Base.invokelatest(CuArray_T, zeros(Float32, n_blocks, n_experts))
            cuda_tk[]  = Base.invokelatest(CuArray_T, zeros(Int32, n_blocks, cache.top_k))
            use_cuda_activity[] = true
        catch err
            @warn "CUDA activity buffer init failed; per-tick activity will run on CPU" error=err
        end
    end

    # SAAQ rows are keyed by block id, not vector position. Some reports
    # include a leading "unassigned" entry or omit blocks entirely.
    saaq_by_block = Dict{Int, SAAQReadinessRecord}(s.block => s for s in bundle.saaq)

    on(current_frame) do frame
        # In fallback (no-cache) mode, the heatmap encodes the selected block's
        # per-token activity row from `simulate_router_frame`. When the cache is
        # active, the heatmap is driven by `update_activity_field!` and should
        # not be overwritten by the inspector frame.
        if cache === nothing
            activity[][frame.block, :] .= frame.expert_activity
            activity[] = activity[]  # notify
        end
    end

    # --- Per-tick activity field update ---------------------------------------
    # Pulls top-k from the cache (or recomputes via simulate_router_frame in the
    # fallback path), then evolves the n_blocks × n_experts activity field via
    # `update_activity_field!`. Reuses preallocated buffers to avoid per-tick
    # allocation in the play loop.
    function _fill_topk_buf!(t::Int)
        cache === nothing && return topk_buf
        slot = clamp(t + 1, 1, cache.n_tokens)
        @inbounds for j in 1:cache.top_k, b in 1:n_blocks
            topk_buf[b, j] = cache.topk[b, j, slot]
        end
        return topk_buf
    end

    function _step_activity!(t::Int)
        if cache === nothing
            # Fallback: only update the selected block row from the heavy frame.
            frame = simulate_router_frame(bundle, selected_block[], t;
                                          backend=backend, seed=seed[])
            activity[][selected_block[], :] .= frame.expert_activity
            activity[] = activity[]
            return
        end
        _fill_topk_buf!(t)
        if use_cuda_activity[]
            try
                Base.invokelatest(copyto!, cuda_tk[], topk_buf)
                update_activity_field!(backend, cuda_act[], cuda_tk[])
                Base.invokelatest(copyto!, activity[], cuda_act[])
            catch err
                @warn "CUDA activity tick failed; switching to CPU activity for this session" error=err
                use_cuda_activity[] = false
                update_activity_field!(CPUBackend(), activity[], topk_buf)
            end
        else
            update_activity_field!(CPUBackend(), activity[], topk_buf)
        end
        activity[] = activity[]
    end

    function _reconstruct_activity!(t::Int)
        cache === nothing && return _step_activity!(t)
        activity[] .= 0f0
        for s in 0:t
            _fill_topk_buf!(s)
            if use_cuda_activity[]
                try
                    Base.invokelatest(copyto!, cuda_act[], activity[])
                    Base.invokelatest(copyto!, cuda_tk[], topk_buf)
                    update_activity_field!(backend, cuda_act[], cuda_tk[])
                    Base.invokelatest(copyto!, activity[], cuda_act[])
                catch err
                    @warn "CUDA reconstruct failed; falling back to CPU" error=err
                    use_cuda_activity[] = false
                    update_activity_field!(CPUBackend(), activity[], topk_buf)
                end
            else
                update_activity_field!(CPUBackend(), activity[], topk_buf)
            end
        end
        activity[] = activity[]
    end

    # --- Figure & Layout ---
    fig = Makie.Figure(size = (1280, 820))
    Label(fig[0, 1], "Grok-1 MoE Atmosphere — XAIDissectViz", fontsize=22, font=:bold)
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

    # Risk / SAAQ side strip — align to 1:n_blocks even when bundle.saaq has
    # extra non-block entries (e.g. "unassigned" with block_index=null/0).
    risk_ax = Axis(grid[1, 3], title = "Router Risk", yticksvisible = false, xticksvisible = false)
    risk_vals = zeros(Float32, n_blocks)
    for s in bundle.saaq
        if 1 <= s.block <= n_blocks
            risk_vals[s.block] = s.risk_score
        end
    end
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
    # Float32 so the pulse animation (32 + 6*sin(...)) doesn't InexactError
    node_sizes  = Observable(fill(18.0f0, 2 + n_experts + 1))

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
                node_sizes[][i] = 26.0f0
            elseif i > 2 && i <= 2 + n_experts
                eid = i - 2
                if eid in top2
                    node_colors[][i] = :limegreen
                    node_sizes[][i] = Float32(32 + 6 * sin(2π * (frame.token_idx % 30) / 30))  # pulse
                else
                    node_colors[][i] = :dimgray
                    node_sizes[][i] = 14.0f0
                end
            else
                node_colors[][i] = i == 1 ? :skyblue : :orange
                node_sizes[][i] = 20.0f0
            end
        end
        node_colors[] = node_colors[]  # notify
        node_sizes[] = node_sizes[]
    end

    onany(selected_block, current_frame) do blk, frame
        if frame.block != blk
            new_frame = simulate_router_frame(bundle, blk, token_idx[]; backend=backend, seed=seed[])
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
    Label(inspector[4, 1], lift(f -> "Logits (first 4): $(round.(f.logits[1:min(4, length(f.logits))]; digits=3)) …", current_frame))
    Label(inspector[5, 1], lift(f -> "Probs: $(round.(f.probs; digits=3))", current_frame))
    Label(inspector[6, 1], lift(f -> "Entropy: $(round(f.entropy; digits=4))", current_frame))
    Label(inspector[7, 1], lift(selected_block) do b
        r = get(saaq_by_block, b, nothing)
        if r === nothing
            "Risk: n/a | Readiness: n/a | (no SAAQ row for block $b)"
        else
            "Risk: $(round(r.risk_score; digits=3)) | Readiness: $(round(r.readiness; digits=3)) | $(r.status)"
        end
    end)
    Label(inspector[8, 1], "Provenance: $(bundle.provenance) — simulated router dynamics on real metadata")
    Label(inspector[9, 1], "Click heatmap row or use slider to change block", fontsize = 10, color = :gray)

    # Performance / runtime label — updated every play tick. Keeps the user
    # honest about which backend is doing the activity-field work.
    perf_label = Label(inspector[10, 1],
        lift(token_idx, selected_block, fps_obs) do t, b, f
            backend_name = backend isa CUDABackend ? "CUDABackend" : "CPUBackend"
            cuda_str = backend isa CUDABackend ? (cuda_avail ? "true" : "false") :
                       "n/a (CPUBackend)"
            cache_str = cache === nothing ? "off (fallback)" :
                "$(cache.n_blocks)×$(cache.n_tokens)×$(cache.top_k)"
            act_path = use_cuda_activity[] ? "CUDA kernels" : "CPU kernels"
            interval_ms = f > 0 ? round(1000.0 / f; digits=1) : 0.0
            string("Backend: ", backend_name, " | CUDA.functional()=", cuda_str,
                   "\nActivity path: ", act_path,
                   " | Frame cache: ", cache_str,
                   "\nToken: ", t, "  Block: ", b,
                   "  FPS: ", round(f; digits=1), " (", interval_ms, " ms/tick)")
        end,
        fontsize = 11, halign = :left)

    # D. Timeline Controls (bottom)
    timeline = grid[3, 1:4] = GridLayout()
    token_slider = Slider(timeline[1, 1],
        range = 0:(cache === nothing ? 300 : cache.n_tokens - 1),
        startvalue = 0, width = 600)
    suppress_slider_cb = Ref(false)
    prev_token = Ref(0)
    on(token_slider.value) do v
        suppress_slider_cb[] && return
        token_idx[] = v
        new_frame = simulate_router_frame(bundle, selected_block[], v; backend=backend, seed=seed[])
        current_frame[] = new_frame
        delta = v - prev_token[]
        if cache !== nothing && (delta < 0 || delta > 1)
            _reconstruct_activity!(v)
        elseif cache !== nothing && v > prev_token[]
            for s in (prev_token[] + 1):v
                _step_activity!(s)
            end
        else
            _step_activity!(v)
        end
        prev_token[] = v
    end

    play_btn = Button(timeline[1, 2], label = lift(p -> p ? "⏸ Pause" : "▶ Play", is_playing))
    on(play_btn.clicks) do _
        is_playing[] = !is_playing[]
        if !is_playing[]
            current_frame[] = simulate_router_frame(bundle, selected_block[], token_idx[];
                                                    backend=backend, seed=seed[])
        end
        if is_playing[]
            if play_task[] !== nothing && !istaskdone(play_task[])
                schedule(play_task[], InterruptException(); error=true)
            end
            last_tick_time[] = time()
            play_task[] = @async begin
                try
                    max_t = cache === nothing ? 300 : cache.n_tokens - 1
                    while is_playing[]
                        nxt = token_idx[] + 1
                        if nxt > max_t
                            nxt = 0
                            _reconstruct_activity!(0)
                        else
                            _step_activity!(nxt)
                        end
                        token_idx[] = nxt
                        prev_token[] = nxt
                        suppress_slider_cb[] = true
                        try
                            token_slider.value[] = nxt
                        finally
                            suppress_slider_cb[] = false
                        end
                        if nxt % 5 == 0
                            current_frame[] = simulate_router_frame(bundle, selected_block[], nxt;
                                                                    backend=backend, seed=seed[])
                        end
                        now = time()
                        dt = now - last_tick_time[]
                        last_tick_time[] = now
                        if dt > 0
                            inst = 1.0 / dt
                            fps_obs[] = fps_obs[] == 0 ? inst : 0.6 * fps_obs[] + 0.4 * inst
                        end
                        sleep(0.08)
                    end
                catch e
                    e isa InterruptException || rethrow(e)
                end
            end
        end
    end

    seed_box = Textbox(timeline[1, 3], placeholder = "Seed", width = 80)
    on(seed_box.stored_string) do s
        try
            seed[] = parse(Int, s)
        catch
        end
    end
    # Re-simulate the inspector frame when the seed changes. The cache itself
    # is keyed off `cache_seed` (set at launch) and is intentionally not
    # rebuilt on every UI seed change to keep things responsive — the cache
    # is for the heatmap atmosphere; the inspector reflects the live UI seed.
    on(seed) do _
        current_frame[] = simulate_router_frame(bundle, selected_block[], token_idx[];
                                                backend=backend, seed=seed[])
    end

    # Initial frame + activity field at t=0.
    current_frame[] = simulate_router_frame(bundle, selected_block[], token_idx[]; backend=backend, seed=seed[])
    _step_activity!(token_idx[])
    refresh_graph!(current_frame[])

    # Final layout tweaks
    colsize!(grid, 1, Auto())
    colsize!(grid, 4, Relative(0.28))
    rowsize!(grid, 1, Relative(0.42))
    rowsize!(grid, 2, Relative(0.38))

    display(fig)
    on(events(fig).window_open) do open
        if !open
            is_playing[] = false
            if play_task[] !== nothing && !istaskdone(play_task[])
                schedule(play_task[], InterruptException(); error=true)
            end
        end
    end
    @info "Grok-1 MoE Atmosphere launched — use mouse on heatmap, play button, slider. Close window to exit."
    return fig
end
