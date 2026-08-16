<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEP Architecture V3.2 - Formal Preregistration Document</title>
    
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Load Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Load MathJax for rendering LaTeX equations -->
    <script>
        MathJax = {
            tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']] },
            svg: { fontCache: 'global' }
        };
    </script>
    <script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

    <style>
        body {
            background-color: #030508;
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
        }
        .mono { font-family: 'Space Mono', monospace; }

        /* Box Styles */
        .info-box { background: rgba(0,0,0,0.6); border-left: 3px solid #22d3ee; padding: 1.5rem; margin: 1.5rem 0; border-radius: 0 8px 8px 0; }
        .hypothesis-box { background: rgba(34, 211, 238, 0.05); border-left: 4px solid #22d3ee; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 1.5rem; }
        .failure-box { background: rgba(239, 68, 68, 0.05); border-left: 4px solid #ef4444; padding: 1.5rem; border-radius: 0 8px 8px 0; margin-bottom: 1.5rem; }
        
        /* MathJax Custom Styling */
        mjx-container[jax="SVG"][display="true"] {
            background: rgba(10, 15, 25, 0.8);
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 8px;
            padding: 1.5rem 0;
            margin: 2rem 0;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.05);
            overflow-x: auto;
        }
        mjx-container svg { filter: drop-shadow(0px 0px 2px rgba(34, 211, 238, 0.5)); color: #22d3ee !important; }

        /* Sticky Nav */
        .sticky-nav {
            position: sticky;
            top: 0;
            z-index: 50;
            background: rgba(3, 5, 8, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .section-divider {
            border-top: 1px dashed rgba(255,255,255,0.1);
            margin: 5rem 0;
            position: relative;
        }
        .section-divider::after {
            content: "◆";
            position: absolute;
            top: -12px;
            left: 50%;
            transform: translateX(-50%);
            color: #22d3ee;
            background: #030508;
            padding: 0 10px;
        }
    </style>
</head>
<body class="antialiased selection:bg-cyan-900 selection:text-cyan-100">

    <!-- STICKY NAVIGATION -->
    <nav class="sticky-nav py-4 px-6 shadow-lg">
        <div class="max-w-4xl mx-auto flex flex-wrap justify-between items-center text-xs md:text-sm font-bold tracking-widest uppercase mono">
            <span class="text-cyan-400">MEP V3.2 Protocol</span>
            <div class="flex gap-4">
                <a href="#summary" class="text-gray-400 hover:text-cyan-400 transition-colors">Summary</a>
                <a href="#optimizer" class="text-gray-400 hover:text-cyan-400 transition-colors">Optimizer</a>
                <a href="#memory" class="text-gray-400 hover:text-cyan-400 transition-colors">Memory</a>
            </div>
        </div>
    </nav>

    <!-- MAIN CONTENT CONTAINER -->
    <main class="max-w-4xl mx-auto px-6 py-16 prose prose-invert prose-cyan lg:prose-lg">

        <header class="mb-16">
            <h1 class="text-4xl md:text-5xl font-bold text-white mb-6 tracking-tight">MEP Architecture V3.2</h1>
            <h2 class="text-2xl text-cyan-400 font-light mt-0 mb-6 border-b border-gray-800 pb-6">Formal Preregistration Document</h2>
            <div class="flex flex-col sm:flex-row gap-4 mono text-sm text-gray-400 bg-gray-900/50 p-4 rounded-lg border border-gray-800">
                <div><strong class="text-gray-200">Author:</strong> 3MOORE. BBB</div>
                <div class="hidden sm:block">|</div>
                <div><strong class="text-gray-200">Date:</strong> August 16, 2026</div>
                <div class="hidden sm:block">|</div>
                <div><strong class="text-cyan-400 animate-pulse">Status: Pre-Data Collection (Timestamped)</strong></div>
            </div>
        </header>

        <!-- SECTION: EXECUTIVE SUMMARY -->
        <section id="summary" class="scroll-mt-24">
            <h3 class="text-xl font-bold text-white uppercase tracking-widest mt-12 mb-4">Executive Summary</h3>
            <p>
                This document serves as the formal preregistration for the experimental validation of the Mandelbrot-Euler-Planck (MEP) Architecture. Following exploratory engineering in V1.0-V3.1, this protocol freezes the hypotheses, benchmark suites, and explicit failure conditions for two core components of the architecture: the <strong>Adaptive Physical Momentum (APM) Optimizer</strong> and the <strong>Topological Generalization (Memory)</strong> mechanism. No experimental data on the held-out tasks has been collected prior to freezing this document.
            </p>
        </section>

        <div class="section-divider"></div>

        <!-- SECTION: OPTIMIZER MECHANISM -->
        <section id="optimizer" class="scroll-mt-24">
            <h2 class="text-3xl font-bold text-white mb-6">Part I: The Optimizer Mechanism</h2>

            <div class="hypothesis-box">
                <strong class="text-cyan-400 uppercase tracking-widest text-xs block mb-2">The Hypothesis</strong>
                <p class="m-0 text-sm leading-relaxed text-gray-200">
                    The previously observed success of the MEP Optimizer is not driven by Langevin thermal noise, but by the interaction between an adaptive squared-gradient preconditioner (acting as a diagonal metric tensor) and underdamped physical momentum (Mass).
                </p>
            </div>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">1. Model & Equations</h3>
            <p>The optimizer defines the update rule for parameters $\theta$:</p>
            
            <div class="info-box">
                <p class="mb-2"><strong>1. Preconditioner (Local Heat Capacity):</strong></p>
                $$v_t = \beta v_{t-1} + (1-\beta)g_t^2$$
                
                <p class="mt-4 mb-2"><strong>2. Bias Correction:</strong></p>
                $$\hat{v}_t = v_t / (1 - \beta^t)$$
                
                <p class="mt-4 mb-2"><strong>3. Underdamped Momentum:</strong></p>
                $$m_t = \mu m_{t-1} + (1-\mu)g_t$$
                
                <p class="mt-4 mb-2"><strong>4. Adaptive Update:</strong></p>
                $$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon} + \mathcal{N}(0, \sigma^2)$$
            </div>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">2. Experimental Design & Ablations</h3>
            <p>To isolate the driving mechanism, the following component configurations will be tested:</p>
            <ul>
                <li><strong>Config A (Full):</strong> Preconditioner + Momentum + Noise ($\sigma > 0$)</li>
                <li><strong>Config B (-Noise):</strong> Preconditioner + Momentum ($\sigma = 0$) <em>(This is V3.1)</em></li>
                <li><strong>Config C (-Momentum):</strong> Preconditioner + Noise ($\mu = 0$, standard gradient)</li>
                <li><strong>Config D (-Preconditioner):</strong> Momentum + Noise ($v_t = 1$)</li>
                <li><strong>Config E (Baseline 1):</strong> Standard PyTorch AdamW (Tuned)</li>
                <li><strong>Config F (Baseline 2):</strong> Standard PyTorch SGD with Momentum (Tuned)</li>
            </ul>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">3. Benchmark Suite (Held-Out Tasks)</h3>
            <p>The optimizers will be evaluated on the following unseen benchmarks:</p>
            <ol>
                <li><strong>High-Dimensional Non-Convex:</strong> 100-dimensional Rastrigin Function (10,000 steps).</li>
                <li><strong>Real-World Image Classification:</strong> ResNet-18 trained on CIFAR-10 from scratch (100 epochs).</li>
            </ol>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">4. Metrics, Tuning, & Seeds</h3>
            <ul>
                <li><strong>Primary Metric:</strong> Final validation loss (CIFAR-10) and final function value (Rastrigin).</li>
                <li><strong>Tuning:</strong> Grid search on learning rate ($\eta \in \{1e-2, 1e-3, 1e-4\}$) using a separate 10% validation split. Momentum ($\mu$) fixed at $0.9$.</li>
                <li><strong>Seeds:</strong> 50 independent random seeds for Rastrigin; 10 independent random seeds for CIFAR-10.</li>
                <li><strong>Minimum Effect Size:</strong> A statistically significant difference ($p < 0.05$, Welch's t-test) resulting in at least a 3% improvement in final loss over baselines.</li>
            </ul>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">5. Explicit Failure Conditions (Falsifiability)</h3>
            <div class="failure-box">
                <strong class="text-red-500 uppercase tracking-widest text-xs block mb-2">Failure Conditions</strong>
                <p class="m-0 text-sm leading-relaxed text-gray-200">The APM optimizer hypothesis will be considered <strong>FAILED</strong> if:</p>
                <ol class="mt-2 mb-0 text-sm">
                    <li>Config E (AdamW) outperforms Config B (-Noise) across the majority of seeds on CIFAR-10.</li>
                    <li>Config C (-Momentum) performs statistically indistinguishably from Config B, meaning the underdamped mass component contributes no actual value beyond standard preconditioning.</li>
                </ol>
            </div>
        </section>

        <div class="section-divider"></div>

        <!-- SECTION: MEMORY MECHANISM -->
        <section id="memory" class="scroll-mt-24">
            <h2 class="text-3xl font-bold text-white mb-6">Part II: Topological Generalization (The Memory Mechanism)</h2>

            <div class="hypothesis-box">
                <strong class="text-cyan-400 uppercase tracking-widest text-xs block mb-2">The Hypothesis</strong>
                <p class="m-0 text-sm leading-relaxed text-gray-200">
                    The architectural mapping operator $\mathcal{M}$ mitigates Catastrophic Forgetting not by perfectly preserving exact prior states, but by confining network weights within an "Attractor Basin." This creates a lossy but structurally stable <strong>geometric scar</strong> that preserves sufficient prior-task knowledge while learning new tasks.
                </p>
            </div>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">1. Operationalizing the "Geometric Scar"</h3>
            <ul>
                <li><strong>Definition:</strong> The "geometric scar" is mathematically defined as the L2 distance between the network's weights after learning Task B ($W_{A+B}$) and the exact local minimum found after learning Task A ($W_A$).</li>
                <li><strong>Measurement:</strong> We measure the structural retention by the classification accuracy on Task A <em>after</em> the network has converged on Task B.</li>
            </ul>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">2. Experimental Design</h3>
            <ul>
                <li><strong>Benchmark Suite:</strong> Split MNIST (Task A: Digits 0-4; Task B: Digits 5-9).</li>
                <li><strong>Baselines:</strong>
                    <ol>
                        <li>Standard SGD (Naive fine-tuning, expected to catastrophically forget).</li>
                        <li>Elastic Weight Consolidation (EWC) (Industry-standard regularization baseline).</li>
                    </ol>
                </li>
            </ul>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">3. Metrics & Seeds</h3>
            <ul>
                <li><strong>Primary Metric:</strong> Retained accuracy on Task A test set after training on Task B.</li>
                <li><strong>Seeds:</strong> 20 independent random initializations.</li>
            </ul>

            <h3 class="text-xl font-bold text-white mt-10 mb-4">4. Explicit Failure Conditions (Falsifiability)</h3>
            <div class="failure-box">
                <strong class="text-red-500 uppercase tracking-widest text-xs block mb-2">Failure Conditions</strong>
                <p class="m-0 text-sm leading-relaxed text-gray-200">The Topological Generalization hypothesis will be considered <strong>FAILED</strong> if:</p>
                <ol class="mt-2 mb-0 text-sm">
                    <li>The MEP Architecture ($\mathcal{M}$) retains less than <strong>80% relative accuracy</strong> on Task A compared to its immediate post-Task A performance.</li>
                    <li>The MEP Architecture performs statistically <strong>no better</strong> than the Elastic Weight Consolidation (EWC) baseline. If it only matches EWC, the complex fractal attractor basin theory is mathematically redundant and unnecessary.</li>
                </ol>
            </div>
        </section>

    </main>

</body>
</html>
