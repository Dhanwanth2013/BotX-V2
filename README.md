# 🧮 botX — AI Math Assistant

> *"Talk to me like a human. I'll solve like a machine."*

**botX** is a full-stack AI-powered math assistant that understands plain English, renders beautiful LaTeX, plots stunning graphs, and handles everything from basic arithmetic to Fourier transforms — all served from a single Python file.

Built with 💙 by **Dhanwanth V** © 2026. All rights reserved.

---

## ✨ What Is botX?

Ever wished your calculator could also understand *"hey, what's the integral of sin(x) from 0 to pi?"* instead of making you remember arcane syntax?

That's exactly what botX does.

It combines:
- A **Computer Algebra System** (powered by SymPy) for symbolic math
- A **full NLP engine** that parses natural language into math operations
- An **optional offline AI brain** via [Ollama](https://ollama.com) for richer conversational responses
- A **Desmos-style interactive graph calculator** built from scratch on HTML Canvas
- A slick, dark-themed web UI that runs entirely in your browser

No cloud subscriptions. No API keys. Just run `python app.py` and do math.

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install flask sympy numpy matplotlib
```

### Run

```bash
python app.py
```

The app opens automatically at **http://localhost:5050** 🎉

### Optional: Supercharge with Ollama AI

Install [Ollama](https://ollama.com) and pull a model for enhanced conversational AI:

```bash
ollama pull gemma3:1b    # Fast, lightweight
ollama pull llama3.2     # Balanced
ollama pull mistral      # Great for math reasoning
```

botX auto-detects whichever model you have installed. No config needed.

---

## 🧠 Supported Operations

### ∫ Calculus

| Operation | Example Input |
|---|---|
| Solve equations | `2x^2 + 3x = 5` |
| Differentiate | `sin(x)/x` |
| Partial derivatives | `x^2*y^3, x, y` |
| Indefinite integral | `sin(x)` |
| Definite integral | `sin(x), 0, pi` |
| Limits | `sin(x)/x, 0` |
| Taylor / Maclaurin series | `exp(x), 8, 0` |
| ODE (2nd order) | `y'' + 3y' + 2y = 0` |

### ✦ Algebra

| Operation | Example Input |
|---|---|
| Factor / Expand / Simplify | `x^2 - 1, factor` |
| System of equations | `x+y=5; x-y=1` |
| Inequalities | `x^2 - 4 < 0` |
| Sequences & Series | `1/n^2, 1, oo` |

### 🔁 Transforms

| Operation | Example Input |
|---|---|
| Laplace Transform | `t*exp(-t)` |
| Inverse Laplace | `1/(s+1), inverse` |
| Fourier Transform | `exp(-x^2)` |
| Inverse Fourier | `exp(-k^2), inverse` |

### 📈 Plotting

| Operation | Example Input |
|---|---|
| 2D Plot (multi-function) | `sin(x); cos(x); x^2/10` |
| Polar Plot | `sin(3*theta)` |
| Parametric Curve | `cos(t); sin(t)` |
| 3D Surface | `sin(x)*cos(y)` |

### ⊞ Linear Algebra

| Operation | Example Input |
|---|---|
| Matrix (det, inv, eigen...) | `[[2,1],[5,3]], det` |
| Vector Calculus | `gradient, x^2+y^2` |
| Divergence | `divergence, x*y, y*z, x*z` |
| Curl | `curl, y, -x, 0` |
| Laplacian | `laplacian, x^2+y^2` |

### 📊 Data Analysis

| Operation | Example Input |
|---|---|
| Statistics | `2, 4, 4, 4, 5, 5, 7, 9` |
| Polynomial Regression | `0,1; 1,3; 2,5; degree=2` |

### ℕ Number Theory

| Operation | Example Input |
|---|---|
| Factorize | `factorize, 360` |
| Is Prime? | `isprime, 97` |
| GCD / LCM | `gcd, 48, 18, 12` |
| Fibonacci | `fibonacci, 20` |
| Euler's Totient | `totient, 36` |
| Primes up to N | `primes, 100` |
| Modular inverse | `modinv, 3, 7` |
| Next/Previous prime | `nextprime, 100` |

### 🤖 Natural Language (NLP Mode)

Just... talk to it. Really.

```
differentiate sin(x) with respect to x
integrate x squared from 0 to 3
what is the limit of sin(x)/x as x approaches 0?
is 97 a prime number?
convert 100 km to miles
what is 15% of 240?
factorize 360
tell me a math joke
who created you?
what is pi?
```

botX understands greetings, jokes, compliments, philosophy, unit conversions, math facts, and more — with graceful fallback responses when Ollama isn't available.

---

## 🧮 Interactive Graph Calculator

Visit **http://localhost:5050/graph** (or click the **🧮 Graph** button) for a full Desmos-style calculator featuring:

- **Multi-function plotting** with color-coded curves
- **Drag to pan**, scroll/pinch to zoom
- **Auto-detect zeros, maxima, minima** and mark them on the graph
- **Table of values** modal
- **Screenshot** export to PNG
- **Custom x/y range inputs**
- Toggle grid, axes, labels, special points
- Touch support for mobile

All built with raw HTML Canvas — no external charting library.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+K` | Command Palette |
| `Ctrl+L` | Clear Chat |
| `Ctrl+T` | Toggle Dark/Light Theme |
| `Ctrl+F` | Formula Cheatsheet |
| `Ctrl+/` | Keyboard Shortcuts |
| `Ctrl+,` | Settings |
| `Ctrl+E` | Export session history (JSON) |
| `↑ / ↓` | Navigate input history |
| `Esc` | Close any modal |

---

## 🏗️ Architecture

```
app.py
├── OfflineAI          — Ollama integration (auto-detects model, graceful fallback)
├── parse_nlp()        — Full NLP engine (greetings, math routing, unit conversions...)
├── preprocess()       — Natural notation → SymPy-safe expression converter
├── solve_equation()   — Symbolic + numerical solver with domain support
├── do_differentiate() — Multi-variable differentiation
├── do_integrate()     — Definite and indefinite integration
├── do_limit()         — Limits with direction support
├── do_taylor()        — Taylor/Maclaurin series
├── do_simplify()      — Factor, expand, cancel, apart, trigsimp, etc.
├── do_laplace()       — Laplace transform (forward & inverse)
├── do_fourier()       — Fourier transform (forward & inverse)
├── do_stats()         — Descriptive statistics (mean, std, IQR, skewness...)
├── do_number_theory() — GCD, LCM, primes, Fibonacci, totient, etc.
├── do_system()        — Systems of equations
├── do_sequence()      — Summations and products
├── do_inequality()    — Inequality solving
├── do_partial_diff()  — Partial differentiation
├── do_vector()        — Gradient, divergence, curl, Laplacian
├── do_regression()    — Polynomial regression with R² and RMSE
├── do_matrix()        — Matrix operations (det, inv, eigenvals, rref, ...)
├── do_ode2()          — 2nd-order ODE solver with initial conditions
├── make_plot()        — 2D function plots (multi-function)
├── make_polar_plot()  — Polar coordinate plots
├── make_parametric_plot() — Parametric curves
├── make_3d_plot()     — 3D surface plots
├── make_stats_plot()  — Histogram + box plot
├── make_regression_plot() — Scatter + regression curve
│
├── Flask Routes
│   ├── GET  /         — Main chat UI
│   ├── POST /compute  — Direct operation endpoint
│   ├── POST /nlp      — Natural language query endpoint
│   ├── GET  /graph    — Interactive graph calculator
│   ├── GET  /ai_status — Ollama status check
│   └── POST /ai_explain — AI explanation for math results
│
└── HTML (embedded)
    ├── Full dark/light themed chat interface
    ├── Command Palette (Ctrl+K)
    ├── Formula Cheatsheet panel
    ├── Settings modal
    ├── Fullscreen plot viewer
    ├── Toast notification system
    └── Desmos-style Graph Calculator (Canvas)
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| Computer Algebra | SymPy |
| Numerical Computing | NumPy |
| Plotting | Matplotlib (Agg backend, base64 PNG) |
| AI (optional) | Ollama (local LLM) |
| Math Rendering | MathJax 3 (LaTeX → SVG) |
| Frontend | Vanilla JS · HTML5 Canvas |
| Fonts | JetBrains Mono · Sora (Google Fonts) |

Zero frontend build step. Zero npm. Zero webpack. One Python file.

---

## 🤖 AI Configuration

botX ships with a two-tier AI system:

**Tier 1 — Rule-based NLP** (always active, zero latency):
Handles greetings, unit conversions, known math facts, operation routing via regex pattern matching. Works offline with no dependencies.

**Tier 2 — Ollama LLM** (optional, requires Ollama):
Enhances conversational responses, handles ambiguous queries, generates step-by-step explanations for results. Falls back to Tier 1 if unavailable.

### Preferred Models (auto-detected in priority order)

`llama3.2` · `llama3.1` · `mistral` · `phi4` · `gemma3` · `qwen2.5` · `deepseek-r1`

### AI Speed Presets (configurable in Settings)

| Preset | Tokens | Best For |
|---|---|---|
| ⚡ Fast | ~80 | Quick one-liners, older hardware |
| ⚖ Balanced | ~150 | Everyday use (default) |
| 💬 Full | ~350 | Rich explanations, complex queries |

---

## 🌟 Features Highlight

- **LaTeX rendering** — every result is rendered as beautiful math via MathJax
- **Copy-to-clipboard** — copy result as text or raw LaTeX
- **Download plots** — save any plot as PNG
- **Fullscreen plot viewer** — click any plot to expand
- **Session statistics** — live computation count and timing in the header
- **Input history** — navigate previous inputs with ↑↓ arrow keys
- **Export session** — download full session history as JSON
- **Formula Cheatsheet** — built-in reference panel with click-to-insert formulas
- **Responsive layout** — works on mobile with collapsible sidebar
- **Dark/Light theme** — persists across sessions via localStorage
- **Settings panel** — precision, plot range, AI speed, step display

---

## 📂 Project Structure

```
botX/
└── app.py     ← Everything. The entire application.
```

Yes, really. It's one file. ~3,000 lines of Python with the full HTML/CSS/JS UI embedded as a string. Dhanwanth went for the mono-file approach and it *works*.

---

## 📋 Requirements

```
flask
sympy
numpy
matplotlib
requests          # for Ollama integration (graceful if missing)
```

Install all at once:

```bash
pip install flask sympy numpy matplotlib requests
```

Python 3.10+ recommended (uses `str | None` type hints).

---

## 💡 Usage Tips

**Multi-function plots** — separate with semicolons:
```
sin(x); cos(x); x^2/10
```

**Definite integral** — separate limits with commas:
```
x^2, 0, 5
```

**ODE with initial conditions**:
```
y'' + y = sin(x), y(0) = 0, y'(0) = 1
```

**Matrix operations** — use Python list syntax:
```
[[1,2,3],[4,5,6],[7,8,10]], eigenvals
```

**Polynomial regression** — semicolon-separated (x,y) pairs:
```
0,0; 1,1; 2,4; 3,9; 4,16; degree=2
```

**System of equations** — semicolon-separated:
```
x + y = 5; 2x - y = 1
```

**Implicit multiplication** is supported: `2x`, `3sin(x)`, `x(x+1)` all work.

---

## 🤩 Fun Facts About botX

- Type `"tell me a joke"` and it knows 10+ math jokes (or generates a new one with Ollama)
- It knows the answer to life, the universe, and everything is **42**
- Ask it `"who is your favourite mathematician?"` for a surprisingly philosophical answer
- It has opinions on Euler, Gauss, Ramanujan, and Turing
- It will gently explain why dividing by zero is still not okay, no matter how nicely you ask

---

## 📜 License & Credits

**© 2026 Dhanwanth V. All rights reserved.**

Unauthorised copying or redistribution is strictly prohibited.

Built with:
- [Flask](https://flask.palletsprojects.com/) — Web framework
- [SymPy](https://www.sympy.org/) — Computer Algebra System
- [NumPy](https://numpy.org/) — Numerical computing
- [Matplotlib](https://matplotlib.org/) — Plot generation
- [Ollama](https://ollama.com/) — Local LLM inference
- [MathJax](https://www.mathjax.org/) — LaTeX rendering

---

*Made with* 🧠 *and excessive amounts of caffeine.*
