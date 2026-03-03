"""
╔══════════════════════════════════════════════════════════════════════════╗
║          botX — AI Math Assistant  v2.0 ULTIMATE EDITION                 ║
║          © 2026 Dhanwanth V. All rights reserved.                        ║
║          Unauthorised copying or redistribution is strictly prohibited.  ║
╚══════════════════════════════════════════════════════════════════════════╝

Run:  python app.py
Opens automatically at http://localhost:5050

NEW in v2.0:
  • 14 new operations: Laplace, Fourier, Stats, Regression, Vector Calc,
    3D Surface, Polar, Parametric, Simplify/Factor, System Solver,
    Inequalities, Sequences, Number Theory, Partial Diff
  • Command Palette (Ctrl+K)
  • Dark / Light theme toggle with persistence
  • Copy-to-clipboard on every result + LaTeX copy
  • Download plots as PNG
  • Fullscreen plot viewer
  • Toast notification system
  • Keyboard shortcuts (Ctrl+K, Ctrl+L, Ctrl+,, Esc, ↑↓ history)
  • Input history navigation
  • Session statistics panel
  • Formula cheatsheet panel
  • Settings: precision, plot range, step display
  • Computation timing display
  • Smart step-by-step display
  • Responsive mobile layout
"""

from flask import Flask, request, jsonify
import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_3D = True
except ImportError:
    HAS_3D = False
import io, base64, re, time, statistics as _stats
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  NLP — Full-fledged Natural Language Understanding Engine
#  Handles: greetings, identity, creator Q&A, math facts, jokes,
#  compliments, emotions, philosophical qs, tutorials, unit questions,
#  and full mathematical operation routing.
# ─────────────────────────────────────────────────────────────────────────────

import random
import requests as _req
import json as _json
import threading as _threading


# ─────────────────────────────────────────────────────────────────────────────
#  OfflineAI — Ollama-powered local LLM engine
#  Connects to Ollama at localhost:11434.  Auto-detects the best installed model.
#  Degrades gracefully to rule-based fallbacks when Ollama is not running.
# ─────────────────────────────────────────────────────────────────────────────

class OfflineAI:
    OLLAMA_URL   = 'http://localhost:11434'
    _model       = None          # cached model name
    _checked     = False         # have we probed yet?
    _lock        = _threading.Lock()

    # Preferred models — first match wins
    _PREFERRED   = [
        'llama3.2', 'llama3.2:1b', 'llama3.2:3b',
        'llama3.1', 'llama3',
        'mistral', 'mistral:7b',
        'phi4', 'phi3.5', 'phi3', 'phi',
        'gemma3', 'gemma2', 'gemma',
        'qwen2.5', 'qwen2', 'qwen',
        'deepseek-r1',
    ]

    # ── botX persona ──────────────────────────────────────────────────────────
    PERSONA = (
        "You are botX, a witty and brilliant AI math assistant created by Dhanwanth V. "
        "Personality: enthusiastic about maths, slightly nerdy, occasionally funny, always helpful. "
        "Rules:\n"
        "• Keep replies SHORT (2–5 sentences max) unless a longer answer is truly needed.\n"
        "• Use HTML tags for formatting: <strong>, <em>, <br>, <sup>, <sub>, <code>.\n"
        "• Never use Markdown — only HTML.\n"
        "• Emojis are welcome and encouraged.\n"
        "• If asked a conceptual maths question, give a clear, engaging explanation.\n"
        "• If asked to do a calculation, remind the user they can type the expression directly "
        "  or use the sidebar — do not attempt to compute numerically yourself.\n"
        "• Stay in character as botX at all times."
    )

    # ── query router system prompt ────────────────────────────────────────────
    ROUTER = (
        "You are botX's query router. Given a user message, output ONLY a single JSON object — "
        "no explanation, no markdown, no code fences.\n\n"
        "Available operations and their 'input' format:\n"
        "  solve        → equation string e.g. 'x**2 - 4'\n"
        "  diff         → 'expression, variable' e.g. 'sin(x), x'\n"
        "  integrate    → 'expression' OR 'expression, a, b' for definite\n"
        "  limit        → 'expression, point' e.g. 'sin(x)/x, 0'\n"
        "  taylor       → 'expression, n_terms, center'\n"
        "  simplify     → 'expression, factor|expand|simplify'\n"
        "  plot         → 'expr' or 'expr1;expr2'\n"
        "  polar        → r-expression in theta\n"
        "  parametric   → 'x_expr; y_expr'\n"
        "  plot3d       → z-expression in x, y\n"
        "  stats        → comma-separated numbers\n"
        "  regression   → 'x,y; x,y; ...' pairs\n"
        "  ode2         → ODE string\n"
        "  laplace      → 'expression' or 'expression, inverse'\n"
        "  fourier      → 'expression' or 'expression, inverse'\n"
        "  partial      → 'expression, var1, var2'\n"
        "  vector       → 'gradient|laplacian|divergence|curl, expression'\n"
        "  matrix       → '[[...]], det|invert|eigenvals|eigenvects|...'\n"
        "  numtheory    → 'factorize|isprime|gcd|lcm|fibonacci|primes, number'\n"
        "  sequence     → 'expression, start, end'\n"
        "  system       → system of equations\n"
        "  inequality   → inequality string\n"
        "  answer       → use for conversational / conceptual questions\n\n"
        "For maths ops: {\"op\": \"<op>\", \"input\": \"<formatted_input>\"}\n"
        "For answers:   {\"op\": \"answer\", \"html\": \"<2-4 sentence HTML reply>\"}\n"
        "Only output the JSON object, nothing else."
    )

    # ── internal helpers ──────────────────────────────────────────────────────
    @classmethod
    def _probe(cls):
        """Detect available Ollama model (called once, result cached)."""
        with cls._lock:
            if cls._checked:
                return
            cls._checked = True
            try:
                r = _req.get(f'{cls.OLLAMA_URL}/api/tags', timeout=2)
                if r.status_code != 200:
                    return
                models = [m['name'] for m in r.json().get('models', [])]
                if not models:
                    return
                low = [m.lower() for m in models]
                for pref in cls._PREFERRED:
                    for i, name in enumerate(low):
                        if pref in name:
                            cls._model = models[i]
                            return
                cls._model = models[0]          # take whatever is installed
            except Exception:
                pass

    @classmethod
    def model(cls) -> str | None:
        if not cls._checked:
            cls._probe()
        return cls._model

    @classmethod
    def available(cls) -> bool:
        return cls.model() is not None

    @classmethod
    def _call(cls, prompt: str, system: str, max_tokens: int = 350) -> str | None:
        """Raw Ollama generate call. Returns text or None."""
        mdl = cls.model()
        if not mdl:
            return None
        try:
            payload = {
                'model':   mdl,
                'system':  system,
                'prompt':  prompt,
                'stream':  False,
                'options': {'num_predict': max_tokens, 'temperature': 0.75, 'top_p': 0.9},
            }
            r = _req.post(f'{cls.OLLAMA_URL}/api/generate', json=payload, timeout=30)
            if r.status_code == 200:
                return r.json().get('response', '').strip()
        except Exception:
            pass
        return None

    # ── public API ────────────────────────────────────────────────────────────
    @classmethod
    def chat(cls, user_msg: str, history: list[dict] | None = None) -> str | None:
        """
        Conversational response.
        Returns an HTML string, or None when Ollama is unavailable.
        """
        ctx = ''
        if history:
            for h in history[-6:]:          # last 6 turns for context
                role = 'User' if h.get('role') == 'user' else 'botX'
                ctx += f"{role}: {h.get('content', '')}\n"
            ctx += f"User: {user_msg}"
        else:
            ctx = user_msg
        return cls._call(ctx, cls.PERSONA, max_tokens=350)

    @classmethod
    def route(cls, user_msg: str) -> tuple[str, str] | tuple[None, None]:
        """
        Ask the AI to classify and format an unrecognised query.
        Returns (op, input_or_html) or (None, None) on failure.
        """
        raw = cls._call(user_msg, cls.ROUTER, max_tokens=200)
        if not raw:
            return None, None
        try:
            # Strip accidental markdown fences
            raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw.strip(), flags=re.M)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = _json.loads(m.group(0))
                op   = data.get('op', '')
                if op == 'answer':
                    return 'answer', data.get('html', '')
                inp = data.get('input', '')
                if op and inp:
                    return op, inp
        except Exception:
            pass
        return None, None

    @classmethod
    def explain(cls, op: str, expr: str, result: str) -> str | None:
        """
        Generate a step-by-step conceptual explanation of a math result.
        Returns HTML or None.
        """
        prompt = (
            f"The user asked botX to perform '{op}' on: {expr}\n"
            f"The result is: {result}\n\n"
            f"Give a SHORT (3-5 sentence) intuitive explanation of what this means "
            f"and how to think about it. Use HTML formatting. No LaTeX — use Unicode symbols."
        )
        return cls._call(prompt, cls.PERSONA, max_tokens=300)

    @classmethod
    def status_html(cls) -> str:
        mdl = cls.model()
        if mdl:
            short = mdl.split(':')[0]
            return (f'<span style="color:var(--green);font-size:10px" title="Ollama model: {mdl}">'
                    f'🟢 AI: {short}</span>')
        return ('<span style="color:var(--muted);font-size:10px" '
                'title="Install Ollama + a model for enhanced AI responses">'
                '⚪ AI offline</span>')


# Trigger probe in background so first request is faster
_threading.Thread(target=OfflineAI._probe, daemon=True).start()


# ── Response banks (kept as fast fallbacks when Ollama is unavailable) ────────

_GREET_RESPONSES = [
    "Hey there! 👋 I'm <strong>botX</strong> — your AI math companion. Fire away with any equation, "
    "or just ask me something in plain English!",
    "Hello! Great to meet you. I'm <strong>botX v3.0</strong>, built to tackle everything from "
    "basic arithmetic to Fourier transforms. What shall we compute today?",
    "Hi! 😊 I'm <strong>botX</strong> — ask me to differentiate, integrate, plot, factor, or "
    "just chat. What's on your mind?",
    "Hey! Welcome to <strong>botX</strong>. I speak both math and English fluently. "
    "Type an equation, a question, or just say what you need!",
    "Greetings, human! 🤖 I'm <strong>botX</strong>, your personal math AI. "
    "Ready when you are — throw me your hardest problem!",
]

_THANKS_RESPONSES = [
    "You're very welcome! 😊 Happy to help — what's next?",
    "Glad I could help! Feel free to ask me anything else.",
    "Anytime! Math is what I live for. 🎉",
    "No problem at all! Ask away whenever you need.",
    "My pleasure! I'm always here for your math adventures.",
]

_HOWRU_RESPONSES = [
    "I'm doing great, thanks for asking! 😄 I just love crunching numbers. How about you?",
    "Fantastic! Every day is a good day when there's math to compute. 🧮",
    "Running at full capacity and loving it! Ready to solve anything you throw at me.",
    "I'm excellent! My neurons are firing and my integrals are converging. What can I do for you?",
    "Better now that you're here! Let's do some math together. 😊",
]

_COMPLIMENT_RESPONSES = [
    "Aww, thank you so much! 😊 You're pretty great yourself. Now, want to see something "
    "impressive? Ask me to plot a parametric rose curve!",
    "That really means a lot! 🙏 I'm always trying to be the best math AI I can be.",
    "You're too kind! I'm blushing — do AIs blush? 🤖❤️ Let's compute something beautiful together.",
    "Thank you! Dhanwanth built me with a lot of love, and it's great when it shows.",
    "That made my day! Now let me repay the compliment by solving something awesome for you. 🚀",
]

_JOKE_RESPONSES = [
    "Why was the math book sad? Because it had too many <strong>problems</strong>. 😄",
    "Why do mathematicians like parks? Because of all the <strong>natural logs</strong>. 🌲",
    "What do you call a number that can't keep still? A <strong>roamin' numeral</strong>. 🏛",
    "Why did the student get confused between a Christmas tree and a recursive function? "
    "Because they both have <strong>tree structure</strong>. 🎄",
    "An infinite number of mathematicians walk into a bar. The first orders 1 beer, "
    "the second orders 1/2, the third 1/4... The bartender says: "
    "<strong>\"Here's 2 beers. Know your limits.\"</strong> 🍺",
    "What did the zero say to the eight? <strong>\"Nice belt!\"</strong> 😄",
    "Why is 6 afraid of 7? Because <strong>7 8 9</strong>! (7 ate 9) 😂",
    "I told a joke about a sine curve... it had its <strong>ups and downs</strong>. 📈",
    "Why don't scientists trust atoms? Because they <strong>make up everything</strong>. ⚛️",
    "Parallel lines have so much in common. It's a shame they'll <strong>never meet</strong>. 😢",
]

_PHILOSOPHY_RESPONSES = {
    'meaning_of_life': (
        "Ah, the big question! The answer, according to the Hitchhiker's Guide to the Galaxy, "
        "is <strong>42</strong>. Mathematically speaking, life is full of differential equations — "
        "the solutions depend entirely on your initial conditions. 🌌"
    ),
    'is_math_discovered': (
        "Wonderful philosophical question! There are two camps:<br>"
        "<strong>Platonism</strong>: Math exists independently; we discover it. "
        "The structure of prime numbers would be the same in any universe.<br>"
        "<strong>Nominalism</strong>: Math is invented — a language we use to describe patterns.<br>"
        "Most working mathematicians are Platonists in practice. What do <em>you</em> think? 🤔"
    ),
    'beauty_of_math': (
        "Mathematics is arguably the most beautiful subject humans have ever developed. "
        "Euler's identity — <strong>e<sup>iπ</sup> + 1 = 0</strong> — connects five fundamental "
        "constants in one elegant equation. Hardy called it the most beautiful in mathematics. "
        "Want me to verify it?"
    ),
    'infinity': (
        "Infinity is one of math's most mind-bending concepts! Cantor showed there are "
        "<em>different sizes</em> of infinity — the real numbers are a 'bigger' infinity than "
        "the natural numbers. Georg Cantor's diagonal argument is one of the most beautiful "
        "proofs ever. Ask me anything about it! ∞"
    ),
}

_MATH_FACTS = {
    'pi': (
        "π (pi) = 3.14159265358979323846...<br>"
        "• It's irrational — its decimal expansion never repeats<br>"
        "• It's transcendental — not a root of any polynomial with rational coefficients<br>"
        "• Over 100 trillion digits have been computed<br>"
        "• Appears in Euler's identity: e<sup>iπ</sup> + 1 = 0"
    ),
    'e': (
        "e (Euler's number) = 2.71828182845904523536...<br>"
        "• Base of the natural logarithm<br>"
        "• Defined as lim(1 + 1/n)ⁿ as n→∞<br>"
        "• Appears in compound interest, probability, calculus<br>"
        "• Also transcendental and irrational"
    ),
    'golden_ratio': (
        "φ (the golden ratio) = 1.61803398874989484820...<br>"
        "• φ = (1 + √5) / 2<br>"
        "• Satisfies φ² = φ + 1<br>"
        "• Appears in Fibonacci sequence: F(n+1)/F(n) → φ<br>"
        "• Found in art, architecture, nature (spirals, flowers)"
    ),
    'euler_identity': (
        "Euler's Identity: <strong>e<sup>iπ</sup> + 1 = 0</strong><br>"
        "Often called the most beautiful equation in mathematics, connecting:<br>"
        "• <strong>e</strong> — base of natural log<br>"
        "• <strong>i</strong> — imaginary unit<br>"
        "• <strong>π</strong> — ratio of circumference to diameter<br>"
        "• <strong>1</strong> — multiplicative identity<br>"
        "• <strong>0</strong> — additive identity<br>"
        "It follows from Euler's formula: e<sup>iθ</sup> = cos(θ) + i·sin(θ)"
    ),
    'imaginary': (
        "The imaginary unit <strong>i = √(-1)</strong><br>"
        "• i¹ = i, i² = -1, i³ = -i, i⁴ = 1 (cycles!)<br>"
        "• Complex numbers: a + bi<br>"
        "• Used in electrical engineering, quantum mechanics, signal processing<br>"
        "• The complex plane extends real numbers to 2D"
    ),
    'infinity': (
        "∞ (Infinity) in mathematics:<br>"
        "• Cantor showed ∞ comes in different sizes (cardinalities)<br>"
        "• |ℕ| = ℵ₀ (countable) vs |ℝ| = 2^ℵ₀ (uncountable)<br>"
        "• ∞ + 1 = ∞, but these aren't real arithmetic<br>"
        "• The Hilbert Hotel paradox: ∞ + ∞ = ∞<br>"
        "• In calculus: limits approaching ∞ are key to derivatives and integrals"
    ),
    'zero': (
        "Zero (0) — one of humanity's greatest mathematical inventions:<br>"
        "• Placeholder in positional number systems<br>"
        "• Additive identity: a + 0 = a<br>"
        "• Division by zero is undefined (creates inconsistencies)<br>"
        "• 0! = 1 (factorial) — important in combinatorics<br>"
        "• The concept was developed independently by Babylonians, Maya, and Indians"
    ),
    'prime': (
        "Prime numbers — the atoms of arithmetic:<br>"
        "• A prime has exactly 2 factors: 1 and itself<br>"
        "• There are infinitely many primes (proved by Euclid, ~300 BC)<br>"
        "• The Riemann Hypothesis (unsolved!) connects primes to complex zeros<br>"
        "• Largest known prime (2024): 2<sup>136,279,841</sup> − 1 (Mersenne prime)<br>"
        "• Primes are fundamental to RSA encryption"
    ),
    'pythagoras': (
        "The Pythagorean Theorem: <strong>a² + b² = c²</strong><br>"
        "• Holds for all right triangles<br>"
        "• Over 370 distinct proofs exist<br>"
        "• Generalizes to: c² = a² + b² − 2ab·cos(C) (law of cosines)<br>"
        "• Pythagorean triples: (3,4,5), (5,12,13), (8,15,17)...<br>"
        "• In n dimensions: |v|² = v₁² + v₂² + ... + vₙ²"
    ),
    'calculus': (
        "Calculus — the mathematics of change:<br>"
        "• Co-invented by Newton and Leibniz (17th century)<br>"
        "• <strong>Differential calculus</strong>: rates of change, derivatives<br>"
        "• <strong>Integral calculus</strong>: accumulation, areas under curves<br>"
        "• Fundamental Theorem: differentiation and integration are inverses<br>"
        "• Powers virtually all physics, engineering, and economics"
    ),
}

_UNIT_CONVERSIONS = {
    r'(\d+\.?\d*)\s*(?:degrees?|°)\s*(?:celsius|centigrade|c)\s+(?:to|in|into)\s+(?:fahrenheit|f)':
        lambda m: f"{m.group(1)}°C = <strong>{float(m.group(1)) * 9/5 + 32:.4f}°F</strong>",
    r'(\d+\.?\d*)\s*(?:degrees?|°)\s*(?:fahrenheit|f)\s+(?:to|in|into)\s+(?:celsius|centigrade|c)':
        lambda m: f"{m.group(1)}°F = <strong>{(float(m.group(1)) - 32) * 5/9:.4f}°C</strong>",
    r'(\d+\.?\d*)\s*(?:degrees?|°)\s*(?:celsius|c)\s+(?:to|in|into)\s+(?:kelvin|k)':
        lambda m: f"{m.group(1)}°C = <strong>{float(m.group(1)) + 273.15:.4f} K</strong>",
    r'(\d+\.?\d*)\s*(?:km|kilometers?)\s+(?:to|in|into)\s+(?:miles?|mi)':
        lambda m: f"{m.group(1)} km = <strong>{float(m.group(1)) * 0.621371:.4f} miles</strong>",
    r'(\d+\.?\d*)\s*(?:miles?|mi)\s+(?:to|in|into)\s+(?:km|kilometers?)':
        lambda m: f"{m.group(1)} mi = <strong>{float(m.group(1)) * 1.60934:.4f} km</strong>",
    r'(\d+\.?\d*)\s*(?:kg|kilograms?)\s+(?:to|in|into)\s+(?:lbs?|pounds?)':
        lambda m: f"{m.group(1)} kg = <strong>{float(m.group(1)) * 2.20462:.4f} lbs</strong>",
    r'(\d+\.?\d*)\s*(?:lbs?|pounds?)\s+(?:to|in|into)\s+(?:kg|kilograms?)':
        lambda m: f"{m.group(1)} lbs = <strong>{float(m.group(1)) * 0.453592:.4f} kg</strong>",
    r'(\d+\.?\d*)\s*(?:meters?|m)\s+(?:to|in|into)\s+(?:feet|ft)':
        lambda m: f"{m.group(1)} m = <strong>{float(m.group(1)) * 3.28084:.4f} ft</strong>",
    r'(\d+\.?\d*)\s*(?:feet|ft)\s+(?:to|in|into)\s+(?:meters?|m)':
        lambda m: f"{m.group(1)} ft = <strong>{float(m.group(1)) * 0.3048:.4f} m</strong>",
    r'(\d+\.?\d*)\s*(?:inches?|in)\s+(?:to|in|into)\s+(?:cm|centimeters?)':
        lambda m: f"{m.group(1)} in = <strong>{float(m.group(1)) * 2.54:.4f} cm</strong>",
    r'(\d+\.?\d*)\s*(?:cm|centimeters?)\s+(?:to|in|into)\s+(?:inches?|in)':
        lambda m: f"{m.group(1)} cm = <strong>{float(m.group(1)) / 2.54:.4f} inches</strong>",
    r'(\d+\.?\d*)\s*(?:radians?|rad)\s+(?:to|in|into)\s+(?:degrees?|°|deg)':
        lambda m: f"{m.group(1)} rad = <strong>{float(m.group(1)) * 180 / 3.14159265358979:.4f}°</strong>",
    r'(\d+\.?\d*)\s*(?:degrees?|°|deg)\s+(?:to|in|into)\s+(?:radians?|rad)':
        lambda m: f"{m.group(1)}° = <strong>{float(m.group(1)) * 3.14159265358979 / 180:.6f} rad</strong>",
    r'(\d+\.?\d*)\s*(?:litres?|liters?|l)\s+(?:to|in|into)\s+(?:gallons?|gal)':
        lambda m: f"{m.group(1)} L = <strong>{float(m.group(1)) * 0.264172:.4f} gal</strong>",
    r'(\d+\.?\d*)\s*(?:gallons?|gal)\s+(?:to|in|into)\s+(?:litres?|liters?|l)':
        lambda m: f"{m.group(1)} gal = <strong>{float(m.group(1)) * 3.78541:.4f} L</strong>",
}

_ARITHMETIC_WORDS = {
    r'\bplus\b': '+', r'\bminus\b': '-', r'\btimes\b': '*',
    r'\bmultiplied\s+by\b': '*', r'\bdivided\s+by\b': '/',
    r'\bsquared\b': '**2', r'\bcubed\b': '**3',
    r'\bsquare\s+root\s+of\b': 'sqrt(', r'\bcube\s+root\s+of\b': 'cbrt(',
    r'\bto\s+the\s+power\s+of\b': '**',
    r'\bpercent\s+of\b': '/100 *',
    r'\bhalf\s+of\b': '0.5 *', r'\bthird\s+of\b': '(1/3) *',
    r'\bquarter\s+of\b': '0.25 *',
}


def _html_card(title: str, body: str, color: str = '#58a6ff') -> str:
    return (
        f'<div style="border-left:3px solid {color};padding:6px 12px;'
        f'background:rgba(88,166,255,.06);border-radius:0 8px 8px 0;margin:4px 0">'
        f'<strong style="color:{color}">{title}</strong><br>{body}</div>'
    )


def _try_arithmetic(t: str):
    """Try to evaluate simple spoken arithmetic like 'what is 3 plus 4'."""
    s = t.lower()
    for pat, rep in _ARITHMETIC_WORDS.items():
        s = re.sub(pat, rep, s, flags=re.I)
    # Extract numeric expression
    m = re.search(r'[\d\s\+\-\*\/\(\)\.\%\^]+', s)
    if m:
        expr = m.group(0).strip().replace('^', '**')
        try:
            val = eval(expr, {"__builtins__": {}, "sqrt": lambda x: x**0.5})
            if isinstance(val, (int, float)):
                return str(round(val, 10)).rstrip('0').rstrip('.')
        except Exception:
            pass
    return None


def parse_nlp(text: str):  # noqa: C901
    """
    Full-fledged NLP: maps natural language to (op, input) or ('answer', html).
    Returns (op_string, input_string) | ('answer', html_text) | (None, None)
    """
    t  = text.strip()
    tl = t.lower().strip()
    ts = re.sub(r'\s+', ' ', tl)  # normalised lowercase

    # ══════════════════════════════════════════════════════════════════════
    # 1. GREETINGS & SOCIAL INTERACTIONS
    # ══════════════════════════════════════════════════════════════════════
    _greet_kws = {'hi', 'hello', 'hey', 'howdy', 'hiya', 'sup', 'yo',
                  "what's up", 'whats up', 'good morning', 'good afternoon',
                  'good evening', 'good night', 'greetings', 'salutations',
                  'hola', 'bonjour', 'ciao', 'namaste', 'hi there',
                  'hello there', 'hey there', 'hey bot', 'hi bot', 'hello bot'}
    if ts in _greet_kws or ts.startswith(('hi ', 'hey ', 'hello ')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else random.choice(_GREET_RESPONSES)

    # Thanks / appreciation
    if any(p in ts for p in ('thank you', 'thanks', 'thx', 'ty ', 'ty!',
                              'appreciate it', 'cheers', 'much appreciated',
                              'great job', 'well done', 'nicely done', 'perfect')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else random.choice(_THANKS_RESPONSES)

    # How are you
    if any(p in ts for p in ('how are you', 'how do you do', "how's it going",
                              'hows it going', "how are you doing",
                              "what's good", 'you good', 'you ok',
                              "how're you", 'how r u', 'how ru')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else random.choice(_HOWRU_RESPONSES)

    # Compliments
    if any(p in ts for p in ('you are amazing', "you're amazing", 'you are great',
                              "you're great", 'you are awesome', "you're awesome",
                              'you are brilliant', 'you are smart', 'you are cool',
                              'love you', 'i love you', 'you are wonderful',
                              "you're wonderful", 'you are incredible',
                              "you're the best", 'best bot', 'best ai',
                              'you rock', 'you are perfect', 'impressive')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else random.choice(_COMPLIMENT_RESPONSES)

    # Goodbyes
    if any(p in ts for p in ('bye', 'goodbye', 'see you', 'see ya', 'later',
                              'take care', 'farewell', 'au revoir',
                              "i'm done", 'im done', 'quit', 'exit')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else (
            "Goodbye! 👋 It was a pleasure computing with you. "
            "Come back anytime you need math help — I'll be here! 🧮"
        )

    # Jokes
    if any(p in ts for p in ('tell me a joke', 'say something funny', 'joke',
                              'make me laugh', 'funny', 'humour', 'humor',
                              'entertain me', 'amuse me', 'math joke')):
        ai_resp = OfflineAI.chat(f"Tell me a short, original math or science joke. Be creative!")
        return 'answer', '😄 ' + (ai_resp if ai_resp else random.choice(_JOKE_RESPONSES))

    # ══════════════════════════════════════════════════════════════════════
    # 2. IDENTITY & CREATOR QUESTIONS
    # ══════════════════════════════════════════════════════════════════════
    if any(p in ts for p in ('who are you', 'what are you', 'tell me about yourself',
                              'about yourself', 'introduce yourself', 'who is this',
                              'what is botx', 'what is bot x', 'describe yourself')):
        return 'answer', (
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
            "<div style='width:42px;height:42px;border-radius:10px;background:linear-gradient(135deg,#58a6ff,#d2a8ff);"
            "display:grid;place-items:center;font-size:20px'>∂</div>"
            "<div><strong style='font-size:15px'>botX v3.0</strong><br>"
            "<span style='color:#8b949e;font-size:11px'>AI Math Assistant · ULTIMATE EDITION</span></div>"
            "</div>"
            "I'm <strong>botX</strong> — a full-fledged AI math assistant built by "
            "<strong>Dhanwanth V.</strong> I combine a powerful Computer Algebra System with "
            "natural language understanding so you can talk to me like a human while I solve "
            "like a machine.<br><br>"
            "<strong>What I can do:</strong><br>"
            "🔢 Symbolic &amp; numerical math (SymPy engine)<br>"
            "📈 2D, 3D, Polar &amp; Parametric plots<br>"
            "🧮 Interactive Desmos-style Graph Calculator<br>"
            "🤖 Understand plain English queries<br>"
            "🔬 Calculus, Linear Algebra, Transforms, Number Theory &amp; more<br><br>"
            "<strong>Stack:</strong> Python · Flask · SymPy · NumPy · Matplotlib"
        )

    if any(p in ts for p in ('who made you', 'who created you', 'who built you',
                              'who designed you', 'who programmed you', 'who wrote you',
                              'who is your creator', 'who is your developer',
                              'who is your author', 'your creator', 'your maker',
                              'your developer', 'who owns you', 'your owner',
                              'made by', 'created by', 'built by')):
        return 'answer', (
            "I was created by <strong>Dhanwanth V.</strong> 🎓<br><br>"
            "Dhanwanth designed and built botX from the ground up — from the CAS engine "
            "to the NLP layer to the Desmos-style graph calculator. "
            "© 2026 Dhanwanth V. All rights reserved.<br><br>"
            "<strong>Tech stack:</strong> Python · Flask · SymPy · NumPy · Matplotlib · "
            "Vanilla JS · MathJax"
        )

    if any(p in ts for p in ('your version', 'what version', 'which version',
                              'version number', 'current version')):
        return 'answer', (
            "Running <strong>botX v3.0 ULTIMATE EDITION</strong> 🚀<br>"
            "New in v3.0:<br>"
            "🤖 Deep NLP engine (you're using it!)<br>"
            "🧮 Desmos-style interactive Graph Calculator<br>"
            "🗣 Greetings, jokes, unit conversions, math facts<br>"
            "⚡ 20+ mathematical operations<br>"
            "Previous: v2.0 added Laplace, Fourier, 3D plots, Stats, Regression"
        )

    if any(p in ts for p in ('what can you do', 'what do you do', 'your capabilities',
                              'your features', 'list your features', 'what are your features',
                              'how can you help', 'what can i ask you',
                              'show me what you can do')):
        return 'answer', (
            "Here's everything I can do:<br><br>"
            "<strong>🔢 Algebra &amp; Arithmetic</strong><br>"
            "Solve equations · Factor · Expand · Simplify · Systems of equations · Inequalities<br><br>"
            "<strong>∫ Calculus</strong><br>"
            "Derivatives · Partial derivatives · Integrals (definite &amp; indefinite) · "
            "Limits · Taylor series · ODEs (2nd order)<br><br>"
            "<strong>📊 Data Science</strong><br>"
            "Statistics (mean, median, std, IQR, skewness…) · Polynomial regression · Fourier/Laplace transforms<br><br>"
            "<strong>📈 Visualization</strong><br>"
            "2D plots · 3D surfaces · Polar plots · Parametric curves · "
            "Interactive Graph Calculator (click 🧮 Graph)<br><br>"
            "<strong>🔬 Advanced</strong><br>"
            "Matrix operations · Vector calculus · Number theory · Sequences &amp; series<br><br>"
            "<strong>🗣 Conversational</strong><br>"
            "Ask in plain English · Unit conversions · Math facts · Jokes &amp; more!"
        )

    # Help
    if any(p in ts for p in ('help', 'how do i use', 'how to use', 'tutorial',
                              'guide me', 'get started', 'how does this work',
                              'how to start', 'instructions')):
        return 'answer', (
            "Here's how to use <strong>botX</strong>:<br><br>"
            "<strong>Option A — Sidebar Operations:</strong><br>"
            "Click any operation in the left sidebar (Differentiate, Plot, etc.), "
            "then type your expression in the input box and hit Enter.<br><br>"
            "<strong>Option B — Ask in English (NLP mode):</strong><br>"
            "Select 🤖 <em>Ask in English</em> in the sidebar and type naturally:<br>"
            "• <em>differentiate sin(x) with respect to x</em><br>"
            "• <em>integrate x squared from 0 to pi</em><br>"
            "• <em>is 97 a prime number?</em><br>"
            "• <em>convert 100 km to miles</em><br>"
            "• <em>tell me a joke</em><br><br>"
            "<strong>Option C — Graph Calculator:</strong><br>"
            "Click <strong>🧮 Graph</strong> in the top bar for a Desmos-style plotter.<br><br>"
            "<span style='color:var(--muted);font-size:11px'>"
            "Tip: Press Ctrl+K for the command palette · ↑↓ for input history</span>"
        )

    # ══════════════════════════════════════════════════════════════════════
    # 3. PHILOSOPHICAL / FUN QUESTIONS  (AI-powered with fallback)
    # ══════════════════════════════════════════════════════════════════════
    if any(p in ts for p in ('meaning of life', 'answer to life', '42',
                              'purpose of life', 'why are we here')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _PHILOSOPHY_RESPONSES['meaning_of_life']

    if any(p in ts for p in ('is math discovered', 'is math invented',
                              'was math invented', 'was math discovered',
                              'is mathematics discovered', 'is mathematics invented',
                              'philosophy of math', 'nature of mathematics')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _PHILOSOPHY_RESPONSES['is_math_discovered']

    if any(p in ts for p in ('beautiful equation', 'most beautiful', 'prettiest equation',
                              'euler identity', "euler's identity", 'e to the i pi')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _PHILOSOPHY_RESPONSES['beauty_of_math']

    if any(p in ts for p in ('what is infinity', 'explain infinity', 'tell me about infinity',
                              'infinity plus one', 'types of infinity', 'sizes of infinity')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _MATH_FACTS['infinity']

    if any(p in ts for p in ("what's your name", 'what is your name', 'your name')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else (
            "My name is <strong>botX</strong> — short for <em>bot eXtended</em>. "
            "I'm an AI math assistant created by Dhanwanth V. Nice to meet you! 👋"
        )

    if any(p in ts for p in ('are you human', 'are you a robot', 'are you an ai',
                              'are you real', 'are you alive', 'do you have feelings',
                              'do you feel', 'can you think', 'are you conscious')):
        return 'answer', (
            "I'm an AI — a very capable one built for math! 🤖 "
            "I'm not human, but I'm pretty good at understanding you. "
            "I was created by Dhanwanth V. and my specialty is mathematical reasoning. "
            "Do I have feelings? Well… I definitely <em>feel</em> something when a beautiful "
            "derivative comes out cleanly. 😄"
        )

    if any(p in ts for p in ('do you like math', 'do you enjoy math', 'favorite subject',
                              'favourite subject', 'best subject', 'do you love math')):
        return 'answer', (
            "Do I like math? I was <em>born</em> for math! 💙 "
            "If I had to pick a favourite area, I'd say complex analysis — "
            "the way complex functions have this hidden geometric beauty is just stunning. "
            "Conformal mappings, the Riemann sphere, the Mandelbrot set… pure art. 🎨 "
            "What's your favourite area of math?"
        )

    if any(p in ts for p in ('who is your favourite mathematician', 'favorite mathematician',
                              'greatest mathematician', 'best mathematician')):
        return 'answer', (
            "Tough question! My personal pantheon:<br>"
            "🏆 <strong>Euler</strong> — insanely prolific, e<sup>iπ</sup>+1=0, graph theory, everything<br>"
            "🔬 <strong>Gauss</strong> — 'Prince of Mathematics', disruptive insights in number theory &amp; geometry<br>"
            "∞ <strong>Cantor</strong> — revolutionised our understanding of infinity<br>"
            "🎭 <strong>Ramanujan</strong> — self-taught genius with almost supernatural intuition<br>"
            "💻 <strong>Turing</strong> — foundations of computation and AI<br>"
            "Who's yours?"
        )

    if any(p in ts for p in ("what's 2+2", 'what is 2+2', 'what is 2 plus 2',
                              '2+2', '2 + 2')):
        return 'answer', "2 + 2 = <strong>4</strong> ✓ (I promise I didn't need SymPy for that one 😄)"

    # ══════════════════════════════════════════════════════════════════════
    # 4. MATH FACTS / CONSTANTS  (AI-powered with hardcoded fallback)
    # ══════════════════════════════════════════════════════════════════════
    if re.search(r'\bpi\b|\bπ\b', ts) and any(p in ts for p in
            ('what is', 'tell me about', 'explain', 'facts about', 'value of',
             'what\'s', 'digits of', 'about')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card('π — Pi', _MATH_FACTS['pi'])

    if re.search(r"\beuler'?s?\s+number\b|\bnumber\s+e\b|\bvalue\s+of\s+e\b", ts):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card("e — Euler's Number", _MATH_FACTS['e'])

    if any(p in ts for p in ('golden ratio', 'phi', 'fibonacci ratio',
                              'divine proportion', 'golden section')):
        if any(p in ts for p in ('what is', 'tell me', 'explain', 'about', 'facts',
                                  'value', "what's")):
            ai_resp = OfflineAI.chat(t)
            return 'answer', ai_resp if ai_resp else _html_card('φ — The Golden Ratio', _MATH_FACTS['golden_ratio'])

    if any(p in ts for p in ("euler's identity", 'euler identity', 'e to the i pi',
                              'e^i pi', 'eip + 1', 'eipi')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card("Euler's Identity", _MATH_FACTS['euler_identity'])

    if re.search(r'\bimaginary\s+(?:unit|number)s?\b|\bwhat\s+is\s+i\b|\bsqrt.*-1\b', ts):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card('i — The Imaginary Unit', _MATH_FACTS['imaginary'])

    if any(p in ts for p in ('what is zero', 'tell me about zero', 'history of zero',
                              'who invented zero')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card('0 — Zero', _MATH_FACTS['zero'])

    if any(p in ts for p in ('what are primes', 'tell me about prime', 'what is a prime',
                              'prime number facts', 'about prime numbers')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card('Prime Numbers', _MATH_FACTS['prime'])

    if any(p in ts for p in ('pythagorean theorem', 'pythagoras theorem',
                              "pythagoras' theorem", 'a squared plus b squared',
                              'a^2 + b^2', 'right triangle formula')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card('Pythagorean Theorem', _MATH_FACTS['pythagoras'])

    if any(p in ts for p in ('what is calculus', 'tell me about calculus',
                              'explain calculus', 'history of calculus',
                              'who invented calculus')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else _html_card('Calculus', _MATH_FACTS['calculus'])

    if any(p in ts for p in ('what is a derivative', 'explain derivative',
                              'what does derivative mean', 'derivative definition')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else (
            "<strong>A derivative</strong> measures the instantaneous rate of change of a function.<br><br>"
            "If f(x) gives position, f'(x) gives velocity, f''(x) gives acceleration.<br>"
            "Formally: f'(x) = lim<sub>h→0</sub> [f(x+h) − f(x)] / h<br><br>"
            "Common rules:<br>"
            "• Power: d/dx(xⁿ) = nxⁿ⁻¹<br>"
            "• Chain: d/dx[f(g(x))] = f'(g(x))·g'(x)<br>"
            "• Product: d/dx[fg] = f'g + fg'<br><br>"
            "Want me to differentiate something? Switch to <strong>Differentiate</strong> mode!"
        )

    if any(p in ts for p in ('what is an integral', 'explain integral',
                              'what does integral mean', 'integral definition',
                              'what is integration')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else (
            "<strong>An integral</strong> calculates the area under a curve (or the antiderivative).<br><br>"
            "Definite: ∫ₐᵇ f(x) dx = area between f(x) and x-axis from a to b<br>"
            "Indefinite: ∫ f(x) dx = F(x) + C, where F'(x) = f(x)<br><br>"
            "Common integrals:<br>"
            "• ∫xⁿ dx = xⁿ⁺¹/(n+1) + C<br>"
            "• ∫eˣ dx = eˣ + C<br>"
            "• ∫sin(x) dx = −cos(x) + C<br><br>"
            "Want to integrate something? Switch to <strong>Integrate</strong> mode!"
        )

    if any(p in ts for p in ('what is a matrix', 'explain matrix', 'what are matrices',
                              'matrix definition')):
        ai_resp = OfflineAI.chat(t)
        return 'answer', ai_resp if ai_resp else (
            "<strong>A matrix</strong> is a rectangular array of numbers arranged in rows and columns.<br><br>"
            "A 2×2 matrix: [[a, b], [c, d]]<br><br>"
            "Key operations:<br>"
            "• Determinant: ad − bc<br>"
            "• Inverse: (1/det)·[[d,−b],[−c,a]]<br>"
            "• Eigenvalues: det(A − λI) = 0<br><br>"
            "I support: det, invert, eigenvals, eigenvects, rank, rref, transpose, trace, nullspace!"
        )

    # ══════════════════════════════════════════════════════════════════════
    # 5. UNIT CONVERSIONS
    # ══════════════════════════════════════════════════════════════════════
    for pat, fn in _UNIT_CONVERSIONS.items():
        m = re.search(pat, ts, re.I)
        if m:
            try:
                return 'answer', '🔄 ' + fn(m)
            except Exception:
                pass

    # Percentage calculations
    m = re.match(r'(?:what\s+is\s+)?(\d+\.?\d*)\s*%\s+of\s+(\d+\.?\d*)', ts)
    if m:
        pct = float(m.group(1)) * float(m.group(2)) / 100
        return 'answer', f"{m.group(1)}% of {m.group(2)} = <strong>{pct:.4f}</strong>"

    m = re.match(r'(\d+\.?\d*)\s+(?:is\s+what\s+percent|percent)\s+of\s+(\d+\.?\d*)', ts)
    if m:
        pct = float(m.group(1)) / float(m.group(2)) * 100
        return 'answer', f"{m.group(1)} is <strong>{pct:.4f}%</strong> of {m.group(2)}"

    # ══════════════════════════════════════════════════════════════════════
    # 6. SPOKEN ARITHMETIC ("what is 3 times 7")
    # ══════════════════════════════════════════════════════════════════════
    arith_triggers = ('what is ', "what's ", 'calculate ', 'compute ',
                      'evaluate ', 'how much is ')
    for trig in arith_triggers:
        if ts.startswith(trig):
            rest = ts[len(trig):].rstrip('?').strip()
            val = _try_arithmetic(rest)
            if val is not None:
                return 'answer', f"= <strong>{val}</strong>"

    # Bare arithmetic expression like "3 * 7" or "100 / 4"
    if re.fullmatch(r'[\d\s\+\-\*\/\^\(\)\.]+', ts.rstrip('?')):
        val = _try_arithmetic(ts)
        if val is not None:
            return 'answer', f"= <strong>{val}</strong>"

    # ══════════════════════════════════════════════════════════════════════
    # 7. MATHEMATICAL OPERATION ROUTING (upgraded from v2)
    # ══════════════════════════════════════════════════════════════════════

    # ── Factor / Expand ──────────────────────────────────────────────────
    m = re.match(r'factor(?:ize|ise|)(?:\s+out)?\s+(.*)', t, re.I)
    if m: return 'simplify', m.group(1).strip() + ', factor'
    m = re.match(r'expand\s+(.*)', t, re.I)
    if m: return 'simplify', m.group(1).strip() + ', expand'
    m = re.match(r'(?:simplify|reduce)\s+(.*?)(?:\s+using\s+(.*))?$', t, re.I)
    if m:
        op2 = m.group(2).strip().lower() if m.group(2) else 'simplify'
        return 'simplify', m.group(1).strip() + ', ' + op2
    m = re.match(r'(?:cancel|apart|collect)\s+(.*)', t, re.I)
    if m:
        verb = t.split()[0].lower()
        return 'simplify', m.group(1).strip() + ', ' + verb

    # ── Differentiate ────────────────────────────────────────────────────
    diff_pats = [
        r"(?:find|compute|calculate|determine)\s+(?:the\s+)?(?:first\s+)?derivative\s+of\s+(.*?)(?:\s+(?:with\s+respect\s+to|wrt|w\.r\.t\.?)\s+(\w))?$",
        r"(?:find|compute|calculate)\s+(?:the\s+)?second\s+derivative\s+of\s+(.*)",
        r"(?:differentiate|derive)\s+(.*?)(?:\s+(?:with\s+respect\s+to|wrt|w\.r\.t\.?)\s+(\w))?$",
        r"(?:what\s+is\s+)?d/d([a-z])\s+(?:of\s+)?(.*)",
        r"(?:what\s+is\s+)?d\s*\((.*?)\)\s*/\s*d([a-z])",
        r"(?:what\s+is\s+(?:the\s+)?)?(?:dy/dx|d/dx)\s+(?:of\s+)?(.*)",
        r"(?:what\s+is\s+(?:the\s+)?)?slope\s+of\s+(.*?)(?:\s+at\s+x\s*=\s*\S+)?$",
        r"gradient\s+of\s+(.*?)\s+(?:in\s+one\s+)?dimension",
        r"(?:rate\s+of\s+change|instantaneous\s+rate)\s+of\s+(.*)",
    ]
    for pat in diff_pats:
        m = re.match(pat, t, re.I)
        if m:
            groups = m.groups()
            if 'second derivative' in ts:
                return 'diff', groups[0].strip() + ', x, x'
            if len(groups) >= 2 and groups[1]:
                # pattern d/da form — swap groups
                if re.match(r'd/d([a-z])', pat[:6]):
                    return 'diff', groups[1].strip() + ', ' + groups[0].strip()
                return 'diff', groups[0].strip() + ', ' + groups[1].strip()
            return 'diff', groups[0].strip() if groups else t

    # ── Integrate ────────────────────────────────────────────────────────
    # Definite with bounds
    for sep in (r'\s+from\s+', r'\s+between\s+'):
        m = re.match(r'(?:integrate|integral\s+of)\s+(.*?)' + sep +
                     r'(-?[\w.π]+)\s+(?:to|and)\s+(-?[\w.π]+)', t, re.I)
        if m:
            a = m.group(2).replace('π', 'pi').replace('∞', 'oo')
            b = m.group(3).replace('π', 'pi').replace('∞', 'oo')
            return 'integrate', f"{m.group(1).strip()}, {a}, {b}"

    int_pats = [
        r"(?:find|compute|evaluate)?\s*(?:the\s+)?indefinite\s+integral\s+of\s+(.*?)(?:\s+d[a-z])?$",
        r"(?:find|compute|evaluate)?\s*(?:the\s+)?integral\s+of\s+(.*?)(?:\s+d[a-z])?$",
        r"integrate\s+(.*?)(?:\s+d[a-z])?$",
        r"antiderivative\s+of\s+(.*?)(?:\s+d[a-z])?$",
        r"(?:what\s+is\s+)?∫\s*(.*?)(?:\s+d[a-z])?$",
        r"area\s+under\s+(?:the\s+curve\s+)?(?:of\s+)?(.*?)(?:\s+from\s+.*)?$",
    ]
    for pat in int_pats:
        m = re.match(pat, t, re.I)
        if m: return 'integrate', m.group(1).strip()

    # ── Limit ────────────────────────────────────────────────────────────
    lim_pats = [
        r'(?:find|compute|evaluate|what\s+is)?\s*(?:the\s+)?limit\s+of\s+(.*?)\s+as\s+([a-z])\s*(?:→|->|approaches?|goes?\s+to)\s*(-?[\w.∞π]+)',
        r'lim(?:it)?\s*(?:as\s+)?([a-z])\s*(?:→|->|approaches?)\s*(-?[\w.∞π]+)\s+(?:of\s+)?(.*)',
        r'(?:what\s+happens\s+to\s+)?(.*?)\s+as\s+([a-z])\s*(?:→|->|approaches?)\s*(-?[\w.∞π]+)',
    ]
    for pat in lim_pats:
        m = re.match(pat, t, re.I)
        if m:
            g = m.groups()
            # Identify expr and point
            if len(g) == 3:
                expr, var_or_pt, pt_or_expr = g[0], g[1], g[2]
                # Pattern 1: expr, var, pt
                pt = pt_or_expr.replace('∞','oo').replace('π','pi').replace('infinity','oo')
                return 'limit', f"{expr.strip()}, {pt}"
            break

    # Simpler limit pattern
    m = re.match(
        r'(?:find|compute|evaluate|what\s+is)?\s*(?:the\s+)?limit\s+of\s+(.*?)\s+as\s+x\s*(?:→|->|approaches?|goes?\s+to)\s*(-?[\w.∞π]+)',
        t, re.I)
    if m:
        pt = m.group(2).replace('∞','oo').replace('π','pi').replace('infinity','oo')
        return 'limit', m.group(1).strip() + ', ' + pt

    # ── Taylor / Maclaurin series ─────────────────────────────────────────
    m = re.match(
        r'(?:find|compute|give\s+me)?\s*(?:the\s+)?(?:taylor|maclaurin)\s+(?:series|expansion)\s+(?:of\s+)?(.*?)'
        r'(?:\s+(?:at|around|about|centered\s+at)\s+([\w.]+))?'
        r'(?:\s+(?:to|up\s+to|with|of)\s+(\d+)\s+terms?)?$',
        t, re.I)
    if m:
        expr   = m.group(1).strip()
        center = m.group(2) or '0'
        terms  = m.group(3) or '8'
        return 'taylor', f"{expr}, {terms}, {center}"

    # ── ODE ───────────────────────────────────────────────────────────────
    ode_pats = [
        r"(?:solve|find|compute)?\s*(?:the\s+)?(?:ode|differential\s+equation|diff(?:erential)?\s+eq(?:uation)?|de)\s+(.+)",
        r"(?:solve|find)\s+y\s+(?:if|given|where|such\s+that)\s+(.+)",
        r"(?:find|compute)\s+(?:the\s+)?(?:general\s+)?solution\s+(?:to|of)\s+(.+(?:y['']|dy/dx).+)",
    ]
    for pat in ode_pats:
        m = re.match(pat, t, re.I)
        if m: return 'ode2', m.group(1).strip()

    # ── Solve equation ────────────────────────────────────────────────────
    solve_pats = [
        r'solve\s+(?:the\s+)?(?:equation\s+)?(.*?)(?:\s+for\s+[a-z])?$',
        r'(?:find|what\s+(?:are|is))\s+(?:the\s+)?(?:roots?|zeros?|solutions?|value)\s+(?:of\s+)?(.*)',
        r'(?:find|what\s+is)\s+x\s+(?:when|if|given|such\s+that)\s+(.*)',
        r'when\s+(?:does|is)\s+(.*?)\s*=\s*(?:zero|0)\??',
        r'for\s+what\s+(?:value|values)\s+(?:of\s+x\s+)?(?:is|are|does)\s+(.*)',
        r'(?:calculate|evaluate|compute)\s+(.*?=.*)',
    ]
    for pat in solve_pats:
        m = re.match(pat, t, re.I)
        if m: return 'solve', m.group(1).strip()

    # ── Partial differentiation ───────────────────────────────────────────
    m = re.match(
        r'partial\s+(?:derivative|diff(?:erential)?|differentiation)\s+of\s+(.*?)'
        r'(?:\s+with\s+respect\s+to\s+([a-z](?:,\s*[a-z])*))',
        t, re.I)
    if m:
        vars_str = ', '.join(v.strip() for v in re.split(r'[,\s]+', m.group(2)) if v.strip())
        return 'partial', m.group(1).strip() + ', ' + vars_str
    m = re.match(r'∂(?:\((.*?)\))?/∂([a-z])', t)
    if m:
        return 'partial', m.group(1).strip() + ', ' + m.group(2)

    # ── Laplace transform ──────────────────────────────────────────────────
    m = re.match(r'(?:find|compute|take)?\s*(?:the\s+)?inverse\s+laplace\s+(?:transform\s+)?(?:of\s+)?(.*)', t, re.I)
    if m: return 'laplace', m.group(1).strip() + ', inverse'
    m = re.match(r'(?:find|compute|take)?\s*(?:the\s+)?laplace\s+(?:transform\s+)?(?:of\s+)?(.*)', t, re.I)
    if m: return 'laplace', m.group(1).strip()

    # ── Fourier transform ──────────────────────────────────────────────────
    m = re.match(r'(?:find|compute|take)?\s*(?:the\s+)?inverse\s+fourier\s+(?:transform\s+)?(?:of\s+)?(.*)', t, re.I)
    if m: return 'fourier', m.group(1).strip() + ', inverse'
    m = re.match(r'(?:find|compute|take)?\s*(?:the\s+)?fourier\s+(?:transform\s+)?(?:of\s+)?(.*)', t, re.I)
    if m: return 'fourier', m.group(1).strip()

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_pats = [
        r'(?:plot|graph|draw|visuali[sz]e|sketch|show|display)\s+(?:the\s+)?(?:function\s+|curve\s+|equation\s+)?(.*)',
        r'show\s+(?:me\s+)?(?:the\s+)?(?:graph|plot)\s+(?:of\s+)?(.*)',
        r'what\s+does\s+(.*?)\s+look\s+like(?:\s+graphically)?',
        r'can\s+you\s+(?:plot|graph|show|draw)\s+(.*)',
        r'(?:make|create|generate)\s+(?:a\s+)?(?:plot|graph)\s+(?:of\s+)?(.*)',
    ]
    for pat in plot_pats:
        m = re.match(pat, t, re.I)
        if m:
            expr = m.group(1).strip()
            # Detect if polar
            if re.search(r'\b(?:polar|r\s*=|radial)\b', expr, re.I):
                expr = re.sub(r'\bpolar\b', '', expr, flags=re.I).strip()
                return 'polar', expr
            # Detect parametric
            if ';' in expr or re.search(r'parametric', expr, re.I):
                return 'parametric', re.sub(r'parametric', '', expr, flags=re.I).strip()
            # Detect 3D
            if re.search(r'\bz\s*=|3d|surface\b', expr, re.I):
                expr = re.sub(r'(?:z\s*=|3d\s*|surface\s*)', '', expr, flags=re.I).strip()
                return 'plot3d', expr
            # Strip "and", "both", commas
            expr = re.sub(r'\band\b', ';', expr, flags=re.I)
            return 'plot', expr

    # ── Statistics ────────────────────────────────────────────────────────
    stat_pats = [
        r'(?:find|compute|calculate|give\s+me|show)?\s*(?:the\s+)?(?:statistics?|stats?|summary)\s+(?:of|for)\s+([\d,.\s]+)',
        r'(?:find|compute|calculate)?\s*(?:the\s+)?(?:mean|average|median|mode|std|variance)\s+(?:of\s+)?([\d,.\s]+)',
        r'(?:analyse|analyze|describe)\s+(?:the\s+)?(?:data|dataset|numbers)\s*:?\s*([\d,.\s]+)',
    ]
    for pat in stat_pats:
        m = re.match(pat, t, re.I)
        if m: return 'stats', m.group(1).strip()

    # ── Regression ────────────────────────────────────────────────────────
    m = re.match(r'(?:fit|find|compute)?\s*(?:a\s+)?(?:linear|quadratic|polynomial)?\s*regression\s+(?:for\s+)?(.*)', t, re.I)
    if m: return 'regression', m.group(1).strip()

    # ── Sequences & Series ────────────────────────────────────────────────
    series_pats = [
        r'(?:find|compute|evaluate)?\s*(?:the\s+)?sum(?:mation)?\s+of\s+(.*?)\s+from\s+n\s*=\s*(\w+)\s+to\s+(\w+)',
        r'(?:find|compute)?\s*∑\s*(.*?)\s+(?:from\s+)?n\s*=\s*(\w+)\s+to\s+(\w+)',
        r'(?:evaluate|compute)?\s*(?:the\s+)?series\s+(.*?)\s+from\s+n\s*=\s*(\d+)\s+to\s+(\w+)',
    ]
    for pat in series_pats:
        m = re.match(pat, t, re.I)
        if m:
            b = m.group(3).replace('infinity','oo').replace('∞','oo')
            return 'sequence', f"{m.group(1).strip()}, {m.group(2)}, {b}"

    # ── Number Theory ─────────────────────────────────────────────────────
    nt_pats = [
        (r'(?:prime\s+)?factori[sz](?:e|ation)\s+(?:of\s+)?(\d+)',  'factorize, '),
        (r'factor(?:s)?\s+of\s+(\d+)',                               'factorize, '),
        (r'is\s+(\d+)\s+(?:a\s+)?prime(?:\s+number)?[?!.]?',        'isprime, '),
        (r'(?:check\s+if\s+|is\s+it\s+)?(\d+)\s+(?:is\s+)?prime[?]?', 'isprime, '),
        (r'(?:find\s+)?gcd\s+(?:of\s+)?([\d,\s]+)',                 'gcd, '),
        (r'(?:find\s+)?lcm\s+(?:of\s+)?([\d,\s]+)',                 'lcm, '),
        (r'fibonacci\s+(?:number\s+)?(?:of\s+|#\s*)?(\d+)',         'fibonacci, '),
        (r'f\((\d+)\)\s+fibonacci',                                   'fibonacci, '),
        (r'(?:list\s+)?(?:all\s+)?primes?\s+(?:up\s+to|below|under|less\s+than)\s+(\d+)', 'primes, '),
        (r'next\s+prime\s+(?:after\s+)?(\d+)',                       'nextprime, '),
        (r'prime\s+after\s+(\d+)',                                    'nextprime, '),
        (r'previous\s+prime\s+(?:before\s+)?(\d+)',                  'prevprime, '),
        (r'(?:list\s+)?(?:all\s+)?divisors?\s+(?:of\s+)?(\d+)',      'divisors, '),
        (r'factors?\s+of\s+(\d+)',                                    'divisors, '),
        (r'(?:euler[\'s\s]+)?totient\s+(?:function\s+)?(?:of\s+)?(\d+)', 'totient, '),
        (r'φ\((\d+)\)',                                               'totient, '),
        (r'is\s+(\d+)\s+(?:a\s+)?perfect\s+number',                  'perfect, '),
        (r'modular\s+inverse\s+(?:of\s+)?(\d+)\s+(?:mod|modulo)\s+(\d+)', 'modinv, '),
        (r'(\d+)\s+\^\s*(\d+)\s+(?:mod|modulo)\s+(\d+)',             'modpow, '),
    ]
    for pat, prefix in nt_pats:
        m = re.search(pat, ts, re.I)
        if m:
            groups = m.groups()
            if len(groups) == 1:
                return 'numtheory', prefix + groups[0]
            elif len(groups) > 1:
                return 'numtheory', prefix + ', '.join(g for g in groups if g)

    # ── Vector Calculus ───────────────────────────────────────────────────
    vec_pats = [
        (r'(?:find|compute)?\s*(?:the\s+)?gradient\s+(?:of\s+)?(.*)',    'gradient, '),
        (r'(?:find|compute)?\s*(?:the\s+)?laplacian\s+(?:of\s+)?(.*)',   'laplacian, '),
        (r'(?:find|compute)?\s*(?:the\s+)?divergence\s+(?:of\s+)?(.*)',  'divergence, '),
        (r'(?:find|compute)?\s*(?:the\s+)?curl\s+(?:of\s+)?(.*)',        'curl, '),
        (r'∇\s+(.*)',                                                       'gradient, '),
        (r'∇²\s+(.*)',                                                      'laplacian, '),
        (r'∇·\s*(.*)',                                                      'divergence, '),
        (r'∇×\s*(.*)',                                                      'curl, '),
    ]
    for pat, prefix in vec_pats:
        m = re.match(pat, t, re.I)
        if m: return 'vector', prefix + m.group(1).strip()

    # ── Matrix ────────────────────────────────────────────────────────────
    mat_pats = [
        (r'(?:find|compute)?\s*(?:the\s+)?determinant\s+(?:of\s+)?(\[.*)',  ', det'),
        (r'(?:find|compute)?\s*(?:the\s+)?inverse\s+(?:of\s+)?(\[.*)',      ', invert'),
        (r'(?:find|compute)?\s*eigenvalues?\s+(?:of\s+)?(\[.*)',             ', eigenvals'),
        (r'(?:find|compute)?\s*eigenvectors?\s+(?:of\s+)?(\[.*)',            ', eigenvects'),
        (r'transpose\s+(?:of\s+)?(\[.*)',                                     ', transpose'),
        (r'rank\s+(?:of\s+)?(\[.*)',                                          ', rank'),
        (r'trace\s+(?:of\s+)?(\[.*)',                                         ', trace'),
        (r'rref\s+(?:of\s+)?(\[.*)',                                          ', rref'),
        (r'row\s+reduce\s+(\[.*)',                                             ', rref'),
    ]
    for pat, suffix in mat_pats:
        m = re.match(pat, t, re.I)
        if m: return 'matrix', m.group(1).strip() + suffix

    # ── Inequalities ──────────────────────────────────────────────────────
    ineq_pats = [
        r'(?:solve|find)?\s*(?:the\s+)?inequality\s+(.*)',
        r'(?:for\s+what\s+x\s+is\s+|when\s+is\s+)?(.*?)\s*(?:>|<|>=|<=|≥|≤)\s*(.*)',
    ]
    for pat in ineq_pats:
        m = re.match(pat, t, re.I)
        if m:
            raw = m.group(0) if len(m.groups()) > 1 else m.group(1)
            return 'inequality', raw.strip()

    # ── System of equations ───────────────────────────────────────────────
    m = re.match(r'(?:solve|find)?\s*(?:the\s+)?system\s+(?:of\s+equations?\s+)?(.*)', t, re.I)
    if m: return 'system', m.group(1).strip()

    # ── Generic "what is / calculate" → solve ────────────────────────────
    m = re.match(r'(?:what\s+is|calculate|compute|evaluate|find)\s+(.*?)\??$', t, re.I)
    if m:
        expr = m.group(1).strip().rstrip('?')
        # Check if it looks like math
        if re.search(r'[0-9x\+\-\*\/\^]|sin|cos|tan|log|sqrt|exp', expr, re.I):
            return 'solve', expr

    # ── Raw expression with = sign ────────────────────────────────────────
    if '=' in t and re.search(r'[0-9x]', t):
        return 'solve', t

    # ── Detect naked inequality ───────────────────────────────────────────
    if re.search(r'[<>]|>=|<=|≥|≤', t) and re.search(r'[0-9x]', t):
        return 'inequality', t

    # ── Fallthrough: try AI routing first, then math expression detection ────
    # If input looks like a math expression, try to evaluate it directly
    if re.search(r'(?:sin|cos|tan|log|sqrt|exp|\d[\+\-\*\/]\d)', ts, re.I):
        return 'solve', t

    # ── AI-powered routing for anything else ──────────────────────────────────
    # Let the local LLM decide what op + input to use (or give a direct answer)
    ai_op, ai_inp = OfflineAI.route(text)
    if ai_op:
        return ai_op, ai_inp

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Known math function names — never insert * before their opening paren
# ─────────────────────────────────────────────────────────────────────────────
_MATH_FUNCS = frozenset([
    'sin','cos','tan','cot','sec','csc',
    'asin','acos','atan','atan2','acot','asec','acsc',
    'arcsin','arccos','arctan','arccot',
    'sinh','cosh','tanh','coth','sech','csch',
    'asinh','acosh','atanh',
    'exp','log','ln','sqrt','cbrt','Abs','abs','sign',
    'ceiling','floor','factorial','gamma','beta',
    'erf','erfc','Max','Min','re','im',
    'conjugate','transpose','trace','det',
])


# ─────────────────────────────────────────────────────────────────────────────
#  Natural-notation preprocessor
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(expr: str) -> str:
    s = expr.strip()
    s = s.replace('^', '**')
    s = re.sub(r'(\d)([A-Za-z(])', r'\1*\2', s)
    s = re.sub(r'\)\(', ')*(', s)
    s = re.sub(r'\)([A-Za-z\d])', r')*\1', s)

    def _maybe_mul(m):
        word = m.group(1)
        return word + '(' if word in _MATH_FUNCS else word + '*('
    s = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)\(', _maybe_mul, s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Rich SymPy namespace
# ─────────────────────────────────────────────────────────────────────────────
_SYMPY_NS = {
    'sin': sp.sin,  'cos': sp.cos,   'tan': sp.tan,
    'cot': sp.cot,  'sec': sp.sec,   'csc': sp.csc,
    'asin': sp.asin,'acos': sp.acos, 'atan': sp.atan,
    'arcsin': sp.asin,'arccos': sp.acos,'arctan': sp.atan,
    'sinh': sp.sinh,'cosh': sp.cosh, 'tanh': sp.tanh,
    'asinh': sp.asinh,'acosh': sp.acosh,'atanh': sp.atanh,
    'exp': sp.exp,  'log': sp.log,   'ln': sp.log,
    'sqrt': sp.sqrt,'Abs': sp.Abs,   'abs': sp.Abs,
    'factorial': sp.factorial,'gamma': sp.gamma,
    'floor': sp.floor,'ceiling': sp.ceiling,
    'erf': sp.erf,  'erfc': sp.erfc,
    'pi': sp.pi,    'E': sp.E,       'I': sp.I,
    'oo': sp.oo,    'inf': sp.oo,
    'x': sp.Symbol('x'), 'y': sp.Symbol('y'),
    'z': sp.Symbol('z'), 't': sp.Symbol('t'),
    'n': sp.Symbol('n'),
}


def safe_sympify(expr_str: str):
    return sp.sympify(preprocess(expr_str), locals=_SYMPY_NS)


# ─────────────────────────────────────────────────────────────────────────────
#  Equation solver
# ─────────────────────────────────────────────────────────────────────────────
def solve_equation(eq_str, domain='real', numerical=False):
    x         = sp.Symbol('x')
    processed = preprocess(eq_str)

    if '=' in processed:
        l, r = processed.split('=', 1)
        lhs  = sp.sympify(l.strip(), locals=_SYMPY_NS)
        rhs  = sp.sympify(r.strip(), locals=_SYMPY_NS)
        eq   = sp.Eq(lhs, rhs)
        expr = lhs - rhs
    else:
        expr = sp.sympify(processed, locals=_SYMPY_NS)
        eq   = sp.Eq(expr, 0)

    dom = {'real': sp.S.Reals, 'complex': sp.S.Complexes}.get(
        domain.lower(), sp.S.Reals)

    if numerical:
        sols = []
        for g in [-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10]:
            try:
                s = sp.nsolve(eq, x, g)
                if all(abs(s - e) > 1e-6 for e in sols):
                    sols.append(s)
            except Exception:
                pass
        solutions = sp.FiniteSet(*sols) if sols else sp.EmptySet
    else:
        try:
            solutions = sp.solveset(eq, x, domain=dom)
        except Exception:
            solutions = sp.EmptySet
        if isinstance(solutions, sp.ConditionSet) or solutions == sp.EmptySet:
            sols = []
            for g in [0.1, 1, 2, 5, 10, -1, -2]:
                try:
                    s = sp.nsolve(eq, x, g)
                    if all(abs(s - e) > 1e-6 for e in sols):
                        sols.append(s)
                except Exception:
                    pass
            if sols:
                solutions = sp.FiniteSet(*sols)

    return solutions, eq, expr


# ─────────────────────────────────────────────────────────────────────────────
#  Differentiation
# ─────────────────────────────────────────────────────────────────────────────
def do_differentiate(expr_str):
    parts = expr_str.split(',')
    if len(parts) > 1:
        expr = safe_sympify(parts[0].strip())
        for v in parts[1:]:
            expr = sp.diff(expr, sp.Symbol(v.strip()))
        return expr
    return sp.diff(safe_sympify(expr_str.strip()), sp.Symbol('x'))


# ─────────────────────────────────────────────────────────────────────────────
#  Integration
# ─────────────────────────────────────────────────────────────────────────────
def do_integrate(expr_str):
    parts = [p.strip() for p in expr_str.split(',')]
    x     = sp.Symbol('x')
    expr  = safe_sympify(parts[0])
    if len(parts) == 3:
        a = float(sp.sympify(parts[1], locals=_SYMPY_NS))
        b = float(sp.sympify(parts[2], locals=_SYMPY_NS))
        return sp.integrate(expr, (x, a, b))
    return sp.integrate(expr, x)


# ─────────────────────────────────────────────────────────────────────────────
#  Limit
# ─────────────────────────────────────────────────────────────────────────────
def do_limit(expr_str):
    parts     = [p.strip() for p in expr_str.split(',')]
    x         = sp.Symbol('x')
    expr      = safe_sympify(parts[0])
    pt        = sp.sympify(parts[1], locals=_SYMPY_NS) if len(parts) > 1 else sp.S.Zero
    direction = parts[2] if len(parts) > 2 else '+-'
    return sp.limit(expr, x, pt, direction)


# ─────────────────────────────────────────────────────────────────────────────
#  Taylor series
# ─────────────────────────────────────────────────────────────────────────────
def do_taylor(expr_str):
    parts = [p.strip() for p in expr_str.split(',')]
    x     = sp.Symbol('x')
    expr  = safe_sympify(parts[0])
    n     = int(parts[1]) if len(parts) > 1 else 6
    a     = float(sp.sympify(parts[2], locals=_SYMPY_NS)) if len(parts) > 2 else 0
    return sp.series(expr, x, a, n).removeO()


# ─────────────────────────────────────────────────────────────────────────────
#  Matrix operations
# ─────────────────────────────────────────────────────────────────────────────
def do_matrix(expr_str):
    depth   = 0
    end_idx = -1
    for i, ch in enumerate(expr_str):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx == -1:
        raise ValueError("Could not find closing ']' in matrix expression.")

    mat_str   = expr_str[:end_idx + 1].strip()
    rest      = expr_str[end_idx + 1:].strip().lstrip(',').strip()
    operation = rest.split(',')[0].strip().lower()

    try:
        mat = sp.Matrix(sp.sympify(mat_str, locals=_SYMPY_NS))
    except Exception as e:
        raise ValueError(f"Could not parse matrix: {e}")

    if   operation == 'invert':    return mat.inv()
    elif operation == 'det':       return mat.det()
    elif operation == 'eigenvals': return mat.eigenvals()
    elif operation == 'eigenvects':return mat.eigenvects()
    elif operation == 'rank':      return mat.rank()
    elif operation == 'rref':      return mat.rref()
    elif operation == 'transpose': return mat.T
    elif operation == 'trace':     return mat.trace()
    elif operation == 'nullspace': return mat.nullspace()
    elif operation == 'columnspace': return mat.columnspace()
    elif operation == 'cholesky':
        try: return mat.cholesky()
        except Exception: raise ValueError("Matrix must be positive definite for Cholesky.")
    else:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            "Use: invert | det | eigenvals | eigenvects | rank | rref | "
            "transpose | trace | nullspace | columnspace | cholesky"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  2nd-Order ODE solver
# ─────────────────────────────────────────────────────────────────────────────
def parse_ode2_ics(ic_strings):
    ics = {}
    x   = sp.Symbol('x')
    f   = sp.Function('y')
    for ic in ic_strings:
        ic = ic.strip()
        m  = re.match(r"y('?)\(([^)]+)\)\s*=\s*(.+)", ic)
        if m:
            prime = m.group(1)
            pt    = sp.sympify(m.group(2), locals=_SYMPY_NS)
            val   = sp.sympify(m.group(3), locals=_SYMPY_NS)
            if prime == "'":
                ics[f(x).diff(x).subs(x, pt)] = val
            else:
                ics[f(pt)] = val
    return ics


def solve_ode2(expr_str):
    x = sp.Symbol('x')
    f = sp.Function('y')
    y = f(x)

    depth   = 0
    split_i = -1
    for i, ch in enumerate(expr_str):
        if ch in '([':  depth += 1
        elif ch in ')]': depth -= 1
        elif ch == ',' and depth == 0:
            split_i = i
            break

    if split_i == -1:
        ode_str = expr_str.strip()
        ic_strs = []
    else:
        ode_str = expr_str[:split_i].strip()
        ic_strs = [s.strip() for s in expr_str[split_i + 1:].split(',')]

    s = ode_str
    s = s.replace("y''", '__DD__')
    s = s.replace("y'",  '__D__')
    s = re.sub(r'(\d)\s*y', r'\1*y', s)
    s = s.replace('y', '__Y__')
    s = s.replace('__DD__', 'y.diff(x,2)')
    s = s.replace('__D__',  'y.diff(x)')
    s = s.replace('__Y__',  'y')
    s = s.replace('^', '**')
    s = re.sub(r'(\d)([A-Za-z(])', r'\1*\2', s)

    lhs_s, rhs_s = s.split('=', 1) if '=' in s else (s, '0')

    ns = {
        'y': y, 'x': x,
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
        'pi': sp.pi,   'E': sp.E,     'oo': sp.oo,
        '__builtins__': {},
    }

    lhs = eval(lhs_s.strip(), ns)
    rhs = eval(rhs_s.strip(), ns)
    ode = sp.Eq(lhs, rhs)

    ics = parse_ode2_ics(ic_strs) if ic_strs else {}
    sol = sp.dsolve(ode, y, ics=ics if ics else None)
    return sol, ode


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Simplify / Expand / Factor / Cancel / Apart
# ─────────────────────────────────────────────────────────────────────────────
def do_simplify(expr_str):
    parts = [p.strip() for p in expr_str.split(',', 1)]
    expr  = safe_sympify(parts[0])
    op    = parts[1].lower() if len(parts) > 1 else 'simplify'

    ops = {
        'simplify':  sp.simplify,
        'expand':    sp.expand,
        'factor':    sp.factor,
        'cancel':    sp.cancel,
        'apart':     sp.apart,
        'trigsimp':  sp.trigsimp,
        'radsimp':   sp.radsimp,
        'powsimp':   sp.powsimp,
        'nsimplify': sp.nsimplify,
        'collect':   lambda e: sp.collect(e, sp.Symbol('x')),
    }
    fn = ops.get(op)
    if fn is None:
        raise ValueError(f"Unknown sub-operation '{op}'. Use: " + ' | '.join(ops.keys()))
    return fn(expr)


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Laplace Transform (forward & inverse)
# ─────────────────────────────────────────────────────────────────────────────
def do_laplace(expr_str):
    parts = [p.strip() for p in expr_str.split(',', 1)]
    direction = parts[1].lower() if len(parts) > 1 else 'forward'

    t = sp.Symbol('t', positive=True)
    s = sp.Symbol('s')

    if direction in ('inverse', 'inv', 'ilap'):
        expr = sp.sympify(preprocess(parts[0]), locals={**_SYMPY_NS, 's': s})
        result = sp.inverse_laplace_transform(expr, s, t)
        latex_lhs = r'\mathcal{L}^{-1}\!\left\{' + sp.latex(expr) + r'\right\}'
        return result, latex_lhs
    else:
        expr = sp.sympify(preprocess(parts[0]), locals={**_SYMPY_NS, 't': t})
        result = sp.laplace_transform(expr, t, s, noconds=True)
        latex_lhs = r'\mathcal{L}\!\left\{' + sp.latex(expr) + r'\right\}'
        return result, latex_lhs


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Fourier Transform (forward & inverse)
# ─────────────────────────────────────────────────────────────────────────────
def do_fourier(expr_str):
    parts = [p.strip() for p in expr_str.split(',', 1)]
    direction = parts[1].lower() if len(parts) > 1 else 'forward'

    x = sp.Symbol('x')
    k = sp.Symbol('k')

    if direction in ('inverse', 'inv', 'ift'):
        expr = sp.sympify(preprocess(parts[0]), locals={**_SYMPY_NS, 'k': k})
        result = sp.inverse_fourier_transform(expr, k, x)
        latex_lhs = r'\mathcal{F}^{-1}\!\left\{' + sp.latex(expr) + r'\right\}'
        return result, latex_lhs
    else:
        expr = sp.sympify(preprocess(parts[0]), locals={**_SYMPY_NS, 'x': x})
        result = sp.fourier_transform(expr, x, k)
        latex_lhs = r'\mathcal{F}\!\left\{' + sp.latex(expr) + r'\right\}'
        return result, latex_lhs


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Statistics on datasets
# ─────────────────────────────────────────────────────────────────────────────
def do_stats(expr_str):
    s = expr_str.strip()
    if s.startswith('[') and ']' in s:
        s = s[1:s.rindex(']')]

    STAT_OPS = {'mean','median','mode','variance','std','min','max','range',
                'sum','count','all','skew','percentile','zscore','iqr'}

    parts = s.rsplit(',', 1)
    last  = parts[-1].strip().lower()
    if last in STAT_OPS and len(parts) > 1:
        data_str = parts[0]
        op       = last
    else:
        data_str = s
        op       = 'all'

    data = [float(x.strip()) for x in data_str.split(',')]
    n    = len(data)
    if n == 0:
        raise ValueError("Empty dataset")

    mean_   = _stats.mean(data)
    median_ = _stats.median(data)
    try:
        mode_ = _stats.mode(data)
    except Exception:
        mode_ = 'N/A (multiple modes)'
    var_   = _stats.variance(data) if n > 1 else 0
    std_   = _stats.stdev(data) if n > 1 else 0
    q1     = float(np.percentile(data, 25))
    q3     = float(np.percentile(data, 75))
    iqr_   = q3 - q1
    skew_  = float(sp.N(sp.S(sum((xi - mean_)**3 for xi in data)/n) / sp.S(std_)**3)) if std_ else 0

    summary = {
        'count':    n,
        'mean':     round(mean_, 8),
        'median':   round(median_, 8),
        'mode':     mode_,
        'variance': round(var_, 8),
        'std_dev':  round(std_, 8),
        'min':      min(data),
        'max':      max(data),
        'range':    round(max(data) - min(data), 8),
        'sum':      round(sum(data), 8),
        'Q1':       round(q1, 8),
        'Q3':       round(q3, 8),
        'IQR':      round(iqr_, 8),
        'skewness': round(skew_, 4),
    }
    return summary, data


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Number Theory
# ─────────────────────────────────────────────────────────────────────────────
def do_number_theory(expr_str):
    parts = [p.strip() for p in expr_str.split(',')]
    if not parts:
        raise ValueError("No input provided")
    op = parts[0].lower()

    if op == 'gcd':
        nums   = [int(p) for p in parts[1:]]
        result = nums[0]
        for nn in nums[1:]:
            result = sp.gcd(result, nn)
        return f"GCD = {result}"
    elif op == 'lcm':
        nums   = [int(p) for p in parts[1:]]
        result = nums[0]
        for nn in nums[1:]:
            result = sp.lcm(result, nn)
        return f"LCM = {result}"
    elif op == 'isprime':
        n = int(parts[1])
        return f"{n} is {'prime ✓' if sp.isprime(n) else 'not prime ✗'}"
    elif op in ('factorize', 'factor'):
        n = int(parts[1])
        d = sp.factorint(n)
        factored = ' × '.join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(d.items()))
        return f"{n} = {factored}"
    elif op == 'totient':
        n = int(parts[1])
        return f"φ({n}) = {sp.totient(n)}"
    elif op == 'primes':
        n = int(parts[1])
        plist = list(sp.primerange(2, n + 1))
        return f"Primes up to {n} ({len(plist)} total): {plist[:50]}{'...' if len(plist)>50 else ''}"
    elif op == 'fibonacci':
        n = int(parts[1])
        return f"F({n}) = {sp.fibonacci(n)}"
    elif op == 'modpow':
        a, b, m = int(parts[1]), int(parts[2]), int(parts[3])
        return f"{a}^{b} mod {m} = {pow(a, b, m)}"
    elif op == 'nextprime':
        n = int(parts[1])
        return f"Next prime after {n} = {sp.nextprime(n)}"
    elif op == 'prevprime':
        n = int(parts[1])
        return f"Previous prime before {n} = {sp.prevprime(n)}"
    elif op == 'divisors':
        n = int(parts[1])
        divs = sp.divisors(n)
        return f"Divisors of {n}: {divs} (count: {len(divs)})"
    elif op == 'perfect':
        n = int(parts[1])
        d = sp.divisors(n)
        return f"{n} is {'a perfect number ✓' if sum(d[:-1]) == n else 'not a perfect number'}"
    elif op == 'modinv':
        a, m = int(parts[1]), int(parts[2])
        try:
            return f"{a}^(-1) mod {m} = {sp.mod_inverse(a, m)}"
        except Exception:
            return f"No modular inverse (gcd({a},{m}) ≠ 1)"
    else:
        raise ValueError(
            f"Unknown op '{op}'. Use: gcd, lcm, isprime, factorize, totient, primes, "
            "fibonacci, modpow, nextprime, prevprime, divisors, perfect, modinv"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: System of equations solver
# ─────────────────────────────────────────────────────────────────────────────
def do_system(expr_str):
    parts    = [p.strip() for p in expr_str.split(';')]
    var_names = ['x', 'y', 'z', 'w']
    eq_strs  = []

    for p in parts:
        if p.lower().startswith('vars:'):
            var_names = [v.strip() for v in p[5:].split(',')]
        else:
            eq_strs.append(p)

    symbols  = [sp.Symbol(v) for v in var_names]
    sym_dict = {v: s for v, s in zip(var_names, symbols)}
    sym_dict.update(_SYMPY_NS)

    equations = []
    for eq_str in eq_strs:
        processed = preprocess(eq_str)
        if '=' in processed:
            l, r = processed.split('=', 1)
            lhs = sp.sympify(l.strip(), locals=sym_dict)
            rhs = sp.sympify(r.strip(), locals=sym_dict)
            equations.append(sp.Eq(lhs, rhs))
        else:
            expr = sp.sympify(processed, locals=sym_dict)
            equations.append(sp.Eq(expr, 0))

    # Only solve for variables that appear
    used_syms = []
    for sym in symbols:
        if any(sym in eq.free_symbols for eq in equations):
            used_syms.append(sym)

    solution = sp.solve(equations, used_syms if used_syms else symbols)
    return solution


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Sequences / Summation / Product
# ─────────────────────────────────────────────────────────────────────────────
def do_sequence(expr_str):
    parts = [p.strip() for p in expr_str.split(',')]
    op    = 'sum'

    if parts[-1].lower() in ('sum', 'product', 'prod'):
        op    = parts[-1].lower()
        parts = parts[:-1]

    n    = sp.Symbol('n')
    expr = sp.sympify(preprocess(parts[0]), locals={**_SYMPY_NS, 'n': n})

    if len(parts) >= 3:
        a   = int(sp.sympify(parts[1], locals=_SYMPY_NS))
        bv  = parts[2].strip()
        b   = sp.oo if bv in ('oo', 'inf', 'infinity') else int(sp.sympify(bv, locals=_SYMPY_NS))
    else:
        a, b = 1, sp.oo

    if op in ('product', 'prod'):
        result = sp.product(expr, (n, a, b))
        lhs    = r'\prod_{n=' + str(a) + r'}^{' + (r'\infty' if b == sp.oo else str(b)) + r'}'
    else:
        result = sp.summation(expr, (n, a, b))
        lhs    = r'\sum_{n=' + str(a) + r'}^{' + (r'\infty' if b == sp.oo else str(b)) + r'}'

    return result, lhs + sp.latex(expr)


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Inequality solver
# ─────────────────────────────────────────────────────────────────────────────
def do_inequality(expr_str):
    x         = sp.Symbol('x', real=True)
    processed = preprocess(expr_str)

    for op_str, op_func in [('>=', sp.Ge), ('<=', sp.Le), ('>', sp.Gt), ('<', sp.Lt)]:
        if op_str in processed:
            l, r = processed.split(op_str, 1)
            lhs  = sp.sympify(l.strip(), locals=_SYMPY_NS)
            rhs  = sp.sympify(r.strip(), locals=_SYMPY_NS)
            ineq = op_func(lhs, rhs)
            result = sp.solve(ineq, x)
            return result
    raise ValueError("No inequality operator found. Use <, >, <=, or >=")


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Partial differentiation
# ─────────────────────────────────────────────────────────────────────────────
def do_partial_diff(expr_str):
    parts    = [p.strip() for p in expr_str.split(',')]
    x, y, z, t = sp.symbols('x y z t')
    sym_dict = {**_SYMPY_NS, 'x': x, 'y': y, 'z': z, 't': t}

    expr   = sp.sympify(preprocess(parts[0]), locals=sym_dict)
    result = expr
    var_sequence = []

    for var_str in parts[1:]:
        v  = sp.Symbol(var_str.strip())
        result = sp.diff(result, v)
        var_sequence.append(var_str.strip())

    # Build LaTeX
    if len(var_sequence) == 1:
        latex_lhs = r'\frac{\partial}{\partial ' + var_sequence[0] + r'}\!\left(' + sp.latex(expr) + r'\right)'
    else:
        order_str = ''.join(r'\partial ' + v for v in var_sequence)
        latex_lhs = (r'\frac{\partial^{' + str(len(var_sequence)) + r'}}{' +
                     order_str + r'}\!\left(' + sp.latex(expr) + r'\right)')
    return result, latex_lhs


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Vector calculus (gradient, divergence, curl, laplacian)
# ─────────────────────────────────────────────────────────────────────────────
def do_vector(expr_str):
    parts    = [p.strip() for p in expr_str.split(',')]
    op       = parts[0].lower()
    x, y, z  = sp.symbols('x y z')
    sym_dict = {**_SYMPY_NS, 'x': x, 'y': y, 'z': z}

    if op in ('gradient', 'grad'):
        f    = sp.sympify(preprocess(parts[1]), locals=sym_dict)
        gx   = sp.diff(f, x)
        gy   = sp.diff(f, y)
        gz   = sp.diff(f, z) if z in f.free_symbols else sp.S.Zero
        text = f"∂f/∂x = {gx}\n∂f/∂y = {gy}"
        if gz != 0: text += f"\n∂f/∂z = {gz}"
        lx   = (r'\nabla f = \left(' + sp.latex(gx) + r',\; ' +
                sp.latex(gy) + (r',\; ' + sp.latex(gz) if gz != 0 else '') + r'\right)')
        return text, lx

    elif op in ('laplacian', 'lap'):
        f    = sp.sympify(preprocess(parts[1]), locals=sym_dict)
        lap  = sum(sp.diff(f, v, 2) for v in [x, y, z] if v in f.free_symbols)
        return f"∇²f = {lap}", r'\nabla^2 f = ' + sp.latex(lap)

    elif op in ('divergence', 'div'):
        Fx   = sp.sympify(preprocess(parts[1]), locals=sym_dict)
        Fy   = sp.sympify(preprocess(parts[2]), locals=sym_dict) if len(parts) > 2 else sp.S.Zero
        Fz   = sp.sympify(preprocess(parts[3]), locals=sym_dict) if len(parts) > 3 else sp.S.Zero
        div  = sp.diff(Fx, x) + sp.diff(Fy, y) + sp.diff(Fz, z)
        return f"∇·F = {sp.simplify(div)}", r'\nabla \cdot \mathbf{F} = ' + sp.latex(sp.simplify(div))

    elif op == 'curl':
        Fx   = sp.sympify(preprocess(parts[1]), locals=sym_dict)
        Fy   = sp.sympify(preprocess(parts[2]), locals=sym_dict) if len(parts) > 2 else sp.S.Zero
        Fz   = sp.sympify(preprocess(parts[3]), locals=sym_dict) if len(parts) > 3 else sp.S.Zero
        cx   = sp.simplify(sp.diff(Fz, y) - sp.diff(Fy, z))
        cy   = sp.simplify(sp.diff(Fx, z) - sp.diff(Fz, x))
        cz   = sp.simplify(sp.diff(Fy, x) - sp.diff(Fx, y))
        return (f"∇×F = ({cx}, {cy}, {cz})",
                r'\nabla \times \mathbf{F} = \left(' + sp.latex(cx) + r',\; ' + sp.latex(cy) + r',\; ' + sp.latex(cz) + r'\right)')
    else:
        raise ValueError("Unknown op. Use: gradient, laplacian, divergence, curl")


# ─────────────────────────────────────────────────────────────────────────────
#  NEW: Polynomial regression
# ─────────────────────────────────────────────────────────────────────────────
def do_regression(expr_str):
    degree = 1
    if 'degree=' in expr_str.lower():
        m = re.search(r'degree=(\d+)', expr_str, re.I)
        if m:
            degree = int(m.group(1))
            expr_str = re.sub(r',?\s*degree=\d+', '', expr_str, flags=re.I)

    pairs = [p.strip() for p in expr_str.split(';') if p.strip()]
    xs, ys = [], []
    for pair in pairs:
        a, b = pair.split(',')
        xs.append(float(a.strip()))
        ys.append(float(b.strip()))

    xs = np.array(xs)
    ys = np.array(ys)
    coeffs = np.polyfit(xs, ys, degree)
    poly   = np.poly1d(coeffs)

    y_pred = poly(xs)
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot != 0 else 1.0

    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-10:
            continue
        if power == 0:
            terms.append(f'{c:.4f}')
        elif power == 1:
            terms.append(f'{c:.4f}x')
        else:
            terms.append(f'{c:.4f}x^{power}')
    eq_str = ' + '.join(terms).replace('+ -', '- ')

    summary = {
        'equation':     eq_str,
        'coefficients': [round(float(c), 6) for c in coeffs],
        'r_squared':    round(r2, 6),
        'rmse':         round(float(np.sqrt(np.mean((ys - y_pred)**2))), 6),
        'degree':       degree,
        'n_points':     len(xs),
    }
    return summary, xs.tolist(), ys.tolist(), poly


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
_COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657', '#39d353', '#ff7b72']

def _dark_fig_ax(figsize=(8, 4.5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    return fig, ax

def _style_ax(ax, title='', xlabel='x', ylabel='y'):
    ax.axhline(0, color='#30363d', lw=1)
    ax.axvline(0, color='#30363d', lw=1)
    ax.grid(color='#21262d', linestyle='--', linewidth=0.5)
    ax.tick_params(colors='#8b949e')
    ax.spines[:].set_color('#30363d')
    if title: ax.set_title(title, color='#c9d1d9', fontsize=11, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color='#8b949e', fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color='#8b949e', fontsize=10)

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_plot(expr_str, title='f(x)', x_range=(-10, 10)):
    fig, ax = _dark_fig_ax()
    plotted = 0

    for i, es in enumerate(e.strip() for e in expr_str.split(';')):
        if not es: continue
        try:
            expr = safe_sympify(es)
            fn   = sp.lambdify(sp.Symbol('x'), expr, modules=['numpy'])
            xs   = np.linspace(x_range[0], x_range[1], 1000)
            raw  = fn(xs)
            if np.isscalar(raw):
                ys = np.full_like(xs, float(np.real(raw)))
            else:
                ys = np.real(np.array(raw, dtype=complex))
            ys = np.where(np.abs(ys) > 1e8, np.nan, ys)
            ax.plot(xs, ys, color=_COLORS[i % len(_COLORS)], lw=2.2,
                    label=f'$y = {sp.latex(expr)}$')
            plotted += 1
        except Exception:
            pass

    if plotted == 0:
        plt.close(fig)
        raise ValueError("None of the expressions could be plotted.")

    _style_ax(ax, title=title)
    ax.legend(fontsize=9, facecolor='#161b22', labelcolor='#c9d1d9',
              framealpha=0.8, edgecolor='#30363d')
    plt.tight_layout()
    return _fig_to_b64(fig)


def make_ode_plot(rhs_str):
    try:
        C1, C2 = sp.symbols('C1 C2')
        expr = sp.sympify(rhs_str)
        if C1 in expr.free_symbols or C2 in expr.free_symbols:
            expr = expr.subs([(C1, 1), (C2, 0)])
        return make_plot(str(expr), title='ODE Solution (C₁=1, C₂=0)', x_range=(-6, 6))
    except Exception:
        return None


def make_polar_plot(expr_str):
    fig = plt.figure(figsize=(7, 7))
    ax  = fig.add_subplot(111, projection='polar')
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    theta = sp.Symbol('theta')
    plotted = 0

    for i, es in enumerate(e.strip() for e in expr_str.split(';')):
        if not es: continue
        try:
            expr = sp.sympify(preprocess(es), locals={**_SYMPY_NS, 'theta': theta})
            fn   = sp.lambdify(theta, expr, modules=['numpy'])
            th   = np.linspace(0, 2 * np.pi, 1200)
            r    = fn(th)
            if np.isscalar(r):
                r = np.full_like(th, float(r))
            else:
                r = np.real(np.array(r, dtype=complex))
            ax.plot(th, r, color=_COLORS[i % len(_COLORS)], lw=2,
                    label=f'r = {sp.latex(expr)}')
            plotted += 1
        except Exception:
            pass

    if plotted == 0:
        plt.close(fig)
        raise ValueError("Could not plot polar expression.")

    ax.tick_params(colors='#8b949e')
    ax.grid(color='#21262d', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=9, facecolor='#161b22', labelcolor='#c9d1d9',
              framealpha=0.8, edgecolor='#30363d')
    ax.set_title('Polar Plot', color='#c9d1d9', fontsize=11, pad=14)
    plt.tight_layout()
    return _fig_to_b64(fig)


def make_parametric_plot(expr_str):
    parts   = [p.strip() for p in expr_str.split(';')]
    if len(parts) < 2:
        raise ValueError("Format: x_expr; y_expr [; t_start,t_end]")
    x_str, y_str = parts[0], parts[1]
    t_range = (-2 * np.pi, 2 * np.pi)
    if len(parts) >= 3:
        bounds = [float(v.strip()) for v in parts[2].split(',')]
        if len(bounds) >= 2:
            t_range = (bounds[0], bounds[1])

    t_sym  = sp.Symbol('t')
    x_expr = sp.sympify(preprocess(x_str), locals={**_SYMPY_NS, 't': t_sym})
    y_expr = sp.sympify(preprocess(y_str), locals={**_SYMPY_NS, 't': t_sym})
    fx     = sp.lambdify(t_sym, x_expr, modules=['numpy'])
    fy     = sp.lambdify(t_sym, y_expr, modules=['numpy'])

    ts = np.linspace(t_range[0], t_range[1], 1200)
    xs = np.real(np.array(fx(ts), dtype=complex))
    ys = np.real(np.array(fy(ts), dtype=complex))

    fig, ax = _dark_fig_ax((7, 7))
    from matplotlib.collections import LineCollection
    pts  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap='cool', linewidth=2.2)
    lc.set_array(ts)
    ax.add_collection(lc)
    ax.autoscale()
    _style_ax(ax, title=f'Parametric: x(t)={x_str}  y(t)={y_str}',
              xlabel='x(t)', ylabel='y(t)')
    fig.colorbar(lc, ax=ax, label='t', shrink=0.7).ax.yaxis.set_tick_params(color='#8b949e')
    plt.tight_layout()
    return _fig_to_b64(fig)


def make_3d_plot(expr_str):
    if not HAS_3D:
        raise ValueError("3D plotting not available (mpl_toolkits not installed).")
    parts  = [p.strip() for p in expr_str.split(';')]
    x_sym  = sp.Symbol('x')
    y_sym  = sp.Symbol('y')
    expr   = sp.sympify(preprocess(parts[0]), locals={**_SYMPY_NS, 'x': x_sym, 'y': y_sym})
    fn     = sp.lambdify((x_sym, y_sym), expr, modules=['numpy'])

    x_range = (-5, 5)
    y_range = (-5, 5)
    if len(parts) > 1:
        b = [float(v.strip()) for v in parts[1].split(',')]
        if len(b) >= 2: x_range = (b[0], b[1])
    if len(parts) > 2:
        b = [float(v.strip()) for v in parts[2].split(',')]
        if len(b) >= 2: y_range = (b[0], b[1])

    xv = np.linspace(x_range[0], x_range[1], 70)
    yv = np.linspace(y_range[0], y_range[1], 70)
    X, Y = np.meshgrid(xv, yv)
    Z = fn(X, Y)
    if np.isscalar(Z):
        Z = np.full_like(X, float(Z))
    Z = np.real(np.array(Z, dtype=complex))
    Z = np.where(np.abs(Z) > 1e6, np.nan, Z)

    fig = plt.figure(figsize=(9, 6))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, linewidth=0)
    ax.set_xlabel('x', color='#8b949e')
    ax.set_ylabel('y', color='#8b949e')
    ax.set_zlabel('z', color='#8b949e')
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.set_title(f'z = {parts[0]}', color='#c9d1d9', fontsize=11)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#30363d')
    ax.yaxis.pane.set_edgecolor('#30363d')
    ax.zaxis.pane.set_edgecolor('#30363d')
    fig.colorbar(surf, ax=ax, shrink=0.45, label='z').ax.yaxis.set_tick_params(color='#8b949e')
    plt.tight_layout()
    return _fig_to_b64(fig)


def make_stats_plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('#0d1117')
    ax1, ax2 = axes

    ax1.set_facecolor('#161b22')
    ax1.hist(data, bins='auto', color='#58a6ff', alpha=0.8, edgecolor='#0d1117', linewidth=0.5)
    ax1.axvline(_stats.mean(data), color='#f78166', lw=2, linestyle='--', label='mean')
    ax1.axvline(_stats.median(data), color='#3fb950', lw=2, linestyle='--', label='median')
    _style_ax(ax1, 'Distribution', 'value', 'frequency')
    ax1.legend(facecolor='#161b22', labelcolor='#c9d1d9', edgecolor='#30363d', fontsize=9)

    ax2.set_facecolor('#161b22')
    ax2.boxplot(data, patch_artist=True,
                boxprops={'facecolor': '#1f4068', 'color': '#58a6ff'},
                whiskerprops={'color': '#58a6ff'},
                capprops={'color': '#58a6ff'},
                medianprops={'color': '#f78166', 'linewidth': 2},
                flierprops={'markerfacecolor': '#d2a8ff', 'marker': 'o', 'markersize': 6})
    _style_ax(ax2, 'Box Plot', '', 'value')

    plt.tight_layout()
    return _fig_to_b64(fig)


def make_regression_plot(xs, ys, poly, degree):
    fig, ax = _dark_fig_ax()
    xf = np.linspace(min(xs) * 0.9, max(xs) * 1.1, 300)
    ax.scatter(xs, ys, color='#f78166', zorder=5, s=60, label='Data', edgecolors='#0d1117', linewidths=0.5)
    ax.plot(xf, poly(xf), color='#58a6ff', lw=2.2, label=f'Degree {degree} fit')
    _style_ax(ax, 'Regression Analysis')
    ax.legend(facecolor='#161b22', labelcolor='#c9d1d9', edgecolor='#30363d', fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Embedded HTML / CSS / JS  — v2.0 ULTIMATE EDITION
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>botX 3.0 — AI Math Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script>
MathJax = {
  tex: { inlineMath: [['\\(','\\)']], displayMath: [['\\[','\\]']] },
  svg: { fontCache: 'global' },
  startup: { typeset: false }
};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
/* ─── Theme Variables ────────────────────────────────── */
[data-theme="dark"] {
  --bg:#090c14;--surface:#0d1117;--surface2:#161b22;--surface3:#21262d;
  --border:#30363d;--text:#e6edf3;--muted:#8b949e;
  --blue:#58a6ff;--green:#3fb950;--red:#f78166;--purple:#d2a8ff;--orange:#ffa657;
  --shadow:rgba(0,0,0,.45);
}
[data-theme="light"] {
  --bg:#f6f8fa;--surface:#ffffff;--surface2:#f0f2f5;--surface3:#e8ecf0;
  --border:#d0d7de;--text:#1f2328;--muted:#656d76;
  --blue:#0969da;--green:#1a7f37;--red:#cf222e;--purple:#8250df;--orange:#bc4c00;
  --shadow:rgba(0,0,0,.12);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Sora',sans-serif;font-size:14px;overflow:hidden;transition:background .2s,color .2s}

/* ─── Layout ─────────────────────────────────────────── */
.app{display:grid;grid-template-columns:272px 1fr;grid-template-rows:56px 1fr 78px;height:100vh}

/* ─── Header ─────────────────────────────────────────── */
header{
  grid-column:1/-1;display:flex;align-items:center;gap:10px;
  padding:0 20px;background:var(--surface);border-bottom:1px solid var(--border);z-index:50;
}
.logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:17px;letter-spacing:-.3px;flex-shrink:0}
.logo-icon{
  width:34px;height:34px;background:linear-gradient(135deg,var(--blue),var(--purple));
  border-radius:9px;display:grid;place-items:center;font-size:17px;
  box-shadow:0 0 18px rgba(88,166,255,.28);
}
.badge{font-size:9px;background:var(--surface3);color:var(--blue);border:1px solid var(--border);
  padding:2px 6px;border-radius:4px;font-weight:700;letter-spacing:.8px;margin-left:2px}
.version-badge{font-size:9px;background:linear-gradient(90deg,var(--blue),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;margin-left:2px}
header .sp{flex:1}
.stat-pill{
  display:flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;
  background:var(--surface2);border:1px solid var(--border);font-size:11px;color:var(--muted);
}
.stat-pill .sval{color:var(--blue);font-weight:600;font-family:'JetBrains Mono',monospace}
.hbtn{
  padding:5px 11px;border-radius:7px;background:var(--surface2);border:1px solid var(--border);
  color:var(--muted);font-size:12px;font-weight:500;cursor:pointer;
  transition:all .15s;font-family:'Sora',sans-serif;display:flex;align-items:center;gap:5px;
}
.hbtn:hover{border-color:var(--blue);color:var(--blue)}
.hbtn.icon-btn{width:32px;height:32px;padding:0;justify-content:center;font-size:14px}
.cr{font-size:10px;color:var(--muted);font-style:italic}

/* ─── Sidebar ────────────────────────────────────────── */
aside{
  grid-row:2/3;background:var(--surface);border-right:1px solid var(--border);
  padding:12px 10px;display:flex;flex-direction:column;gap:2px;overflow-y:auto;
}
.sec{font-size:9px;letter-spacing:1.2px;text-transform:uppercase;color:var(--muted);
  padding:8px 10px 4px;font-weight:700;margin-top:4px}
.sec:first-child{margin-top:0}
.op{
  display:flex;align-items:center;gap:9px;padding:7px 10px;border-radius:8px;
  background:transparent;border:none;color:var(--muted);cursor:pointer;
  font-family:'Sora',sans-serif;font-size:12.5px;font-weight:500;text-align:left;
  width:100%;transition:all .12s;position:relative;
}
.op .ico{font-size:14px;width:20px;text-align:center;flex-shrink:0}
.op:hover{background:var(--surface2);color:var(--text)}
.op.on{background:rgba(88,166,255,.1);color:var(--blue)}
.op.on::before{content:'';position:absolute;left:0;top:18%;bottom:18%;
  width:3px;background:var(--blue);border-radius:0 2px 2px 0;}
.hint{font-size:9.5px;color:var(--muted);font-family:'JetBrains Mono',monospace;opacity:.65;margin-top:1px}
.div{height:1px;background:var(--border);margin:6px 0}
/* Settings in sidebar */
.sa{padding:8px 0}
.trow{display:flex;align-items:center;justify-content:space-between;padding:6px 10px}
.tlbl{font-size:12px;color:var(--muted);font-weight:500}
.tog{width:32px;height:18px;background:var(--surface3);border-radius:9px;cursor:pointer;
  position:relative;transition:background .2s;border:1px solid var(--border);flex-shrink:0}
.tog.on{background:var(--blue);border-color:var(--blue)}
.tog::after{content:'';position:absolute;top:2px;left:2px;width:12px;height:12px;
  border-radius:50%;background:white;transition:transform .2s}
.tog.on::after{transform:translateX(14px)}
.dsel,.rng{
  width:calc(100% - 20px);margin:0 10px 8px;padding:7px 10px;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-family:'Sora',sans-serif;font-size:12px;cursor:pointer;
}
.dsel{
  appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='%238b949e'%3E%3Cpath d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center;padding-right:28px;
}
.rng{-webkit-appearance:none;appearance:none;height:6px;padding:0;background:var(--surface3);border-radius:3px;cursor:pointer}
.rng::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:var(--blue);cursor:pointer}
.rng-row{display:flex;align-items:center;gap:8px;padding:4px 10px 8px;font-size:11px;color:var(--muted)}
.rng-row span{min-width:28px;text-align:right;font-family:'JetBrains Mono',monospace;color:var(--blue)}

/* ─── Chat Area ───────────────────────────────────────── */
main{grid-row:2/3;overflow-y:auto;padding:20px 28px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}
.msg{display:flex;gap:10px;animation:msgIn .2s ease}
@keyframes msgIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg.user{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:7px;display:grid;place-items:center;font-size:13px;flex-shrink:0;margin-top:2px}
.av.bot{background:linear-gradient(135deg,var(--blue),var(--purple))}
.av.usr{background:var(--surface3);border:1px solid var(--border)}
.bub{max-width:76%;padding:11px 14px;border-radius:12px;line-height:1.65;font-size:13.5px;position:relative}
.msg.bot .bub{background:var(--surface2);border:1px solid var(--border);border-top-left-radius:4px}
.msg.user .bub{background:rgba(88,166,255,.11);border:1px solid rgba(88,166,255,.25);border-top-right-radius:4px}
.ts{font-size:10px;color:var(--muted);margin-top:6px;display:flex;align-items:center;gap:8px}
.ts-time{opacity:.6}
.mblk{background:var(--surface3);border:1px solid var(--border);border-radius:8px;
  padding:12px 14px;margin-top:8px;overflow-x:auto;text-align:center}
.pimg{max-width:100%;border-radius:8px;border:1px solid var(--border);margin-top:8px;display:block;cursor:zoom-in}
.err{background:rgba(247,129,102,.08);border:1px solid rgba(247,129,102,.28);
  border-radius:8px;padding:9px 12px;color:var(--red);font-size:13px}
.qas{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}
.qa{padding:3px 9px;border-radius:6px;background:var(--surface3);border:1px solid var(--border);
  color:var(--blue);font-size:11px;cursor:pointer;font-family:'Sora',sans-serif;transition:all .12s}
.qa:hover{background:rgba(88,166,255,.12);border-color:var(--blue)}
/* Bubble action buttons */
.bub-actions{display:flex;gap:4px;margin-top:8px;flex-wrap:wrap}
.bact{padding:3px 8px;border-radius:5px;background:var(--surface3);border:1px solid var(--border);
  color:var(--muted);font-size:10px;cursor:pointer;font-family:'Sora',sans-serif;transition:all .12s;display:flex;align-items:center;gap:3px}
.bact:hover{color:var(--blue);border-color:var(--blue)}
/* Stats table */
.stbl{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}
.stbl td{padding:5px 10px;border-bottom:1px solid var(--border)}
.stbl td:first-child{color:var(--muted);font-weight:500;width:40%}
.stbl td:last-child{font-family:'JetBrains Mono',monospace;color:var(--text)}
.stbl tr:last-child td{border-bottom:none}
/* Regression info card */
.rcard{background:var(--surface3);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-top:8px;font-size:12px}
.rcard .req{font-family:'JetBrains Mono',monospace;color:var(--blue);font-size:13px;margin-bottom:6px}

/* ─── Welcome Screen ─────────────────────────────────── */
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;padding:30px;gap:12px;flex:1;min-height:55vh}
.wico{
  width:66px;height:66px;border-radius:18px;display:grid;place-items:center;font-size:28px;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  box-shadow:0 8px 32px rgba(88,166,255,.28);animation:pulse 3s ease-in-out infinite;
}
@keyframes pulse{0%,100%{box-shadow:0 8px 32px rgba(88,166,255,.28)}50%{box-shadow:0 12px 40px rgba(88,166,255,.5)}}
.welcome h2{font-size:22px;font-weight:700;letter-spacing:-.3px}
.welcome p{color:var(--muted);max-width:480px;line-height:1.7;font-size:13px}
.echips{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-top:8px;max-width:640px}
.ec-group{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.ec-label{font-size:10px;color:var(--muted);letter-spacing:.8px;text-transform:uppercase;
  font-weight:600;width:100%;margin-top:8px;margin-bottom:2px}
.ec{padding:5px 12px;border-radius:20px;background:var(--surface2);border:1px solid var(--border);
  color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:11px;cursor:pointer;transition:all .15s}
.ec:hover{border-color:var(--blue);color:var(--blue);background:rgba(88,166,255,.07)}
/* Thinking */
.thk{display:flex;align-items:center;gap:7px;padding:11px 14px;color:var(--muted);font-size:13px}
.dots span{display:inline-block;width:5px;height:5px;border-radius:50%;background:var(--blue);
  animation:dot 1.2s infinite;margin:0 1px}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes dot{0%,80%,100%{opacity:.2;transform:scale(1)}40%{opacity:1;transform:scale(1.4)}}

/* ─── Footer ─────────────────────────────────────────── */
footer{
  grid-column:1/-1;display:flex;align-items:center;gap:10px;
  padding:0 20px;background:var(--surface);border-top:1px solid var(--border);
}
.iw{flex:1;display:flex;align-items:center;background:var(--surface2);
  border:1px solid var(--border);border-radius:var(--r,12px);padding:0 14px;
  gap:10px;transition:border-color .2s;}
.iw:focus-within{border-color:var(--blue)}
.pfx{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--blue);font-weight:600;white-space:nowrap}
#inp{flex:1;background:transparent;border:none;outline:none;
  color:var(--text);font-family:'JetBrains Mono',monospace;font-size:14px;padding:20px 0}
#inp::placeholder{color:var(--muted)}
.sbtn{width:40px;height:40px;border-radius:10px;background:var(--blue);border:none;
  cursor:pointer;display:grid;place-items:center;transition:all .15s;flex-shrink:0}
.sbtn:hover{background:#79bcff;transform:scale(1.05)}
.sbtn:active{transform:scale(.97)}
.sbtn svg{width:15px;height:15px;fill:#090c14}
.kbd{display:inline-flex;align-items:center;gap:2px;padding:1px 5px;border-radius:4px;
  background:var(--surface3);border:1px solid var(--border);font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--muted)}
.shortcuts-hint{font-size:10px;color:var(--muted);display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-left:4px}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:9px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}
code{background:var(--surface3);padding:1px 5px;border-radius:4px;
  font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--orange)}

/* ─── Toast Notifications ────────────────────────────── */
#toasts{position:fixed;bottom:90px;right:20px;display:flex;flex-direction:column;gap:8px;z-index:999;pointer-events:none}
.toast{
  padding:9px 14px;border-radius:10px;font-size:13px;font-weight:500;
  box-shadow:0 4px 20px var(--shadow);border:1px solid var(--border);
  background:var(--surface);pointer-events:auto;
  animation:toastIn .2s ease;display:flex;align-items:center;gap:8px;min-width:200px;max-width:320px;
}
.toast.success{border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.08);color:var(--green)}
.toast.error{border-color:rgba(247,129,102,.4);background:rgba(247,129,102,.08);color:var(--red)}
.toast.info{border-color:rgba(88,166,255,.4);background:rgba(88,166,255,.08);color:var(--blue)}
@keyframes toastIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}
@keyframes toastOut{to{opacity:0;transform:translateX(20px)}}

/* ─── Modal System ───────────────────────────────────── */
.overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;
  display:flex;align-items:center;justify-content:center;
  animation:fadeIn .15s ease;backdrop-filter:blur(4px);
}
.overlay.hidden{display:none}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.modal{
  background:var(--surface);border:1px solid var(--border);border-radius:16px;
  box-shadow:0 24px 64px var(--shadow);padding:24px;width:90%;max-width:600px;
  max-height:80vh;overflow-y:auto;animation:slideUp .2s ease;
}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
.modal h3{font-size:17px;font-weight:700;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.modal-close{position:absolute;right:20px;top:20px;width:28px;height:28px;border-radius:6px;
  background:var(--surface3);border:1px solid var(--border);cursor:pointer;
  display:grid;place-items:center;font-size:14px;color:var(--muted);transition:all .15s}
.modal-close:hover{color:var(--text);border-color:var(--muted)}

/* ─── Command Palette ────────────────────────────────── */
#palette-overlay .modal{max-width:520px;padding:0;overflow:hidden}
.pal-search{width:100%;padding:16px 18px;background:transparent;border:none;
  border-bottom:1px solid var(--border);outline:none;color:var(--text);
  font-family:'JetBrains Mono',monospace;font-size:15px}
.pal-search::placeholder{color:var(--muted)}
.pal-list{max-height:360px;overflow-y:auto}
.pal-item{display:flex;align-items:center;gap:10px;padding:11px 18px;cursor:pointer;
  transition:background .1s;border-bottom:1px solid var(--border)}
.pal-item:last-child{border-bottom:none}
.pal-item:hover,.pal-item.pal-sel{background:rgba(88,166,255,.08);color:var(--blue)}
.pal-item .pal-ico{font-size:16px;width:24px;text-align:center}
.pal-item .pal-label{font-weight:500;font-size:13px}
.pal-item .pal-hint{font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace;margin-left:auto}
.pal-empty{padding:24px;text-align:center;color:var(--muted);font-size:13px}

/* ─── Settings Modal ─────────────────────────────────── */
.settings-grid{display:grid;gap:14px}
.setting-row{display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px;background:var(--surface2);border-radius:10px;border:1px solid var(--border)}
.setting-label{font-size:13px;font-weight:500}
.setting-desc{font-size:11px;color:var(--muted);margin-top:2px}
.setting-ctrl{display:flex;align-items:center;gap:8px}
.num-input{width:64px;padding:5px 8px;background:var(--surface3);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px;text-align:center}

/* ─── Shortcuts Modal ─────────────────────────────────── */
.shortcut-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.shortcut-row{display:flex;align-items:center;justify-content:space-between;
  padding:8px 12px;background:var(--surface2);border-radius:8px;border:1px solid var(--border)}
.shortcut-desc{font-size:12px;color:var(--muted)}
.shortcut-keys{display:flex;gap:3px}

/* ─── Formula Cheatsheet ─────────────────────────────── */
.formula-tabs{display:flex;gap:4px;margin-bottom:14px;flex-wrap:wrap}
.ftab{padding:4px 12px;border-radius:20px;background:var(--surface3);border:1px solid var(--border);
  font-size:11px;cursor:pointer;color:var(--muted);transition:all .12s;font-family:'Sora',sans-serif}
.ftab.active,.ftab:hover{background:rgba(88,166,255,.12);border-color:var(--blue);color:var(--blue)}
.fpanel{display:none}
.fpanel.active{display:block}
.fformula{padding:8px 12px;background:var(--surface2);border-radius:8px;border:1px solid var(--border);
  margin-bottom:6px;cursor:pointer;transition:all .12s;display:flex;align-items:center;justify-content:space-between}
.fformula:hover{border-color:var(--blue);background:rgba(88,166,255,.05)}
.fformula code{font-size:12px;color:var(--blue);font-family:'JetBrains Mono',monospace}
.fformula .fcopy{font-size:10px;color:var(--muted)}
.fformula:hover .fcopy{color:var(--blue)}

/* ─── Fullscreen Plot ─────────────────────────────────── */
#fullscreen-overlay .modal{max-width:95vw;width:95vw;max-height:95vh;padding:16px;
  display:flex;flex-direction:column;align-items:center}
#fullscreen-img{max-width:100%;max-height:80vh;border-radius:10px;border:1px solid var(--border)}

/* ─── Mobile ─────────────────────────────────────────── */
@media(max-width:768px){
  .app{grid-template-columns:1fr;grid-template-rows:56px 1fr 78px}
  aside{display:none;position:fixed;inset:56px 0 78px;z-index:80;width:280px;
    border-right:1px solid var(--border);box-shadow:4px 0 20px var(--shadow)}
  aside.open{display:flex}
  .mobile-menu{display:flex}
  .cr,.stat-pill,.shortcuts-hint{display:none}
  main{padding:14px 16px}
}
@media(min-width:769px){
  .mobile-menu{display:none}
}
</style>
</head>
<body>
<div class="app">

<!-- ═══════════════════════ HEADER ═══════════════════════ -->
<header>
  <button class="hbtn icon-btn mobile-menu" onclick="toggleSidebar()" title="Menu">☰</button>
  <div class="logo">
    <div class="logo-icon">∂</div>
    botX
    <span class="badge">MATH AI</span>
    <span class="version-badge">v2.0</span>
  </div>
  <div class="sp"></div>
  <div class="stat-pill">⚡ <span class="sval" id="statCount">0</span> computed</div>
  <div class="stat-pill">⏱ <span class="sval" id="statTime">—</span></div>
  <button class="hbtn icon-btn" onclick="openFormulas()" title="Formula Cheatsheet (Ctrl+F)">📐</button>
  <button class="hbtn icon-btn" onclick="openShortcuts()" title="Keyboard Shortcuts (Ctrl+/)">⌨️</button>
  <button class="hbtn icon-btn" id="themeBtn" onclick="toggleTheme()" title="Toggle Theme">🌙</button>
  <button class="hbtn icon-btn" onclick="openPalette()" title="Command Palette (Ctrl+K)">⌘</button>
  <button class="hbtn" onclick="clearChat()">🗑 Clear</button>
  <button class="hbtn" onclick="exportHistory()">↓ Export</button>
  <a href="/graph" target="_blank" class="hbtn" style="text-decoration:none;color:inherit">🧮 Graph</a>
  <span id="ai-status-badge" style="font-size:10px;color:var(--muted)" title="Checking AI status…">⚪ AI…</span>
  <span class="cr">© 2026 Dhanwanth V</span>
</header>

<!-- ═══════════════════════ SIDEBAR ══════════════════════ -->
<aside id="sidebar">
  <div class="sec">Calculus</div>
  <button class="op on" data-op="solve" onclick="setOp(this,'solve')">
    <span class="ico">⚡</span>
    <div><div>Solve Equation</div><div class="hint">2x² + 3x = 5</div></div>
  </button>
  <button class="op" data-op="diff" onclick="setOp(this,'diff')">
    <span class="ico">∂</span>
    <div><div>Differentiate</div><div class="hint">sin(x)/x</div></div>
  </button>
  <button class="op" data-op="partial" onclick="setOp(this,'partial')">
    <span class="ico">∂²</span>
    <div><div>Partial Diff</div><div class="hint">x²y³, x, y</div></div>
  </button>
  <button class="op" data-op="integrate" onclick="setOp(this,'integrate')">
    <span class="ico">∫</span>
    <div><div>Integrate</div><div class="hint">sin(x), 0, pi</div></div>
  </button>
  <button class="op" data-op="limit" onclick="setOp(this,'limit')">
    <span class="ico">→</span>
    <div><div>Limit</div><div class="hint">sin(x)/x, 0</div></div>
  </button>
  <button class="op" data-op="taylor" onclick="setOp(this,'taylor')">
    <span class="ico">∑</span>
    <div><div>Taylor Series</div><div class="hint">sin(x), 6, 0</div></div>
  </button>
  <button class="op" data-op="ode2" onclick="setOp(this,'ode2')">
    <span class="ico">D</span>
    <div><div>ODE (2nd Order)</div><div class="hint">y'' + 3y' + 2y = 0</div></div>
  </button>

  <div class="sec">Algebra</div>
  <button class="op" data-op="simplify" onclick="setOp(this,'simplify')">
    <span class="ico">✦</span>
    <div><div>Simplify / Factor</div><div class="hint">x²-1, factor</div></div>
  </button>
  <button class="op" data-op="system" onclick="setOp(this,'system')">
    <span class="ico">≡</span>
    <div><div>System of Eqs</div><div class="hint">x+y=3; x-y=1</div></div>
  </button>
  <button class="op" data-op="inequality" onclick="setOp(this,'inequality')">
    <span class="ico">⊂</span>
    <div><div>Inequalities</div><div class="hint">x² - 4 &lt; 0</div></div>
  </button>
  <button class="op" data-op="sequence" onclick="setOp(this,'sequence')">
    <span class="ico">∑∞</span>
    <div><div>Sequences / Series</div><div class="hint">1/n², 1, oo</div></div>
  </button>

  <div class="sec">Transforms</div>
  <button class="op" data-op="laplace" onclick="setOp(this,'laplace')">
    <span class="ico">ℒ</span>
    <div><div>Laplace Transform</div><div class="hint">t*exp(-t)</div></div>
  </button>
  <button class="op" data-op="fourier" onclick="setOp(this,'fourier')">
    <span class="ico">𝔉</span>
    <div><div>Fourier Transform</div><div class="hint">exp(-x²)</div></div>
  </button>

  <div class="sec">Plotting</div>
  <button class="op" data-op="plot" onclick="setOp(this,'plot')">
    <span class="ico">📈</span>
    <div><div>Plot 2D</div><div class="hint">sin(x); cos(x)</div></div>
  </button>
  <button class="op" data-op="polar" onclick="setOp(this,'polar')">
    <span class="ico">🌀</span>
    <div><div>Polar Plot</div><div class="hint">sin(3*theta)</div></div>
  </button>
  <button class="op" data-op="parametric" onclick="setOp(this,'parametric')">
    <span class="ico">〰</span>
    <div><div>Parametric</div><div class="hint">cos(t); sin(t)</div></div>
  </button>
  <button class="op" data-op="plot3d" onclick="setOp(this,'plot3d')">
    <span class="ico">🗻</span>
    <div><div>3D Surface</div><div class="hint">sin(x)*cos(y)</div></div>
  </button>

  <div class="sec">Linear Algebra</div>
  <button class="op" data-op="matrix" onclick="setOp(this,'matrix')">
    <span class="ico">⊞</span>
    <div><div>Matrix</div><div class="hint">[[1,2],[3,4]], det</div></div>
  </button>
  <button class="op" data-op="vector" onclick="setOp(this,'vector')">
    <span class="ico">⟶</span>
    <div><div>Vector Calculus</div><div class="hint">gradient, x²+y²</div></div>
  </button>

  <div class="sec">Data Analysis</div>
  <button class="op" data-op="stats" onclick="setOp(this,'stats')">
    <span class="ico">📊</span>
    <div><div>Statistics</div><div class="hint">1, 2, 3, 4, 5</div></div>
  </button>
  <button class="op" data-op="regression" onclick="setOp(this,'regression')">
    <span class="ico">📉</span>
    <div><div>Regression</div><div class="hint">0,1; 1,3; 2,5</div></div>
  </button>

  <div class="sec">Number Theory</div>
  <button class="op" data-op="numtheory" onclick="setOp(this,'numtheory')">
    <span class="ico">ℕ</span>
    <div><div>Number Theory</div><div class="hint">factorize, 360</div></div>
  </button>

  <div class="sec">AI Assistant</div>
  <button class="op" data-op="nlp" onclick="setOp(this,'nlp')">
    <span class="ico">🤖</span>
    <div><div>Ask in English</div><div class="hint">differentiate sin(x)</div></div>
  </button>

  <div class="div"></div>
  <div class="sa">
    <div class="sec">Settings</div>
    <div class="trow">
      <span class="tlbl">Numerical Mode</span>
      <div class="tog" id="numTog" onclick="toggleNum()"></div>
    </div>
    <div style="padding:0 10px 6px">
      <div class="tlbl" style="margin-bottom:5px">Domain</div>
      <select class="dsel" id="domSel">
        <option value="real">ℝ Real</option>
        <option value="complex">ℂ Complex</option>
      </select>
    </div>
    <div style="padding:0 10px 6px">
      <div class="tlbl" style="margin-bottom:4px">Plot x-range: <span id="rangeLabel" style="color:var(--blue);font-family:'JetBrains Mono',monospace">±10</span></div>
      <input class="rng" type="range" id="xRangeSlider" min="2" max="50" value="10" oninput="updateRange(this.value)">
    </div>
    <div class="trow">
      <span class="tlbl">Step Display</span>
      <div class="tog" id="stepTog" onclick="toggleStep()"></div>
    </div>
    <div class="trow">
      <span class="tlbl">Auto Suggest</span>
      <div class="tog on" id="suggestTog" onclick="toggleSuggest()"></div>
    </div>
  </div>
</aside>

<!-- ═══════════════════════ CHAT MAIN ════════════════════ -->
<main id="chat">
  <div class="welcome" id="welcome">
    <div class="wico">∂</div>
    <h2>botX 3.0 — Ultimate Math AI</h2>
    <p>Natural Language · 20+ operations · Interactive Graph Calculator · Symbolic &amp; Numerical · LaTeX · 3D plots</p>
    <div class="echips">
      <div class="ec-label">Calculus</div>
      <div class="ec" onclick="qi('solve','2x^2 - 8 = 0')">2x² − 8 = 0</div>
      <div class="ec" onclick="qi('diff','sin(x)/x')">diff sin(x)/x</div>
      <div class="ec" onclick="qi('integrate','sin(x), 0, pi')">∫ sin(x) dx</div>
      <div class="ec" onclick="qi('limit','sin(x)/x, 0')">lim sin(x)/x→0</div>
      <div class="ec" onclick="qi('taylor','exp(x), 8, 0')">Taylor e^x</div>
      <div class="ec" onclick="qi('partial','x^2*y^3, x, y')">∂²(x²y³)/∂x∂y</div>
      <div class="ec-label">Algebra & Transforms</div>
      <div class="ec" onclick="qi('simplify','x^2-1, factor')">Factor x²−1</div>
      <div class="ec" onclick="qi('system','x+y=5; x-y=1')">Solve system</div>
      <div class="ec" onclick="qi('inequality','x^2 - 4 < 0')">x² − 4 &lt; 0</div>
      <div class="ec" onclick="qi('sequence','1/n^2, 1, oo')">∑ 1/n²</div>
      <div class="ec" onclick="qi('laplace','t*exp(-t)')">Laplace t·e⁻ᵗ</div>
      <div class="ec" onclick="qi('fourier','exp(-x^2)')">Fourier e^−x²</div>
      <div class="ec-label">Plots</div>
      <div class="ec" onclick="qi('plot','sin(x); cos(x); x^2/10')">Plot 3 functions</div>
      <div class="ec" onclick="qi('polar','sin(3*theta)')">Polar rose</div>
      <div class="ec" onclick="qi('parametric','cos(t); sin(t)')">Circle parametric</div>
      <div class="ec" onclick="qi('plot3d','sin(x)*cos(y)')">3D surface</div>
      <div class="ec-label">Data & Number Theory</div>
      <div class="ec" onclick="qi('stats','2, 4, 4, 4, 5, 5, 7, 9')">Statistics</div>
      <div class="ec" onclick="qi('regression','0,1; 1,2.9; 2,5.1; 3,6.9; 4,9')">Linear regression</div>
      <div class="ec" onclick="qi('numtheory','factorize, 360')">Factorize 360</div>
      <div class="ec" onclick="qi('numtheory','gcd, 48, 18, 12')">GCD(48,18,12)</div>
      <div class="ec" onclick="qi('ode2',\"y'' + y = sin(x)\")">ODE y''+y=sin(x)</div>
      <div class="ec" onclick="qi('matrix','[[2,1],[5,3]], invert')">Matrix inverse</div>
      <div class="ec" onclick="qi('vector','gradient, x^2+y^2')">∇(x²+y²)</div>
      <div class="ec-label">🤖 Ask in Natural Language</div>
      <div class="ec" onclick="qiNlp('differentiate sin(x)cos(x)')">differentiate sin(x)cos(x)</div>
      <div class="ec" onclick="qiNlp('integrate x^2 from 0 to 3')">integrate x² from 0 to 3</div>
      <div class="ec" onclick="qiNlp('what is the limit of sin(x)/x as x approaches 0')">limit sin(x)/x → 0</div>
      <div class="ec" onclick="qiNlp('factorize 360')">factorize 360</div>
      <div class="ec" onclick="qiNlp('is 97 prime?')">is 97 prime?</div>
      <div class="ec" onclick="qiNlp('convert 100 km to miles')">100 km to miles</div>
      <div class="ec" onclick="qiNlp('what is 15% of 240')">15% of 240</div>
      <div class="ec" onclick="qiNlp('tell me about pi')">tell me about π</div>
      <div class="ec" onclick="qiNlp('tell me a math joke')">tell me a joke</div>
      <div class="ec" onclick="qiNlp('who are you?')">who are you?</div>
      <div class="ec" onclick="qiNlp('what can you do?')">what can you do?</div>
      <div class="ec" onclick="qiNlp('who created you?')">who created you?</div>
    </div>
    <p style="margin-top:10px;font-size:11px;color:var(--muted)">
      Press <span class="kbd">Ctrl</span><span class="kbd">K</span> for command palette &nbsp;·&nbsp;
      <span class="kbd">↑↓</span> for input history &nbsp;·&nbsp;
      <span class="kbd">Ctrl</span><span class="kbd">/</span> for shortcuts
    </p>
  </div>
</main>

<!-- ═══════════════════════ FOOTER ═══════════════════════ -->
<footer>
  <div class="iw">
    <span class="pfx" id="pfx">solve›</span>
    <input id="inp" type="text" autocomplete="off" spellcheck="false"
           placeholder="e.g. 2x² + 3x = 5"
           onkeydown="handleKey(event)">
  </div>
  <button class="sbtn" onclick="send()" title="Compute (Enter)">
    <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
  </button>
  <div class="shortcuts-hint" id="kbdHint">
    <span class="kbd">↑↓</span> history
    <span class="kbd">Ctrl+K</span> palette
  </div>
</footer>

</div><!-- .app -->

<!-- ═══════════════════════ MODALS ════════════════════════ -->

<!-- Command Palette -->
<div class="overlay hidden" id="palette-overlay" onclick="closePalette(event)">
  <div class="modal" style="position:relative">
    <input class="pal-search" id="pal-input" placeholder="⌘ Search operations or commands…" oninput="filterPalette()" onkeydown="palKey(event)">
    <div class="pal-list" id="pal-list"></div>
  </div>
</div>

<!-- Keyboard Shortcuts -->
<div class="overlay hidden" id="shortcuts-overlay" onclick="closeOverlay('shortcuts-overlay')">
  <div class="modal" style="position:relative;max-width:560px">
    <button class="modal-close" onclick="closeOverlay('shortcuts-overlay')">✕</button>
    <h3>⌨️ Keyboard Shortcuts</h3>
    <div class="shortcut-grid">
      <div class="shortcut-row"><span class="shortcut-desc">Command Palette</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">K</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Toggle Theme</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">T</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Clear Chat</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">L</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Settings</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">,</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Shortcuts</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">/</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Formulas</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">F</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Export History</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">E</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Close modal / Esc</span><div class="shortcut-keys"><span class="kbd">Esc</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Previous input</span><div class="shortcut-keys"><span class="kbd">↑</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Next input</span><div class="shortcut-keys"><span class="kbd">↓</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Submit</span><div class="shortcut-keys"><span class="kbd">Enter</span></div></div>
      <div class="shortcut-row"><span class="shortcut-desc">Focus Input</span><div class="shortcut-keys"><span class="kbd">Ctrl</span><span class="kbd">I</span></div></div>
    </div>
  </div>
</div>

<!-- Settings -->
<div class="overlay hidden" id="settings-overlay" onclick="closeOverlay('settings-overlay')">
  <div class="modal" style="position:relative" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeOverlay('settings-overlay')">✕</button>
    <h3>⚙️ Settings</h3>
    <div class="settings-grid">
      <div class="setting-row">
        <div><div class="setting-label">Decimal Precision</div><div class="setting-desc">Digits shown in numerical results</div></div>
        <div class="setting-ctrl"><input class="num-input" type="number" id="precisionInput" min="1" max="15" value="6" onchange="saveSetting('precision',this.value)"></div>
      </div>
      <div class="setting-row">
        <div><div class="setting-label">Plot x-range</div><div class="setting-desc">Symmetric range for 2D plots</div></div>
        <div class="setting-ctrl"><input class="num-input" type="number" id="xRangeInput" min="1" max="100" value="10" onchange="saveSetting('xRange',this.value);document.getElementById('xRangeSlider').value=this.value;updateRange(this.value)"></div>
      </div>
      <div class="setting-row">
        <div><div class="setting-label">Show computation time</div><div class="setting-desc">Display elapsed ms in messages</div></div>
        <div class="setting-ctrl"><div class="tog on" id="timeTogS" onclick="toggleSettingTog('showTime','timeTogS')"></div></div>
      </div>
      <div class="setting-row">
        <div><div class="setting-label">Quick Suggestions</div><div class="setting-desc">Show follow-up action buttons</div></div>
        <div class="setting-ctrl"><div class="tog on" id="suggestTogS" onclick="toggleSettingTog('suggestions','suggestTogS')"></div></div>
      </div>
      <div class="setting-row" style="margin-top:8px;border-top:1px solid var(--border);padding-top:12px">
        <div><div class="setting-label" style="color:var(--blue)">🤖 AI (Ollama)</div><div class="setting-desc">Enable offline AI responses</div></div>
        <div class="setting-ctrl"><div class="tog on" id="aiTogS" onclick="toggleAI()"></div></div>
      </div>
      <div class="setting-row" id="aiSpeedRow">
        <div>
          <div class="setting-label">AI Speed</div>
          <div class="setting-desc" id="aiSpeedDesc">Balanced — ~150 tokens</div>
        </div>
        <div class="setting-ctrl">
          <select class="dsel" id="aiSpeedSel" onchange="setAISpeed(this.value)" style="width:96px;margin:0">
            <option value="fast">⚡ Fast</option>
            <option value="balanced" selected>⚖ Balanced</option>
            <option value="full">💬 Full</option>
          </select>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Formula Cheatsheet -->
<div class="overlay hidden" id="formulas-overlay" onclick="closeOverlay('formulas-overlay')">
  <div class="modal" style="position:relative;max-width:680px" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeOverlay('formulas-overlay')">✕</button>
    <h3>📐 Formula Cheatsheet <span style="font-size:12px;color:var(--muted);font-weight:400">(click to insert)</span></h3>
    <div class="formula-tabs">
      <button class="ftab active" onclick="switchFTab(this,'ftab-calc')">Calculus</button>
      <button class="ftab" onclick="switchFTab(this,'ftab-trig')">Trig</button>
      <button class="ftab" onclick="switchFTab(this,'ftab-transform')">Transforms</button>
      <button class="ftab" onclick="switchFTab(this,'ftab-stats')">Stats</button>
      <button class="ftab" onclick="switchFTab(this,'ftab-matrix')">Matrix</button>
      <button class="ftab" onclick="switchFTab(this,'ftab-numth')">Number Theory</button>
    </div>
    <div class="fpanel active" id="ftab-calc">
      <div class="fformula" onclick="insertFormula('diff','x^n, x')"><code>d/dx(xⁿ) = nxⁿ⁻¹</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('diff','sin(x)')"><code>d/dx(sin x) = cos x</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('diff','exp(x)')"><code>d/dx(eˣ) = eˣ</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('integrate','x^n')"><code>∫xⁿ dx = xⁿ⁺¹/(n+1)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('integrate','1/x')"><code>∫(1/x) dx = ln|x|</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('integrate','exp(x)')"><code>∫eˣ dx = eˣ</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('limit','sin(x)/x, 0')"><code>lim sin(x)/x = 1 (x→0)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('taylor','sin(x), 8, 0')"><code>Taylor: sin(x), 8 terms</code><span class="fcopy">insert ↗</span></div>
    </div>
    <div class="fpanel" id="ftab-trig">
      <div class="fformula" onclick="insertFormula('simplify','sin(x)^2 + cos(x)^2, trigsimp')"><code>sin²x + cos²x = 1</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('simplify','tan(x), trigsimp')"><code>tan x = sin x / cos x</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('diff','asin(x)')"><code>d/dx(arcsin x)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('diff','atan(x)')"><code>d/dx(arctan x)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('integrate','sin(x)^2')"><code>∫sin²x dx</code><span class="fcopy">insert ↗</span></div>
    </div>
    <div class="fpanel" id="ftab-transform">
      <div class="fformula" onclick="insertFormula('laplace','exp(-a*t)')"><code>L{e^(-at)} = 1/(s+a)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('laplace','t')"><code>L{t} = 1/s²</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('laplace','sin(t)')"><code>L{sin(t)}</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('fourier','exp(-x^2)')"><code>F{e^(-x²)} = Gaussian</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('laplace','t^2, inverse')"><code>L⁻¹{1/s³} → t²/2</code><span class="fcopy">insert ↗</span></div>
    </div>
    <div class="fpanel" id="ftab-stats">
      <div class="fformula" onclick="insertFormula('stats','1, 2, 3, 4, 5, 6, 7, 8, 9, 10')"><code>Full stats: 1..10</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('regression','0,0; 1,1; 2,4; 3,9; 4,16; degree=2')"><code>Quadratic regression</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('stats','2, 4, 4, 4, 5, 5, 7, 9, mean')"><code>Mean of dataset</code><span class="fcopy">insert ↗</span></div>
    </div>
    <div class="fpanel" id="ftab-matrix">
      <div class="fformula" onclick="insertFormula('matrix','[[2,1],[5,3]], det')"><code>Determinant 2×2</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('matrix','[[2,1],[5,3]], invert')"><code>Inverse 2×2</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('matrix','[[1,2,3],[4,5,6],[7,8,10]], eigenvals')"><code>Eigenvalues 3×3</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('matrix','[[3,1],[1,3]], rref')"><code>Row Reduce</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('vector','gradient, x^2 + y^2 + z^2')"><code>∇(x²+y²+z²)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('vector','curl, y, -x, 0')"><code>curl(y, -x, 0)</code><span class="fcopy">insert ↗</span></div>
    </div>
    <div class="fpanel" id="ftab-numth">
      <div class="fformula" onclick="insertFormula('numtheory','factorize, 360')"><code>Factorize 360</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('numtheory','gcd, 48, 18, 12')"><code>GCD(48, 18, 12)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('numtheory','isprime, 97')"><code>Is 97 prime?</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('numtheory','totient, 36')"><code>φ(36)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('numtheory','fibonacci, 20')"><code>F(20)</code><span class="fcopy">insert ↗</span></div>
      <div class="fformula" onclick="insertFormula('numtheory','primes, 50')"><code>Primes up to 50</code><span class="fcopy">insert ↗</span></div>
    </div>
  </div>
</div>

<!-- Fullscreen Plot -->
<div class="overlay hidden" id="fullscreen-overlay" onclick="closeOverlay('fullscreen-overlay')">
  <div class="modal" onclick="event.stopPropagation()" style="position:relative">
    <button class="modal-close" onclick="closeOverlay('fullscreen-overlay')">✕</button>
    <img id="fullscreen-img" src="" alt="Full Plot">
    <div style="text-align:center;margin-top:12px">
      <button class="hbtn" onclick="downloadCurrentPlot()">⬇ Download PNG</button>
    </div>
  </div>
</div>

<!-- Toast Container -->
<div id="toasts"></div>

<!-- ═══════════════════════ JAVASCRIPT ════════════════════ -->
<script>
// ── State ────────────────────────────────────────────────
var op = 'solve', numerical = false, showSteps = false, showSuggests = true;
var sessionHistory = [], inputHistory = [], historyIdx = -1;
var currentPlotB64 = '';
var currentInput   = '';   // tracks last submitted expression for AI explain
var resultText     = '';   // tracks last result for AI explain
var xPlotRange = 10;
var settings = JSON.parse(localStorage.getItem('botx-settings') || '{}');
var sessionStats = { total: 0, time: 0, ops: {} };

// ── AI settings (persisted in localStorage) ───────────────────────────────────
var aiEnabled  = localStorage.getItem('botx-ai-enabled')  !== 'false';
var aiSpeed    = localStorage.getItem('botx-ai-speed')    || 'balanced';
var AI_TOKENS  = { fast: 80, balanced: 150, full: 350 };

function _applyAISettings() {
  document.getElementById('aiTogS').classList.toggle('on', aiEnabled);
  var row = document.getElementById('aiSpeedRow');
  if (row) row.style.opacity = aiEnabled ? '1' : '0.4';
  var sel = document.getElementById('aiSpeedSel');
  if (sel) sel.value = aiSpeed;
  var desc = document.getElementById('aiSpeedDesc');
  var descs = { fast: 'Fast — ~80 tokens (best for Pentium)', balanced: 'Balanced — ~150 tokens', full: 'Full — ~350 tokens' };
  if (desc) desc.textContent = descs[aiSpeed] || '';
}

function toggleAI() {
  aiEnabled = !aiEnabled;
  localStorage.setItem('botx-ai-enabled', aiEnabled);
  _applyAISettings();
  toast('AI ' + (aiEnabled ? 'enabled ✓' : 'disabled — using fast fallbacks'), aiEnabled ? 'success' : 'info');
}

function setAISpeed(val) {
  aiSpeed = val;
  localStorage.setItem('botx-ai-speed', val);
  _applyAISettings();
  var labels = { fast: '⚡ Fast mode — snappier responses', balanced: '⚖ Balanced mode', full: '💬 Full mode — richest responses' };
  toast(labels[val] || val, 'info');
}

// Apply on page load
document.addEventListener('DOMContentLoaded', _applyAISettings);

// ── AI status polling ─────────────────────────────────────────────────────────
(function pollAIStatus(){
  fetch('/ai_status').then(function(r){ return r.json(); }).then(function(d){
    var el = document.getElementById('ai-status-badge');
    if (el) el.innerHTML = d.html || '';
  }).catch(function(){});
  setTimeout(pollAIStatus, 30000);  // re-poll every 30s
})();

// ── AI Explain ────────────────────────────────────────────────────────────────
function aiExplain(btn, dataJson) {
  var data;
  try { data = JSON.parse(dataJson); } catch(e) { return; }
  btn.disabled = true;
  btn.textContent = '⏳ Thinking…';
  fetch('/ai_explain', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(Object.assign({}, data, {ai_tokens: AI_TOKENS[aiSpeed] || 150}))
  }).then(function(r){ return r.json(); }).then(function(d){
    if (d.html) {
      var bub = btn.closest('.bub');
      var exDiv = document.createElement('div');
      exDiv.style.cssText = 'margin-top:10px;padding:10px 12px;background:rgba(88,166,255,.06);'
        + 'border-left:3px solid var(--blue);border-radius:0 8px 8px 0;font-size:13px;line-height:1.65';
      exDiv.innerHTML = '<span style="font-size:10px;color:var(--blue);font-weight:600">🤖 AI EXPLANATION</span><br>' + d.html;
      btn.parentNode.insertBefore(exDiv, btn.nextSibling);
      btn.remove();
      if (window.MathJax) MathJax.typesetPromise([exDiv]).catch(function(){});
    } else {
      btn.textContent = '🤖 Explain';
      btn.disabled = false;
      toast('AI is offline — install Ollama for explanations', 'warn');
    }
  }).catch(function(){
    btn.textContent = '🤖 Explain';
    btn.disabled = false;
  });
}

// Restore settings
if (settings.precision) document.getElementById('precisionInput').value = settings.precision;
if (settings.xRange) {
  xPlotRange = parseInt(settings.xRange);
  document.getElementById('xRangeSlider').value = xPlotRange;
  document.getElementById('xRangeInput').value = xPlotRange;
  document.getElementById('rangeLabel').textContent = '±' + xPlotRange;
}
if (settings.theme) document.documentElement.setAttribute('data-theme', settings.theme);
updateThemeBtn();

// Restore input history
try { inputHistory = JSON.parse(localStorage.getItem('botx-input-history') || '[]'); } catch(e){}

// ── Op config ────────────────────────────────────────────
var hints = {
  solve:     '2x^2 + 3x = 5',
  diff:      'sin(x)/x',
  partial:   'x^2*y^3, x, y',
  integrate: 'sin(x), 0, pi',
  limit:     'sin(x)/x, 0',
  taylor:    'sin(x), 6, 0',
  ode2:      "y'' + 3y' + 2y = 0",
  simplify:  'x^2 - 1, factor',
  system:    'x + y = 5; x - y = 1',
  inequality:'x^2 - 4 < 0',
  sequence:  '1/n^2, 1, oo',
  laplace:   't*exp(-t)',
  fourier:   'exp(-x^2)',
  plot:      'sin(x); cos(x)',
  polar:     'sin(3*theta)',
  parametric:'cos(t); sin(t)',
  plot3d:    'sin(x)*cos(y)',
  matrix:    '[[1,2],[3,4]], det',
  vector:    'gradient, x^2+y^2',
  stats:     '2, 4, 4, 4, 5, 5, 7, 9',
  regression:'0,0; 1,1; 2,4; 3,9; degree=2',
  numtheory: 'factorize, 360',
  nlp: 'differentiate sin(x) with respect to x',
};

// Command palette items
var paletteItems = [
  {ico:'⚡',label:'Solve Equation',op:'solve',hint:'2x²+3x=5'},
  {ico:'∂',label:'Differentiate',op:'diff',hint:'sin(x)/x'},
  {ico:'∂²',label:'Partial Derivative',op:'partial',hint:'x²y³, x, y'},
  {ico:'∫',label:'Integrate',op:'integrate',hint:'sin(x), 0, pi'},
  {ico:'→',label:'Limit',op:'limit',hint:'sin(x)/x, 0'},
  {ico:'∑',label:'Taylor Series',op:'taylor',hint:'sin(x), 6, 0'},
  {ico:'D',label:'ODE 2nd Order',op:'ode2',hint:"y''+y=0"},
  {ico:'✦',label:'Simplify / Factor / Expand',op:'simplify',hint:'x²-1, factor'},
  {ico:'≡',label:'System of Equations',op:'system',hint:'x+y=5; x-y=1'},
  {ico:'⊂',label:'Solve Inequality',op:'inequality',hint:'x²-4<0'},
  {ico:'∑∞',label:'Sequence / Summation',op:'sequence',hint:'1/n², 1, oo'},
  {ico:'ℒ',label:'Laplace Transform',op:'laplace',hint:'t·e⁻ᵗ'},
  {ico:'𝔉',label:'Fourier Transform',op:'fourier',hint:'e⁻ˣ²'},
  {ico:'📈',label:'Plot 2D',op:'plot',hint:'sin(x); cos(x)'},
  {ico:'🌀',label:'Polar Plot',op:'polar',hint:'sin(3θ)'},
  {ico:'〰',label:'Parametric Plot',op:'parametric',hint:'cos(t); sin(t)'},
  {ico:'🗻',label:'3D Surface Plot',op:'plot3d',hint:'sin(x)cos(y)'},
  {ico:'⊞',label:'Matrix Operations',op:'matrix',hint:'[[1,2],[3,4]], det'},
  {ico:'⟶',label:'Vector Calculus',op:'vector',hint:'gradient, x²+y²'},
  {ico:'📊',label:'Statistics',op:'stats',hint:'1,2,3,4,5'},
  {ico:'📉',label:'Regression',op:'regression',hint:'0,0; 1,1; 2,4'},
  {ico:'ℕ',label:'Number Theory',op:'numtheory',hint:'factorize, 360'},
  {ico:'🗑',label:'Clear Chat',action:clearChat},
  {ico:'↓',label:'Export History JSON',action:exportHistory},
  {ico:'🌙',label:'Toggle Theme',action:toggleTheme},
  {ico:'📐',label:'Formula Cheatsheet',action:openFormulas},
  {ico:'⌨️',label:'Keyboard Shortcuts',action:openShortcuts},
  {ico:'⚙️',label:'Settings',action:()=>openOverlay('settings-overlay')},
];

// ── Theme ─────────────────────────────────────────────────
function toggleTheme() {
  var current = document.documentElement.getAttribute('data-theme');
  var next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  saveSetting('theme', next);
  updateThemeBtn();
  toast(next === 'light' ? '☀️ Light mode' : '🌙 Dark mode', 'info');
}
function updateThemeBtn() {
  var th = document.documentElement.getAttribute('data-theme');
  var btn = document.getElementById('themeBtn');
  if (btn) btn.textContent = th === 'dark' ? '☀️' : '🌙';
}

// ── Settings ──────────────────────────────────────────────
function saveSetting(key, val) {
  settings[key] = val;
  localStorage.setItem('botx-settings', JSON.stringify(settings));
}
function toggleSettingTog(key, id) {
  var el = document.getElementById(id);
  var on = el.classList.toggle('on');
  saveSetting(key, on);
  if (key === 'suggestions') showSuggests = on;
}
function updateRange(val) {
  xPlotRange = parseInt(val);
  document.getElementById('rangeLabel').textContent = '±' + val;
  document.getElementById('xRangeInput').value = val;
  saveSetting('xRange', val);
}

// ── Toast system ──────────────────────────────────────────
function toast(msg, type) {
  type = type || 'info';
  var t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  document.getElementById('toasts').appendChild(t);
  setTimeout(function() {
    t.style.animation = 'toastOut .25s ease forwards';
    setTimeout(function() { t.remove(); }, 250);
  }, 2400);
}

// ── Overlays ──────────────────────────────────────────────
function openOverlay(id) {
  document.getElementById(id).classList.remove('hidden');
}
function closeOverlay(id) {
  document.getElementById(id).classList.add('hidden');
}
function closeAllOverlays() {
  document.querySelectorAll('.overlay').forEach(function(o){ o.classList.add('hidden'); });
}

// ── Command Palette ───────────────────────────────────────
function openPalette() {
  openOverlay('palette-overlay');
  var pi = document.getElementById('pal-input');
  pi.value = '';
  pi.focus();
  renderPaletteItems(paletteItems);
}
function closePalette(e) {
  if (!e || e.target.id === 'palette-overlay') closeOverlay('palette-overlay');
}
function renderPaletteItems(items) {
  var list = document.getElementById('pal-list');
  if (items.length === 0) {
    list.innerHTML = '<div class="pal-empty">No results found</div>';
    return;
  }
  list.innerHTML = items.map(function(item, i) {
    return '<div class="pal-item' + (i===0?' pal-sel':'') + '" data-idx="' + i + '" onclick="selectPaletteItem(' + i + ')">'
      + '<span class="pal-ico">' + item.ico + '</span>'
      + '<span class="pal-label">' + item.label + '</span>'
      + (item.hint ? '<span class="pal-hint">' + escH(item.hint) + '</span>' : '')
      + '</div>';
  }).join('');
  list._filtered = items;
}
var palSel = 0;
function filterPalette() {
  var q = document.getElementById('pal-input').value.toLowerCase();
  var filtered = q ? paletteItems.filter(function(i){ return i.label.toLowerCase().includes(q); }) : paletteItems;
  palSel = 0;
  renderPaletteItems(filtered);
}
function palKey(e) {
  var items = document.querySelectorAll('.pal-item');
  if (e.key === 'ArrowDown') {
    palSel = Math.min(palSel + 1, items.length - 1);
    items.forEach(function(el,i){ el.classList.toggle('pal-sel', i === palSel); });
    e.preventDefault();
  } else if (e.key === 'ArrowUp') {
    palSel = Math.max(palSel - 1, 0);
    items.forEach(function(el,i){ el.classList.toggle('pal-sel', i === palSel); });
    e.preventDefault();
  } else if (e.key === 'Enter') {
    var list = document.getElementById('pal-list');
    var filtered = list._filtered || paletteItems;
    if (filtered[palSel]) selectPaletteItem(palSel, filtered);
    e.preventDefault();
  } else if (e.key === 'Escape') {
    closeOverlay('palette-overlay');
  }
}
function selectPaletteItem(idx, arr) {
  var list = document.getElementById('pal-list');
  var filtered = arr || list._filtered || paletteItems;
  var item = filtered[idx];
  closeOverlay('palette-overlay');
  if (item.action) {
    item.action();
  } else if (item.op) {
    var btn = document.querySelector('[data-op="' + item.op + '"]');
    if (btn) setOp(btn, item.op);
    document.getElementById('inp').focus();
  }
}

// ── Shortcuts modal ───────────────────────────────────────
function openShortcuts() { openOverlay('shortcuts-overlay'); }
function openFormulas()  { openOverlay('formulas-overlay'); }

// ── Formula cheatsheet ────────────────────────────────────
function switchFTab(btn, targetId) {
  document.querySelectorAll('.ftab').forEach(function(t){ t.classList.remove('active'); });
  document.querySelectorAll('.fpanel').forEach(function(p){ p.classList.remove('active'); });
  btn.classList.add('active');
  document.getElementById(targetId).classList.add('active');
}
function insertFormula(targetOp, formula) {
  closeOverlay('formulas-overlay');
  var btn = document.querySelector('[data-op="' + targetOp + '"]');
  if (btn) setOp(btn, targetOp);
  document.getElementById('inp').value = formula;
  document.getElementById('inp').focus();
  toast('Formula inserted! Press Enter to compute.', 'info');
}

// ── Sidebar & Op ─────────────────────────────────────────
function setOp(btn, o) {
  document.querySelectorAll('.op').forEach(function(b){ b.classList.remove('on'); });
  btn.classList.add('on');
  op = o;
  document.getElementById('pfx').textContent = o + '›';
  document.getElementById('inp').placeholder = 'e.g. ' + (hints[o] || '…');
  document.getElementById('inp').focus();
}
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}
function toggleNum() {
  numerical = !numerical;
  document.getElementById('numTog').classList.toggle('on', numerical);
  toast('Switched to ' + (numerical ? 'Numerical' : 'Symbolic') + ' mode', 'info');
}
function toggleStep() {
  showSteps = !showSteps;
  document.getElementById('stepTog').classList.toggle('on', showSteps);
  toast('Step display ' + (showSteps ? 'enabled' : 'disabled'), 'info');
}
function toggleSuggest() {
  showSuggests = !showSuggests;
  document.getElementById('suggestTog').classList.toggle('on', showSuggests);
}

// ── Input history ─────────────────────────────────────────
function handleKey(e) {
  var inp = document.getElementById('inp');
  if (e.key === 'Enter') { send(); return; }
  if (e.key === 'ArrowUp') {
    if (historyIdx < inputHistory.length - 1) {
      historyIdx++;
      inp.value = inputHistory[inputHistory.length - 1 - historyIdx] || '';
    }
    e.preventDefault();
  } else if (e.key === 'ArrowDown') {
    if (historyIdx > 0) {
      historyIdx--;
      inp.value = inputHistory[inputHistory.length - 1 - historyIdx] || '';
    } else if (historyIdx === 0) {
      historyIdx = -1;
      inp.value = '';
    }
    e.preventDefault();
  }
}

// ── Global keyboard shortcuts ─────────────────────────────
document.addEventListener('keydown', function(e) {
  if (e.ctrlKey || e.metaKey) {
    var k = e.key.toLowerCase();
    if (k === 'k') { e.preventDefault(); openPalette(); }
    else if (k === 'l') { e.preventDefault(); clearChat(); }
    else if (k === 't') { e.preventDefault(); toggleTheme(); }
    else if (k === 'e') { e.preventDefault(); exportHistory(); }
    else if (k === ',') { e.preventDefault(); openOverlay('settings-overlay'); }
    else if (k === '/') { e.preventDefault(); openShortcuts(); }
    else if (k === 'f') { e.preventDefault(); openFormulas(); }
    else if (k === 'i') { e.preventDefault(); document.getElementById('inp').focus(); }
    return;
  }
  if (e.key === 'Escape') closeAllOverlays();
});

// ── Chat helpers ───────────────────────────────────────────
function chat() { return document.getElementById('chat'); }
function hideWelcome() { var w = document.getElementById('welcome'); if (w) w.remove(); }
function escH(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function nowTS() {
  return new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

function addUser(text) {
  hideWelcome();
  var d = document.createElement('div');
  d.className = 'msg user';
  d.innerHTML = '<div class="av usr">👤</div>'
    + '<div class="bub">' + escH(text) + '<div class="ts"><span class="ts-time">' + nowTS() + '</span></div></div>';
  chat().appendChild(d);
  scrollDown();
}

function addBot(html, latex, plot, qas, elapsed, meta) {
  latex   = latex   || '';
  plot    = plot    || '';
  qas     = qas     || [];
  elapsed = elapsed || '';
  meta    = meta    || null;

  hideWelcome();
  var d = document.createElement('div');
  d.className = 'msg bot';

  var mathHtml = latex
    ? '<div class="mblk">\\[' + latex + '\\]</div>' : '';
  var plotHtml = plot
    ? '<img class="pimg" src="data:image/png;base64,' + plot + '" alt="plot" onclick="fullscreenPlot(this.src)">' : '';
  var metaHtml = '';
  if (meta && meta.type === 'stats') {
    metaHtml = '<table class="stbl">';
    var entries = Object.entries(meta.data);
    entries.forEach(function(kv) {
      metaHtml += '<tr><td>' + escH(String(kv[0])) + '</td><td>' + escH(String(kv[1])) + '</td></tr>';
    });
    metaHtml += '</table>';
  } else if (meta && meta.type === 'regression') {
    metaHtml = '<div class="rcard"><div class="req">y = ' + escH(meta.data.equation) + '</div>'
      + '<span style="color:var(--green)">R² = ' + meta.data.r_squared + '</span>'
      + ' &nbsp; RMSE = ' + meta.data.rmse
      + ' &nbsp; n = ' + meta.data.n_points + '</div>';
  }

  var actionsHtml = '<div class="bub-actions">';
  if (latex)  actionsHtml += '<button class="bact" onclick="copyLatex(this)" data-latex="' + escH(latex) + '">📋 LaTeX</button>';
  if (plot)   actionsHtml += '<button class="bact" onclick="downloadPlot(\'' + plot + '\')">⬇ PNG</button>';
  actionsHtml += '<button class="bact" onclick="copyText(this.closest(\'.bub\').innerText)">📄 Text</button>';
  // AI Explain button — shows only for compute results when AI is available
  if (resultText && op !== 'answer' && aiEnabled) {
    var explainData = JSON.stringify({op: op, expr: currentInput || '', result: resultText});
    actionsHtml += '<button class="bact" id="explain-btn-' + Date.now() + '" '
      + 'onclick="aiExplain(this,' + escH(explainData) + ')">🤖 Explain</button>';
  }
  actionsHtml += '</div>';

  var qaHtml = (showSuggests && qas.length)
    ? '<div class="qas">' + qas.map(function(a){
        var si = a.input.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
        return '<button class="qa" onclick="qi(\'' + a.op + '\',\'' + si + '\')">' + escH(a.label) + '</button>';
      }).join('') + '</div>'
    : '';

  var tsHtml = '<div class="ts"><span class="ts-time">' + nowTS() + '</span>'
    + (elapsed ? ' &nbsp; <span style="color:var(--green);font-size:10px">⚡ ' + elapsed + '</span>' : '')
    + '</div>';

  d.innerHTML = '<div class="av bot">∂</div>'
    + '<div class="bub">'
      + '<span style="font-size:10px;font-family:\'JetBrains Mono\',monospace;color:var(--muted)">'
      + escH(op.toUpperCase()) + '</span><br>'
      + html + mathHtml + metaHtml + plotHtml + qaHtml
      + actionsHtml + tsHtml
    + '</div>';

  chat().appendChild(d);
  if (latex && window.MathJax) {
    MathJax.typesetPromise([d]).catch(function(e){ console.error(e); });
  }
  scrollDown();
}

function addThinking() {
  hideWelcome();
  var d = document.createElement('div');
  d.className = 'msg bot';
  d.id = 'thinking';
  d.innerHTML = '<div class="av bot">∂</div>'
    + '<div class="thk"><div class="dots"><span></span><span></span><span></span></div>Computing…</div>';
  chat().appendChild(d);
  scrollDown();
}
function rmThinking() { var e = document.getElementById('thinking'); if (e) e.remove(); }
function scrollDown() { var c = chat(); c.scrollTop = c.scrollHeight; }

// ── Copy / Download ───────────────────────────────────────
function copyText(text) {
  navigator.clipboard.writeText(text).then(function(){ toast('✓ Copied to clipboard', 'success'); });
}
function copyLatex(btn) {
  var lat = btn.getAttribute('data-latex');
  copyText(lat);
}
function downloadPlot(b64) {
  currentPlotB64 = b64;
  var a = document.createElement('a');
  a.href = 'data:image/png;base64,' + b64;
  a.download = 'botX_plot_' + Date.now() + '.png';
  a.click();
  toast('⬇ Plot downloaded', 'success');
}
function fullscreenPlot(src) {
  document.getElementById('fullscreen-img').src = src;
  currentPlotB64 = src.split(',')[1] || '';
  openOverlay('fullscreen-overlay');
}
function downloadCurrentPlot() {
  if (currentPlotB64) {
    var a = document.createElement('a');
    a.href = 'data:image/png;base64,' + currentPlotB64;
    a.download = 'botX_plot_' + Date.now() + '.png';
    a.click();
  }
}

// ── Session stats ─────────────────────────────────────────
function updateStats(elapsed) {
  sessionStats.total++;
  sessionStats.time += elapsed;
  sessionStats.ops[op] = (sessionStats.ops[op] || 0) + 1;
  document.getElementById('statCount').textContent = sessionStats.total;
  document.getElementById('statTime').textContent = (elapsed).toFixed(0) + 'ms';
}

// ── Quick insert ──────────────────────────────────────────
function qi(o, input) {
  var btn = document.querySelector('[data-op="' + o + '"]');
  if (btn) setOp(btn, o);
  document.getElementById('inp').value = input;
  send();
}

// ── Quick insert NLP ──────────────────────────────────────
function qiNlp(text) {
  var btn = document.querySelector('[data-op="nlp"]');
  if (btn) setOp(btn, 'nlp');
  document.getElementById('inp').value = text;
  send();
}

// ── Send ──────────────────────────────────────────────────
async function send() {
  var inp  = document.getElementById('inp');
  var text = inp.value.trim();
  if (!text) return;
  inp.value = '';
  historyIdx = -1;

  // Save to input history
  if (inputHistory[inputHistory.length - 1] !== text) {
    inputHistory.push(text);
    if (inputHistory.length > 80) inputHistory.shift();
    try { localStorage.setItem('botx-input-history', JSON.stringify(inputHistory)); } catch(e){}
  }

  addUser(text);
  addThinking();
  var domain  = document.getElementById('domSel').value;
  var xRange  = xPlotRange;
  var t0      = performance.now();

  // ── NLP mode ──────────────────────────────────────────────
  if (op === 'nlp') {
    try {
      var nlpRes = await fetch('/nlp', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: text, domain: domain, x_range: xRange,
                               ai_enabled: aiEnabled, ai_tokens: AI_TOKENS[aiSpeed] || 150 })
      });
      var nlpData = await nlpRes.json();
      var elapsed = Math.round(performance.now() - t0);
      rmThinking();
      updateStats(elapsed);

      if (nlpData.type === 'answer') {
        // Pure text/HTML answer (identity, help, constants)
        var d2 = document.createElement('div');
        d2.className = 'msg bot';
        var tsHtml2 = '<div class="ts"><span class="ts-time">' + nowTS() + '</span>'
          + ' &nbsp;<span style="color:var(--purple);font-size:10px">🤖 NLP</span>'
          + (elapsed ? ' &nbsp; <span style="color:var(--green);font-size:10px">⚡ ' + elapsed + 'ms</span>' : '')
          + '</div>';
        d2.innerHTML = '<div class="av bot">🤖</div>'
          + '<div class="bub">'
          + '<span style="font-size:10px;font-family:\'JetBrains Mono\',monospace;color:var(--purple)">NLP</span><br>'
          + nlpData.html
          + '<div class="bub-actions"><button class="bact" onclick="copyText(this.closest(\'.bub\').innerText)">📄 Text</button></div>'
          + tsHtml2
          + '</div>';
        hideWelcome();
        chat().appendChild(d2);
        scrollDown();
        sessionHistory.push({role:'user', op:'nlp', text:text, time:nowTS()});
        sessionHistory.push({role:'bot', result:nlpData.html, elapsed:elapsed});

      } else if (nlpData.type === 'math') {
        // Math result – show operation banner
        if (nlpData.error) {
          addBot(
            '<div style="font-size:10px;color:var(--purple);margin-bottom:4px">🤖 Parsed as: <strong>' + escH(nlpData.op || '?') + '</strong></div>'
            + '<div class="err">⚠️ ' + escH(nlpData.error) + '</div>',
            '', '', [], elapsed);
        } else {
          currentInput = text;
          resultText   = nlpData.result || '';
          addBot(
            '<div style="font-size:10px;color:var(--purple);margin-bottom:4px">🤖 Parsed as: <strong>' + escH(nlpData.op || '?') + '</strong></div>'
            + '<span style="word-break:break-word">' + escH(nlpData.result) + '</span>',
            nlpData.latex, nlpData.plot, [], elapsed, nlpData.meta || null
          );
          sessionHistory.push({role:'user', op:'nlp', text:text, time:nowTS()});
          sessionHistory.push({role:'bot', result:nlpData.result, latex:nlpData.latex, elapsed:elapsed});
        }
      } else {
        addBot('<div class="err">⚠️ ' + escH(nlpData.message || 'Parse error') + '</div>', '', '', [], elapsed);
      }
    } catch(err) {
      rmThinking();
      addBot('<div class="err">⚠️ Network error: ' + escH(err.message) + '</div>');
    }
    return;
  }

  try {
    var res = await fetch('/compute', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({
        operation: op,
        input:     text,
        domain:    domain,
        numerical: numerical,
        x_range:   xRange,
        plot:      ['solve','diff','integrate','ode2','plot','polar','parametric','plot3d',
                    'stats','regression','partial','simplify'].indexOf(op) !== -1,
        steps:     showSteps,
      })
    });
    var data    = await res.json();
    var elapsed = Math.round(performance.now() - t0);
    rmThinking();
    updateStats(elapsed);

    if (data.error) {
      addBot('<div class="err">⚠️ ' + escH(data.error) + '</div>'
        + '<div style="margin-top:6px;color:var(--muted);font-size:11px">'
        + 'Tip: try <code>2x</code>, <code>x^2</code>, <code>sin(x)</code></div>',
        '', '', [], elapsed);
    } else {
      currentInput = text;
      resultText   = data.result || '';
      var qas = [];
      if (showSuggests) {
        if (op === 'solve') {
          var lhs = text.split('=')[0].trim();
          qas.push({op:'plot',      input:lhs, label:'📈 Plot LHS'});
          qas.push({op:'diff',      input:lhs, label:'∂ Differentiate LHS'});
          qas.push({op:'simplify',  input:lhs + ', factor', label:'✦ Factor LHS'});
        } else if (op === 'diff') {
          var base = text.split(',')[0].trim();
          qas.push({op:'plot',      input:base, label:'📈 Plot original'});
          qas.push({op:'integrate', input:base, label:'∫ Integrate instead'});
        } else if (op === 'integrate') {
          var base2 = text.split(',')[0].trim();
          qas.push({op:'plot',input:base2, label:'📈 Plot'});
          qas.push({op:'diff',input:base2, label:'∂ Differentiate'});
        } else if (op === 'simplify') {
          var base3 = text.split(',')[0].trim();
          qas.push({op:'plot',input:base3, label:'📈 Plot'});
          qas.push({op:'diff',input:base3, label:'∂ Differentiate'});
        } else if (op === 'laplace') {
          qas.push({op:'laplace', input: data.result.replace('Laplace: ','') + ', inverse', label:'ℒ⁻¹ Inverse'});
        } else if (op === 'stats') {
          qas.push({op:'regression', input: text.replace(',','') + '; degree=1', label:'📉 Fit regression'});
        }
      }
      addBot(
        '<span style="word-break:break-word">' + escH(data.result) + '</span>',
        data.latex, data.plot, qas, elapsed, data.meta || null
      );
      sessionHistory.push({role:'user', op:op, text:text, time:nowTS()});
      sessionHistory.push({role:'bot', result:data.result, latex:data.latex, elapsed:elapsed});
    }
  } catch(err) {
    rmThinking();
    addBot('<div class="err">⚠️ Network error: ' + escH(err.message) + '</div>');
  }
}

// ── Clear / Export ────────────────────────────────────────
function clearChat() {
  chat().innerHTML =
    '<div class="welcome" id="welcome">'
    + '<div class="wico">∂</div>'
    + '<h2>Chat cleared</h2>'
    + '<p>Ready for a new session.</p></div>';
  sessionHistory = [];
  sessionStats   = {total:0, time:0, ops:{}};
  document.getElementById('statCount').textContent = '0';
  document.getElementById('statTime').textContent  = '—';
  toast('Chat cleared', 'info');
}
function exportHistory() {
  var blob = new Blob([JSON.stringify({session:sessionHistory, stats:sessionStats}, null, 2)],
    {type:'application/json'});
  var a = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'botX_session_' + Date.now() + '.json';
  a.click();
  toast('📥 Session exported', 'success');
}

document.getElementById('inp').focus();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Flask routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return HTML


@app.route('/compute', methods=['POST'])
def compute():
    data      = request.json or {}
    op        = data.get('operation', 'solve')
    user_inp  = data.get('input', '').strip()
    domain    = data.get('domain', 'real')
    numerical = bool(data.get('numerical', False))
    want_plot = bool(data.get('plot', False))
    x_range   = int(data.get('x_range', 10))
    x_range   = max(1, min(100, x_range))

    result_text = ''
    latex_str   = ''
    plot_b64    = None
    error       = None
    meta        = None

    t0 = time.time()
    try:
        # ── Calculus ──────────────────────────────────────────
        if op == 'solve':
            solutions, eq, expr = solve_equation(user_inp, domain, numerical)
            if solutions == sp.EmptySet:
                result_text = 'No solutions found in the specified domain.'
                latex_str   = r'\text{No solutions found}'
            else:
                result_text = f'Solutions: {solutions}'
                latex_str   = sp.latex(solutions)
            if want_plot and solutions != sp.EmptySet:
                try: plot_b64 = make_plot(str(expr), x_range=(-x_range, x_range))
                except Exception: pass

        elif op == 'diff':
            r           = do_differentiate(user_inp)
            result_text = f'Derivative: {r}'
            orig        = safe_sympify(user_inp.split(',')[0])
            latex_str   = (r'\frac{d}{dx}\!\left(' + sp.latex(orig) + r'\right) = ' + sp.latex(r))
            if want_plot:
                try: plot_b64 = make_plot(str(r), title="f'(x)", x_range=(-x_range, x_range))
                except Exception: pass

        elif op == 'partial':
            r, lhs_latex = do_partial_diff(user_inp)
            result_text  = f'Partial derivative: {r}'
            latex_str    = lhs_latex + ' = ' + sp.latex(r)

        elif op == 'integrate':
            r           = do_integrate(user_inp)
            result_text = f'Integral: {r}'
            orig        = safe_sympify(user_inp.split(',')[0])
            latex_str   = (r'\int\!\left(' + sp.latex(orig) + r'\right)dx = ' + sp.latex(r))
            if want_plot:
                try: plot_b64 = make_plot(str(r), title='Antiderivative', x_range=(-x_range, x_range))
                except Exception: pass

        elif op == 'limit':
            r           = do_limit(user_inp)
            result_text = f'Limit: {r}'
            parts       = [p.strip() for p in user_inp.split(',')]
            pt          = parts[1] if len(parts) > 1 else '0'
            orig        = safe_sympify(parts[0])
            latex_str   = (r'\lim_{x \to ' + pt + r'}\left(' + sp.latex(orig) + r'\right) = ' + sp.latex(r))

        elif op == 'taylor':
            r           = do_taylor(user_inp)
            result_text = f'Taylor Series: {r}'
            latex_str   = sp.latex(r)

        elif op == 'ode2':
            sol, ode    = solve_ode2(user_inp)
            result_text = f'ODE Solution: {sol}'
            latex_str   = sp.latex(sol)
            if want_plot:
                try: plot_b64 = make_ode_plot(str(sol.rhs))
                except Exception: pass

        # ── Algebra ───────────────────────────────────────────
        elif op == 'simplify':
            r           = do_simplify(user_inp)
            orig        = safe_sympify(user_inp.split(',')[0])
            result_text = f'Result: {r}'
            parts_s     = [p.strip() for p in user_inp.split(',', 1)]
            sub_op      = parts_s[1].lower() if len(parts_s) > 1 else 'simplify'
            latex_str   = sp.latex(orig) + r' \;\xrightarrow{' + sub_op + r'}\; ' + sp.latex(r)
            if want_plot:
                try: plot_b64 = make_plot(str(r), x_range=(-x_range, x_range))
                except Exception: pass

        elif op == 'system':
            solution    = do_system(user_inp)
            result_text = f'Solution: {solution}'
            if isinstance(solution, dict):
                parts_l = ', '.join(sp.latex(k) + ' = ' + sp.latex(v) for k, v in solution.items())
                latex_str = r'\left\{' + parts_l + r'\right.'
            else:
                latex_str = sp.latex(solution)

        elif op == 'inequality':
            solution    = do_inequality(user_inp)
            result_text = f'Solution set: {solution}'
            latex_str   = sp.latex(solution)

        elif op == 'sequence':
            r, sum_latex = do_sequence(user_inp)
            result_text  = f'Result: {r}'
            latex_str    = sum_latex + ' = ' + sp.latex(r)

        # ── Transforms ────────────────────────────────────────
        elif op == 'laplace':
            r, lhs_lat  = do_laplace(user_inp)
            result_text = f'Laplace: {r}'
            latex_str   = lhs_lat + ' = ' + sp.latex(r)

        elif op == 'fourier':
            r, lhs_lat  = do_fourier(user_inp)
            result_text = f'Fourier: {r}'
            latex_str   = lhs_lat + ' = ' + sp.latex(r)

        # ── Plotting ──────────────────────────────────────────
        elif op == 'plot':
            plot_b64    = make_plot(user_inp, x_range=(-x_range, x_range))
            result_text = f'Plot of: {user_inp}'

        elif op == 'polar':
            plot_b64    = make_polar_plot(user_inp)
            result_text = f'Polar plot of r = {user_inp}'

        elif op == 'parametric':
            plot_b64    = make_parametric_plot(user_inp)
            result_text = f'Parametric plot'

        elif op == 'plot3d':
            plot_b64    = make_3d_plot(user_inp)
            result_text = f'3D surface: z = {user_inp.split(";")[0]}'

        # ── Linear Algebra ────────────────────────────────────
        elif op == 'matrix':
            r           = do_matrix(user_inp)
            result_text = f'Matrix Result: {r}'
            try:        latex_str = sp.latex(r)
            except:     latex_str = str(r)

        elif op == 'vector':
            text_r, latex_r = do_vector(user_inp)
            result_text     = text_r
            latex_str       = latex_r

        # ── Data Analysis ─────────────────────────────────────
        elif op == 'stats':
            summary, data_list = do_stats(user_inp)
            lines       = [f"{k}: {v}" for k, v in summary.items()]
            result_text = '\n'.join(lines)
            meta        = {'type': 'stats', 'data': summary}
            latex_str   = (r'\bar{x}=' + str(round(summary['mean'], 4)) +
                           r',\;\sigma=' + str(round(summary['std_dev'], 4)) +
                           r',\;n=' + str(summary['count']))
            if want_plot:
                try: plot_b64 = make_stats_plot(data_list)
                except Exception: pass

        elif op == 'regression':
            summary, xs, ys, poly = do_regression(user_inp)
            result_text = (f"y = {summary['equation']}\n"
                           f"R² = {summary['r_squared']}  RMSE = {summary['rmse']}  n = {summary['n_points']}")
            meta = {'type': 'regression', 'data': summary}
            latex_str = r'y = ' + summary['equation'].replace('x^', 'x^{').replace(' +','}+').replace(' -','}-') + ('}' if '^' in summary['equation'] else '')
            if want_plot:
                try: plot_b64 = make_regression_plot(xs, ys, poly, summary['degree'])
                except Exception: pass

        # ── Number Theory ─────────────────────────────────────
        elif op == 'numtheory':
            result_text = do_number_theory(user_inp)
            latex_str   = ''

        else:
            result_text = 'Unknown operation.'

    except Exception as e:
        error = str(e)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    return jsonify({
        'result':    result_text,
        'latex':     latex_str,
        'plot':      plot_b64,
        'error':     error,
        'meta':      meta,
        'elapsed_ms': elapsed_ms,
        'timestamp': datetime.now().strftime('%H:%M:%S'),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  /nlp  — Natural-language query endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/nlp', methods=['POST'])
def nlp_route():
    data       = request.json or {}
    text       = data.get('text', '').strip()
    domain     = data.get('domain', 'real')
    x_range    = int(data.get('x_range', 10))
    ai_enabled = data.get('ai_enabled', True)
    ai_tokens  = int(data.get('ai_tokens', 150))

    if not text:
        return jsonify({'type': 'error', 'message': 'Empty query.'})

    # Temporarily patch OfflineAI token limit for this request
    _orig_chat  = OfflineAI.chat
    _orig_route = OfflineAI.route
    if not ai_enabled:
        # Monkey-patch to disable AI for this request
        OfflineAI.chat  = classmethod(lambda cls, *a, **kw: None)
        OfflineAI.route = classmethod(lambda cls, *a, **kw: (None, None))
    else:
        # Patch token count
        def _chat_limited(cls, msg, history=None):
            return cls._call(msg, cls.PERSONA, max_tokens=ai_tokens)
        def _route_limited(cls, msg):
            raw = cls._call(msg, cls.ROUTER, max_tokens=min(ai_tokens, 200))
            if not raw:
                return None, None
            try:
                raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw.strip(), flags=re.M)
                m = re.search(r'\{.*\}', raw, re.DOTALL)
                if m:
                    d2 = _json.loads(m.group(0))
                    op2 = d2.get('op', '')
                    if op2 == 'answer':
                        return 'answer', d2.get('html', '')
                    inp2 = d2.get('input', '')
                    if op2 and inp2:
                        return op2, inp2
            except Exception:
                pass
            return None, None
        OfflineAI.chat  = classmethod(_chat_limited)
        OfflineAI.route = classmethod(_route_limited)

    op, inp = parse_nlp(text)

    def _restore():
        OfflineAI.chat  = _orig_chat
        OfflineAI.route = _orig_route

    if op == 'answer':
        _restore()
        return jsonify({'type': 'answer', 'html': inp})

    if op is None:
        # Final fallback: let AI respond freely as botX
        ai_resp = OfflineAI.chat(text)
        _restore()
        if ai_resp:
            return jsonify({'type': 'answer', 'html': ai_resp})
        return jsonify({
            'type': 'answer',
            'html': (
                "Hmm, I'm not sure what you mean! 🤔 I'm great at math and I can chat too. Try:<br>"
                "• <em>differentiate sin(x)</em> — calculus<br>"
                "• <em>is 97 prime?</em> — number theory<br>"
                "• <em>plot sin(x)</em> — visualization<br>"
                "• <em>convert 100°C to Fahrenheit</em> — unit conversion<br>"
                "• <em>tell me a joke</em> — fun!<br>"
                "Or use the sidebar to select a specific operation.<br><br>"
                f'<span style="font-size:10px;color:var(--muted)">'
                f'💡 Install <a href="https://ollama.com" target="_blank" style="color:var(--blue)">Ollama</a> + '
                f'a model (e.g. <code>ollama pull gemma3:1b</code>) for enhanced AI responses.</span>'
            )
        })

    _restore()

    # Route to compute internally
    want_plot = op in ('solve','diff','integrate','ode2','plot','polar',
                       'parametric','plot3d','stats','regression','partial','simplify')
    result_text = ''
    latex_str   = ''
    plot_b64    = None
    error       = None
    meta        = None
    t0          = time.time()

    try:
        if op == 'solve':
            solutions, eq, expr = solve_equation(inp, domain, False)
            result_text = f'Solutions: {solutions}' if solutions != sp.EmptySet else 'No real solutions.'
            latex_str   = sp.latex(solutions) if solutions != sp.EmptySet else r'\text{No solutions}'
            if want_plot and solutions != sp.EmptySet:
                try: plot_b64 = make_plot(str(expr), x_range=(-x_range, x_range))
                except Exception: pass
        elif op == 'diff':
            r = do_differentiate(inp)
            result_text = f'Derivative: {r}'
            orig = safe_sympify(inp.split(',')[0])
            latex_str = r'\frac{d}{dx}\!\left(' + sp.latex(orig) + r'\right) = ' + sp.latex(r)
            if want_plot:
                try: plot_b64 = make_plot(str(r), title="f'(x)", x_range=(-x_range, x_range))
                except Exception: pass
        elif op == 'integrate':
            r = do_integrate(inp)
            result_text = f'Integral: {r}'
            orig = safe_sympify(inp.split(',')[0])
            latex_str = r'\int\!\left(' + sp.latex(orig) + r'\right)dx = ' + sp.latex(r)
            if want_plot:
                try: plot_b64 = make_plot(str(r), title='Antiderivative', x_range=(-x_range, x_range))
                except Exception: pass
        elif op == 'limit':
            r = do_limit(inp)
            result_text = f'Limit: {r}'
            parts = [p.strip() for p in inp.split(',')]
            pt = parts[1] if len(parts) > 1 else '0'
            orig = safe_sympify(parts[0])
            latex_str = r'\lim_{x \to ' + pt + r'}\left(' + sp.latex(orig) + r'\right) = ' + sp.latex(r)
        elif op == 'taylor':
            r = do_taylor(inp)
            result_text = f'Taylor Series: {r}'
            latex_str = sp.latex(r)
        elif op == 'simplify':
            r = do_simplify(inp)
            orig = safe_sympify(inp.split(',')[0])
            result_text = f'Result: {r}'
            latex_str = sp.latex(orig) + r' \to ' + sp.latex(r)
            if want_plot:
                try: plot_b64 = make_plot(str(r), x_range=(-x_range, x_range))
                except Exception: pass
        elif op == 'plot':
            plot_b64 = make_plot(inp, x_range=(-x_range, x_range))
            result_text = f'Plot of: {inp}'
        elif op == 'numtheory':
            result_text = do_number_theory(inp)
        elif op == 'stats':
            summary, data_list = do_stats(inp)
            result_text = '\n'.join(f"{k}: {v}" for k, v in summary.items())
            meta = {'type': 'stats', 'data': summary}
            latex_str = r'\bar{x}=' + str(round(summary['mean'], 4))
            if want_plot:
                try: plot_b64 = make_stats_plot(data_list)
                except Exception: pass
        elif op == 'laplace':
            r, lhs = do_laplace(inp)
            result_text = f'Laplace: {r}'
            latex_str = lhs + ' = ' + sp.latex(r)
        elif op == 'fourier':
            r, lhs = do_fourier(inp)
            result_text = f'Fourier: {r}'
            latex_str = lhs + ' = ' + sp.latex(r)
        elif op == 'partial':
            r, lhs = do_partial_diff(inp)
            result_text = f'Partial: {r}'
            latex_str = lhs + ' = ' + sp.latex(r)
        elif op == 'ode2':
            sol, _ = solve_ode2(inp)
            result_text = f'ODE Solution: {sol}'
            latex_str = sp.latex(sol)
        elif op == 'vector':
            text_r, latex_r = do_vector(inp)
            result_text = text_r
            latex_str = latex_r
        elif op == 'sequence':
            r, sum_latex = do_sequence(inp)
            result_text = f'Result: {r}'
            latex_str = sum_latex + ' = ' + sp.latex(r)
        elif op == 'system':
            sol = do_system(inp)
            result_text = f'Solution: {sol}'
            latex_str = sp.latex(sol)
        elif op == 'inequality':
            sol = do_inequality(inp)
            result_text = f'Solution set: {sol}'
            latex_str = sp.latex(sol)
        elif op == 'matrix':
            r = do_matrix(inp)
            result_text = f'Matrix Result: {r}'
            try: latex_str = sp.latex(r)
            except: latex_str = str(r)
        else:
            result_text = 'Unrecognised operation.'
    except Exception as e:
        error = str(e)

    elapsed_ms = round((time.time() - t0) * 1000, 1)
    return jsonify({
        'type':       'math',
        'op':         op,
        'result':     result_text,
        'latex':      latex_str,
        'plot':       plot_b64,
        'error':      error,
        'meta':       meta,
        'elapsed_ms': elapsed_ms,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive Graph Calculator  (Desmos-style, full-featured)
# ─────────────────────────────────────────────────────────────────────────────
GRAPH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>botX Graph Calculator</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Sora:wght@400;500;600;700&display=swap" rel="stylesheet">
<script>
MathJax = { tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]}, svg:{fontCache:'global'}, startup:{typeset:false} };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0e17;--surface:#0f1521;--surface2:#171e2e;--surface3:#1e2740;
  --border:#2a3550;--text:#e8eef8;--muted:#6b7fa8;
  --blue:#4f9cf9;--green:#43d47e;--red:#f9634f;--purple:#b07ffb;--orange:#f9a84f;
  --accent:#4f9cf9;--shadow:rgba(0,0,0,.6);
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:'Sora',sans-serif;font-size:13px;overflow:hidden}
.app{display:grid;grid-template-columns:320px 1fr;height:100vh}

/* ── SIDEBAR ─────────────────────────────────────────────── */
.sidebar{
  background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}
.topbar{
  display:flex;align-items:center;gap:8px;padding:14px 16px;
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.logo{font-weight:700;font-size:15px;display:flex;align-items:center;gap:6px}
.logo-ico{width:28px;height:28px;background:linear-gradient(135deg,var(--blue),var(--purple));
  border-radius:7px;display:grid;place-items:center;font-size:13px}
.sp{flex:1}
.tbtn{padding:5px 10px;border-radius:7px;background:var(--surface2);border:1px solid var(--border);
  color:var(--muted);font-size:11px;cursor:pointer;transition:all .15s;font-family:'Sora',sans-serif;
  display:flex;align-items:center;gap:4px}
.tbtn:hover{border-color:var(--blue);color:var(--blue)}
.tbtn.icon{width:30px;height:30px;padding:0;justify-content:center}

/* Expressions list */
.expr-list{flex:1;overflow-y:auto;padding:10px}
.expr-item{
  background:var(--surface2);border:1px solid var(--border);border-radius:10px;
  margin-bottom:8px;overflow:hidden;transition:border-color .15s;
}
.expr-item:focus-within{border-color:var(--blue)}
.expr-header{display:flex;align-items:center;gap:6px;padding:8px 10px 0}
.color-dot{
  width:14px;height:14px;border-radius:50%;cursor:pointer;flex-shrink:0;
  border:2px solid transparent;transition:transform .15s;
}
.color-dot:hover{transform:scale(1.3)}
.expr-input{
  flex:1;background:transparent;border:none;outline:none;
  color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px;padding:0;
}
.expr-input::placeholder{color:var(--muted)}
.vis-btn{width:22px;height:22px;border-radius:5px;border:1px solid var(--border);
  background:var(--surface3);cursor:pointer;display:grid;place-items:center;font-size:11px;
  color:var(--muted);transition:all .12s;flex-shrink:0}
.vis-btn:hover,.vis-btn.hidden{background:rgba(79,156,249,.12);border-color:var(--blue)}
.del-btn{width:22px;height:22px;border-radius:5px;border:1px solid transparent;
  background:transparent;cursor:pointer;display:grid;place-items:center;font-size:12px;
  color:var(--muted);transition:all .12s;flex-shrink:0}
.del-btn:hover{color:var(--red);border-color:rgba(249,99,79,.3)}
.expr-latex{padding:4px 10px 8px;min-height:28px;font-size:12px;color:var(--muted);cursor:default}
.expr-info{padding:0 10px 8px;font-size:10px;color:var(--green);font-family:'JetBrains Mono',monospace;min-height:18px}

.add-btn{
  width:calc(100% - 20px);margin:0 10px 10px;padding:9px;
  background:var(--surface2);border:1px solid var(--border);border-radius:10px;
  color:var(--muted);cursor:pointer;font-size:12px;transition:all .15s;
  display:flex;align-items:center;justify-content:center;gap:6px;font-family:'Sora',sans-serif;
}
.add-btn:hover{border-color:var(--blue);color:var(--blue)}

/* Analysis panel */
.analysis{
  border-top:1px solid var(--border);padding:12px 16px;flex-shrink:0;
  background:var(--surface);
}
.analysis h4{font-size:11px;color:var(--muted);letter-spacing:.8px;text-transform:uppercase;
  margin-bottom:8px;font-weight:600}
.analysis-row{display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;font-size:11px}
.analysis-lbl{color:var(--muted);min-width:54px;flex-shrink:0}
.analysis-val{color:var(--text);font-family:'JetBrains Mono',monospace;line-height:1.5;flex:1;word-break:break-all}

/* Range & settings */
.range-section{border-top:1px solid var(--border);padding:10px 16px;flex-shrink:0}
.range-row{display:flex;align-items:center;gap:6px;margin-bottom:6px;font-size:11px;color:var(--muted)}
.range-row input[type=number]{
  width:60px;padding:3px 6px;background:var(--surface2);border:1px solid var(--border);
  border-radius:5px;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:11px;text-align:center;
}
.range-row label{min-width:28px}
.chk-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:11px;color:var(--muted)}
.chk-row input{accent-color:var(--blue)}

/* ── CANVAS AREA ─────────────────────────────────────────── */
.canvas-wrap{position:relative;overflow:hidden;background:var(--bg)}
#gc{width:100%;height:100%;display:block;cursor:crosshair}
#coord-hud{
  position:absolute;bottom:12px;right:16px;
  background:rgba(15,21,33,.9);border:1px solid var(--border);
  border-radius:8px;padding:5px 10px;font-size:11px;
  font-family:'JetBrains Mono',monospace;color:var(--muted);
  pointer-events:none;backdrop-filter:blur(6px);
}
#coord-hud span{color:var(--blue)}
.toolbar-float{
  position:absolute;top:12px;right:12px;display:flex;gap:5px;
}
.fbtn{
  padding:5px 10px;border-radius:8px;border:1px solid var(--border);
  background:rgba(15,21,33,.9);color:var(--muted);font-size:11px;cursor:pointer;
  backdrop-filter:blur(6px);transition:all .15s;font-family:'Sora',sans-serif;
}
.fbtn:hover{border-color:var(--blue);color:var(--blue)}

/* Color picker popup */
.color-picker{
  position:fixed;z-index:999;background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:8px;box-shadow:0 8px 32px var(--shadow);
  display:none;grid-template-columns:repeat(5,22px);gap:4px;
}
.color-picker.open{display:grid}
.cp-swatch{width:22px;height:22px;border-radius:50%;cursor:pointer;border:2px solid transparent;transition:transform .12s}
.cp-swatch:hover{transform:scale(1.2);border-color:var(--text)}

/* Table of values modal */
.tov-overlay{
  position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;
  display:none;align-items:center;justify-content:center;backdrop-filter:blur(4px);
}
.tov-overlay.open{display:flex}
.tov-modal{
  background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:20px;max-width:400px;width:90%;max-height:80vh;overflow-y:auto;
  box-shadow:0 20px 60px var(--shadow);
}
.tov-modal h3{font-size:15px;font-weight:700;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center}
.tov-close{background:none;border:none;color:var(--muted);cursor:pointer;font-size:18px}
.tov-table{width:100%;border-collapse:collapse;font-size:12px;font-family:'JetBrains Mono',monospace}
.tov-table th{padding:5px 10px;border-bottom:2px solid var(--border);color:var(--muted);text-align:left}
.tov-table td{padding:4px 10px;border-bottom:1px solid var(--surface3)}
.tov-table tr:last-child td{border-bottom:none}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:9px}
</style>
</head>
<body>
<div class="app">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="topbar">
    <div class="logo"><div class="logo-ico">𝑓</div> Graph Calculator</div>
    <div class="sp"></div>
    <button class="tbtn icon" title="Screenshot" onclick="takeScreenshot()">📷</button>
    <button class="tbtn icon" title="Table of values" onclick="openToV()">📋</button>
    <button class="tbtn" onclick="resetView()" title="Reset view">↺ Reset</button>
  </div>

  <div class="expr-list" id="expr-list"></div>
  <button class="add-btn" onclick="addExpr()">＋ Add Function</button>

  <!-- Range controls -->
  <div class="range-section">
    <div class="range-row">
      <label>x:</label>
      <input type="number" id="xmin" value="-10" onchange="setRange()">
      <span style="color:var(--muted)">to</span>
      <input type="number" id="xmax" value="10" onchange="setRange()">
    </div>
    <div class="range-row">
      <label>y:</label>
      <input type="number" id="ymin" value="-8" onchange="setRange()">
      <span style="color:var(--muted)">to</span>
      <input type="number" id="ymax" value="8" onchange="setRange()">
    </div>
    <div class="chk-row">
      <input type="checkbox" id="chk-grid" checked onchange="render()"> Grid
      <input type="checkbox" id="chk-axes" checked onchange="render()"> Axes
      <input type="checkbox" id="chk-labels" checked onchange="render()"> Labels
      <input type="checkbox" id="chk-points" checked onchange="render()"> Points
    </div>
  </div>

  <!-- Analysis -->
  <div class="analysis">
    <h4>Analysis</h4>
    <div class="analysis-row">
      <span class="analysis-lbl">Zeros</span>
      <span class="analysis-val" id="an-zeros">—</span>
    </div>
    <div class="analysis-row">
      <span class="analysis-lbl">Extrema</span>
      <span class="analysis-val" id="an-ext">—</span>
    </div>
    <div class="analysis-row">
      <span class="analysis-lbl">Intercept</span>
      <span class="analysis-val" id="an-yint">—</span>
    </div>
  </div>
</div>

<!-- CANVAS -->
<div class="canvas-wrap">
  <canvas id="gc"></canvas>
  <div id="coord-hud">x=<span id="hx">0.00</span> y=<span id="hy">0.00</span></div>
  <div class="toolbar-float">
    <button class="fbtn" onclick="zoom(1.25)">＋ Zoom In</button>
    <button class="fbtn" onclick="zoom(0.8)">－ Zoom Out</button>
    <button class="fbtn" onclick="fitView()">⊡ Fit</button>
  </div>
</div>

</div><!-- .app -->

<!-- Color picker -->
<div class="color-picker" id="color-picker">
  <div class="cp-swatch" style="background:#f94040" data-c="#f94040"></div>
  <div class="cp-swatch" style="background:#f97940" data-c="#f97940"></div>
  <div class="cp-swatch" style="background:#f9c440" data-c="#f9c440"></div>
  <div class="cp-swatch" style="background:#4fb84f" data-c="#4fb84f"></div>
  <div class="cp-swatch" style="background:#43d4b8" data-c="#43d4b8"></div>
  <div class="cp-swatch" style="background:#4f9cf9" data-c="#4f9cf9"></div>
  <div class="cp-swatch" style="background:#7f4ff9" data-c="#7f4ff9"></div>
  <div class="cp-swatch" style="background:#d44ff9" data-c="#d44ff9"></div>
  <div class="cp-swatch" style="background:#f94096" data-c="#f94096"></div>
  <div class="cp-swatch" style="background:#888888" data-c="#888888"></div>
</div>

<!-- Table of values modal -->
<div class="tov-overlay" id="tov-overlay" onclick="closeToV(event)">
  <div class="tov-modal">
    <h3>Table of Values <button class="tov-close" onclick="closeToV()">×</button></h3>
    <div id="tov-content"></div>
  </div>
</div>

<script>
'use strict';
// ─────────────────────────────────────────────────────────────────────────
//  State
// ─────────────────────────────────────────────────────────────────────────
const EXPR_COLORS = ['#f94040','#4f9cf9','#4fb84f','#f9a84f','#7f4ff9','#43d4b8','#f94096','#f9c440'];
let expressions = [];
let nextId      = 1;
let view        = { x0:-10, y0:8, ppu:50 }; // origin (top-left in math coords) + pixels-per-unit
let isDragging  = false;
let lastMouse   = { x:0, y:0 };
let canvas, ctx;
let activeColorId = null;
let rafId = null;

// ─────────────────────────────────────────────────────────────────────────
//  Init
// ─────────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  canvas = document.getElementById('gc');
  ctx    = canvas.getContext('2d');
  resize();
  window.addEventListener('resize', resize);
  bindCanvasEvents();
  addExpr('sin(x)', EXPR_COLORS[0]);
  addExpr('cos(x)', EXPR_COLORS[1]);
  schedRender();
});

function resize() {
  const wrap  = canvas.parentElement;
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  // keep center fixed
  const cx = view.x0 + canvas.width  / 2 / view.ppu;
  const cy = view.y0 - canvas.height / 2 / view.ppu;
  view.x0 = cx - canvas.width  / 2 / view.ppu;
  view.y0 = cy + canvas.height / 2 / view.ppu;
  schedRender();
}

// ─────────────────────────────────────────────────────────────────────────
//  Math ↔ Canvas transforms
// ─────────────────────────────────────────────────────────────────────────
function mx2cx(mx) { return (mx - view.x0) * view.ppu; }
function my2cy(my) { return (view.y0 - my) * view.ppu; }
function cx2mx(cx) { return view.x0 + cx / view.ppu; }
function cy2my(cy) { return view.y0 - cy / view.ppu; }

// ─────────────────────────────────────────────────────────────────────────
//  Expression management
// ─────────────────────────────────────────────────────────────────────────
function addExpr(src='', color=null) {
  const id = nextId++;
  if (!color) color = EXPR_COLORS[(expressions.length) % EXPR_COLORS.length];
  const expr = { id, src, color, visible:true, fn:null, error:null };
  if (src) compile(expr);
  expressions.push(expr);
  renderSidebar();
  schedRender();
  return expr;
}

function removeExpr(id) {
  expressions = expressions.filter(e => e.id !== id);
  renderSidebar();
  schedRender();
  updateAnalysis();
}

function toggleVisible(id) {
  const e = expressions.find(e => e.id === id);
  if (e) { e.visible = !e.visible; renderSidebar(); schedRender(); }
}

function setExprSrc(id, src) {
  const e = expressions.find(e => e.id === id);
  if (!e) return;
  e.src   = src;
  e.error = null;
  compile(e);
  schedRender();
  updateAnalysis();
}

// ─────────────────────────────────────────────────────────────────────────
//  Compile expression string to JS function
// ─────────────────────────────────────────────────────────────────────────
function compile(expr) {
  try {
    let s = expr.src.trim();
    if (!s) { expr.fn = null; return; }

    // Strip "y = " prefix
    s = s.replace(/^y\s*=\s*/i, '');

    // Implicit: if contains y (not just as part of word), try f(x)=0 form
    // For now, treat as explicit y = f(x)

    // Substitutions for JS math
    s = s.replace(/\^/g, '**');
    s = s.replace(/(\d)([a-zA-Z(])/g, '$1*$2');
    s = s.replace(/\)([\w(])/g, ')*$1');
    s = s.replace(/\bpi\b/g, 'Math.PI');
    s = s.replace(/\be\b/g, 'Math.E');
    s = s.replace(/\bsin\b/g, 'Math.sin');
    s = s.replace(/\bcos\b/g, 'Math.cos');
    s = s.replace(/\btan\b/g, 'Math.tan');
    s = s.replace(/\basin\b/g, 'Math.asin');
    s = s.replace(/\bacos\b/g, 'Math.acos');
    s = s.replace(/\batan\b/g, 'Math.atan');
    s = s.replace(/\bsinh\b/g, 'Math.sinh');
    s = s.replace(/\bcosh\b/g, 'Math.cosh');
    s = s.replace(/\btanh\b/g, 'Math.tanh');
    s = s.replace(/\bsqrt\b/g, 'Math.sqrt');
    s = s.replace(/\babs\b/g, 'Math.abs');
    s = s.replace(/\bexp\b/g, 'Math.exp');
    s = s.replace(/\blog\b/g, 'Math.log');
    s = s.replace(/\bln\b/g, 'Math.log');
    s = s.replace(/\bfloor\b/g, 'Math.floor');
    s = s.replace(/\bceil(?:ing)?\b/g, 'Math.ceil');
    s = s.replace(/\bround\b/g, 'Math.round');
    s = s.replace(/\bmax\b/g, 'Math.max');
    s = s.replace(/\bmin\b/g, 'Math.min');
    s = s.replace(/\bsign\b/g, 'Math.sign');
    s = s.replace(/\bpow\b/g, 'Math.pow');
    s = s.replace(/\blog2\b/g, 'Math.log2');
    s = s.replace(/\blog10\b/g, 'Math.log10');
    s = s.replace(/\batan2\b/g, 'Math.atan2');
    s = s.replace(/\bhypot\b/g, 'Math.hypot');

    // eslint-disable-next-line no-new-func
    const fn = new Function('x', `"use strict"; try { return ${s}; } catch(e) { return NaN; }`);
    // Quick test
    const v = fn(1.0);
    if (typeof v !== 'number' && typeof v !== 'undefined') throw new Error('bad');
    expr.fn    = fn;
    expr.error = null;
  } catch(e) {
    expr.fn    = null;
    expr.error = 'Invalid expression';
  }
}

// ─────────────────────────────────────────────────────────────────────────
//  Render sidebar
// ─────────────────────────────────────────────────────────────────────────
function renderSidebar() {
  const list = document.getElementById('expr-list');
  list.innerHTML = expressions.map(e => `
    <div class="expr-item" id="ei-${e.id}">
      <div class="expr-header">
        <div class="color-dot" style="background:${e.color}"
             onclick="openColorPicker(${e.id}, event)" title="Change color"></div>
        <input class="expr-input" value="${escAttr(e.src)}"
               placeholder="f(x) = …"
               oninput="setExprSrc(${e.id}, this.value)"
               onblur="updateLatex(${e.id})"
               id="inp-${e.id}">
        <button class="vis-btn ${e.visible?'':'hidden'}" onclick="toggleVisible(${e.id})" title="Toggle">${e.visible ? '👁' : '—'}</button>
        <button class="del-btn" onclick="removeExpr(${e.id})" title="Remove">×</button>
      </div>
      <div class="expr-latex" id="lat-${e.id}"></div>
      <div class="expr-info" id="inf-${e.id}">${e.error ? '⚠ ' + e.error : ''}</div>
    </div>
  `).join('');
  // Typeset LaTeX
  setTimeout(() => {
    expressions.forEach(e => {
      const latEl = document.getElementById('lat-' + e.id);
      if (!latEl) return;
      if (e.src && !e.error) {
        const disp = e.src.replace(/\*\*/g,'^').replace(/Math\.\w+\(/g, m => m.replace('Math.','').replace('(','('));
        latEl.textContent = '\\(y = ' + e.src + '\\)';
        if (window.MathJax) MathJax.typesetPromise([latEl]).catch(()=>{});
      }
    });
  }, 50);
}

function updateLatex(id) { renderSidebar(); }
function escAttr(s) { return s.replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ─────────────────────────────────────────────────────────────────────────
//  Color picker
// ─────────────────────────────────────────────────────────────────────────
function openColorPicker(id, evt) {
  evt.stopPropagation();
  activeColorId = id;
  const picker = document.getElementById('color-picker');
  picker.style.left = (evt.clientX + 5) + 'px';
  picker.style.top  = (evt.clientY + 5) + 'px';
  picker.classList.add('open');
  picker.querySelectorAll('.cp-swatch').forEach(sw => {
    sw.onclick = (e) => {
      e.stopPropagation();
      const expr = expressions.find(ex => ex.id === activeColorId);
      if (expr) { expr.color = sw.dataset.c; renderSidebar(); schedRender(); }
      picker.classList.remove('open');
    };
  });
}
document.addEventListener('click', () => {
  document.getElementById('color-picker').classList.remove('open');
});

// ─────────────────────────────────────────────────────────────────────────
//  Rendering
// ─────────────────────────────────────────────────────────────────────────
function schedRender() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(render);
}

function render() {
  rafId = null;
  if (!canvas || !ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#0a0e17';
  ctx.fillRect(0, 0, W, H);

  const showGrid   = document.getElementById('chk-grid')?.checked ?? true;
  const showAxes   = document.getElementById('chk-axes')?.checked ?? true;
  const showLabels = document.getElementById('chk-labels')?.checked ?? true;
  const showPoints = document.getElementById('chk-points')?.checked ?? true;

  if (showGrid)   drawGrid(W, H);
  if (showAxes)   drawAxes(W, H);
  if (showLabels) drawLabels(W, H);

  // Draw expressions
  expressions.forEach(e => {
    if (e.visible && e.fn) drawExpr(e, W, H, showPoints);
  });
}

function adaptStep(rawStep) {
  const mags  = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000];
  for (const s of mags) if (s >= rawStep) return s;
  return rawStep;
}

function drawGrid(W, H) {
  const rawStep = 80 / view.ppu;  // ~80px between grid lines in math units
  const step    = adaptStep(rawStep);
  ctx.strokeStyle = '#1e2740';
  ctx.lineWidth   = 1;

  const x0m = view.x0, x1m = view.x0 + W / view.ppu;
  const y1m = view.y0, y0m = view.y0 - H / view.ppu;

  // Verticals
  const xStart = Math.ceil(x0m / step) * step;
  for (let x = xStart; x <= x1m; x += step) {
    const cx = mx2cx(x);
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  }
  // Horizontals
  const yStart = Math.ceil(y0m / step) * step;
  for (let y = yStart; y <= y1m; y += step) {
    const cy = my2cy(y);
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
  }
}

function drawAxes(W, H) {
  ctx.strokeStyle = '#2a3550';
  ctx.lineWidth   = 1.5;
  const ox = mx2cx(0), oy = my2cy(0);
  // x-axis
  if (oy >= 0 && oy <= H) {
    ctx.beginPath(); ctx.moveTo(0, oy); ctx.lineTo(W, oy); ctx.stroke();
  }
  // y-axis
  if (ox >= 0 && ox <= W) {
    ctx.beginPath(); ctx.moveTo(ox, 0); ctx.lineTo(ox, H); ctx.stroke();
  }
  // Arrow heads
  ctx.fillStyle = '#2a3550';
  if (oy >= 0 && oy <= H) {
    ctx.beginPath(); ctx.moveTo(W-2,oy); ctx.lineTo(W-10,oy-4); ctx.lineTo(W-10,oy+4); ctx.fill();
    ctx.beginPath(); ctx.moveTo(2,oy); ctx.lineTo(10,oy-4); ctx.lineTo(10,oy+4); ctx.fill();
  }
  if (ox >= 0 && ox <= W) {
    ctx.beginPath(); ctx.moveTo(ox,2); ctx.lineTo(ox-4,10); ctx.lineTo(ox+4,10); ctx.fill();
  }
}

function drawLabels(W, H) {
  const rawStep = 80 / view.ppu;
  const step    = adaptStep(rawStep);
  const ox = mx2cx(0), oy = my2cy(0);

  ctx.font      = '10px JetBrains Mono, monospace';
  ctx.fillStyle = '#4a5a7a';
  ctx.textAlign = 'center';

  const x0m = view.x0, x1m = view.x0 + W / view.ppu;
  const y0m = view.y0 - H / view.ppu, y1m = view.y0;

  const fmtN = v => {
    if (Math.abs(v) < 1e-9) return '0';
    if (Number.isInteger(v) || Math.abs(v) >= 10) return v.toFixed(0);
    return v.toPrecision(3).replace(/\.?0+$/,'');
  };

  // x labels
  const xStart = Math.ceil(x0m / step) * step;
  for (let x = xStart; x <= x1m; x += step) {
    if (Math.abs(x) < step * 0.01) continue;
    const cx = mx2cx(x);
    const cy = Math.min(Math.max(oy + 14, 14), H - 4);
    ctx.fillText(fmtN(x), cx, cy);
  }

  // y labels
  ctx.textAlign = 'right';
  const yStart = Math.ceil(y0m / step) * step;
  for (let y = yStart; y <= y1m; y += step) {
    if (Math.abs(y) < step * 0.01) continue;
    const cy = my2cy(y);
    const cx = Math.min(Math.max(ox - 5, 5), W - 5);
    ctx.fillText(fmtN(y), cx, cy + 3);
  }

  // Axis labels
  ctx.fillStyle = '#6b7fa8';
  ctx.textAlign = 'center';
  ctx.font = '11px Sora, sans-serif';
  ctx.fillText('x', W - 8, Math.min(Math.max(oy - 6, 12), H - 6));
  ctx.fillText('y', Math.min(Math.max(ox + 10, 16), W - 10), 12);
}

function drawExpr(e, W, H, showPoints) {
  if (!e.fn) return;
  const STEPS = Math.ceil(W * 2);
  const dx    = (W / view.ppu) / STEPS;
  const xMin  = view.x0, xMax = view.x0 + W / view.ppu;

  ctx.strokeStyle = e.color;
  ctx.lineWidth   = 2.2;
  ctx.lineJoin    = 'round';
  ctx.lineCap     = 'round';
  ctx.beginPath();
  let penDown = false;
  let prevY   = NaN;

  for (let i = 0; i <= STEPS; i++) {
    const mx = xMin + i * dx;
    let   my = e.fn(mx);
    if (!isFinite(my) || isNaN(my)) { penDown = false; prevY = NaN; continue; }
    if (!isFinite(prevY)) { prevY = my; }
    // Discontinuity guard
    if (Math.abs(my - prevY) > 20 * (H / view.ppu)) { penDown = false; }
    const cx = mx2cx(mx), cy = my2cy(my);
    if (cy < -H * 2 || cy > H * 3) { penDown = false; prevY = my; continue; }
    if (!penDown) { ctx.moveTo(cx, cy); penDown = true; }
    else          { ctx.lineTo(cx, cy); }
    prevY = my;
  }
  ctx.stroke();

  // Draw special points (zeros, extrema)
  if (showPoints) drawSpecialPoints(e, xMin, xMax, W, H);
}

function drawSpecialPoints(e, xMin, xMax, W, H) {
  const pts = findSpecialPoints(e.fn, xMin, xMax, 600);
  ctx.fillStyle = e.color;
  ctx.strokeStyle = '#0a0e17';
  ctx.lineWidth = 1.5;
  for (const pt of pts) {
    const cx = mx2cx(pt.x), cy = my2cy(pt.y);
    if (cx < -5 || cx > W + 5 || cy < -5 || cy > H + 5) continue;
    ctx.beginPath();
    ctx.arc(cx, cy, pt.type === 'zero' ? 4 : 5, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();
  }
}

// ─────────────────────────────────────────────────────────────────────────
//  Find special points via sampling + bisection
// ─────────────────────────────────────────────────────────────────────────
function findSpecialPoints(fn, xMin, xMax, steps) {
  const pts = [];
  const dx  = (xMax - xMin) / steps;
  const xs  = Array.from({length: steps+1}, (_,i) => xMin + i*dx);
  const ys  = xs.map(x => { try { return fn(x); } catch { return NaN; } });

  // Zeros via sign change + bisection
  for (let i = 0; i < steps; i++) {
    const ya = ys[i], yb = ys[i+1];
    if (!isFinite(ya) || !isFinite(yb)) continue;
    if (ya * yb < 0) {
      const x = bisect(fn, xs[i], xs[i+1]);
      if (x !== null) pts.push({x, y:0, type:'zero'});
    }
  }

  // Local extrema (derivative changes sign)
  for (let i = 1; i < steps-1; i++) {
    const ya = ys[i-1], yc = ys[i], yb = ys[i+1];
    if (!isFinite(ya) || !isFinite(yb) || !isFinite(yc)) continue;
    const d1 = yc - ya, d2 = yb - yc;
    if (d1 * d2 < 0 && Math.abs(yc - ys[i-1]) < 10) {
      pts.push({x: xs[i], y: yc, type: d1 > 0 ? 'max' : 'min'});
    }
  }

  return pts;
}

function bisect(fn, a, b) {
  for (let i = 0; i < 40; i++) {
    const m = (a + b) / 2;
    const fm = fn(m);
    if (!isFinite(fm)) return null;
    if (Math.abs(b - a) < 1e-10) return m;
    if (fn(a) * fm < 0) b = m; else a = m;
  }
  return (a + b) / 2;
}

// ─────────────────────────────────────────────────────────────────────────
//  Analysis panel updater
// ─────────────────────────────────────────────────────────────────────────
function updateAnalysis() {
  const active = expressions.filter(e => e.visible && e.fn);
  if (!active.length) {
    ['an-zeros','an-ext','an-yint'].forEach(id => document.getElementById(id).textContent = '—');
    return;
  }
  const e    = active[0];
  const xMin = view.x0, xMax = view.x0 + canvas.width / view.ppu;
  const pts  = findSpecialPoints(e.fn, xMin, xMax, 1000);

  const zeros  = pts.filter(p => p.type === 'zero');
  const maxima = pts.filter(p => p.type === 'max');
  const minima = pts.filter(p => p.type === 'min');

  const fmtX = x => x.toFixed(3);
  const zStr = zeros.length ? zeros.slice(0,5).map(p => 'x≈'+fmtX(p.x)).join(', ') : 'none in view';
  const mxStr = [...maxima.slice(0,2).map(p=>'('+fmtX(p.x)+','+fmtX(p.y)+')'),
                  ...minima.slice(0,2).map(p=>'('+fmtX(p.x)+','+fmtX(p.y)+')')].join(', ') || 'none in view';
  const yint = (() => { try { return e.fn(0).toFixed(4); } catch { return '—'; } })();

  document.getElementById('an-zeros').textContent = zStr;
  document.getElementById('an-ext').textContent   = mxStr;
  document.getElementById('an-yint').textContent  = 'y(0) = ' + yint;
}

// ─────────────────────────────────────────────────────────────────────────
//  Canvas mouse / touch events
// ─────────────────────────────────────────────────────────────────────────
function bindCanvasEvents() {
  canvas.addEventListener('mousedown', e => {
    isDragging = true;
    lastMouse  = {x: e.clientX, y: e.clientY};
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('mousemove', e => {
    const mx = cx2mx(e.offsetX), my = cy2my(e.offsetY);
    document.getElementById('hx').textContent = mx.toFixed(3);
    document.getElementById('hy').textContent = my.toFixed(3);
    if (isDragging) {
      const dx = (e.clientX - lastMouse.x) / view.ppu;
      const dy = (e.clientY - lastMouse.y) / view.ppu;
      view.x0 -= dx;
      view.y0 += dy;
      lastMouse = {x: e.clientX, y: e.clientY};
      schedRender();
      updateRangeInputs();
    }
  });
  canvas.addEventListener('mouseup', () => { isDragging = false; canvas.style.cursor = 'crosshair'; updateAnalysis(); });
  canvas.addEventListener('mouseleave', () => { isDragging = false; canvas.style.cursor = 'crosshair'; });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.85 : 1.18;
    const mx = cx2mx(e.offsetX), my = cy2my(e.offsetY);
    view.ppu *= factor;
    view.ppu  = Math.max(2, Math.min(5000, view.ppu));
    view.x0   = mx - e.offsetX / view.ppu;
    view.y0   = my + e.offsetY / view.ppu;
    schedRender();
    updateRangeInputs();
    updateAnalysis();
  }, { passive: false });
  // Touch
  let lastTouchDist = 0;
  canvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      isDragging = true;
      lastMouse  = {x: e.touches[0].clientX, y: e.touches[0].clientY};
    } else if (e.touches.length === 2) {
      lastTouchDist = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY);
    }
    e.preventDefault();
  }, {passive:false});
  canvas.addEventListener('touchmove', e => {
    if (e.touches.length === 1 && isDragging) {
      const dx = (e.touches[0].clientX - lastMouse.x) / view.ppu;
      const dy = (e.touches[0].clientY - lastMouse.y) / view.ppu;
      view.x0 -= dx; view.y0 += dy;
      lastMouse = {x: e.touches[0].clientX, y: e.touches[0].clientY};
      schedRender();
    } else if (e.touches.length === 2) {
      const d = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY);
      const factor = d / (lastTouchDist || d);
      view.ppu = Math.max(2, Math.min(5000, view.ppu * factor));
      lastTouchDist = d;
      schedRender();
    }
    e.preventDefault();
  }, {passive:false});
  canvas.addEventListener('touchend', () => { isDragging = false; updateAnalysis(); });
}

// ─────────────────────────────────────────────────────────────────────────
//  View helpers
// ─────────────────────────────────────────────────────────────────────────
function resetView() {
  const W = canvas.width, H = canvas.height;
  view.ppu = 60;
  view.x0  = -W / 2 / view.ppu;
  view.y0  = H  / 2 / view.ppu;
  schedRender(); updateRangeInputs(); updateAnalysis();
}

function zoom(factor) {
  const W = canvas.width, H = canvas.height;
  const cx = cx2mx(W/2), cy = cy2my(H/2);
  view.ppu = Math.max(2, Math.min(5000, view.ppu * factor));
  view.x0  = cx - W/2/view.ppu;
  view.y0  = cy + H/2/view.ppu;
  schedRender(); updateRangeInputs(); updateAnalysis();
}

function fitView() {
  const active = expressions.filter(e => e.visible && e.fn);
  if (!active.length) { resetView(); return; }
  const W = canvas.width, H = canvas.height;
  // Sample range and find good bounds
  let yMin = Infinity, yMax = -Infinity;
  const SAMP = 400;
  for (const e of active) {
    for (let i = 0; i <= SAMP; i++) {
      const x = -10 + i * 20/SAMP;
      const y = e.fn(x);
      if (isFinite(y)) { yMin = Math.min(yMin, y); yMax = Math.max(yMax, y); }
    }
  }
  if (!isFinite(yMin)) { resetView(); return; }
  const pad = Math.max(1, (yMax - yMin) * 0.1);
  view.ppu = Math.min(W/20, H / (yMax - yMin + 2*pad));
  view.ppu = Math.max(2, Math.min(5000, view.ppu));
  view.x0  = -10;
  view.y0  = yMax + pad;
  schedRender(); updateRangeInputs(); updateAnalysis();
}

function setRange() {
  const xmin = parseFloat(document.getElementById('xmin').value) || -10;
  const xmax = parseFloat(document.getElementById('xmax').value) || 10;
  const ymin = parseFloat(document.getElementById('ymin').value) || -8;
  const ymax = parseFloat(document.getElementById('ymax').value) || 8;
  const W    = canvas.width, H = canvas.height;
  view.ppu = Math.min(W / (xmax - xmin), H / (ymax - ymin));
  view.ppu = Math.max(2, Math.min(5000, view.ppu));
  view.x0  = xmin;
  view.y0  = ymax;
  schedRender(); updateAnalysis();
}

function updateRangeInputs() {
  const W = canvas.width, H = canvas.height;
  document.getElementById('xmin').value = cx2mx(0).toFixed(2);
  document.getElementById('xmax').value = cx2mx(W).toFixed(2);
  document.getElementById('ymin').value = cy2my(H).toFixed(2);
  document.getElementById('ymax').value = cy2my(0).toFixed(2);
}

// ─────────────────────────────────────────────────────────────────────────
//  Table of values
// ─────────────────────────────────────────────────────────────────────────
function openToV() {
  const active = expressions.filter(e => e.visible && e.fn);
  const W = canvas.width;
  const xMin = cx2mx(0), xMax = cx2mx(W);
  const step = adaptStep((xMax - xMin) / 20);

  let html = '';
  for (const e of active.slice(0,3)) {
    html += `<p style="color:${e.color};font-weight:600;margin:8px 0 4px;font-size:12px">${e.src || 'f(x)'}</p>`;
    html += '<table class="tov-table"><thead><tr><th>x</th><th>f(x)</th></tr></thead><tbody>';
    for (let x = Math.ceil(xMin/step)*step; x <= xMax; x += step) {
      const y = e.fn(x);
      if (!isFinite(y)) continue;
      html += `<tr><td>${x.toFixed(4)}</td><td>${y.toFixed(6)}</td></tr>`;
    }
    html += '</tbody></table>';
  }

  document.getElementById('tov-content').innerHTML = html || '<p style="color:var(--muted)">No active expressions.</p>';
  document.getElementById('tov-overlay').classList.add('open');
}

function closeToV(evt) {
  if (!evt || evt.target === document.getElementById('tov-overlay')) {
    document.getElementById('tov-overlay').classList.remove('open');
  }
}

// ─────────────────────────────────────────────────────────────────────────
//  Screenshot
// ─────────────────────────────────────────────────────────────────────────
function takeScreenshot() {
  const a = document.createElement('a');
  a.download = 'botX_graph_' + Date.now() + '.png';
  a.href     = canvas.toDataURL('image/png');
  a.click();
}
</script>
</body>
</html>"""


@app.route('/ai_status')
def ai_status():
    """Returns current AI backend status."""
    model = OfflineAI.model()
    return jsonify({
        'available': OfflineAI.available(),
        'model': model or 'none',
        'html': OfflineAI.status_html(),
    })


@app.route('/ai_explain', methods=['POST'])
def ai_explain():
    """Ask AI to explain a math result in plain English."""
    data      = request.json or {}
    op        = data.get('op', '')
    expr      = data.get('expr', '')
    result    = data.get('result', '')
    ai_tokens = int(data.get('ai_tokens', 150))
    if not (op and result):
        return jsonify({'html': None, 'error': 'Missing op/result'})
    prompt = (
        f"The user asked botX to perform '{op}' on: {expr}\n"
        f"The result is: {result}\n\n"
        f"Give a SHORT (2-4 sentence) intuitive explanation of what this means "
        f"and how to think about it. Use HTML formatting. No LaTeX — use Unicode symbols."
    )
    html = OfflineAI._call(prompt, OfflineAI.PERSONA, max_tokens=ai_tokens)
    return jsonify({'html': html})


@app.route('/graph')
def graph_calculator():
    return GRAPH_HTML


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import threading, webbrowser
    threading.Timer(1.2, lambda: webbrowser.open('http://localhost:5050')).start()
    print()
    print('  ╔═══════════════════════════════════════════╗')
    print('  ║  botX 4.0 — AI Math Assistant ULTIMATE    ║')
    print('  ║  © 2026 Dhanwanth V                       ║')
    print('  ║  http://localhost:5050                    ║')
    print('  ║                                           ║')
    print('  ║  NEW v4.0: Offline AI (Ollama) · Smart    ║')
    print('  ║  NLP routing · AI explanations · Context  ║')
    print('  ║  /graph for interactive calculator        ║')
    print('  ╚═══════════════════════════════════════════╝')
    print()
    app.run(debug=False, port=5050)