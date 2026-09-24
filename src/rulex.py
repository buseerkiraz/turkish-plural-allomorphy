# -*- coding: utf-8 -*-
"""RULEX (Nosofsky, Palmeri & McKinley 1994), standard library only.

Rule-plus-exception model. A learner tests one hypothesis at a time, keeps no
memory of past exemplars, and ends up with at most one rule plus a short list
of exceptions. Learning is stochastic, so a single run is one simulated
subject; predictions are averages over many runs.

Learning stages (paper, pp. 55-57 and Appendix):

  1. PERFECT single-dimension rule. Sample a dimension (probability proportional
     to its weight, without replacement), form the rule consistent with the
     current exemplar, keep it while it makes no errors. Discard on the first
     error and sample another dimension.
  2. IMPERFECT single-dimension rule. Sample dimensions the same way. Counters
     per value, incremented with probability pstor, classify by majority.
     After lwind trials the rule is dropped if accuracy < lcrit; at uwind
     trials it becomes permanent if accuracy >= scrit, else it is dropped.
  3. CONJUNCTIVE rule over a pair of dimensions, sampled with probability
     proportional to the product of weights. On a trial with pattern (a, b)
     in category K the counters for (a, b) -> K, (a, ~b) -> ~K and (~a, b) -> ~K
     are incremented, with probability pstor ** 2; (~a, ~b) is left alone.
     Same windows as stage 2, strict criterion ccrit.
  Stages 2 and 3 run in that order with probability `branch`, otherwise the
  conjunctive search comes first (default branch = 1.0).
  4. EXCEPTIONS, once a permanent rule exists or every dimension and pair is
     exhausted. When the rule is wrong or does not apply, the item's rule
     dimensions are sampled with probability 1 and every other dimension with
     probability pstor. An exception over n dimensions, with m already stored,
     is kept with probability pstor ** n * capac ** m. An exception that later
     gives a wrong answer is discarded.

Classification: applicable exceptions first (conflicts resolved by the
proportion that point to each category), then the rule, then a guess.

Choices the paper leaves open, made explicit here:
  * A perfect rule becomes permanent once it has survived uwind trials
    without an error; the paper only says it is kept "as long as it works".
  * Rule accuracy during a test window is scored on the response the tentative
    rule gives BEFORE that trial's counters update, and only on trials where it
    gives one. Scoring an empty or tied counter as half-right would make the
    default ccrit = 1.0 unreachable, so no conjunctive rule could ever become
    permanent and Type II of Shepard et al. (1961) could not be learned.
  * While a rule is still being tested, responses use that tentative rule, so
    a subject still searching at the end of training is tested on its
    current hypothesis.
  * A new exception is attempted only when no correct stored exception
    already covers the item, and an exact duplicate is not stored twice.
"""
import itertools


class Rulex:
    def __init__(self, n_dims, rng, pstor, scrit, lwind, ccrit=1.0, lcrit=0.55,
                 uwind=None, weights=None, capac=1.0, branch=1.0):
        self.n_dims = n_dims
        self.rng = rng
        self.pstor, self.scrit, self.ccrit, self.lcrit = pstor, scrit, ccrit, lcrit
        self.lwind = lwind
        self.uwind = uwind if uwind is not None else 2 * lwind
        self.w = list(weights) if weights else [1.0] * n_dims
        self.capac = capac

        self.stage = "perfect"
        self._plan = (["imperfect", "conjunctive"] if rng.random() < branch
                      else ["conjunctive", "imperfect"])
        self._dims_left = [d for d in range(n_dims) if self.w[d] > 0]
        self._pairs_left = [p for p in itertools.combinations(self._dims_left, 2)]
        self.hyp = None          # dict(dims, counts, n, correct) while testing
        self.rule = None         # dict(dims, table) once permanent
        self.exceptions = []     # list of (tuple of (dim, value), category)
        self.history = []        # stages passed through, for reporting

    # ------------------------------------------------------------ sampling
    def _pick(self, items, weight):
        total = sum(weight(i) for i in items)
        r = self.rng.random() * total
        for i in items:
            r -= weight(i)
            if r <= 0:
                return i
        return items[-1]

    def _new_hypothesis(self):
        """Start testing the next dimension or pair, moving to the next search
        stage when a pool runs dry. Returns False once nothing is left."""
        while True:
            if self.stage in ("perfect", "imperfect") and self._dims_left:
                d = self._pick(self._dims_left, lambda i: self.w[i])
                self._dims_left.remove(d)
                self.hyp = dict(dims=(d,), counts={}, n=0, scored=0, correct=0)
                return True
            if self.stage == "conjunctive" and self._pairs_left:
                p = self._pick(self._pairs_left,
                               lambda q: self.w[q[0]] * self.w[q[1]])
                self._pairs_left.remove(p)
                self.hyp = dict(dims=p, counts={}, n=0, scored=0, correct=0)
                return True
            if not self._plan:
                self._enter_exceptions(None)
                return False
            self.stage = self._plan.pop(0)
            if self.stage == "imperfect":
                self._dims_left = [d for d in range(self.n_dims) if self.w[d] > 0]

    def _enter_exceptions(self, rule):
        self.rule = rule
        self.hyp = None
        self.history.append(self.stage)
        self.stage = "exceptions"

    # ------------------------------------------------------------ rules
    @staticmethod
    def _key(dims, x):
        return tuple(x[d] for d in dims)

    @staticmethod
    def _majority(counts, key):
        c = counts.get(key)
        if not c or c[0] == c[1]:
            return None
        return 0 if c[0] > c[1] else 1

    def _rule_response(self, x):
        """Category the rule (tentative or permanent) gives, or None."""
        if self.rule is not None:
            return self.rule["table"].get(self._key(self.rule["dims"], x))
        if self.hyp is not None:
            return self._majority(self.hyp["counts"], self._key(self.hyp["dims"], x))
        return None

    def _increment(self, x, y, perfect=False):
        h = self.hyp
        dims = h["dims"]
        if len(dims) == 1:
            if not perfect and self.rng.random() >= self.pstor:
                return
            v = x[dims[0]]
            for val, cat in ((v, y), (1 - v, 1 - y)):
                h["counts"].setdefault((val,), [0, 0])[cat] += 1
        else:
            if self.rng.random() >= self.pstor ** 2:
                return
            a, b = x[dims[0]], x[dims[1]]
            for pat, cat in (((a, b), y), ((a, 1 - b), 1 - y), ((1 - a, b), 1 - y)):
                h["counts"].setdefault(pat, [0, 0])[cat] += 1

    # ------------------------------------------------------------ exceptions
    def _applicable(self, x):
        return [e for e in self.exceptions if all(x[d] == v for d, v in e[0])]

    def _try_store_exception(self, x, y):
        must = set(self.rule["dims"]) if self.rule else set()
        pattern = tuple((d, x[d]) for d in range(self.n_dims)
                        if d in must or self.rng.random() < self.pstor)
        if not pattern:
            return
        if any(e[0] == pattern for e in self.exceptions):
            return
        p = self.pstor ** len(pattern) * self.capac ** len(self.exceptions)
        if self.rng.random() < p:
            self.exceptions.append((pattern, y))

    # ------------------------------------------------------------ public
    def p_category(self, x):
        """[P(0), P(1)] for x under the current state. Deterministic except
        for guessing and conflicting exceptions."""
        app = self._applicable(x)
        if app:
            p1 = sum(1 for e in app if e[1] == 1) / len(app)
            return [1 - p1, p1]
        r = self._rule_response(x)
        if r is None:
            return [0.5, 0.5]
        return [1.0 - r, float(r)]

    def train(self, x, y):
        """One learning trial. Returns [P(0), P(1)] from BEFORE learning."""
        probs = self.p_category(x)

        if self.stage == "exceptions":
            app = self._applicable(x)
            wrong = [e for e in app if e[1] != y]
            if wrong:
                self.exceptions = [e for e in self.exceptions if e not in wrong]
            covered = any(e[1] == y for e in app)
            if not covered and self._rule_response(x) != y:
                self._try_store_exception(x, y)
            return probs

        if self.hyp is None:
            if not self._new_hypothesis():
                return probs            # moved to exceptions; nothing to test
            if self.stage == "perfect":
                self._increment(x, y, perfect=True)
                self.hyp["n"] = 1
                return probs

        h = self.hyp
        pred = self._majority(h["counts"], self._key(h["dims"], x))

        if self.stage == "perfect":
            if pred != y:
                self.hyp = None
                return probs
            h["n"] += 1
            if h["n"] >= self.uwind:
                self._enter_exceptions(dict(dims=h["dims"], table=self._table(h)))
            return probs

        h["n"] += 1
        if pred is not None:
            h["scored"] += 1
            h["correct"] += pred == y
        self._increment(x, y)
        acc = h["correct"] / h["scored"] if h["scored"] else 0.0
        crit = self.scrit if self.stage == "imperfect" else self.ccrit
        if h["n"] >= self.lwind and acc < self.lcrit:
            self.hyp = None
        elif h["n"] >= self.uwind:
            if acc >= crit:
                self._enter_exceptions(dict(dims=h["dims"], table=self._table(h)))
            else:
                self.hyp = None
        return probs

    def _table(self, h):
        return {k: self._majority(h["counts"], k) for k in h["counts"]
                if self._majority(h["counts"], k) is not None}
