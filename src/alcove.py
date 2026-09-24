# -*- coding: utf-8 -*-
"""ALCOVE (Kruschke 1992), standard library only.

Attention Learning COVEring map. An exemplar model: one hidden node per stored
exemplar, activated by similarity to the input; a learned weight from each
exemplar to each category; and a learned attention weight per input dimension.
There is no rule and no exception list anywhere in it.

Equations (Kruschke 1992, eqs. 1-6), with the city-block metric (r = q = 1)
that Kruschke uses for separable dimensions such as binary features:

    hidden     a_j   = exp(-c * sum_i alpha_i * |h_ji - x_i|)
    output     o_k   = sum_j w_kj * a_j
    choice     P(k)  = exp(phi * o_k) / sum_m exp(phi * o_m)
    teacher    t_k   = max(+1, o_k) if k is correct else min(-1, o_k)   ("humble")
    weights    dw_kj = lambda_w * (t_k - o_k) * a_j
    attention  dalpha_i = -lambda_a * sum_j [sum_k (t_k - o_k) * w_kj] * a_j * c * |h_ji - x_i|

Both updates are computed from the pre-trial values, and attention is clipped
at zero. Weights start at 0 and attention starts equal, summing to 1.
"""
import math


class Alcove:
    def __init__(self, exemplars, n_categories, c, phi, lambda_w, lambda_a):
        self.h = [tuple(float(v) for v in e) for e in exemplars]
        self.n_dims = len(self.h[0])
        self.n_cat = n_categories
        self.c, self.phi = c, phi
        self.lambda_w, self.lambda_a = lambda_w, lambda_a
        self.alpha = [1.0 / self.n_dims] * self.n_dims
        self.w = [[0.0] * len(self.h) for _ in range(n_categories)]
        self._dist = {}

    def _diffs(self, x):
        """|h_ji - x_i| for every hidden node j. Exemplars never move, so this
        is cached per input; a speed-up only, the equations are unchanged."""
        x = tuple(x)
        d = self._dist.get(x)
        if d is None:
            d = self._dist[x] = [tuple(abs(hi - xi) for hi, xi in zip(h, x))
                                 for h in self.h]
        return d

    def _hidden(self, x):
        a = self.alpha
        return [math.exp(-self.c * sum(ai * di for ai, di in zip(a, dj)))
                for dj in self._diffs(x)]

    def _outputs(self, act):
        return [sum(wk * aj for wk, aj in zip(row, act)) for row in self.w]

    def _choice(self, out):
        top = max(out)                      # subtract the max for stability
        e = [math.exp(self.phi * (o - top)) for o in out]
        s = sum(e)
        return [v / s for v in e]

    def predict(self, x):
        """Category choice probabilities for input x. Does not learn."""
        return self._choice(self._outputs(self._hidden(x)))

    def train(self, x, category):
        """One learning trial. Returns the choice probabilities from BEFORE the
        update, which is what a learner's response on this trial would be."""
        act = self._hidden(x)
        out = self._outputs(act)
        probs = self._choice(out)

        teach = [max(1.0, o) if k == category else min(-1.0, o)
                 for k, o in enumerate(out)]
        err = [t - o for t, o in zip(teach, out)]

        # Attention gradient uses the old weights, so compute it first.
        back = [sum(err[k] * self.w[k][j] for k in range(self.n_cat))
                for j in range(len(self.h))]
        if self.lambda_a:
            diffs = self._diffs(x)
            ba = [b * a for b, a in zip(back, act)]
            for i in range(self.n_dims):
                g = self.c * sum(baj * dj[i] for baj, dj in zip(ba, diffs))
                self.alpha[i] = max(0.0, self.alpha[i] - self.lambda_a * g)

        for k in range(self.n_cat):
            row = self.w[k]
            for j in range(len(self.h)):
                row[j] += self.lambda_w * err[k] * act[j]
        return probs
