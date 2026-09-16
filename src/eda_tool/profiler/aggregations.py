"""Mergeable centered moments and pairwise covariance (float64 arithmetic)."""
import numpy as np


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = self.m3 = self.m4 = 0.0
        self.minimum = self.maximum = None

    def update(self, values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return
        other = Moments()
        other.n = len(values)
        other.mean = float(values.mean())
        centered = values - other.mean
        other.m2 = float(np.sum(centered ** 2))
        other.m3 = float(np.sum(centered ** 3))
        other.m4 = float(np.sum(centered ** 4))
        other.minimum, other.maximum = float(values.min()), float(values.max())
        self.merge(other)

    def merge(self, other):
        if not other.n:
            return self
        if not self.n:
            self.__dict__.update(other.__dict__)
            return self
        a, b = self.n, other.n
        n = a + b
        delta = other.mean - self.mean
        m2, m3 = self.m2, self.m3
        self.m4 += (other.m4 + delta**4 * a*b*(a*a-a*b+b*b)/n**3
                    + 6*delta**2*(a*a*other.m2+b*b*m2)/n**2
                    + 4*delta*(a*other.m3-b*m3)/n)
        self.m3 += other.m3 + delta**3*a*b*(a-b)/n**2 + 3*delta*(a*other.m2-b*m2)/n
        self.m2 += other.m2 + delta**2*a*b/n
        self.mean += delta*b/n
        self.n = n
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        return self

    def result(self):
        n = self.n
        variance = max(0.0, self.m2/(n-1)) if n > 1 else None
        skew = ((n * np.sqrt(n-1)/(n-2))*self.m3/self.m2**1.5
                if n > 2 and self.m2 > 0 else None)
        kurtosis = (((n-1)/((n-2)*(n-3))) * ((n+1)*n*self.m4/self.m2**2-3*(n-1))
                    if n > 3 and self.m2 > 0 else None)
        result = {'count': n, 'mean': self.mean if n else None,
                  'variance': variance, 'std': np.sqrt(variance) if variance is not None else None,
                  'skew': skew, 'kurtosis': kurtosis, 'min': self.minimum, 'max': self.maximum,
                  'sum': self.mean*n if n else 0.0}
        if any(v is not None and not np.isfinite(v) for v in result.values()):
            raise ArithmeticError('Numeric moments overflowed float64; rescale the input')
        return result


class Covariance:
    def __init__(self):
        self.n = 0
        self.x = self.y = self.xx = self.yy = self.xy = 0.0

    def update(self, x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        b = len(x)
        if not b:
            return
        mx, my = float(x.mean()), float(y.mean())
        dx, dy = x-mx, y-my
        a, n = self.n, self.n+b
        delta_x, delta_y = mx-self.x, my-self.y
        self.xx += float(dx @ dx) + delta_x**2*a*b/n
        self.yy += float(dy @ dy) + delta_y**2*a*b/n
        self.xy += float(dx @ dy) + delta_x*delta_y*a*b/n
        self.x += delta_x*b/n
        self.y += delta_y*b/n
        self.n = n

    def correlation(self):
        if self.n < 2 or self.xx <= 0 or self.yy <= 0:
            return None
        value = self.xy / np.sqrt(self.xx) / np.sqrt(self.yy)
        if not np.isfinite(value):
            raise ArithmeticError('Covariance overflowed float64; rescale the input')
        return float(np.clip(value, -1, 1))
