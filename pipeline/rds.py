"""Minimal pure-Python reader for R .rds files holding a data.frame (XDR, version 2/3)."""
import lzma, gzip, bz2, struct, sys
import numpy as np, pandas as pd

NA_INT = -2147483648

class R:
    def __init__(self, b):
        self.b = b; self.p = 0; self.refs = []
    def i(self):
        v = struct.unpack_from('>i', self.b, self.p)[0]; self.p += 4; return v
    def raw(self, n):
        v = self.b[self.p:self.p+n]; self.p += n; return v
    def length(self):
        n = self.i()
        if n == -1:
            hi = self.i(); lo = self.i(); n = (hi << 32) + lo
        return n
    def item(self):
        flags = self.i()
        t = flags & 0xFF
        has_attr = bool(flags & (1 << 9)); has_tag = bool(flags & (1 << 10))
        if t == 254: return None            # NILVALUE
        if t == 242: return None            # EMPTYENV
        if t == 253: return {'env': 'global'}
        if t == 251: return None            # MISSINGARG
        if t == 252: return None            # UNBOUNDVALUE
        if t == 250: return {'env': 'base'}
        if t == 255:                        # REFSXP
            idx = flags >> 8
            if idx == 0: idx = self.i()
            return self.refs[idx-1]
        if t == 1:                          # SYMSXP
            s = self.item(); self.refs.append(s); return s
        if t == 9:                          # CHARSXP
            n = self.i()
            return None if n == -1 else self.raw(n).decode('utf-8', 'replace')
        if t == 238:                        # ALTREP
            info = self.item(); state = self.item(); attr = self.item()
            cls = info[0] if isinstance(info, list) else info
            return self.altrep(cls, state)
        if t in (2, 6, 17):                 # LISTSXP / LANGSXP pairlists
            attr = self.item() if has_attr else None
            tag = self.item() if has_tag else None
            car = self.item(); cdr = self.item()
            out = [(tag, car)]
            if isinstance(cdr, PL): out += cdr.items
            return PL(out)
        if t == 4:                          # ENVSXP
            locked = self.i(); env = {}; self.refs.append(env)
            self.item(); self.item(); self.item(); self.item()
            return env
        if t in (13, 14, 10, 16, 19, 24, 20):
            n = self.length()
            if t == 13 or t == 10:
                a = np.frombuffer(self.raw(4*n), dtype='>i4').astype(np.int64)
                v = a
            elif t == 14:
                v = np.frombuffer(self.raw(8*n), dtype='>f8').astype(np.float64)
            elif t == 24:
                v = self.raw(n)
            elif t == 16:
                v = [self.item() for _ in range(n)]
            else:  # VECSXP/EXPRSXP
                v = [self.item() for _ in range(n)]
            attrs = self.item() if has_attr else None
            return Vec(t, v, attrs)
        raise ValueError(f'unsupported type {t} at {self.p}')
    def altrep(self, cls, state):
        name = cls if isinstance(cls, str) else str(cls)
        if name == 'compact_intseq':
            n, start, step = state.v if isinstance(state, Vec) else state
            return Vec(13, np.arange(int(start), int(start)+int(n)*int(step), int(step)), None)
        if name == 'compact_realseq':
            n, start, step = state.v
            return Vec(14, np.arange(start, start+n*step, step), None)
        if name in ('wrap_real', 'wrap_integer', 'wrap_string', 'wrap_logical', 'wrap_list'):
            return state.items[0][1] if isinstance(state, PL) else state.v[0]
        if name == 'deferred_string':
            arg = state.items[0][1] if isinstance(state, PL) else state
            return Vec(16, [None if (isinstance(x,(float,np.floating)) and np.isnan(x)) else str(x) for x in arg.v], None)
        raise ValueError('altrep ' + name)

class PL:
    def __init__(self, items): self.items = items
    def get(self, k):
        for t, v in self.items:
            if t == k: return v

class Vec:
    def __init__(self, t, v, attrs): self.t, self.v, self.attrs = t, v, attrs
    def attr(self, k): return self.attrs.get(k) if isinstance(self.attrs, PL) else None

def to_series(vec):
    if vec.t == 16: return pd.Series(vec.v, dtype=object)
    if vec.t in (13, 10):
        a = vec.v.astype(float); a[vec.v == NA_INT] = np.nan
        lev = vec.attr('levels')
        if lev is not None:
            labels = np.array(lev.v + [None], dtype=object)
            idx = np.where(np.isnan(a), len(lev.v), a - 1).astype(int)
            return pd.Series(labels[idx])
        if vec.t == 10: return pd.Series(np.where(np.isnan(a), np.nan, a == 1), dtype=object)
        return pd.Series(a)
    if vec.t == 14: return pd.Series(vec.v)
    return pd.Series([None]*len(vec.v))

def read_rds(path):
    data = open(path, 'rb').read()
    if data[:6] == b'\xfd7zXZ\x00': data = lzma.decompress(data)
    elif data[:2] == b'\x1f\x8b': data = gzip.decompress(data)
    elif data[:3] == b'BZh': data = bz2.decompress(data)
    r = R(data)
    assert r.raw(2) == b'X\n'
    ver = r.i(); r.i(); r.i()
    if ver == 3:
        n = r.i(); r.raw(n)
    obj = r.item()
    names = obj.attr('names').v
    cols = {}
    for nm, col in zip(names, obj.v):
        if isinstance(col, Vec): cols[nm] = to_series(col)
    return pd.DataFrame(cols)

if __name__ == '__main__':
    df = read_rds(sys.argv[1])
    print(df.shape); print(list(df.columns))
