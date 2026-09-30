"""Literal, deterministic value replacement; no evaluated expressions or regex."""
from .base import Transform
import pandas as pd
import math

class FindReplace(Transform):
    def __init__(self, columns, find, replace, mode='exact'):
        super().__init__(columns)
        if not columns:raise ValueError('Choose columns for find and replace.')
        if mode not in {'exact','substring'}:raise ValueError('Choose exact or substring replacement.')
        if not isinstance(find,(str,int,float,bool)) or not isinstance(replace,(str,int,float,bool,type(None))):raise ValueError('Find and replace must be scalar values.')
        if any(isinstance(v,float) and not math.isfinite(v) for v in (find,replace)):raise ValueError('Replacement values must be finite.')
        if mode=='substring' and (not isinstance(find,str) or not find or not isinstance(replace,str)):raise ValueError('Substring replacement requires nonempty search text and replacement text.')
        self.find,self.replace,self.mode=find,replace,mode

    def apply(self, frame):
        self.output_columns(frame.columns);out=frame.copy()
        for c in self.columns:
            if self.mode=='substring':out[c]=out[c].map(lambda v:v.replace(self.find,self.replace) if isinstance(v,str) else v)
            else:
                # Object dtype permits explicit cross-type replacement without silent casting.
                values=out[c].astype(object);mask=values.eq(self.find).fillna(False)
                values.loc[mask]=self.replace;out[c]=values.infer_objects()
        return out
