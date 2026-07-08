"""Re-export the shared in-process predictor for the serving layer.

The implementation lives in the core package (`occuwise.predictor`) so the CLI
(`occuwise.predict`) and the web app share exactly one code path.
"""

from occuwise.predictor import InProcessPredictor, InProcessPredictor as PocPredictor

__all__ = ["InProcessPredictor", "PocPredictor"]
