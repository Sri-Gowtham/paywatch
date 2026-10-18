"""PayWatch UPI toolkit: a bring-your-own-data path for real UPI traffic (shadow-mode pilots).

Everything here runs on numpy + lightgbm (+ pydantic/fastapi for serving). Training and serving share one feature code path
(`features.UpiFeaturizer`), so the features a model trains on are produced exactly the way they are produced live.
No real UPI data ships with this repo; `synthetic.py` makes clearly labelled synthetic traffic for tests and demos only.
"""
