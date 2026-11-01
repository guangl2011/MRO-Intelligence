"""MRO-Intelligence MVP.

A read-only intelligence layer over mid-size distributor ERPs:

- ``mro_intelligence.ingest``  — CSV/SFTP batch connectors (P21/Eclipse/Infor templates)
- ``mro_intelligence.catalog`` — M1: SKU normalization, attribute extraction, dedup
- ``mro_intelligence.graph``   — M2: versioned equipment→failure→repair→part graph
- ``mro_intelligence.demand``  — M3: intermittent long-tail demand sensing
- ``mro_intelligence.health``  — SKU health report, ROI + counterfactual simulation
- ``mro_intelligence.synthetic`` — synthetic AXON-like dataset generator
- ``mro_intelligence.eval``    — NSF success-metric evaluation harness

Scope guard: this MVP never writes to a tenant ERP. All recommendations surface
through platform reports and exports; ERP write-back is out of scope by design
(blueprint locked decision).
"""

__version__ = "0.1.0"
