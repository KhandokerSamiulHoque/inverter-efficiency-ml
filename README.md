# Machine Learning for Inverter Efficiency, with a Data-Leakage Audit

This is the first stage of an independent research project on machine learning for inverter performance and fault diagnosis, by Khandoker Samiul Hoque (ORCID 0009-0008-0756-2395).

## What it does
The code predicts inverter efficiency classes and values from the public California Energy Commission (CEC) inverter database, with the Sandia inverter model parameters retrieved through pvlib. It checks deliberately for **data leakage**, where a model looks accurate only because the answer is already encoded in its inputs.
- Classification F1 is **0.9994** when the target leaks into the inputs.
- F1 is **0.9975** with the target's algebraic components as inputs.
- F1 is **0.9736** with only the raw Sandia parameters.
- A regression on the raw parameters reaches **R-squared 0.928**.

These results show both what an honest model achieves and how easily leakage inflates apparent accuracy.

## Reproduce
```bash
pip install -r requirements.txt
python 04_CEC_Inverter_Leakage_Audit/code/analysis.py
```
The script reads `results/cec_inverter_snapshot.csv`, the exact data snapshot used. If that file is removed, the script rebuilds it from pvlib.

**Rerun on 2026-10-15:** the cross-validation and holdout tables matched the saved results exactly. The largest difference was 2e-16, which is floating-point rounding.

## Limits
- The target is computed from public Sandia-model parameters. It is **not** an independent laboratory measurement and **not** a UL 1741 pass/fail outcome.
- Inclusion in the CEC database is not NRTL certification status.
- This work uses only public data. It uses no employer data, equipment, or procedures.

## Next stages
- A public inverter fault taxonomy (from January 2027).
- Physics-based fault simulation.
- Interpretable models that rank the likely causes of inverter anomalies.

License: MIT (code). CEC/Sandia data are public.
