# Strange Fragmentation Kaon Tagging for |Vub| Measurement

A machine learning pipeline for improving background rejection 
in B_s → Kμν decays at LHCb, using strange fragmentation 
tracks as a novel discriminating handle.

## Motivation

The CKM matrix element |Vub| has a longstanding tension between 
inclusive and exclusive measurements. 
This project takes advantage an unused property of B_s⁰ decays:because 
the B_s⁰ carries a strange quark, its fragmentation environment 
preferentially produces charged kaons. Background B⁺/B⁰ decays 
do not. This difference is not used by any existing analysis.

## Data

Simulated proton-proton collisions at √s = 13 TeV
- LHCb detector, Run 2 conditions (2015–2018)
- >11.5 million reconstructed tracks (Monte Carlo)
- Data obtained from CERN — not included in this repository

To run this code, place your `.root` file in the `data/` folder 
and update the path in `src/config.py`.

## Acknowledgements

Data: CERN / LHCb Collaboration  
Mentor: Dr. Basem Khanji  
Institution: Ibn Rushd National Academy, Jordan
