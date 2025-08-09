# ExoplANNET-2.0
A convolutional neural network to detect planetary signals on radial velocity time series


# Abstract

As my thesis project, i worked on a [previous work](https://www.aanda.org/articles/aa/full_html/2023/09/aa46417-23/aa46417-23.html) done by Luis A. Nieto and Rodrigo F. Diaz. 

On the thesis, we simulated radial velocity (RV) time series on multiple real calendars of observations, typical on the search of exoplanets. The RV time series included both stellar activity and planetary signals. We computed the GLS periodogram for each one and trained a neural network (NN) to detect if the maximum peak of each one was due to the presence of a planet companion.

<img width="1826" height="776" alt="Image" src="https://github.com/user-attachments/assets/009cb1a4-529a-49ec-ac5d-8fa65efe3c9e" />

Comparing with the false alarm positive (FAP) method, we obtained 52% false positives without increasing the false negatives. We also tested it on real data, obtaining a good performance. We showed that the NN method may be able to overtake the statistically method currently used.

## Upgrades made with respect to the previous work:
- Generalized the calendar observations to apply the neural network on real time series, obtaining a good performance on real data.
- Trained with an extended range of planetary periods.
- FAP exclusively calculated with the Montecarlo method (for robustess and confiability).
- Added an attention layer in seek for interpretability that also resulted on a better overall performance.
