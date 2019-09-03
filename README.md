# Nion_inference
Inferring ionizing emissivity from reionization constraints

This repository contains the codes and posterior chains required to reproduce the plots in [Mason+19b](https://ui.adsabs.harvard.edu/abs/2019arXiv190711332M/abstract)

## Posterior draws

Draws from the posterior are located in `chains/res_*_U[-1,1]_49-53_samples.npy` and can be loaded as `numpy` arrays using `np.load`. These are N x 11 arrays when N is the number of samples and 11 is the number of redshift bins.

## Notebook description

- `Nion_plots.ipynb` makes the paper plots. Unusual prerequisites are: `seaborn, astropy`
- `nonparametric_Nion_inference.ipynb` does the inference. It takes a while... (15 hours on 7 3.7 GHz cores). The runs used in the paper are saved as posterior draws in `chains/` and are used as input to the plotting notebook, so you don't need to rerun it. Unusual prerequisites are: `dynesty, astropy`

## Credits

If you use the posterior draws please cite [Mason+19](https://ui.adsabs.harvard.edu/abs/2019arXiv190711332M/abstract) and also the following papers which contain constraints used in the likelihood:

- [Planck 2018](https://ui.adsabs.harvard.edu/abs/2018arXiv180706209P/abstract) CMB optical depth
- Galaxy Lyman-alpha EW evolution ([Mason+18](https://ui.adsabs.harvard.edu/abs/2018ApJ...856....2M/abstract),[+19a](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.3947M/abstract), [Hoag+19](https://ui.adsabs.harvard.edu/abs/2019arXiv190109001H/abstract))
- Lyman-alpha emitter clustering ([Sobacchi & Mesinger 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.453.1843S/abstract))
- QSO Lyman-alpha damping wings ([Davies+18](https://ui.adsabs.harvard.edu/abs/2018ApJ...864..142D/abstract), [Greig+19](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.5094G/abstract))


