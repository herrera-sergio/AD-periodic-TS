# Anomaly detection in quasi-periodic energy consumption data series: a comparison of algorithms

The diffusion of domotics solutions and of smart appliances and meters enables the monitoring of energy consumption at a very fine level and the development of forecasting and diagnostic applications. Anomaly detection (AD) in energy consumption data streams helps identify data points or intervals in which the behavior of an appliance deviates from normality, which may prevent energy losses and break downs. Many statistical and learning approaches have been applied to the task, but the need remains of comparing their performances with data sets of different characteristics. This paper focuses on anomaly detection on quasi-periodic energy consumption data series and contrasts 12 statistical and machine learning algorithms on a data set containing the power consumption signals of fridges. The assessment also evaluates the impact of the length of the series used for training and of the size of the sliding window employed to detect the anomalies. The generalization ability of the top five methods is also evaluated by applying them to an appliance different from that used for training. 

This repository contains the material of the paper, the material is as follows:
* Performance folder: contains the summary of the performances of the 144 experiments.
* Figures folder: contains full-size version of the figures included in the article.
* All Figures folder: contains full-size version of the figures that were not included in the article but that were generated to respond Q2 and Q3.
* Code Folder: Contains the code used for the experiments

## Cite this work
If you use our code or wish to refer it, please use the following BibTex entry.

```
@article{zangrando2022anomaly,
  title={Anomaly detection in quasi-periodic energy consumption data series: a comparison of algorithms},
  author={Zangrando, Niccol{\`o} and Fraternali, Piero and Petri, Marco and Pinciroli Vago, Nicol{\`o} Oreste and Herrera Gonz{\'a}lez, Sergio Luis},
  journal={Energy Informatics},
  volume={5},
  number={4},
  pages={1--22},
  year={2022},
  publisher={SpringerOpen}
}
```
